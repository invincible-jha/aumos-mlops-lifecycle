"""Model packager adapter for aumos-mlops-lifecycle.

Handles model serialisation to multiple formats (pickle, safetensors, ONNX),
Dockerfile generation for model serving, dependency extraction, multi-stage
Docker build orchestration, and image push to a container registry.

Configuration:
    AUMOS_MLOPS_DOCKER_REGISTRY        — Container registry hostname
    AUMOS_MLOPS_DOCKER_REGISTRY_TOKEN  — Registry authentication token
    AUMOS_MLOPS_SERVING_BASE_IMAGE     — Base image for the serving stage
"""

import asyncio
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

_DOCKERFILE_TEMPLATE = """\
# syntax=docker/dockerfile:1.6
# Stage 1: build dependencies
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: runtime
FROM {base_image} AS runtime
LABEL maintainer="AumOS MLOps <mlops@muvera.ai>"
LABEL model.name="{model_name}"
LABEL model.version="{model_version}"
LABEL model.framework="{framework}"

WORKDIR /app
COPY --from=builder /install /usr/local
COPY model/ /app/model/
COPY serve.py /app/serve.py

ENV MODEL_PATH=/app/model
ENV MODEL_VERSION={model_version}
ENV PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["python", "-u", "serve.py"]
"""

_SERVE_PY_TEMPLATE = """\
\"\"\"Auto-generated serving entrypoint for {model_name} v{model_version}.\"\"\"
import os
import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model")
PORT = int(os.environ.get("PORT", 8080))


def load_model():
    import pickle
    model_file = os.path.join(MODEL_PATH, "model.pkl")
    if os.path.exists(model_file):
        with open(model_file, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError(f"Model not found at {{model_file}}")


MODEL = load_model()
logger.info("Model loaded: {model_name} v{model_version}")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({{"status": "ok"}}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/predict":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            prediction = MODEL.predict([body["features"]])
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({{"prediction": prediction[0]}}).encode())


if __name__ == "__main__":
    logger.info("Starting server on port %d", PORT)
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
"""


class ModelPackager:
    """Docker-based model packager with multi-format serialisation support.

    Generates Dockerfiles, serialises models, extracts Python dependencies,
    builds multi-stage Docker images, and pushes them to a container registry.
    All subprocess calls are run via asyncio to avoid blocking the event loop.

    Args:
        registry_url: Container registry hostname (e.g. registry.muvera.ai).
        registry_token: Authentication token for the container registry.
        serving_base_image: Base image for the model serving runtime stage.
    """

    def __init__(
        self,
        registry_url: str,
        registry_token: str,
        serving_base_image: str = "python:3.11-slim",
    ) -> None:
        """Initialise the model packager.

        Args:
            registry_url: Container registry hostname.
            registry_token: Registry push authentication token.
            serving_base_image: Base container image for the serving runtime.
        """
        self._registry_url = registry_url
        self._registry_token = registry_token
        self._base_image = serving_base_image

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    async def serialise_model(
        self,
        model: Any,
        output_dir: str,
        format: str = "pickle",
        model_name: str = "model",
    ) -> str:
        """Serialise a model to the specified format.

        Supported formats:
          - "pickle"       — Python pickle (fastest, not cross-language)
          - "safetensors"  — Safe, language-agnostic tensor serialisation
          - "onnx"         — Open Neural Network Exchange format

        Args:
            model: The Python model object to serialise.
            output_dir: Directory where the serialised artefact is written.
            format: Serialisation format. One of "pickle", "safetensors", "onnx".
            model_name: Base filename for the output artefact (without extension).

        Returns:
            Absolute path to the serialised model file.
        """
        loop = asyncio.get_event_loop()
        output_path: str = await loop.run_in_executor(
            None,
            partial(
                self._serialise_sync,
                model=model,
                output_dir=output_dir,
                format=format,
                model_name=model_name,
            ),
        )
        return output_path

    def _serialise_sync(
        self,
        model: Any,
        output_dir: str,
        format: str,
        model_name: str,
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)

        if format == "pickle":
            import pickle

            output_path = os.path.join(output_dir, f"{model_name}.pkl")
            with open(output_path, "wb") as file_handle:
                pickle.dump(model, file_handle, protocol=pickle.HIGHEST_PROTOCOL)

        elif format == "safetensors":
            from safetensors.torch import save_model  # type: ignore[import-untyped]

            output_path = os.path.join(output_dir, f"{model_name}.safetensors")
            save_model(model, output_path)

        elif format == "onnx":
            import torch  # type: ignore[import-untyped]

            output_path = os.path.join(output_dir, f"{model_name}.onnx")
            dummy_input = torch.zeros(1, 3, 224, 224)
            torch.onnx.export(model, dummy_input, output_path, opset_version=17)

        else:
            raise ValueError(f"Unsupported serialisation format: {format}")

        logger.info("Model serialised", format=format, output_path=output_path)
        return output_path

    # ------------------------------------------------------------------ #
    # Dependency extraction                                                #
    # ------------------------------------------------------------------ #

    async def extract_dependencies(self, output_path: str) -> str:
        """Capture the current Python environment as a requirements.txt.

        Runs `pip freeze` in a thread pool and writes the output to
        output_path.

        Args:
            output_path: Absolute path where requirements.txt should be written.

        Returns:
            Content of the generated requirements.txt as a string.
        """
        loop = asyncio.get_event_loop()
        requirements: str = await loop.run_in_executor(
            None, partial(self._freeze_requirements_sync, output_path=output_path)
        )
        return requirements

    @staticmethod
    def _freeze_requirements_sync(output_path: str) -> str:
        result = subprocess.run(["pip", "freeze"], capture_output=True, text=True, check=True)
        requirements = result.stdout
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as requirements_file:
            requirements_file.write(requirements)
        logger.info("Requirements extracted", line_count=requirements.count("\n"))
        return requirements

    # ------------------------------------------------------------------ #
    # Dockerfile generation                                                #
    # ------------------------------------------------------------------ #

    async def generate_dockerfile(
        self,
        model_name: str,
        model_version: str,
        framework: str,
        output_dir: str,
    ) -> str:
        """Generate a multi-stage Dockerfile for model serving.

        Also generates an accompanying serve.py health check endpoint.

        Args:
            model_name: Human-readable model name (stored as image label).
            model_version: Semantic version tag for the model.
            framework: ML framework identifier (sklearn, pytorch, transformers).
            output_dir: Directory where Dockerfile and serve.py are written.

        Returns:
            Absolute path to the generated Dockerfile.
        """
        dockerfile_content = _DOCKERFILE_TEMPLATE.format(
            base_image=self._base_image,
            model_name=model_name,
            model_version=model_version,
            framework=framework,
        )
        serve_content = _SERVE_PY_TEMPLATE.format(
            model_name=model_name,
            model_version=model_version,
        )

        os.makedirs(output_dir, exist_ok=True)
        dockerfile_path = os.path.join(output_dir, "Dockerfile")
        serve_path = os.path.join(output_dir, "serve.py")

        with open(dockerfile_path, "w") as dockerfile:
            dockerfile.write(dockerfile_content)
        with open(serve_path, "w") as serve_file:
            serve_file.write(serve_content)

        logger.info("Dockerfile generated", dockerfile_path=dockerfile_path, model=model_name)
        return dockerfile_path

    # ------------------------------------------------------------------ #
    # Docker build and push                                                #
    # ------------------------------------------------------------------ #

    async def build_image(
        self,
        build_context_dir: str,
        model_name: str,
        model_version: str,
        tenant_id: str,
    ) -> str:
        """Build a Docker image from the model package build context.

        Args:
            build_context_dir: Directory containing Dockerfile and model artefacts.
            model_name: Model name used to construct the image tag.
            model_version: Model version tag.
            tenant_id: Tenant UUID string embedded in the image tag namespace.

        Returns:
            Full image tag string (registry_url/tenant_id/model_name:version).
        """
        image_tag = self._build_image_tag(model_name=model_name, model_version=model_version, tenant_id=tenant_id)
        logger.info("Building Docker image", image_tag=image_tag, context=build_context_dir)

        process = await asyncio.create_subprocess_exec(
            "docker",
            "build",
            "--tag",
            image_tag,
            "--label",
            f"aumos.tenant_id={tenant_id}",
            "--label",
            f"aumos.model_version={model_version}",
            build_context_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await process.communicate()
        if process.returncode != 0:
            logger.error("Docker build failed", stderr=stderr_bytes.decode())
            raise RuntimeError(f"Docker build failed for {image_tag}: {stderr_bytes.decode()}")

        logger.info("Docker image built", image_tag=image_tag)
        return image_tag

    async def push_image(self, image_tag: str) -> str:
        """Push a built Docker image to the configured container registry.

        Authenticates to the registry using the configured token before pushing.

        Args:
            image_tag: Full image tag string (as returned by build_image).

        Returns:
            Image digest string (sha256:...) after successful push.
        """
        logger.info("Pushing Docker image", image_tag=image_tag)

        # Registry login
        login_process = await asyncio.create_subprocess_exec(
            "docker",
            "login",
            self._registry_url,
            "--username",
            "token",
            "--password-stdin",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, login_stderr = await login_process.communicate(input=self._registry_token.encode())
        if login_process.returncode != 0:
            raise RuntimeError(f"Registry login failed: {login_stderr.decode()}")

        # Push
        push_process = await asyncio.create_subprocess_exec(
            "docker",
            "push",
            image_tag,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        push_stdout, push_stderr = await push_process.communicate()
        if push_process.returncode != 0:
            raise RuntimeError(f"Docker push failed for {image_tag}: {push_stderr.decode()}")

        # Extract digest from push output
        digest = self._extract_digest(push_stdout.decode())
        logger.info("Docker image pushed", image_tag=image_tag, digest=digest)
        return digest

    # ------------------------------------------------------------------ #
    # Full packaging pipeline                                              #
    # ------------------------------------------------------------------ #

    async def package_and_push(
        self,
        model: Any,
        model_name: str,
        model_version: str,
        framework: str,
        serialisation_format: str,
        tenant_id: str,
    ) -> dict[str, Any]:
        """End-to-end model packaging: serialise → Dockerfile → build → push.

        Args:
            model: Python model object to package.
            model_name: Human-readable model name.
            model_version: Semantic version string.
            framework: ML framework (sklearn, pytorch, transformers).
            serialisation_format: "pickle" | "safetensors" | "onnx".
            tenant_id: Tenant UUID string.

        Returns:
            Package result dict with image_tag, digest, and model_path.
        """
        with tempfile.TemporaryDirectory() as build_dir:
            model_dir = os.path.join(build_dir, "model")
            os.makedirs(model_dir)

            model_path = await self.serialise_model(model, model_dir, serialisation_format)
            await self.extract_dependencies(os.path.join(build_dir, "requirements.txt"))
            await self.generate_dockerfile(model_name, model_version, framework, build_dir)

            image_tag = await self.build_image(build_dir, model_name, model_version, tenant_id)
            digest = await self.push_image(image_tag)

        return {
            "image_tag": image_tag,
            "digest": digest,
            "model_path": model_path,
            "packaged_at": datetime.now(tz=timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _build_image_tag(self, model_name: str, model_version: str, tenant_id: str) -> str:
        safe_tenant = tenant_id.replace("-", "")[:16]
        safe_name = model_name.lower().replace(" ", "-").replace("_", "-")
        return f"{self._registry_url}/{safe_tenant}/{safe_name}:{model_version}"

    @staticmethod
    def _extract_digest(push_output: str) -> str:
        for line in push_output.splitlines():
            if "digest:" in line.lower() or "sha256:" in line:
                parts = line.split()
                for part in parts:
                    if part.startswith("sha256:"):
                        return part
        return "sha256:unknown"
