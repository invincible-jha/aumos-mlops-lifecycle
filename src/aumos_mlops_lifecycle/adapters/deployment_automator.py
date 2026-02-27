"""Deployment automator adapter for aumos-mlops-lifecycle.

Manages canary rollouts, A/B test configurations, automatic rollback on
metric degradation, and traffic shifting via Kubernetes Deployments and
Istio VirtualService resources.

Configuration:
    AUMOS_MLOPS_K8S_NAMESPACE       — Kubernetes namespace for serving workloads
    AUMOS_MLOPS_ROLLBACK_ERROR_PCT  — Error rate (%) threshold for auto-rollback
    AUMOS_MLOPS_CANARY_STEP_PCT     — Default canary traffic increment per step
"""

import asyncio
import uuid
from datetime import datetime, timezone
from functools import partial
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

_DEFAULT_NAMESPACE = "aumos-serving"
_DEFAULT_ERROR_THRESHOLD = 5.0  # percent


class DeploymentAutomator:
    """Kubernetes + Istio deployment automator with canary and A/B support.

    Creates and manages K8s Deployments and Istio VirtualService resources
    to implement traffic splitting for canary rollouts and A/B experiments.
    Monitors real-time error rates and triggers automatic rollbacks when
    the configured threshold is breached.

    Args:
        namespace: Kubernetes namespace for serving resources.
        kubeconfig_path: Optional kubeconfig path for non-cluster environments.
        rollback_error_threshold_pct: Error rate percentage that triggers rollback.
        canary_step_pct: Default traffic increment per canary step.
    """

    def __init__(
        self,
        namespace: str = _DEFAULT_NAMESPACE,
        kubeconfig_path: str | None = None,
        rollback_error_threshold_pct: float = _DEFAULT_ERROR_THRESHOLD,
        canary_step_pct: int = 10,
    ) -> None:
        """Initialise the deployment automator.

        Args:
            namespace: Kubernetes namespace for serving workloads.
            kubeconfig_path: Optional kubeconfig path.
            rollback_error_threshold_pct: Percentage at which to trigger rollback.
            canary_step_pct: Default traffic shift increment per step.
        """
        self._namespace = namespace
        self._kubeconfig_path = kubeconfig_path
        self._error_threshold = rollback_error_threshold_pct
        self._canary_step_pct = canary_step_pct
        self._k8s_apps_api: Any | None = None
        self._k8s_custom_api: Any | None = None
        # In-process deployment state: deployment_id → state dict
        self._deployments: dict[str, dict[str, Any]] = {}

    def _get_apis(self) -> tuple[Any, Any]:
        """Lazily initialise Kubernetes AppsV1Api and CustomObjectsApi.

        Returns:
            Tuple of (AppsV1Api, CustomObjectsApi) instances.
        """
        if self._k8s_apps_api is None or self._k8s_custom_api is None:
            try:
                from kubernetes import client as k8s_client, config as k8s_config  # type: ignore[import-untyped]

                if self._kubeconfig_path:
                    k8s_config.load_kube_config(config_file=self._kubeconfig_path)
                else:
                    try:
                        k8s_config.load_incluster_config()
                    except k8s_config.ConfigException:
                        k8s_config.load_kube_config()

                self._k8s_apps_api = k8s_client.AppsV1Api()
                self._k8s_custom_api = k8s_client.CustomObjectsApi()
            except ImportError as exc:
                msg = "kubernetes package is required: pip install kubernetes>=28.0.0"
                raise ImportError(msg) from exc
        return self._k8s_apps_api, self._k8s_custom_api

    # ------------------------------------------------------------------ #
    # Canary deployment                                                    #
    # ------------------------------------------------------------------ #

    async def create_canary_deployment(
        self,
        model_name: str,
        stable_version: str,
        canary_version: str,
        initial_canary_pct: int,
        tenant_id: str,
        image: str,
        replicas: int = 2,
    ) -> dict[str, Any]:
        """Create a canary deployment with initial traffic split.

        Creates a K8s Deployment for the canary version and an Istio
        VirtualService to split traffic between stable and canary pods.

        Args:
            model_name: Logical model name used to construct K8s resource names.
            stable_version: Current production version tag.
            canary_version: New candidate version tag to route canary traffic to.
            initial_canary_pct: Initial percentage of traffic routed to canary.
            tenant_id: Tenant UUID string for labelling and isolation.
            image: Container image URI for the canary pod.
            replicas: Number of replica pods for the canary Deployment.

        Returns:
            Deployment state dict with deployment_id, model_name, traffic_split,
            and status.
        """
        deployment_id = str(uuid.uuid4())
        canary_name = f"{model_name}-canary-{canary_version}"
        stable_pct = 100 - initial_canary_pct

        logger.info(
            "Creating canary deployment",
            deployment_id=deployment_id,
            canary_version=canary_version,
            initial_canary_pct=initial_canary_pct,
            tenant_id=tenant_id,
        )

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            partial(
                self._create_k8s_deployment_sync,
                name=canary_name,
                image=image,
                version=canary_version,
                model_name=model_name,
                tenant_id=tenant_id,
                replicas=replicas,
            ),
        )
        await loop.run_in_executor(
            None,
            partial(
                self._apply_virtual_service_sync,
                model_name=model_name,
                stable_version=stable_version,
                canary_version=canary_version,
                stable_pct=stable_pct,
                canary_pct=initial_canary_pct,
                tenant_id=tenant_id,
            ),
        )

        state: dict[str, Any] = {
            "deployment_id": deployment_id,
            "model_name": model_name,
            "stable_version": stable_version,
            "canary_version": canary_version,
            "traffic_split": {"stable": stable_pct, "canary": initial_canary_pct},
            "status": "canary_in_progress",
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "tenant_id": tenant_id,
            "canary_name": canary_name,
        }
        self._deployments[deployment_id] = state
        return state

    def _create_k8s_deployment_sync(
        self,
        name: str,
        image: str,
        version: str,
        model_name: str,
        tenant_id: str,
        replicas: int,
    ) -> None:
        from kubernetes import client as k8s_client  # type: ignore[import-untyped]

        apps_api, _ = self._get_apis()

        labels = {
            "app": model_name,
            "version": version,
            "aumos.ai/tenant": tenant_id[:63],
        }

        container = k8s_client.V1Container(
            name="model-server",
            image=image,
            ports=[k8s_client.V1ContainerPort(container_port=8080)],
            readiness_probe=k8s_client.V1Probe(
                http_get=k8s_client.V1HTTPGetAction(path="/health", port=8080),
                initial_delay_seconds=10,
                period_seconds=5,
            ),
        )

        deployment = k8s_client.V1Deployment(
            metadata=k8s_client.V1ObjectMeta(name=name, namespace=self._namespace, labels=labels),
            spec=k8s_client.V1DeploymentSpec(
                replicas=replicas,
                selector=k8s_client.V1LabelSelector(match_labels={"app": model_name, "version": version}),
                template=k8s_client.V1PodTemplateSpec(
                    metadata=k8s_client.V1ObjectMeta(labels=labels),
                    spec=k8s_client.V1PodSpec(containers=[container]),
                ),
            ),
        )

        apps_api.create_namespaced_deployment(namespace=self._namespace, body=deployment)
        logger.info("K8s Deployment created", name=name)

    def _apply_virtual_service_sync(
        self,
        model_name: str,
        stable_version: str,
        canary_version: str,
        stable_pct: int,
        canary_pct: int,
        tenant_id: str,
    ) -> None:
        _, custom_api = self._get_apis()

        virtual_service = {
            "apiVersion": "networking.istio.io/v1beta1",
            "kind": "VirtualService",
            "metadata": {
                "name": model_name,
                "namespace": self._namespace,
                "labels": {"aumos.ai/tenant": tenant_id[:63]},
            },
            "spec": {
                "hosts": [model_name],
                "http": [
                    {
                        "route": [
                            {
                                "destination": {"host": model_name, "subset": stable_version},
                                "weight": stable_pct,
                            },
                            {
                                "destination": {"host": model_name, "subset": canary_version},
                                "weight": canary_pct,
                            },
                        ]
                    }
                ],
            },
        }

        try:
            custom_api.replace_namespaced_custom_object(
                group="networking.istio.io",
                version="v1beta1",
                namespace=self._namespace,
                plural="virtualservices",
                name=model_name,
                body=virtual_service,
            )
        except Exception:
            custom_api.create_namespaced_custom_object(
                group="networking.istio.io",
                version="v1beta1",
                namespace=self._namespace,
                plural="virtualservices",
                body=virtual_service,
            )
        logger.info("Istio VirtualService applied", model_name=model_name, stable=stable_pct, canary=canary_pct)

    # ------------------------------------------------------------------ #
    # A/B test configuration                                               #
    # ------------------------------------------------------------------ #

    async def create_ab_test(
        self,
        model_name: str,
        variant_a_version: str,
        variant_b_version: str,
        variant_a_pct: int,
        tenant_id: str,
        image_a: str,
        image_b: str,
        replicas: int = 2,
    ) -> dict[str, Any]:
        """Configure an A/B test between two model variants.

        Creates K8s Deployments for both variants and an Istio VirtualService
        routing traffic according to the specified percentage split.

        Args:
            model_name: Logical model name.
            variant_a_version: Version tag for variant A.
            variant_b_version: Version tag for variant B.
            variant_a_pct: Percentage of traffic routed to variant A.
            tenant_id: Tenant UUID string.
            image_a: Container image for variant A.
            image_b: Container image for variant B.
            replicas: Replica count for each variant Deployment.

        Returns:
            A/B test deployment state dict.
        """
        deployment_id = str(uuid.uuid4())
        variant_b_pct = 100 - variant_a_pct

        logger.info(
            "Creating A/B test",
            deployment_id=deployment_id,
            model_name=model_name,
            variant_a=variant_a_version,
            variant_b=variant_b_version,
            split=f"{variant_a_pct}/{variant_b_pct}",
        )

        loop = asyncio.get_event_loop()
        for version, image in [(variant_a_version, image_a), (variant_b_version, image_b)]:
            await loop.run_in_executor(
                None,
                partial(
                    self._create_k8s_deployment_sync,
                    name=f"{model_name}-{version}",
                    image=image,
                    version=version,
                    model_name=model_name,
                    tenant_id=tenant_id,
                    replicas=replicas,
                ),
            )

        await loop.run_in_executor(
            None,
            partial(
                self._apply_virtual_service_sync,
                model_name=model_name,
                stable_version=variant_a_version,
                canary_version=variant_b_version,
                stable_pct=variant_a_pct,
                canary_pct=variant_b_pct,
                tenant_id=tenant_id,
            ),
        )

        state: dict[str, Any] = {
            "deployment_id": deployment_id,
            "model_name": model_name,
            "variant_a": variant_a_version,
            "variant_b": variant_b_version,
            "traffic_split": {"variant_a": variant_a_pct, "variant_b": variant_b_pct},
            "status": "ab_test_running",
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "tenant_id": tenant_id,
        }
        self._deployments[deployment_id] = state
        return state

    # ------------------------------------------------------------------ #
    # Traffic shifting                                                     #
    # ------------------------------------------------------------------ #

    async def shift_traffic(
        self,
        deployment_id: str,
        canary_pct: int,
    ) -> dict[str, Any]:
        """Shift canary traffic to a new percentage.

        Updates the Istio VirtualService weights and the in-process state.
        Marks the deployment as completed when canary reaches 100%.

        Args:
            deployment_id: Deployment state ID.
            canary_pct: New canary traffic percentage (0–100).

        Returns:
            Updated deployment state dict.

        Raises:
            KeyError: If the deployment_id is not found.
        """
        state = self._deployments[deployment_id]
        stable_pct = 100 - canary_pct

        logger.info(
            "Shifting canary traffic",
            deployment_id=deployment_id,
            canary_pct=canary_pct,
        )

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            partial(
                self._apply_virtual_service_sync,
                model_name=state["model_name"],
                stable_version=state["stable_version"],
                canary_version=state["canary_version"],
                stable_pct=stable_pct,
                canary_pct=canary_pct,
                tenant_id=state["tenant_id"],
            ),
        )

        state["traffic_split"] = {"stable": stable_pct, "canary": canary_pct}
        if canary_pct >= 100:
            state["status"] = "completed"
        return state

    # ------------------------------------------------------------------ #
    # Auto-rollback                                                        #
    # ------------------------------------------------------------------ #

    async def check_and_rollback_if_degraded(
        self,
        deployment_id: str,
        current_error_rate_pct: float,
    ) -> dict[str, Any]:
        """Evaluate current error rate and rollback if it exceeds threshold.

        Args:
            deployment_id: Deployment state ID.
            current_error_rate_pct: Current error rate as a percentage (0–100).

        Returns:
            Updated deployment state dict. If rollback was triggered, status
            will be "rolled_back".
        """
        state = self._deployments.get(deployment_id)
        if state is None:
            raise KeyError(f"Deployment '{deployment_id}' not found")

        if current_error_rate_pct >= self._error_threshold:
            logger.warning(
                "Error threshold exceeded — triggering auto-rollback",
                deployment_id=deployment_id,
                error_rate_pct=current_error_rate_pct,
                threshold_pct=self._error_threshold,
            )
            return await self.rollback_deployment(deployment_id, reason="auto: error rate threshold exceeded")

        return state

    async def rollback_deployment(self, deployment_id: str, reason: str) -> dict[str, Any]:
        """Rollback a deployment by restoring 100% traffic to the stable version.

        Args:
            deployment_id: Deployment state ID.
            reason: Human-readable rollback reason for audit logging.

        Returns:
            Updated deployment state dict with status "rolled_back".
        """
        state = self._deployments[deployment_id]
        logger.info("Rolling back deployment", deployment_id=deployment_id, reason=reason)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            partial(
                self._apply_virtual_service_sync,
                model_name=state["model_name"],
                stable_version=state.get("stable_version", "stable"),
                canary_version=state.get("canary_version", "canary"),
                stable_pct=100,
                canary_pct=0,
                tenant_id=state["tenant_id"],
            ),
        )

        state["traffic_split"] = {"stable": 100, "canary": 0}
        state["status"] = "rolled_back"
        state["rollback_reason"] = reason
        state["rolled_back_at"] = datetime.now(tz=timezone.utc).isoformat()
        return state

    # ------------------------------------------------------------------ #
    # Status tracking                                                      #
    # ------------------------------------------------------------------ #

    async def get_deployment_status(self, deployment_id: str) -> dict[str, Any] | None:
        """Get the current state of a deployment.

        Args:
            deployment_id: Deployment state ID.

        Returns:
            Deployment state dict or None if not found.
        """
        return self._deployments.get(deployment_id)

    async def list_active_deployments(self, tenant_id: str) -> list[dict[str, Any]]:
        """List all non-terminal deployments for a tenant.

        Args:
            tenant_id: Tenant UUID string to filter by.

        Returns:
            List of active (non-rolled_back, non-completed) deployment state dicts.
        """
        terminal_statuses = {"completed", "rolled_back", "failed"}
        return [
            state
            for state in self._deployments.values()
            if state["tenant_id"] == tenant_id and state["status"] not in terminal_statuses
        ]
