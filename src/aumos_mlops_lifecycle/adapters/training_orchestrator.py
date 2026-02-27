"""Training orchestrator adapter for aumos-mlops-lifecycle.

Manages Kubernetes Jobs for distributed ML training workloads. Supports
PyTorch DDP and Horovod multi-node configurations, GPU resource allocation,
pod status monitoring, auto-scaling based on queue depth, and graceful
cleanup of completed training resources.

Configuration:
    AUMOS_MLOPS_K8S_NAMESPACE     — Kubernetes namespace for training jobs
    AUMOS_MLOPS_K8S_KUBECONFIG    — Optional kubeconfig path (uses in-cluster config if unset)
    AUMOS_MLOPS_TRAINING_IMAGE    — Default training container image
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from functools import partial
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

_DEFAULT_NAMESPACE = "aumos-training"
_JOB_LABEL_PREFIX = "aumos.ai/mlops"


class TrainingOrchestrator:
    """Kubernetes-backed distributed training job orchestrator.

    Creates K8s Jobs for PyTorch DDP and Horovod workloads, monitors pod
    status, handles auto-scaling queue signals, and recovers from pod
    failures via restart policies.

    Args:
        namespace: Kubernetes namespace where training jobs are created.
        kubeconfig_path: Path to kubeconfig file. If None, in-cluster
                         service account credentials are used.
        default_image: Default container image for training jobs.
    """

    def __init__(
        self,
        namespace: str = _DEFAULT_NAMESPACE,
        kubeconfig_path: str | None = None,
        default_image: str = "aumos/trainer:latest",
    ) -> None:
        """Initialise the training orchestrator.

        Args:
            namespace: Kubernetes namespace for training jobs.
            kubeconfig_path: Optional kubeconfig path for local development.
            default_image: Default training container image URI.
        """
        self._namespace = namespace
        self._kubeconfig_path = kubeconfig_path
        self._default_image = default_image
        self._k8s_batch_api: Any | None = None
        self._k8s_core_api: Any | None = None

    def _get_apis(self) -> tuple[Any, Any]:
        """Lazily initialise the Kubernetes API clients.

        Returns:
            Tuple of (BatchV1Api, CoreV1Api) Kubernetes client instances.
        """
        if self._k8s_batch_api is None or self._k8s_core_api is None:
            try:
                from kubernetes import client as k8s_client, config as k8s_config  # type: ignore[import-untyped]

                if self._kubeconfig_path:
                    k8s_config.load_kube_config(config_file=self._kubeconfig_path)
                else:
                    try:
                        k8s_config.load_incluster_config()
                    except k8s_config.ConfigException:
                        k8s_config.load_kube_config()

                self._k8s_batch_api = k8s_client.BatchV1Api()
                self._k8s_core_api = k8s_client.CoreV1Api()
            except ImportError as exc:
                msg = "kubernetes package is required: pip install kubernetes>=28.0.0"
                raise ImportError(msg) from exc
        return self._k8s_batch_api, self._k8s_core_api

    # ------------------------------------------------------------------ #
    # Job creation                                                         #
    # ------------------------------------------------------------------ #

    async def create_training_job(
        self,
        experiment_id: str,
        run_id: str,
        image: str | None,
        command: list[str],
        gpu_count: int,
        memory_gb: int,
        cpu_count: int,
        num_nodes: int,
        framework: str,
        env_vars: dict[str, str] | None,
        tenant_id: str,
    ) -> dict[str, Any]:
        """Create a Kubernetes Job to run a distributed training workload.

        Builds a K8s Job manifest with the correct resource requests, DDP or
        Horovod environment variables, and restartPolicy, then submits it via
        the Batch API.

        Args:
            experiment_id: Owning MLflow experiment ID (stored as label).
            run_id: MLflow run ID associated with this training job.
            image: Container image URI. If None uses self._default_image.
            command: Container entrypoint command list.
            gpu_count: Number of NVIDIA GPUs to allocate per pod.
            memory_gb: RAM limit in gigabytes per pod.
            cpu_count: CPU request per pod.
            num_nodes: Number of parallel worker pods (distributed training).
            framework: "pytorch_ddp" | "horovod" — controls env injection.
            env_vars: Additional environment variables to inject into pods.
            tenant_id: Tenant UUID string, stored as a K8s label.

        Returns:
            Job status dict with job_name, namespace, uid, and created_at.
        """
        job_name = f"train-{run_id[:8]}-{uuid.uuid4().hex[:6]}"
        logger.info(
            "Creating training job",
            job_name=job_name,
            framework=framework,
            num_nodes=num_nodes,
            gpu_count=gpu_count,
            tenant_id=tenant_id,
        )

        loop = asyncio.get_event_loop()
        status: dict[str, Any] = await loop.run_in_executor(
            None,
            partial(
                self._create_job_sync,
                job_name=job_name,
                experiment_id=experiment_id,
                run_id=run_id,
                image=image or self._default_image,
                command=command,
                gpu_count=gpu_count,
                memory_gb=memory_gb,
                cpu_count=cpu_count,
                num_nodes=num_nodes,
                framework=framework,
                env_vars=env_vars or {},
                tenant_id=tenant_id,
            ),
        )
        return status

    def _create_job_sync(
        self,
        job_name: str,
        experiment_id: str,
        run_id: str,
        image: str,
        command: list[str],
        gpu_count: int,
        memory_gb: int,
        cpu_count: int,
        num_nodes: int,
        framework: str,
        env_vars: dict[str, str],
        tenant_id: str,
    ) -> dict[str, Any]:
        from kubernetes import client as k8s_client  # type: ignore[import-untyped]

        batch_api, _ = self._get_apis()

        framework_env = self._build_framework_env(framework=framework, num_nodes=num_nodes)
        all_env = {**framework_env, **env_vars}

        k8s_env = [k8s_client.V1EnvVar(name=key, value=value) for key, value in all_env.items()]

        resources = k8s_client.V1ResourceRequirements(
            requests={
                "cpu": str(cpu_count),
                "memory": f"{memory_gb}Gi",
                **({"nvidia.com/gpu": str(gpu_count)} if gpu_count > 0 else {}),
            },
            limits={
                "cpu": str(cpu_count),
                "memory": f"{memory_gb}Gi",
                **({"nvidia.com/gpu": str(gpu_count)} if gpu_count > 0 else {}),
            },
        )

        container = k8s_client.V1Container(
            name="trainer",
            image=image,
            command=command,
            env=k8s_env,
            resources=resources,
        )

        labels = {
            f"{_JOB_LABEL_PREFIX}/experiment-id": experiment_id[:63],
            f"{_JOB_LABEL_PREFIX}/run-id": run_id[:63],
            f"{_JOB_LABEL_PREFIX}/tenant-id": tenant_id[:63],
            f"{_JOB_LABEL_PREFIX}/framework": framework,
        }

        pod_spec = k8s_client.V1PodSpec(
            containers=[container],
            restart_policy="OnFailure",
        )

        pod_template = k8s_client.V1PodTemplateSpec(
            metadata=k8s_client.V1ObjectMeta(labels=labels),
            spec=pod_spec,
        )

        job_spec = k8s_client.V1JobSpec(
            completions=num_nodes,
            parallelism=num_nodes,
            backoff_limit=3,
            template=pod_template,
        )

        job = k8s_client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=k8s_client.V1ObjectMeta(name=job_name, namespace=self._namespace, labels=labels),
            spec=job_spec,
        )

        created_job = batch_api.create_namespaced_job(namespace=self._namespace, body=job)
        logger.info("K8s training job created", job_name=job_name, uid=str(created_job.metadata.uid))

        return {
            "job_name": job_name,
            "namespace": self._namespace,
            "uid": str(created_job.metadata.uid),
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
        }

    @staticmethod
    def _build_framework_env(framework: str, num_nodes: int) -> dict[str, str]:
        """Build distributed training environment variables for the chosen framework.

        Args:
            framework: "pytorch_ddp" or "horovod".
            num_nodes: Number of worker nodes in the training cluster.

        Returns:
            Dict of environment variable name → value strings.
        """
        if framework == "pytorch_ddp":
            return {
                "WORLD_SIZE": str(num_nodes),
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "29500",
                "OMP_NUM_THREADS": "1",
            }
        if framework == "horovod":
            return {
                "HOROVOD_NUM_PROC": str(num_nodes),
                "HOROVOD_TIMELINE": "/tmp/horovod_timeline.json",
            }
        return {}

    # ------------------------------------------------------------------ #
    # Monitoring                                                           #
    # ------------------------------------------------------------------ #

    async def get_job_status(self, job_name: str) -> dict[str, Any]:
        """Retrieve the current status of a Kubernetes training job.

        Args:
            job_name: K8s Job name to query.

        Returns:
            Status dict with active, succeeded, failed pod counts, and
            overall phase (running | succeeded | failed | pending).
        """
        loop = asyncio.get_event_loop()
        status: dict[str, Any] = await loop.run_in_executor(
            None,
            partial(self._get_job_status_sync, job_name=job_name),
        )
        return status

    def _get_job_status_sync(self, job_name: str) -> dict[str, Any]:
        batch_api, _ = self._get_apis()
        job = batch_api.read_namespaced_job(name=job_name, namespace=self._namespace)
        job_status = job.status
        active = job_status.active or 0
        succeeded = job_status.succeeded or 0
        failed = job_status.failed or 0

        if failed > 0:
            phase = "failed"
        elif succeeded > 0 and active == 0:
            phase = "succeeded"
        elif active > 0:
            phase = "running"
        else:
            phase = "pending"

        return {
            "job_name": job_name,
            "active_pods": active,
            "succeeded_pods": succeeded,
            "failed_pods": failed,
            "phase": phase,
            "start_time": job_status.start_time.isoformat() if job_status.start_time else None,
            "completion_time": job_status.completion_time.isoformat() if job_status.completion_time else None,
        }

    async def stream_pod_logs(self, job_name: str, tail_lines: int = 100) -> list[dict[str, str]]:
        """Retrieve recent log lines from all pods belonging to a job.

        Args:
            job_name: K8s Job name whose pods should be queried.
            tail_lines: Number of tail log lines to return per pod.

        Returns:
            List of dicts with pod_name and log_lines keys.
        """
        loop = asyncio.get_event_loop()
        logs: list[dict[str, str]] = await loop.run_in_executor(
            None,
            partial(self._stream_logs_sync, job_name=job_name, tail_lines=tail_lines),
        )
        return logs

    def _stream_logs_sync(self, job_name: str, tail_lines: int) -> list[dict[str, str]]:
        _, core_api = self._get_apis()
        pods = core_api.list_namespaced_pod(
            namespace=self._namespace,
            label_selector=f"job-name={job_name}",
        )
        pod_logs = []
        for pod in pods.items:
            try:
                log_output = core_api.read_namespaced_pod_log(
                    name=pod.metadata.name,
                    namespace=self._namespace,
                    tail_lines=tail_lines,
                )
                pod_logs.append({"pod_name": pod.metadata.name, "log_lines": log_output})
            except Exception:
                logger.exception("Failed to retrieve pod logs", pod_name=pod.metadata.name)
        return pod_logs

    # ------------------------------------------------------------------ #
    # Auto-scaling                                                         #
    # ------------------------------------------------------------------ #

    async def estimate_required_nodes(self, queue_depth: int, max_nodes: int = 8) -> int:
        """Estimate the number of training nodes needed given the queue depth.

        Uses a simple linear scaling heuristic: one node per 5 queued jobs,
        capped at max_nodes.

        Args:
            queue_depth: Number of training jobs waiting to be scheduled.
            max_nodes: Maximum number of nodes to scale to.

        Returns:
            Recommended node count (1 minimum, max_nodes maximum).
        """
        recommended = max(1, min((queue_depth + 4) // 5, max_nodes))
        logger.info("Auto-scale estimate", queue_depth=queue_depth, recommended_nodes=recommended)
        return recommended

    # ------------------------------------------------------------------ #
    # Cleanup                                                              #
    # ------------------------------------------------------------------ #

    async def delete_job(self, job_name: str) -> bool:
        """Delete a completed or failed training job and its pods.

        Args:
            job_name: K8s Job name to delete.

        Returns:
            True if deletion was accepted by the K8s API.
        """
        loop = asyncio.get_event_loop()
        success: bool = await loop.run_in_executor(
            None,
            partial(self._delete_job_sync, job_name=job_name),
        )
        return success

    def _delete_job_sync(self, job_name: str) -> bool:
        from kubernetes import client as k8s_client  # type: ignore[import-untyped]

        batch_api, _ = self._get_apis()
        try:
            batch_api.delete_namespaced_job(
                name=job_name,
                namespace=self._namespace,
                body=k8s_client.V1DeleteOptions(propagation_policy="Foreground"),
            )
            logger.info("Training job deleted", job_name=job_name)
            return True
        except Exception:
            logger.exception("Failed to delete training job", job_name=job_name)
            return False

    async def cleanup_completed_jobs(self, tenant_id: str) -> int:
        """Delete all completed and failed jobs for a tenant.

        Args:
            tenant_id: Tenant UUID string used to filter jobs by label.

        Returns:
            Number of jobs deleted.
        """
        loop = asyncio.get_event_loop()
        deleted_count: int = await loop.run_in_executor(
            None,
            partial(self._cleanup_jobs_sync, tenant_id=tenant_id),
        )
        return deleted_count

    def _cleanup_jobs_sync(self, tenant_id: str) -> int:
        from kubernetes import client as k8s_client  # type: ignore[import-untyped]

        batch_api, _ = self._get_apis()
        label_selector = f"{_JOB_LABEL_PREFIX}/tenant-id={tenant_id}"
        jobs = batch_api.list_namespaced_job(namespace=self._namespace, label_selector=label_selector)
        deleted = 0
        for job in jobs.items:
            status = job.status
            if (status.succeeded or 0) > 0 or (status.failed or 0) > 0:
                try:
                    batch_api.delete_namespaced_job(
                        name=job.metadata.name,
                        namespace=self._namespace,
                        body=k8s_client.V1DeleteOptions(propagation_policy="Foreground"),
                    )
                    deleted += 1
                except Exception:
                    logger.exception("Failed to delete job during cleanup", job_name=job.metadata.name)
        logger.info("Cleanup complete", tenant_id=tenant_id, jobs_deleted=deleted)
        return deleted
