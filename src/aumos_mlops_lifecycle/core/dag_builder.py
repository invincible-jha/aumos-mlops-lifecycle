"""Pipeline DAG construction for aumos-mlops-lifecycle.

Builds React Flow–compatible graph data from the lineage of an ML pipeline:
  Experiment → Model Version → Deployment → Drift Monitor

Returned format matches the React Flow node/edge schema so the
aumos-platform-ui can render the DAG directly.

GAP-163: Pipeline DAG Visualization
"""

from __future__ import annotations

import uuid
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class DAGNode:
    """A node in the pipeline DAG.

    Attributes:
        id: Unique node identifier (UUID as string).
        node_type: One of 'experiment', 'deployment', 'retraining_job', 'feature_set'.
        label: Human-readable display label.
        data: Additional node metadata for tooltip rendering.
        position: Suggested layout position {x, y}.
    """

    def __init__(
        self,
        node_id: str,
        node_type: str,
        label: str,
        data: dict[str, Any] | None = None,
        position: dict[str, int] | None = None,
    ) -> None:
        """Initialise a DAG node.

        Args:
            node_id: Unique identifier.
            node_type: Semantic node type.
            label: Display label.
            data: Supplemental metadata.
            position: Layout hint {x, y}.
        """
        self.id = node_id
        self.node_type = node_type
        self.label = label
        self.data = data or {}
        self.position = position or {"x": 0, "y": 0}

    def to_dict(self) -> dict[str, Any]:
        """Convert to React Flow node format."""
        return {
            "id": self.id,
            "type": self.node_type,
            "data": {"label": self.label, **self.data},
            "position": self.position,
        }


class DAGEdge:
    """A directed edge in the pipeline DAG.

    Attributes:
        source: ID of the source node.
        target: ID of the target node.
        label: Optional edge label (e.g., 'triggered by').
    """

    def __init__(self, source: str, target: str, label: str = "") -> None:
        """Initialise a DAG edge.

        Args:
            source: Source node ID.
            target: Target node ID.
            label: Relationship label.
        """
        self.source = source
        self.target = target
        self.label = label

    def to_dict(self) -> dict[str, Any]:
        """Convert to React Flow edge format."""
        edge_id = f"{self.source}->{self.target}"
        result: dict[str, Any] = {"id": edge_id, "source": self.source, "target": self.target}
        if self.label:
            result["label"] = self.label
        return result


class PipelineDAGBuilder:
    """Builds a React Flow–compatible DAG for the ML pipeline lineage.

    Args:
        experiment_repo: Repository for loading experiments.
        deployment_repo: Repository for loading deployments.
        retraining_repo: Repository for loading retraining jobs.
        feature_set_repo: Repository for loading feature sets.
    """

    def __init__(
        self,
        experiment_repo: Any,
        deployment_repo: Any,
        retraining_repo: Any,
        feature_set_repo: Any,
    ) -> None:
        """Initialise the DAG builder with repository dependencies.

        Args:
            experiment_repo: IExperimentRepository implementation.
            deployment_repo: IDeploymentRepository implementation.
            retraining_repo: IRetrainingJobRepository implementation.
            feature_set_repo: IFeatureSetRepository implementation.
        """
        self._experiments = experiment_repo
        self._deployments = deployment_repo
        self._retraining = retraining_repo
        self._feature_sets = feature_set_repo

    async def build(
        self,
        model_id: str,
        tenant: Any,
        session: Any,
    ) -> dict[str, list[dict[str, Any]]]:
        """Construct the pipeline DAG for a given model.

        Args:
            model_id: UUID of the model whose lineage to visualise.
            tenant: TenantContext for RLS isolation.
            session: Async SQLAlchemy session.

        Returns:
            Dict with 'nodes' and 'edges' lists in React Flow format.
        """
        nodes: list[DAGNode] = []
        edges: list[DAGEdge] = []

        # Collect experiments linked to this model via tags
        experiments, _ = await self._experiments.list_all(tenant, page=1, page_size=100)
        model_experiments = [
            e for e in experiments
            if (e.tags or {}).get("model_id") == model_id
        ]

        y_offset = 0
        for exp in model_experiments:
            exp_node = DAGNode(
                node_id=str(exp.id),
                node_type="experiment",
                label=exp.name,
                data={"status": exp.status, "model_id": model_id},
                position={"x": 0, "y": y_offset},
            )
            nodes.append(exp_node)
            y_offset += 150

        # Collect deployments for this model
        deployments, _ = await self._deployments.list_all(tenant, page=1, page_size=100)
        model_deployments = [
            d for d in deployments
            if str(getattr(d, "model_id", "")) == model_id
        ]

        dep_x_offset = 300
        for dep in model_deployments:
            dep_node = DAGNode(
                node_id=str(dep.id),
                node_type="deployment",
                label=f"{getattr(dep, 'strategy', 'unknown')} → {getattr(dep, 'target_environment', '?')}",
                data={
                    "status": getattr(dep, "status", "unknown"),
                    "strategy": getattr(dep, "strategy", ""),
                    "environment": getattr(dep, "target_environment", ""),
                },
                position={"x": dep_x_offset, "y": 0},
            )
            nodes.append(dep_node)
            dep_x_offset += 150

            # Link experiment → deployment if experiment exists
            if model_experiments:
                edges.append(
                    DAGEdge(
                        source=str(model_experiments[0].id),
                        target=str(dep.id),
                        label="produces",
                    )
                )

        # Collect retraining jobs
        retraining_jobs, _ = await self._retraining.list_all(tenant, page=1, page_size=100)
        model_jobs = [
            j for j in retraining_jobs
            if str(getattr(j, "model_id", "")) == model_id
        ]

        job_y_offset = 300
        for job in model_jobs:
            job_node = DAGNode(
                node_id=str(job.id),
                node_type="retraining_job",
                label=f"Retrain [{getattr(job, 'trigger_type', 'manual')}]",
                data={
                    "status": getattr(job, "status", "unknown"),
                    "trigger_type": getattr(job, "trigger_type", ""),
                },
                position={"x": 150, "y": job_y_offset},
            )
            nodes.append(job_node)
            job_y_offset += 150

            if model_deployments:
                edges.append(
                    DAGEdge(
                        source=str(model_deployments[0].id),
                        target=str(job.id),
                        label="triggers retraining",
                    )
                )

        logger.info(
            "dag_built",
            model_id=model_id,
            node_count=len(nodes),
            edge_count=len(edges),
        )

        return {
            "nodes": [n.to_dict() for n in nodes],
            "edges": [e.to_dict() for e in edges],
        }
