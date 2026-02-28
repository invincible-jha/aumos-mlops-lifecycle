"""HTTP client for the aumos-testing-harness evaluation results API.

Used by ModelCardService (GAP-160) to fetch the latest evaluation results
for a given model when generating model cards.

GAP-160: Model Card Generation
"""

from __future__ import annotations

from typing import Any

import httpx

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class TestingHarnessClient:
    """Async HTTP client for aumos-testing-harness evaluation results.

    Args:
        base_url: Base URL of the aumos-testing-harness service.
        api_key: Internal service-to-service API key.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 15.0,
    ) -> None:
        """Initialise the testing harness client.

        Args:
            base_url: Testing harness service base URL.
            api_key: Bearer token for service authentication.
            timeout: HTTP timeout in seconds.
        """
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    async def get_latest_results(
        self,
        model_id: str,
        tenant_id: str,
    ) -> dict[str, Any]:
        """Fetch the most recent evaluation results for a model.

        Args:
            model_id: UUID of the model whose evaluation results to fetch.
            tenant_id: Tenant UUID for scoped access.

        Returns:
            Dictionary of metric_name → score from the latest evaluation run.
            Returns an empty dict if no evaluation results exist.
        """
        url = f"{self._base_url}/api/v1/runs"
        params = {"model_id": model_id, "page_size": 1, "sort_by": "created_at", "sort_order": "desc"}
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "X-Tenant-ID": tenant_id,
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(url, params=params, headers=headers)
                if response.status_code == 404:
                    return {}
                response.raise_for_status()
                data = response.json()
                items = data.get("items", [])
                if not items:
                    return {}
                run_id = items[0]["id"]
                results_url = f"{self._base_url}/api/v1/runs/{run_id}/results"
                results_response = await client.get(results_url, headers=headers)
                results_response.raise_for_status()
                results_data = results_response.json()
                metrics: dict[str, Any] = {}
                for result in results_data.get("items", []):
                    metrics[result["metric_name"]] = result.get("score", 0.0)
                return metrics
        except Exception as exc:
            logger.warning(
                "testing_harness_client_get_latest_results_failed",
                model_id=model_id,
                error=str(exc),
            )
            return {}
