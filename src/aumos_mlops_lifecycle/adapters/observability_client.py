"""HTTP client for the aumos-observability metrics API.

Used by CanaryScheduler (GAP-159) to query per-deployment error rates
during canary progression checks.

GAP-159: Auto-Scaling Canary Logic
"""

from __future__ import annotations

import httpx

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class ObservabilityClient:
    """Async HTTP client for querying aumos-observability metrics.

    Args:
        base_url: Base URL of the aumos-observability service.
        api_key: Internal service-to-service API key.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 10.0,
    ) -> None:
        """Initialise the observability client.

        Args:
            base_url: Observability service base URL.
            api_key: Bearer token for service authentication.
            timeout: HTTP timeout in seconds.
        """
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    async def get_error_rate(
        self,
        model_id: str,
        deployment_id: str,
        window_minutes: int = 5,
    ) -> float:
        """Query the error rate for a deployment over the last N minutes.

        Args:
            model_id: UUID of the model being monitored.
            deployment_id: UUID of the deployment.
            window_minutes: Look-back window in minutes.

        Returns:
            Error rate as a float in [0, 1]. Returns 0.0 on query failure
            (fail-open so canary is not incorrectly rolled back on metric
            collection errors).
        """
        url = f"{self._base_url}/api/v1/metrics/error-rate"
        params = {
            "model_id": model_id,
            "deployment_id": deployment_id,
            "window_minutes": window_minutes,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()
                rate: float = float(data.get("error_rate", 0.0))
                return rate
        except Exception as exc:
            logger.warning(
                "observability_client_error_rate_failed",
                deployment_id=deployment_id,
                error=str(exc),
            )
            return 0.0
