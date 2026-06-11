"""HTTP communication between a bank client and the coordination server.

The ``httpx.Client`` is injectable: production passes a real client with
the server's base URL; tests pass FastAPI's TestClient (which *is* an
httpx client over an in-process ASGI transport). The FL client logic is
thereby tested against the real server code with zero sockets.
"""

from __future__ import annotations

import logging
import time

import httpx
import torch
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save

from server.schemas import ServerStatus, SubmitResponse

__all__ = ["ServerCommunicator", "ServerUnavailable"]

logger = logging.getLogger(__name__)


class ServerUnavailable(ConnectionError):
    """Server could not be reached within the allotted time."""


class ServerCommunicator:
    """Typed wrapper over the coordination server's REST API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        http: httpx.Client | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._owns_http = http is None
        self._http = http or httpx.Client(base_url=base_url, timeout=timeout)

    # ------------------------------------------------------------------ #
    def wait_for_server(self, deadline_seconds: float = 60.0) -> None:
        """Block until /health responds — clients may start before the server.

        Raises:
            ServerUnavailable: If the deadline expires.
        """
        start = time.monotonic()
        delay = 0.5
        while time.monotonic() - start < deadline_seconds:
            try:
                if self._http.get("/health").status_code == 200:
                    return
            except httpx.TransportError:
                pass
            time.sleep(delay)
            delay = min(delay * 2, 5.0)  # exponential backoff, capped
        raise ServerUnavailable(f"server not reachable after {deadline_seconds:.0f}s")

    def get_status(self) -> ServerStatus:
        resp = self._http.get("/status")
        resp.raise_for_status()
        return ServerStatus.model_validate(resp.json())

    def download_global_model(self) -> tuple[int, dict[str, torch.Tensor]]:
        """Fetch current global weights.

        Returns:
            ``(round_number, state_dict)``.
        """
        resp = self._http.get("/model/current")
        resp.raise_for_status()
        round_num = int(resp.headers["X-Round"])
        return round_num, st_load(resp.content)

    def submit_update(
        self, client_id: str, num_samples: int, state_dict: dict[str, torch.Tensor]
    ) -> SubmitResponse:
        """Upload locally trained weights for the current round.

        Raises:
            httpx.HTTPStatusError: On rejection (422) — deliberately not
                swallowed: a rejected update means a bug or an attack,
                and the client must not retry-loop a poisoned payload.
        """
        blob = st_save({k: v.contiguous() for k, v in state_dict.items()})
        resp = self._http.post(
            "/round/submit",
            data={"client_id": client_id, "num_samples": str(num_samples)},
            files={"weights": ("weights.safetensors", blob, "application/octet-stream")},
        )
        resp.raise_for_status()
        return SubmitResponse.model_validate(resp.json())

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> "ServerCommunicator":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
