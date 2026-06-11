"""FastAPI coordination server.

Run (from repo root):

    uvicorn server.main:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health           liveness probe (Docker healthcheck, Week 7)
    GET  /status           round state — clients poll this
    GET  /model/current    global weights (safetensors bytes, X-Round header)
    POST /round/submit     multipart: client_id, num_samples, weights file
    WS   /ws/metrics       pushes RoundMetrics JSON after every round

Transport security note: this server speaks plain HTTP. The spec says
HTTPS — in deployment TLS terminates at a reverse proxy (or Docker
network boundary in our simulation). Bank-grade deployments would add
mTLS so the server authenticates clients by certificate instead of a
self-declared ``client_id`` string; with HTTP any process that can reach
the port can impersonate bank_a. Acceptable in simulation, called out
because it IS the threat model Week 5 plays with.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import AsyncIterator

import torch
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile, WebSocket, WebSocketDisconnect
from pydantic_settings import BaseSettings, SettingsConfigDict
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save

from model.fraud_mlp import FraudDetectorMLP
from server.aggregation import AggregationError
from server.evaluation import GlobalEvaluator
from server.mlflow_logger import MLflowLogger
from server.model_registry import ModelRegistry, RegistryError
from server.round_manager import RoundManager
from server.schemas import RoundMetrics, ServerStatus, SubmitResponse

logger = logging.getLogger("fraudshield.server")

REPO_ROOT = Path(__file__).resolve().parent.parent

# Hard cap on uploaded weight payloads. The MLP serializes to ~500KB;
# 10MB leaves headroom for bigger models while bounding memory abuse.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class Settings(BaseSettings):
    """Server configuration — mirrors .env.example keys."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    expected_clients: str = "bank_a,bank_b,bank_c"
    num_rounds: int = 50
    checkpoint_dir: Path = REPO_ROOT / "checkpoints"
    global_test_csv: Path = REPO_ROOT / "data" / "global_test.csv"
    mlflow_enabled: bool = True
    mlflow_tracking_uri: str | None = None
    seed: int = 42

    @property
    def client_list(self) -> list[str]:
        return [c.strip() for c in self.expected_clients.split(",") if c.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


class WebSocketHub:
    """Tracks dashboard connections; broadcasts round metrics."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._connections:
            self._connections.remove(ws)

    async def broadcast(self, metrics: RoundMetrics) -> None:
        dead: list[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_json(metrics.model_dump())
            except Exception:  # noqa: BLE001 — any send failure means a dead socket
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


def build_round_manager(settings: Settings, hub: WebSocketHub) -> RoundManager:
    """Wire up registry, evaluator, MLflow, and the round manager."""
    registry = ModelRegistry(settings.checkpoint_dir)
    try:
        registry.latest_round()
    except RegistryError:
        # Fresh start: round 0 = deterministic initial weights so every
        # client begins from the identical model.
        torch.manual_seed(settings.seed)
        registry.save(
            FraudDetectorMLP().state_dict(), 0, note="initial model (seeded)"
        )
        logger.info("registry empty — created seeded round-0 model")

    evaluator = GlobalEvaluator(settings.global_test_csv)
    mlflow = MLflowLogger(
        run_name=f"fedavg_{settings.num_rounds}r",
        tracking_uri=settings.mlflow_tracking_uri,
        enabled=settings.mlflow_enabled,
    )
    mlflow.log_params(
        {
            "strategy": "fedavg",
            "num_rounds": settings.num_rounds,
            "clients": settings.expected_clients,
            "seed": settings.seed,
        }
    )
    return RoundManager(
        registry=registry,
        evaluator=evaluator,
        expected_clients=settings.client_list,
        total_rounds=settings.num_rounds,
        mlflow=mlflow,
        broadcast=hub.broadcast,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.hub = WebSocketHub()
    app.state.manager = build_round_manager(settings, app.state.hub)
    logger.info(
        "server up: %d clients expected, %d rounds",
        len(settings.client_list), settings.num_rounds,
    )
    yield
    app.state.manager.mlflow.close()


app = FastAPI(title="FraudShield Coordination Server", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/status", response_model=ServerStatus)
async def status() -> ServerStatus:
    return app.state.manager.status()


@app.get("/model/current")
async def model_current() -> Response:
    round_num, sd = app.state.manager.current_global_model()
    payload = st_save({k: v.contiguous() for k, v in sd.items()})
    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={"X-Round": str(round_num)},
    )


@app.post("/round/submit", response_model=SubmitResponse)
async def round_submit(
    client_id: str = Form(...),
    num_samples: int = Form(...),
    weights: UploadFile = File(...),
) -> SubmitResponse:
    blob = await weights.read()
    if len(blob) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="weights payload too large")
    try:
        state_dict = st_load(blob)
    except Exception as exc:  # noqa: BLE001 — any parse failure is the client's fault
        raise HTTPException(status_code=422, detail=f"invalid safetensors payload: {exc}")
    try:
        return await app.state.manager.submit(client_id, num_samples, state_dict)
    except AggregationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/metrics/history", response_model=list[RoundMetrics])
async def metrics_history() -> list[RoundMetrics]:
    """Completed-round metrics — lets a late-joining dashboard backfill."""
    return app.state.manager.history


@app.websocket("/ws/metrics")
async def ws_metrics(ws: WebSocket) -> None:
    hub: WebSocketHub = app.state.hub
    await hub.connect(ws)
    try:
        while True:
            # Keep the connection alive; we only push, clients needn't send.
            await ws.receive_text()
    except WebSocketDisconnect:
        hub.disconnect(ws)
