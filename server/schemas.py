"""Pydantic schemas for the coordination API.

Binary model weights travel as raw ``application/octet-stream``
(safetensors bytes), NOT inside JSON — base64-ing ~400KB of float32 per
client per round would add 33% overhead and pointless parsing. JSON is
reserved for control-plane metadata, which these schemas validate.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

__all__ = ["RoundPhase", "ServerStatus", "SubmitResponse", "RoundMetrics"]


class RoundPhase(StrEnum):
    """Lifecycle of the current round."""

    COLLECTING = "collecting"   # waiting for client submissions
    FINISHED = "finished"       # total_rounds reached, training over


class ServerStatus(BaseModel):
    """GET /status response — everything a client or dashboard needs to act."""

    current_round: int = Field(description="Round currently collecting submissions (1-based)")
    total_rounds: int
    phase: RoundPhase
    expected_clients: list[str]
    received_clients: list[str]
    latest_checkpoint_round: int = Field(
        description="Round number of the newest global model (0 = initial weights)"
    )


class SubmitResponse(BaseModel):
    """POST /round/submit response."""

    accepted: bool
    round: int = Field(description="Round the submission was counted toward")
    round_completed: bool = Field(
        description="True if this submission was the last one and aggregation ran"
    )
    message: str = ""


class RoundMetrics(BaseModel):
    """Per-round global evaluation — logged to MLflow and pushed over WebSocket."""

    round: int
    auc_roc: float
    precision: float
    recall: float
    f1: float
    test_loss: float
    n_clients: int
    client_weights: dict[str, float] = Field(
        description="FedAvg weight (n_k/N) per client — dashboard 'contribution' panel"
    )
