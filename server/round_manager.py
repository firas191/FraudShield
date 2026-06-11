"""Round manager: the FL training state machine.

Pull-based protocol (deliberate deviation from the spec's "server
broadcasts via POST /round/start"):
    Push would require every bank client to run its own HTTP server for
    the coordinator to call into — more moving parts, and in real
    deployments banks' inbound firewalls make push impractical anyway.
    Instead clients poll ``GET /status``, download the current global
    model, train, and submit. Functionally identical rounds, half the
    infrastructure. (Flower uses long-lived gRPC streams for the same
    reason.) Document this as a conscious architecture decision.

Concurrency model:
    Everything mutating round state goes through one ``asyncio.Lock``.
    FastAPI runs async endpoints on a single event loop, so the lock
    fully serializes submissions; aggregation + evaluation for our model
    take ~100ms, acceptable to run inline. If the model were large this
    would move to a background task with a COLLECTING→AGGREGATING phase.

Round lifecycle::

    round t COLLECTING ── all expected clients submitted ──▶ aggregate
      ▲                                                        │ validate → fedavg
      │                                                        │ evaluate on global test
      │                                                        │ checkpoint round t
      │                                                        │ MLflow log + WS broadcast
      └──────────────── t+1 ≤ total_rounds ────────────────────┘
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

import torch

from model.utils import flatten_state_dict
from server.aggregation import AggregationError, ClientUpdate, fedavg
from server.evaluation import GlobalEvaluator
from server.mlflow_logger import MLflowLogger
from server.model_registry import ModelRegistry
from server.schemas import RoundMetrics, RoundPhase, ServerStatus, SubmitResponse

__all__ = ["RoundManager"]

logger = logging.getLogger(__name__)

BroadcastFn = Callable[[RoundMetrics], Awaitable[None]]


class RoundManager:
    """Coordinates federated rounds: collect → aggregate → evaluate → advance."""

    def __init__(
        self,
        registry: ModelRegistry,
        evaluator: GlobalEvaluator,
        expected_clients: list[str],
        total_rounds: int,
        mlflow: MLflowLogger | None = None,
        broadcast: BroadcastFn | None = None,
    ) -> None:
        if total_rounds <= 0:
            raise ValueError(f"total_rounds must be positive, got {total_rounds}")
        if len(expected_clients) < 1:
            raise ValueError("need at least one expected client")
        self.registry = registry
        self.evaluator = evaluator
        self.expected_clients = sorted(expected_clients)
        self.total_rounds = total_rounds
        self.mlflow = mlflow or MLflowLogger(enabled=False)
        self.broadcast = broadcast

        self._lock = asyncio.Lock()
        self._submissions: dict[str, ClientUpdate] = {}
        # Resume support: if the registry already has rounds 0..t, the
        # next collecting round is t+1 — a server restart mid-training
        # continues instead of restarting from round 1.
        self.current_round = self.registry.latest_round() + 1
        _, global_sd = self.registry.load_latest()
        _, self._reference_spec = flatten_state_dict(global_sd)
        self.history: list[RoundMetrics] = []

    # ------------------------------------------------------------------ #
    @property
    def finished(self) -> bool:
        return self.current_round > self.total_rounds

    def status(self) -> ServerStatus:
        return ServerStatus(
            current_round=min(self.current_round, self.total_rounds),
            total_rounds=self.total_rounds,
            phase=RoundPhase.FINISHED if self.finished else RoundPhase.COLLECTING,
            expected_clients=self.expected_clients,
            received_clients=sorted(self._submissions.keys()),
            latest_checkpoint_round=self.registry.latest_round(),
        )

    def current_global_model(self) -> tuple[int, dict[str, torch.Tensor]]:
        """Latest global weights and their round number (for client download)."""
        return self.registry.load_latest()

    # ------------------------------------------------------------------ #
    async def submit(
        self, client_id: str, num_samples: int, state_dict: dict[str, torch.Tensor]
    ) -> SubmitResponse:
        """Accept one client's update; aggregate when the round is complete.

        Validation failures raise ``AggregationError`` immediately at
        submission time (HTTP 422 at the API layer) — a malformed update
        is the *submitting client's* problem and must not stall the
        round for everyone else.
        """
        async with self._lock:
            if self.finished:
                return SubmitResponse(
                    accepted=False,
                    round=self.total_rounds,
                    round_completed=False,
                    message="training finished",
                )
            if client_id not in self.expected_clients:
                raise AggregationError(f"unknown client '{client_id}'")
            if client_id in self._submissions:
                raise AggregationError(
                    f"{client_id} already submitted for round {self.current_round}"
                )

            update = ClientUpdate(
                client_id=client_id, num_samples=num_samples, state_dict=state_dict
            )
            # Validate at the door (raises AggregationError on bad updates).
            from server.aggregation import validate_update

            validate_update(update, self._reference_spec)
            self._submissions[client_id] = update
            logger.info(
                "round %d: received %s (%d/%d)",
                self.current_round, client_id,
                len(self._submissions), len(self.expected_clients),
            )

            if set(self._submissions) != set(self.expected_clients):
                return SubmitResponse(
                    accepted=True,
                    round=self.current_round,
                    round_completed=False,
                    message=f"waiting for {len(self.expected_clients) - len(self._submissions)} more",
                )

            metrics = await self._complete_round()
            return SubmitResponse(
                accepted=True,
                round=metrics.round,
                round_completed=True,
                message=f"round {metrics.round} aggregated: AUC={metrics.auc_roc:.4f}",
            )

    async def _complete_round(self) -> RoundMetrics:
        """Aggregate, evaluate, checkpoint, log, broadcast, advance."""
        updates = list(self._submissions.values())
        total_n = sum(u.num_samples for u in updates)
        client_weights = {u.client_id: u.num_samples / total_n for u in updates}

        new_global = fedavg(updates, self._reference_spec)
        eval_metrics = self.evaluator.evaluate(new_global)

        round_num = self.current_round
        self.registry.save(
            new_global, round_num, metrics=eval_metrics.as_dict(),
            note=f"fedavg over {len(updates)} clients",
        )
        self.mlflow.log_round(round_num, eval_metrics.as_dict())

        metrics = RoundMetrics(
            round=round_num,
            auc_roc=eval_metrics.auc_roc,
            precision=eval_metrics.precision,
            recall=eval_metrics.recall,
            f1=eval_metrics.f1,
            test_loss=eval_metrics.test_loss,
            n_clients=len(updates),
            client_weights=client_weights,
        )
        self.history.append(metrics)
        self._submissions.clear()
        self.current_round += 1

        logger.info(
            "round %d complete: AUC=%.4f best_f1=%.4f (thr=%.3f) loss=%.4f%s",
            round_num, eval_metrics.auc_roc, eval_metrics.best_f1,
            eval_metrics.best_threshold, eval_metrics.test_loss,
            " — TRAINING FINISHED" if self.finished else "",
        )
        if self.broadcast is not None:
            try:
                await self.broadcast(metrics)
            except Exception as exc:  # noqa: BLE001 — dashboards must not break training
                logger.warning("websocket broadcast failed: %s", exc)
        return metrics
