"""Thin MLflow wrapper: per-round metric logging with graceful degradation.

Why a wrapper instead of calling mlflow directly from the round manager:

* MLflow import is heavy (~1s) and its absence/misconfiguration must
  never take down the FL server — experiment tracking is observability,
  not a dependency of correctness. The wrapper degrades to logging a
  warning once and becoming a no-op.
* Tests run with ``enabled=False`` and assert against the round
  manager's own state instead of a tracking store.

With no MLFLOW_TRACKING_URI set, MLflow writes to the local ``mlruns/``
file store — no tracking server needed. Inspect with ``mlflow ui``.
"""

from __future__ import annotations

import logging
from types import TracebackType
from typing import Any

__all__ = ["MLflowLogger"]

logger = logging.getLogger(__name__)


class MLflowLogger:
    """Lifecycle-managed MLflow run for one federated training session."""

    def __init__(
        self,
        experiment_name: str = "fraudshield",
        run_name: str | None = None,
        tracking_uri: str | None = None,
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled
        self._mlflow: Any | None = None
        if not enabled:
            return
        try:
            import mlflow

            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            mlflow.start_run(run_name=run_name)
            self._mlflow = mlflow
            logger.info(
                "mlflow: experiment=%s uri=%s", experiment_name, mlflow.get_tracking_uri()
            )
        except Exception as exc:  # noqa: BLE001 — observability must not crash training
            logger.warning("mlflow unavailable, metrics will not be tracked: %s", exc)
            self._enabled = False

    def log_params(self, params: dict[str, Any]) -> None:
        """Log run-level configuration (strategy, rounds, DP settings...)."""
        if self._mlflow is not None:
            try:
                self._mlflow.log_params(params)
            except Exception as exc:  # noqa: BLE001
                logger.warning("mlflow log_params failed: %s", exc)

    def log_round(self, round_num: int, metrics: dict[str, float]) -> None:
        """Log one round's metrics with the round as the step axis."""
        if self._mlflow is not None:
            try:
                clean = {k: float(v) for k, v in metrics.items()}
                self._mlflow.log_metrics(clean, step=round_num)
            except Exception as exc:  # noqa: BLE001
                logger.warning("mlflow log_metrics failed (round %d): %s", round_num, exc)

    def close(self) -> None:
        """End the run. Safe to call multiple times."""
        if self._mlflow is not None:
            try:
                self._mlflow.end_run()
            finally:
                self._mlflow = None

    def __enter__(self) -> "MLflowLogger":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
