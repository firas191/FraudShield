"""Model registry: versioned global-model checkpoints with rollback.

Why this exists (spec §4.2): aggregation can *degrade* the global model —
a bad round on Non-IID data, an undetected poisoned update, an unlucky
DP-noise draw. The registry makes every round's model durable and
addressable so the round manager can roll back to the last good round
instead of retraining from scratch.

Layout on disk::

    checkpoints/
    ├── registry.json                 # round → metadata (metrics, timestamps)
    ├── global_round_0000.safetensors # initial model
    ├── global_round_0001.safetensors
    └── ...

Design decisions:

* **safetensors for weights, JSON for metadata.** Weights are opaque
  tensors; metadata (round number, eval metrics, aggregation strategy)
  must stay human-readable for debugging and is tiny. Mixing them into
  one pickle file would re-introduce the deserialization attack surface
  we removed in Week 1.

* **Atomic metadata writes.** ``registry.json`` is rewritten via a
  temp-file + ``os.replace`` so a crash mid-write can't leave a corrupt
  index pointing at checkpoints that exist (or vice versa). The
  checkpoint file itself is written before the index references it.

* **Rollback = pointer move, not deletion.** ``rollback_to(round)``
  marks later rounds as superseded but keeps their files — forensics on
  *why* a round degraded (Week 5's Byzantine experiments) needs the bad
  weights.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from torch import Tensor

from model.utils import load_weights, save_weights

__all__ = ["CheckpointMeta", "ModelRegistry", "RegistryError"]

logger = logging.getLogger(__name__)


class RegistryError(RuntimeError):
    """Registry invariant violated (missing checkpoint, bad round, ...)."""


@dataclass
class CheckpointMeta:
    """Metadata for one global-model checkpoint."""

    round: int
    filename: str
    created_at: float
    superseded: bool = False
    metrics: dict[str, float] = field(default_factory=dict)
    note: str = ""


class ModelRegistry:
    """Stores and retrieves per-round global model checkpoints.

    Thread-safety note: the FastAPI server runs single-process and the
    round manager serializes round completion, so no file locking is
    needed. If the server is ever scaled out, the registry must move
    behind a single writer (or a real artifact store like MLflow's).
    """

    INDEX_NAME = "registry.json"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / self.INDEX_NAME
        self._index: dict[int, CheckpointMeta] = {}
        if self._index_path.is_file():
            self._load_index()

    # ------------------------------------------------------------------ #
    def save(
        self,
        state_dict: dict[str, Tensor],
        round_num: int,
        metrics: dict[str, float] | None = None,
        note: str = "",
    ) -> Path:
        """Persist a global model for ``round_num``.

        Args:
            state_dict: Global model weights.
            round_num: FL round (0 = initial model before any training).
            metrics: Optional eval metrics to store alongside.
            note: Free-text annotation (e.g. "post-rollback re-aggregation").

        Returns:
            Path of the written checkpoint.

        Raises:
            RegistryError: If the round already exists — rounds are
                immutable; overwriting one would falsify history.
        """
        if round_num < 0:
            raise RegistryError(f"round must be >= 0, got {round_num}")
        if round_num in self._index:
            raise RegistryError(
                f"round {round_num} already checkpointed — rounds are immutable"
            )
        filename = f"global_round_{round_num:04d}.safetensors"
        path = self.root / filename
        save_weights(state_dict, path)  # write blob first...
        self._index[round_num] = CheckpointMeta(
            round=round_num,
            filename=filename,
            created_at=time.time(),
            metrics=dict(metrics or {}),
            note=note,
        )
        self._write_index()  # ...then the index that references it
        logger.info("registry: saved round %d (%s)", round_num, filename)
        return path

    def load(self, round_num: int) -> dict[str, Tensor]:
        """Load the checkpoint for a specific round.

        Raises:
            RegistryError: If the round is unknown or its file vanished.
        """
        meta = self._index.get(round_num)
        if meta is None:
            raise RegistryError(f"no checkpoint for round {round_num}")
        path = self.root / meta.filename
        if not path.is_file():
            raise RegistryError(f"index references missing file: {path}")
        return load_weights(path)

    def latest_round(self) -> int:
        """Highest non-superseded round number.

        Raises:
            RegistryError: If the registry is empty.
        """
        active = [r for r, m in self._index.items() if not m.superseded]
        if not active:
            raise RegistryError("registry is empty")
        return max(active)

    def load_latest(self) -> tuple[int, dict[str, Tensor]]:
        """Load the current (non-superseded) global model."""
        r = self.latest_round()
        return r, self.load(r)

    def rollback_to(self, round_num: int, note: str = "") -> int:
        """Mark every round after ``round_num`` as superseded.

        Subsequent :meth:`latest_round` calls return ``round_num``; the
        superseded checkpoints stay on disk for forensics.

        Returns:
            Number of rounds superseded.

        Raises:
            RegistryError: If ``round_num`` has no checkpoint.
        """
        if round_num not in self._index:
            raise RegistryError(f"cannot roll back to unknown round {round_num}")
        count = 0
        for r, meta in self._index.items():
            if r > round_num and not meta.superseded:
                meta.superseded = True
                meta.note = (meta.note + " | " if meta.note else "") + (
                    note or f"superseded by rollback to round {round_num}"
                )
                count += 1
        self._write_index()
        logger.warning(
            "registry: rolled back to round %d (%d rounds superseded)", round_num, count
        )
        return count

    def history(self) -> list[CheckpointMeta]:
        """All checkpoint metadata, ordered by round."""
        return [self._index[r] for r in sorted(self._index)]

    # ------------------------------------------------------------------ #
    def _write_index(self) -> None:
        payload = {str(r): asdict(m) for r, m in self._index.items()}
        tmp = self._index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, self._index_path)  # atomic on POSIX and NTFS

    def _load_index(self) -> None:
        try:
            raw = json.loads(self._index_path.read_text())
        except json.JSONDecodeError as exc:
            raise RegistryError(f"corrupt registry index: {self._index_path}") from exc
        self._index = {int(r): CheckpointMeta(**meta) for r, meta in raw.items()}
        logger.info("registry: loaded %d checkpoints from %s", len(self._index), self.root)
