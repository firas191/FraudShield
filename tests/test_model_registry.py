"""Tests for the model registry: persistence, immutability, rollback."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from model.fraud_mlp import FraudDetectorMLP
from server.model_registry import ModelRegistry, RegistryError


@pytest.fixture()
def sd() -> dict[str, torch.Tensor]:
    torch.manual_seed(1)
    return FraudDetectorMLP().state_dict()


def scaled(sd: dict[str, torch.Tensor], factor: float) -> dict[str, torch.Tensor]:
    return {k: v * factor for k, v in sd.items()}


class TestSaveLoad:
    def test_round_trip(self, tmp_path: Path, sd) -> None:
        reg = ModelRegistry(tmp_path)
        reg.save(sd, round_num=0)
        loaded = reg.load(0)
        for k in sd:
            assert torch.equal(sd[k], loaded[k])

    def test_metrics_persisted(self, tmp_path: Path, sd) -> None:
        reg = ModelRegistry(tmp_path)
        reg.save(sd, 0, metrics={"auc_roc": 0.97, "epsilon": 1.2})
        meta = reg.history()[0]
        assert meta.metrics["auc_roc"] == pytest.approx(0.97)

    def test_rounds_are_immutable(self, tmp_path: Path, sd) -> None:
        reg = ModelRegistry(tmp_path)
        reg.save(sd, 0)
        with pytest.raises(RegistryError, match="immutable"):
            reg.save(sd, 0)

    def test_unknown_round_raises(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path)
        with pytest.raises(RegistryError, match="no checkpoint"):
            reg.load(3)

    def test_empty_registry_latest_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RegistryError, match="empty"):
            ModelRegistry(tmp_path).latest_round()


class TestPersistenceAcrossInstances:
    def test_index_survives_restart(self, tmp_path: Path, sd) -> None:
        """Simulates a server restart: a new instance must see all rounds."""
        reg1 = ModelRegistry(tmp_path)
        reg1.save(sd, 0)
        reg1.save(scaled(sd, 2.0), 1, metrics={"auc_roc": 0.9})

        reg2 = ModelRegistry(tmp_path)
        assert reg2.latest_round() == 1
        loaded = reg2.load(1)
        assert torch.allclose(loaded[next(iter(sd))], sd[next(iter(sd))] * 2.0)

    def test_corrupt_index_raises(self, tmp_path: Path, sd) -> None:
        reg = ModelRegistry(tmp_path)
        reg.save(sd, 0)
        (tmp_path / ModelRegistry.INDEX_NAME).write_text("{not json")
        with pytest.raises(RegistryError, match="corrupt"):
            ModelRegistry(tmp_path)


class TestRollback:
    def test_rollback_moves_latest_pointer(self, tmp_path: Path, sd) -> None:
        reg = ModelRegistry(tmp_path)
        for r in range(4):
            reg.save(scaled(sd, float(r + 1)), r)
        assert reg.latest_round() == 3

        superseded = reg.rollback_to(1, note="round 2+ degraded by byzantine client")
        assert superseded == 2
        assert reg.latest_round() == 1

        r, weights = reg.load_latest()
        assert r == 1
        key = next(iter(sd))
        assert torch.allclose(weights[key], sd[key] * 2.0)

    def test_rollback_keeps_files_for_forensics(self, tmp_path: Path, sd) -> None:
        reg = ModelRegistry(tmp_path)
        for r in range(3):
            reg.save(scaled(sd, float(r + 1)), r)
        reg.rollback_to(0)
        # superseded checkpoints remain loadable by explicit round number
        loaded = reg.load(2)
        key = next(iter(sd))
        assert torch.allclose(loaded[key], sd[key] * 3.0)

    def test_rollback_survives_restart(self, tmp_path: Path, sd) -> None:
        reg = ModelRegistry(tmp_path)
        for r in range(3):
            reg.save(scaled(sd, float(r + 1)), r)
        reg.rollback_to(0)
        assert ModelRegistry(tmp_path).latest_round() == 0

    def test_rollback_to_unknown_round_raises(self, tmp_path: Path, sd) -> None:
        reg = ModelRegistry(tmp_path)
        reg.save(sd, 0)
        with pytest.raises(RegistryError, match="unknown round"):
            reg.rollback_to(5)

    def test_new_rounds_after_rollback(self, tmp_path: Path, sd) -> None:
        """After rolling back to round 1, training resumes at round 4+
        (round numbers are never reused — history stays unambiguous)."""
        reg = ModelRegistry(tmp_path)
        for r in range(4):
            reg.save(scaled(sd, float(r + 1)), r)
        reg.rollback_to(1)
        reg.save(scaled(sd, 9.0), 4, note="re-aggregated after rollback")
        assert reg.latest_round() == 4
