"""Tests for LocalTrainer: convergence, DP mechanics, ε accounting vs theory.

All on synthetic separable data — fast, CPU-only, no Kaggle CSV needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from client.data_loader import BankDataConfig, BankDataModule
from client.trainer import (
    LocalTrainer,
    TrainerConfig,
    theoretical_epsilon_from_history,
)
from model.fraud_mlp import FraudDetectorMLP

N_ROWS = 4_000
FRAUD_RATE = 0.05  # generous so DP runs still see positives per batch


@pytest.fixture(scope="module")
def partition_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Separable synthetic partition: frauds have V1–V6 shifted by +2σ."""
    rng = np.random.default_rng(3)
    y = (rng.uniform(size=N_ROWS) < FRAUD_RATE).astype(np.int64)
    df = pd.DataFrame({f"V{i}": rng.normal(size=N_ROWS) for i in range(1, 29)})
    for i in range(1, 7):
        df[f"V{i}"] += 2.0 * y
    df.insert(0, "Time", rng.uniform(0, 172_800, size=N_ROWS))
    df["Amount"] = rng.lognormal(mean=3.0, sigma=1.5, size=N_ROWS)
    df["Class"] = y
    path = tmp_path_factory.mktemp("partitions") / "bank_synth.csv"
    df.to_csv(path, index=False)
    return path


def make_data(partition_csv: Path, batch_size: int = 256) -> BankDataModule:
    dm = BankDataModule(BankDataConfig(csv_path=partition_csv, batch_size=batch_size))
    dm.setup()
    return dm


class TestStandardTraining:
    def test_loss_decreases_and_model_learns(self, partition_csv: Path) -> None:
        torch.manual_seed(0)
        data = make_data(partition_csv)
        trainer = LocalTrainer(
            FraudDetectorMLP(), data, TrainerConfig(epochs=4, device="cpu")
        )
        history = trainer.fit()
        assert history[-1].train_loss < history[0].train_loss
        assert history[-1].val.auc_roc > 0.95, "separable data must reach high AUC"
        assert history[-1].epsilon is None

    def test_state_dict_loadable_into_fresh_model(self, partition_csv: Path) -> None:
        torch.manual_seed(0)
        data = make_data(partition_csv)
        trainer = LocalTrainer(
            FraudDetectorMLP(), data, TrainerConfig(epochs=1, device="cpu")
        )
        trainer.fit()
        fresh = FraudDetectorMLP()
        fresh.load_state_dict(trainer.state_dict())  # raises on key mismatch

    def test_nonfinite_loss_raises(self, partition_csv: Path) -> None:
        torch.manual_seed(0)
        data = make_data(partition_csv)
        trainer = LocalTrainer(
            FraudDetectorMLP(), data, TrainerConfig(epochs=1, lr=1e6, device="cpu")
        )
        with pytest.raises(RuntimeError, match="non-finite|empty"):
            trainer.fit()


class TestDPTraining:
    def test_dp_run_trains_and_reports_epsilon(self, partition_csv: Path) -> None:
        torch.manual_seed(0)
        data = make_data(partition_csv)
        trainer = LocalTrainer(
            FraudDetectorMLP(),
            data,
            TrainerConfig(epochs=2, device="cpu", enable_dp=True),
        )
        history = trainer.fit()
        assert history[-1].epsilon is not None
        assert history[-1].epsilon > 0

    def test_epsilon_grows_with_steps(self, partition_csv: Path) -> None:
        torch.manual_seed(0)
        data = make_data(partition_csv)
        trainer = LocalTrainer(
            FraudDetectorMLP(),
            data,
            TrainerConfig(epochs=3, device="cpu", enable_dp=True),
        )
        history = trainer.fit()
        epsilons = [h.epsilon for h in history]
        assert all(e is not None for e in epsilons)
        assert epsilons == sorted(epsilons), "privacy budget must be monotone increasing"
        assert epsilons[0] < epsilons[-1]

    def test_optimizer_and_model_are_wrapped(self, partition_csv: Path) -> None:
        from opacus import GradSampleModule
        from opacus.optimizers import DPOptimizer

        torch.manual_seed(0)
        data = make_data(partition_csv)
        trainer = LocalTrainer(
            FraudDetectorMLP(),
            data,
            TrainerConfig(epochs=1, device="cpu", enable_dp=True),
        )
        assert isinstance(trainer.model, GradSampleModule)
        assert isinstance(trainer.optimizer, DPOptimizer)

    def test_state_dict_keys_are_clean_under_dp(self, partition_csv: Path) -> None:
        torch.manual_seed(0)
        data = make_data(partition_csv)
        trainer = LocalTrainer(
            FraudDetectorMLP(),
            data,
            TrainerConfig(epochs=1, device="cpu", enable_dp=True),
        )
        trainer.fit()
        sd = trainer.state_dict()
        assert not any(k.startswith("_module.") for k in sd)
        FraudDetectorMLP().load_state_dict(sd)

    def test_dp_weights_differ_from_non_dp(self, partition_csv: Path) -> None:
        """Same seed, same data: noise must make DP weights diverge."""
        data = make_data(partition_csv)

        torch.manual_seed(0)
        t_plain = LocalTrainer(
            FraudDetectorMLP(), data, TrainerConfig(epochs=1, device="cpu")
        )
        t_plain.fit()

        torch.manual_seed(0)
        t_dp = LocalTrainer(
            FraudDetectorMLP(),
            data,
            TrainerConfig(epochs=1, device="cpu", enable_dp=True),
        )
        t_dp.fit()

        sd_plain, sd_dp = t_plain.state_dict(), t_dp.state_dict()
        diffs = [
            (sd_plain[k] - sd_dp[k]).abs().max().item() for k in sd_plain
        ]
        assert max(diffs) > 1e-4, "DP-SGD produced identical weights — noise not applied?"


class TestEpsilonAccounting:
    """Week 2 deliverable: accounted ε must match RDP composition theory."""

    def test_engine_epsilon_matches_theory(self, partition_csv: Path) -> None:
        torch.manual_seed(0)
        data = make_data(partition_csv)
        config = TrainerConfig(epochs=2, device="cpu", enable_dp=True, delta=1e-5)
        trainer = LocalTrainer(FraudDetectorMLP(), data, config)
        trainer.fit()

        engine_eps = trainer.epsilon
        theory_eps = theoretical_epsilon_from_history(
            trainer.accountant_history(), delta=config.delta
        )
        assert engine_eps is not None
        assert engine_eps == pytest.approx(theory_eps, rel=0.01)

    def test_more_noise_means_less_epsilon(self, partition_csv: Path) -> None:
        data = make_data(partition_csv)

        def eps_for(sigma: float) -> float:
            torch.manual_seed(0)
            trainer = LocalTrainer(
                FraudDetectorMLP(),
                data,
                TrainerConfig(
                    epochs=1, device="cpu", enable_dp=True, noise_multiplier=sigma
                ),
            )
            trainer.fit()
            eps = trainer.epsilon
            assert eps is not None
            return eps

        assert eps_for(2.0) < eps_for(0.8), "higher σ must yield lower ε"

    def test_history_requires_dp(self, partition_csv: Path) -> None:
        data = make_data(partition_csv)
        trainer = LocalTrainer(FraudDetectorMLP(), data, TrainerConfig(device="cpu"))
        with pytest.raises(RuntimeError, match="enable_dp"):
            trainer.accountant_history()


class TestConfigValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"epochs": 0},
            {"lr": -0.1},
            {"enable_dp": True, "noise_multiplier": 0.0},
            {"enable_dp": True, "max_grad_norm": -1.0},
            {"enable_dp": True, "delta": 1.5},
            {"pos_weight_scale": 0.0},
        ],
    )
    def test_invalid_configs_rejected(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            TrainerConfig(**kwargs)
