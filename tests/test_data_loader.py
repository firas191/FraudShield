"""Tests for the per-bank data module — uses a synthetic partition CSV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from client.data_loader import FEATURE_COLUMNS, BankDataConfig, BankDataModule

N_ROWS = 5_000
FRAUD_RATE = 0.02


@pytest.fixture(scope="module")
def partition_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({f"V{i}": rng.normal(size=N_ROWS) for i in range(1, 29)})
    df.insert(0, "Time", rng.uniform(0, 172_800, size=N_ROWS))
    df["Amount"] = rng.lognormal(mean=3.0, sigma=1.6, size=N_ROWS)
    df["Class"] = (rng.uniform(size=N_ROWS) < FRAUD_RATE).astype(np.int64)
    path = tmp_path_factory.mktemp("partitions") / "bank_test.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def module(partition_csv: Path) -> BankDataModule:
    dm = BankDataModule(BankDataConfig(csv_path=partition_csv, batch_size=256))
    dm.setup()
    return dm


class TestSetup:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        dm = BankDataModule(BankDataConfig(csv_path=tmp_path / "nope.csv"))
        with pytest.raises(FileNotFoundError):
            dm.setup()

    def test_missing_columns_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.csv"
        pd.DataFrame({"Amount": [1.0, 2.0], "Class": [0, 1]}).to_csv(path, index=False)
        dm = BankDataModule(BankDataConfig(csv_path=path))
        with pytest.raises(ValueError, match="missing columns"):
            dm.setup()

    def test_use_before_setup_raises(self, partition_csv: Path) -> None:
        dm = BankDataModule(BankDataConfig(csv_path=partition_csv))
        with pytest.raises(RuntimeError, match="setup"):
            dm.train_loader()

    def test_invalid_config_rejected(self, partition_csv: Path) -> None:
        with pytest.raises(ValueError):
            BankDataConfig(csv_path=partition_csv, batch_size=0)
        with pytest.raises(ValueError):
            BankDataConfig(csv_path=partition_csv, val_fraction=0.7)


class TestTensors:
    def test_batch_shapes_and_dtypes(self, module: BankDataModule) -> None:
        x, y = next(iter(module.train_loader()))
        assert x.shape == (256, len(FEATURE_COLUMNS))
        assert y.shape == (256, 1)
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32

    def test_split_sizes(self, module: BankDataModule) -> None:
        n_train = module.train_size
        n_val = sum(len(y) for _, y in module.val_loader())
        assert n_train + n_val == N_ROWS
        assert n_val == pytest.approx(0.1 * N_ROWS, rel=0.02)

    def test_pos_weight_matches_train_imbalance(self, module: BankDataModule) -> None:
        ys = torch.cat([y for _, y in module.train_loader()])
        n_pos = ys.sum().item()
        expected = (len(ys) - n_pos) / n_pos
        assert module.pos_weight.item() == pytest.approx(expected, rel=1e-6)

    def test_features_are_scaled(self, module: BankDataModule) -> None:
        """After log1p + RobustScaler, no feature should retain raw-Amount magnitude."""
        xs = torch.cat([x for x, _ in module.train_loader()])
        assert xs.abs().max().item() < 50.0
        medians = xs.median(dim=0).values
        assert medians.abs().max().item() < 1.0  # RobustScaler centers on median

    def test_val_has_fraud_cases(self, module: BankDataModule) -> None:
        """Stratification must guarantee positives in val, else AUC is undefined."""
        ys = torch.cat([y for _, y in module.val_loader()])
        assert ys.sum().item() >= 2

    def test_train_loader_shuffles_deterministically(self, module: BankDataModule) -> None:
        first_a = next(iter(module.train_loader()))[0]
        first_b = next(iter(module.train_loader()))[0]
        assert torch.equal(first_a, first_b)  # same seed → same order

    def test_scaler_fit_on_train_only(self, partition_csv: Path) -> None:
        """The scaler's center must equal the TRAIN split's medians exactly.

        Reconstructs the same stratified split (same seed) and compares
        the fitted ``RobustScaler.center_`` against train-only medians.
        If someone 'fixes' the pipeline to fit on the full dataframe,
        the centers shift and this fails.
        """
        import numpy as np
        from sklearn.model_selection import train_test_split

        cfg = BankDataConfig(csv_path=partition_csv)
        dm = BankDataModule(cfg)
        dm.setup()

        df = pd.read_csv(partition_csv)
        train_df, _ = train_test_split(
            df, test_size=cfg.val_fraction, stratify=df["Class"], random_state=cfg.seed
        )
        raw = train_df.loc[:, list(FEATURE_COLUMNS)].copy()
        raw["Amount"] = np.log1p(raw["Amount"])
        expected_center = np.median(raw.to_numpy(dtype=np.float64), axis=0)

        full = df.loc[:, list(FEATURE_COLUMNS)].copy()
        full["Amount"] = np.log1p(full["Amount"])
        full_center = np.median(full.to_numpy(dtype=np.float64), axis=0)

        assert dm._scaler is not None
        np.testing.assert_allclose(dm._scaler.center_, expected_center, rtol=1e-9)
        # Sanity: train-only and full-data centers genuinely differ.
        assert not np.allclose(expected_center, full_center)
