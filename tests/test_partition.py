"""Tests for the Non-IID partitioner — run on synthetic data, no Kaggle CSV needed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.partitioner import (
    PartitionConfig,
    partition_by_bank,
    split_global_test,
    summarize_partitions,
)

N_ROWS = 20_000
FRAUD_RATE = 0.01  # higher than reality so small splits still carry positives


@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    """Dataset shaped like creditcard.csv with a realistic Amount skew."""
    rng = np.random.default_rng(0)
    amounts = rng.lognormal(mean=3.0, sigma=1.6, size=N_ROWS)  # heavy right tail
    df = pd.DataFrame({f"V{i}": rng.normal(size=N_ROWS) for i in range(1, 29)})
    df.insert(0, "Time", rng.uniform(0, 172_800, size=N_ROWS))
    df["Amount"] = amounts
    df["Class"] = (rng.uniform(size=N_ROWS) < FRAUD_RATE).astype(np.int64)
    return df


@pytest.fixture(scope="module")
def config() -> PartitionConfig:
    return PartitionConfig(seed=42)


class TestGlobalTestSplit:
    def test_sizes(self, synthetic_df: pd.DataFrame, config: PartitionConfig) -> None:
        train, test = split_global_test(synthetic_df, config)
        assert len(train) + len(test) == len(synthetic_df)
        assert len(test) == pytest.approx(config.test_fraction * len(synthetic_df), rel=0.01)

    def test_stratification_preserves_fraud_rate(
        self, synthetic_df: pd.DataFrame, config: PartitionConfig
    ) -> None:
        train, test = split_global_test(synthetic_df, config)
        global_rate = synthetic_df["Class"].mean()
        assert test["Class"].mean() == pytest.approx(global_rate, rel=0.15)
        assert train["Class"].mean() == pytest.approx(global_rate, rel=0.15)

    def test_deterministic(self, synthetic_df: pd.DataFrame, config: PartitionConfig) -> None:
        _, test1 = split_global_test(synthetic_df, config)
        _, test2 = split_global_test(synthetic_df, config)
        pd.testing.assert_frame_equal(test1, test2)

    def test_requires_class_column(self, config: PartitionConfig) -> None:
        with pytest.raises(ValueError, match="Class"):
            split_global_test(pd.DataFrame({"Amount": [1.0]}), config)


class TestBankPartition:
    def test_every_row_assigned_exactly_once(
        self, synthetic_df: pd.DataFrame, config: PartitionConfig
    ) -> None:
        train, _ = split_global_test(synthetic_df, config)
        parts = partition_by_bank(train, config)
        assert sum(len(p) for p in parts.values()) == len(train)

    def test_deterministic_given_seed(
        self, synthetic_df: pd.DataFrame, config: PartitionConfig
    ) -> None:
        train, _ = split_global_test(synthetic_df, config)
        parts1 = partition_by_bank(train, config)
        parts2 = partition_by_bank(train, config)
        for bank in parts1:
            pd.testing.assert_frame_equal(parts1[bank], parts2[bank])

    def test_different_seed_changes_assignment(
        self, synthetic_df: pd.DataFrame, config: PartitionConfig
    ) -> None:
        train, _ = split_global_test(synthetic_df, config)
        parts1 = partition_by_bank(train, config)
        parts2 = partition_by_bank(train, PartitionConfig(seed=7))
        assert len(parts1["bank_a"]) != len(parts2["bank_a"]) or not parts1[
            "bank_a"
        ].equals(parts2["bank_a"])

    def test_banks_are_non_iid_in_amount(
        self, synthetic_df: pd.DataFrame, config: PartitionConfig
    ) -> None:
        """Bank A must hold a visibly fatter Amount tail than Bank C."""
        train, _ = split_global_test(synthetic_df, config)
        parts = partition_by_bank(train, config)
        p95_a = parts["bank_a"]["Amount"].quantile(0.95)
        p95_c = parts["bank_c"]["Amount"].quantile(0.95)
        assert p95_a > p95_c * 1.5, (
            f"expected bank_a tail >> bank_c tail, got p95 {p95_a:.1f} vs {p95_c:.1f}"
        )

    def test_high_band_concentrated_in_bank_a(
        self, synthetic_df: pd.DataFrame, config: PartitionConfig
    ) -> None:
        train, _ = split_global_test(synthetic_df, config)
        parts = partition_by_bank(train, config)
        high_counts = {b: int((p["Amount"] > 500).sum()) for b, p in parts.items()}
        total_high = sum(high_counts.values())
        assert high_counts["bank_a"] / total_high > 0.8

    def test_summary_table(self, synthetic_df: pd.DataFrame, config: PartitionConfig) -> None:
        train, _ = split_global_test(synthetic_df, config)
        parts = partition_by_bank(train, config)
        summary = summarize_partitions(parts, total_rows=len(train))
        assert summary["share"].sum() == pytest.approx(1.0)
        assert (summary["frauds"] > 0).all()


class TestPartitionConfigValidation:
    def test_default_is_valid(self) -> None:
        PartitionConfig()

    def test_rejects_columns_not_summing_to_one(self) -> None:
        with pytest.raises(ValueError, match="sum to 1"):
            PartitionConfig(
                assignment={
                    "bank_a": (0.5, 0.5, 0.5),
                    "bank_b": (0.4, 0.5, 0.5),
                    "bank_c": (0.2, 0.0, 0.0),
                }
            )

    def test_rejects_negative_probability(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            PartitionConfig(
                assignment={
                    "bank_a": (1.1, 0.6, 0.9),
                    "bank_b": (-0.1, 0.4, 0.1),
                }
            )

    def test_rejects_bad_test_fraction(self) -> None:
        with pytest.raises(ValueError, match="test_fraction"):
            PartitionConfig(test_fraction=0.9)
