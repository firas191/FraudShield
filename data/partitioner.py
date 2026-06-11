"""Non-IID partitioner: split transactions across three simulated banks.

The problem (discovered in Week 1 exploration):
    The spec wants Bank A = 60% of data with a "large transactions" profile,
    but only 3.2% of real transactions exceed €500. Hard Amount thresholds
    would give Bank A ~9k rows instead of ~145k. So banks cannot literally
    *be* their amount band — they can only be *skewed toward* it.

The mechanism (band-conditional assignment):
    Each row is assigned to a bank by sampling from P(bank | amount_band),
    a 3×3 column-stochastic matrix. The columns (bands LOW/MID/HIGH) are
    tuned so that, combined with the empirical band shares (~80/17/3), the
    marginal bank sizes land on 60/25/15 **in expectation** while each
    bank's amount distribution is distinctly skewed:

        Bank A receives 90% of all HIGH rows → fat right tail (corporate)
        Bank B receives 35% of MID rows      → mid-heavy (SME)
        Bank C is ~99% LOW rows              → retail, card-not-present

    This is statistical heterogeneity (feature-distribution skew), the
    realistic kind of Non-IID: a corporate bank still processes many small
    transactions, but its mix differs. It is what causes client drift in
    FedAvg and what FedProx (Week 4) is built to handle.

Order of operations — test set FIRST:
    The global test set is split off (stratified by Class) *before*
    partitioning. If we partitioned first and carved test data out of the
    banks, any later change to partition parameters would silently move
    rows between train and test, invalidating every previous benchmark.
    A fixed, seed-locked test set is the only trustworthy yardstick for
    the ≥90% AUC-ROC goal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

__all__ = [
    "AMOUNT_BAND_EDGES",
    "BAND_LABELS",
    "PartitionConfig",
    "split_global_test",
    "partition_by_bank",
    "summarize_partitions",
]

logger = logging.getLogger(__name__)

# Band edges over Amount (EUR): LOW ≤ 100 < MID ≤ 500 < HIGH  (spec §4.1)
AMOUNT_BAND_EDGES: tuple[float, float] = (100.0, 500.0)
BAND_LABELS: tuple[str, str, str] = ("LOW", "MID", "HIGH")

BankName = str


@dataclass(frozen=True)
class PartitionConfig:
    """Configuration for the Non-IID split.

    Attributes:
        assignment: ``bank → (P(bank|LOW), P(bank|MID), P(bank|HIGH))``.
            Each band column must sum to 1 across banks. Defaults are
            tuned against the empirical band shares (80.2/16.6/3.2%) to
            yield ~60/25/15 bank sizes.
        test_fraction: Share of the full dataset held out as the global
            test set before partitioning.
        seed: Seed for both the test split and bank assignment. The same
            seed must always reproduce byte-identical partitions —
            otherwise experiments are not comparable across weeks.
    """

    assignment: dict[BankName, tuple[float, float, float]] = field(
        default_factory=lambda: {
            "bank_a": (0.588, 0.600, 0.90),
            "bank_b": (0.237, 0.350, 0.07),
            "bank_c": (0.175, 0.050, 0.03),
        }
    )
    test_fraction: float = 0.15
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0.0 < self.test_fraction < 0.5:
            raise ValueError(f"test_fraction must be in (0, 0.5), got {self.test_fraction}")
        if len(self.assignment) < 2:
            raise ValueError("need at least two banks")
        matrix = np.array(list(self.assignment.values()), dtype=np.float64)
        if matrix.shape[1] != len(BAND_LABELS):
            raise ValueError(
                f"each bank needs {len(BAND_LABELS)} band probabilities, got {matrix.shape[1]}"
            )
        if (matrix < 0).any() or (matrix > 1).any():
            raise ValueError("assignment probabilities must lie in [0, 1]")
        col_sums = matrix.sum(axis=0)
        if not np.allclose(col_sums, 1.0, atol=1e-6):
            raise ValueError(
                f"P(bank | band) must sum to 1 per band; column sums are {col_sums.tolist()}"
            )

    @property
    def bank_names(self) -> list[BankName]:
        return list(self.assignment.keys())


def _band_index(amounts: pd.Series) -> np.ndarray:
    """Map Amount values to band indices 0=LOW, 1=MID, 2=HIGH."""
    return np.digitize(amounts.to_numpy(), bins=np.array(AMOUNT_BAND_EDGES), right=True)


def split_global_test(
    df: pd.DataFrame, config: PartitionConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the global test set, stratified on the Class label.

    Stratification matters: with a 0.17% positive rate, a random 15%
    split could by chance contain too few frauds to estimate AUC-ROC
    with acceptable variance. Stratified splitting guarantees the test
    set carries its proportional ~74 fraud cases.

    Args:
        df: Full dataset including the ``Class`` column.
        config: Partition configuration (uses ``test_fraction``, ``seed``).

    Returns:
        ``(train_df, test_df)`` with reset indices.
    """
    if "Class" not in df.columns:
        raise ValueError("dataframe must contain a 'Class' column")
    train_df, test_df = train_test_split(
        df,
        test_size=config.test_fraction,
        stratify=df["Class"],
        random_state=config.seed,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def partition_by_bank(
    train_df: pd.DataFrame, config: PartitionConfig
) -> dict[BankName, pd.DataFrame]:
    """Assign every training row to a bank via P(bank | amount_band).

    Vectorized: one multinomial draw per band rather than per row, so
    partitioning 240k rows takes milliseconds and is reproducible from
    the seed.

    Args:
        train_df: Training split (after the global test set is removed).
        config: Partition configuration.

    Returns:
        ``bank_name → dataframe`` (indices reset, original row order
        preserved within each bank). Every input row appears in exactly
        one bank.
    """
    if "Amount" not in train_df.columns:
        raise ValueError("dataframe must contain an 'Amount' column")

    rng = np.random.default_rng(config.seed)
    banks = config.bank_names
    matrix = np.array([config.assignment[b] for b in banks], dtype=np.float64)

    bands = _band_index(train_df["Amount"])
    assigned = np.empty(len(train_df), dtype=np.int64)
    for band in range(len(BAND_LABELS)):
        mask = bands == band
        n = int(mask.sum())
        if n == 0:
            continue
        # Renormalize defensively against float drift; validated at config time.
        probs = matrix[:, band] / matrix[:, band].sum()
        assigned[mask] = rng.choice(len(banks), size=n, p=probs)

    partitions: dict[BankName, pd.DataFrame] = {}
    for i, bank in enumerate(banks):
        part = train_df.iloc[assigned == i].reset_index(drop=True)
        if part.empty:
            raise RuntimeError(f"partition for {bank} is empty — check assignment matrix")
        partitions[bank] = part
    return partitions


def summarize_partitions(
    partitions: dict[BankName, pd.DataFrame], total_rows: int
) -> pd.DataFrame:
    """Build a per-bank summary table (size, share, fraud stats, amount mix).

    Used both for CLI reporting and for assertions in tests. Fraud counts
    per bank matter operationally: a bank with too few positive samples
    cannot learn anything fraud-specific locally, which is precisely the
    scenario where federation helps — but we need to *know* that's the
    case rather than discover it from confusing training curves.

    Args:
        partitions: Output of :func:`partition_by_bank`.
        total_rows: Row count of the pre-partition training split.

    Returns:
        One row per bank with columns: rows, share, frauds, fraud_rate,
        pct_low, pct_mid, pct_high, amount_p50, amount_p95.
    """
    records = []
    for bank, df in partitions.items():
        bands = _band_index(df["Amount"])
        records.append(
            {
                "bank": bank,
                "rows": len(df),
                "share": len(df) / total_rows,
                "frauds": int(df["Class"].sum()),
                "fraud_rate": float(df["Class"].mean()),
                "pct_low": float((bands == 0).mean()),
                "pct_mid": float((bands == 1).mean()),
                "pct_high": float((bands == 2).mean()),
                "amount_p50": float(df["Amount"].quantile(0.50)),
                "amount_p95": float(df["Amount"].quantile(0.95)),
            }
        )
    return pd.DataFrame.from_records(records).set_index("bank")
