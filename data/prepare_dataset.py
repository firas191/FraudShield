"""Download, explore, and partition the Kaggle Credit Card Fraud dataset.

Usage (from repo root, venv active):

    python -m data.prepare_dataset download
    python -m data.prepare_dataset explore
    python -m data.prepare_dataset partition [--seed 42] [--test-fraction 0.15]

The ``download`` command uses ``kagglehub``, which caches the dataset under
``~/.cache/kagglehub`` and copies ``creditcard.csv`` into ``data/raw/``.
The dataset is public; if Kaggle still demands credentials, place your
``kaggle.json`` (Account → Create New Token) in ``~/.kaggle/`` — it is
git-ignored here and must never be committed.

The ``explore`` command prints the dataset facts that drive design
decisions in later weeks: class imbalance (→ pos_weight), Amount
distribution per class (→ Non-IID partition thresholds in Week 2), and
feature scale ranges (→ which features need scaling before DP training,
since unscaled Amount/Time would dominate the gradient norm and get
disproportionately clipped by DP-SGD).

The ``partition`` command (Week 2) holds out a stratified global test
set, then assigns the remaining rows to three banks via band-conditional
sampling — see ``data/partitioner.py`` for the rationale.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

from data.partitioner import (
    PartitionConfig,
    partition_by_bank,
    split_global_test,
    summarize_partitions,
)

logger = logging.getLogger("fraudshield.data")

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
RAW_CSV = RAW_DIR / "creditcard.csv"
PARTITIONS_DIR = REPO_ROOT / "data" / "partitions"
GLOBAL_TEST_CSV = REPO_ROOT / "data" / "global_test.csv"
KAGGLE_DATASET = "mlg-ulb/creditcardfraud"

EXPECTED_ROWS = 284_807
EXPECTED_FRAUDS = 492
EXPECTED_COLUMNS = 31  # Time, V1..V28, Amount, Class


def download() -> Path:
    """Fetch the dataset via kagglehub and place creditcard.csv in data/raw/.

    Returns:
        Path to the local ``creditcard.csv``.

    Raises:
        FileNotFoundError: If the downloaded archive does not contain
            ``creditcard.csv``.
        RuntimeError: If kagglehub fails (network, auth).
    """
    if RAW_CSV.is_file():
        logger.info("dataset already present at %s — skipping download", RAW_CSV)
        return RAW_CSV

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is not installed — run: pip install -r requirements.txt"
        ) from exc

    logger.info("downloading %s via kagglehub ...", KAGGLE_DATASET)
    try:
        dataset_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    except Exception as exc:
        raise RuntimeError(
            f"kagglehub download failed: {exc}\n"
            "Fallback: download manually from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            f"and place creditcard.csv at {RAW_CSV}"
        ) from exc

    source = next(dataset_dir.rglob("creditcard.csv"), None)
    if source is None:
        raise FileNotFoundError(f"creditcard.csv not found inside {dataset_dir}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, RAW_CSV)
    logger.info("dataset ready at %s", RAW_CSV)
    return RAW_CSV


def validate(df: pd.DataFrame) -> None:
    """Sanity-check the dataset matches the published Kaggle version.

    Catches silently truncated downloads — a corrupted CSV would
    otherwise surface much later as inexplicable training behavior.

    Raises:
        ValueError: On row/column/fraud-count mismatch or NaNs.
    """
    if df.shape != (EXPECTED_ROWS, EXPECTED_COLUMNS):
        raise ValueError(
            f"unexpected shape {df.shape}, expected ({EXPECTED_ROWS}, {EXPECTED_COLUMNS}) "
            "— the download may be truncated; delete data/raw/ and retry"
        )
    fraud_count = int(df["Class"].sum())
    if fraud_count != EXPECTED_FRAUDS:
        raise ValueError(f"expected {EXPECTED_FRAUDS} fraud rows, found {fraud_count}")
    if df.isna().any().any():
        raise ValueError("dataset contains NaNs — corrupted download")


def explore() -> dict[str, float]:
    """Print the exploration summary and return key statistics.

    Returns:
        Dict with ``fraud_rate``, ``pos_weight``, and Amount quantiles —
        the numbers Week 2 (partitioner) and Week 4 (loss weighting) need.
    """
    if not RAW_CSV.is_file():
        raise FileNotFoundError(
            f"{RAW_CSV} missing — run: python -m data.prepare_dataset download"
        )
    df = pd.read_csv(RAW_CSV)
    validate(df)

    n = len(df)
    frauds = int(df["Class"].sum())
    fraud_rate = frauds / n
    # pos_weight for BCEWithLogitsLoss: ratio of negatives to positives.
    pos_weight = (n - frauds) / frauds

    amount = df["Amount"]
    fraud_amount = df.loc[df["Class"] == 1, "Amount"]
    stats: dict[str, float] = {
        "rows": float(n),
        "frauds": float(frauds),
        "fraud_rate": fraud_rate,
        "pos_weight": pos_weight,
        "amount_p50": float(amount.quantile(0.50)),
        "amount_p95": float(amount.quantile(0.95)),
        "amount_max": float(amount.max()),
        "fraud_amount_p50": float(fraud_amount.quantile(0.50)),
        "time_span_hours": float((df["Time"].max() - df["Time"].min()) / 3600.0),
    }

    print("\n=== Kaggle Credit Card Fraud — exploration ===")
    print(f"rows: {n:,}   frauds: {frauds}   fraud rate: {fraud_rate:.4%}")
    print(f"suggested BCEWithLogitsLoss pos_weight: {pos_weight:.1f}")
    print(f"Amount  p50={stats['amount_p50']:.2f}  p95={stats['amount_p95']:.2f}  "
          f"max={stats['amount_max']:.2f}")
    print(f"Fraud Amount p50={stats['fraud_amount_p50']:.2f}")
    print(f"Time span: {stats['time_span_hours']:.1f} hours (2 days)")

    print("\nNon-IID partition preview (spec §4.1 thresholds):")
    bands = {
        "Bank A (Tunis)  Amount > 500": df["Amount"] > 500,
        "Bank B (Sfax)   100 < Amount <= 500": (df["Amount"] > 100) & (df["Amount"] <= 500),
        "Bank C (Sousse) Amount <= 100": df["Amount"] <= 100,
    }
    for label, mask in bands.items():
        sub = df[mask]
        share = len(sub) / n
        rate = sub["Class"].mean()
        print(f"  {label:42s} rows={len(sub):>7,} ({share:5.1%})  fraud rate={rate:.4%}")
    print(
        "\nNote: raw Amount thresholds give a different size split than the "
        "spec's 60/25/15 target — Week 2's partitioner reconciles this by "
        "sampling within bands rather than using hard thresholds alone."
    )
    return stats


def partition(config: PartitionConfig) -> None:
    """Create bank partitions and the global test set on disk.

    Writes ``data/partitions/bank_{a,b,c}.csv`` and
    ``data/global_test.csv``, then prints the per-bank summary. Refuses
    to overwrite existing partitions unless they are deleted first —
    silently regenerating partitions mid-project would invalidate every
    experiment logged against the old split.
    """
    if not RAW_CSV.is_file():
        raise FileNotFoundError(
            f"{RAW_CSV} missing — run: python -m data.prepare_dataset download"
        )
    existing = sorted(PARTITIONS_DIR.glob("*.csv")) if PARTITIONS_DIR.exists() else []
    if existing or GLOBAL_TEST_CSV.exists():
        raise FileExistsError(
            "partitions already exist — delete data/partitions/ and "
            "data/global_test.csv explicitly if you really intend to regenerate "
            "(this invalidates all previously logged experiments)"
        )

    df = pd.read_csv(RAW_CSV)
    validate(df)

    train_df, test_df = split_global_test(df, config)
    partitions = partition_by_bank(train_df, config)

    PARTITIONS_DIR.mkdir(parents=True, exist_ok=True)
    for bank, part in partitions.items():
        out = PARTITIONS_DIR / f"{bank}.csv"
        part.to_csv(out, index=False)
        logger.info("wrote %s (%d rows)", out, len(part))
    test_df.to_csv(GLOBAL_TEST_CSV, index=False)
    logger.info("wrote %s (%d rows, %d frauds)", GLOBAL_TEST_CSV, len(test_df),
                int(test_df["Class"].sum()))

    summary = summarize_partitions(partitions, total_rows=len(train_df))
    print(f"\n=== Non-IID partitions (seed={config.seed}, "
          f"test_fraction={config.test_fraction}) ===")
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 140):
        print(summary)
    print(
        f"\nglobal test: {len(test_df):,} rows, {int(test_df['Class'].sum())} frauds "
        f"({test_df['Class'].mean():.4%})"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="fetch creditcard.csv into data/raw/")
    sub.add_parser("explore", help="print dataset statistics")
    part_parser = sub.add_parser("partition", help="create Non-IID bank partitions")
    part_parser.add_argument("--seed", type=int, default=42)
    part_parser.add_argument("--test-fraction", type=float, default=0.15)
    args = parser.parse_args(argv)

    if args.command == "download":
        download()
    elif args.command == "explore":
        explore()
    elif args.command == "partition":
        partition(PartitionConfig(test_fraction=args.test_fraction, seed=args.seed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
