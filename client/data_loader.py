"""Per-bank dataset loading and preprocessing.

Privacy-critical design decision — **all preprocessing statistics are
computed from the bank's own training rows only**:

* The ``RobustScaler`` is fit on the local train split, never on global
  data. In a real deployment no global statistics exist (banks can't
  pool their data — that's the entire premise), so fitting locally is
  both the honest simulation and a privacy requirement: scaler medians/
  quantiles computed across banks would already be a small cross-
  institution data leak outside the DP mechanism.
* The trade-off: each bank's features land on slightly different scales,
  adding to the Non-IID drift. That is realistic, and it is FedProx's
  job (Week 4) to absorb it — not the data pipeline's job to hide it.

Preprocessing choices:

* ``Amount → log1p(Amount)`` before scaling. Amounts span 0–25,691 with a
  heavy right tail; without the log, a single large transaction dominates
  the feature and, later under DP-SGD, inflates per-sample gradient norms
  so the clipping threshold C wastes its budget on scale rather than
  signal.
* ``RobustScaler`` (median/IQR) instead of ``StandardScaler`` (mean/std):
  fraud data is outlier-rich by nature, and mean/std are exactly the
  statistics outliers distort.
* V1–V28 are PCA outputs and already roughly centered; scaling them again
  is a no-op-ish transformation that keeps the pipeline uniform.
* Train/val split is stratified on Class: with fraud rates near 0.1%, an
  unstratified 10% val split could contain zero frauds, making AUC-ROC
  undefined.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset

__all__ = ["FEATURE_COLUMNS", "LABEL_COLUMN", "BankDataConfig", "BankDataModule"]

logger = logging.getLogger(__name__)

# Canonical feature order — must match ModelConfig.input_dim == 30 and be
# identical on every client, otherwise aggregated weights are meaningless.
FEATURE_COLUMNS: tuple[str, ...] = ("Time", *(f"V{i}" for i in range(1, 29)), "Amount")
LABEL_COLUMN: str = "Class"


@dataclass(frozen=True)
class BankDataConfig:
    """Configuration for one bank's local data pipeline.

    Attributes:
        csv_path: Path to this bank's partition CSV.
        batch_size: Training batch size. 512 is a deliberate default:
            large batches improve the signal-to-noise ratio under DP-SGD
            (noise is added once per batch, signal scales with batch
            size), and the MLP is small enough that 4GB VRAM handles
            per-sample gradients at this size.
        val_fraction: Local validation share, stratified on Class.
        seed: Reproducibility seed for the split and loader shuffling.
        device: Used only to decide ``pin_memory`` for host→GPU copies.
    """

    csv_path: str | Path
    batch_size: int = 512
    val_fraction: float = 0.1
    seed: int = 42
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if not 0.0 < self.val_fraction < 0.5:
            raise ValueError(f"val_fraction must be in (0, 0.5), got {self.val_fraction}")


class BankDataModule:
    """Loads one bank's partition and exposes train/val DataLoaders.

    Lifecycle: construct → :meth:`setup` → use ``train_loader()`` /
    ``val_loader()`` / ``pos_weight``. Setup is explicit (not done in
    ``__init__``) so construction stays cheap and failures surface where
    they can be handled — inside the client's round loop.
    """

    def __init__(self, config: BankDataConfig) -> None:
        self.config = config
        self._scaler: RobustScaler | None = None
        self._train_ds: TensorDataset | None = None
        self._val_ds: TensorDataset | None = None
        self._pos_weight: float | None = None

    # ------------------------------------------------------------------ #
    def setup(self) -> None:
        """Read the CSV, split, fit the scaler on train, build tensors.

        Raises:
            FileNotFoundError: If the partition CSV is missing.
            ValueError: If required columns are absent or a split ends up
                with fewer than 2 fraud cases (training would be
                meaningless and AUC ill-defined).
        """
        path = Path(self.config.csv_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"partition not found: {path} — run: python -m data.prepare_dataset partition"
            )
        df = pd.read_csv(path)
        missing = [c for c in (*FEATURE_COLUMNS, LABEL_COLUMN) if c not in df.columns]
        if missing:
            raise ValueError(f"partition {path.name} missing columns: {missing}")

        train_df, val_df = train_test_split(
            df,
            test_size=self.config.val_fraction,
            stratify=df[LABEL_COLUMN],
            random_state=self.config.seed,
        )
        for name, split in (("train", train_df), ("val", val_df)):
            n_pos = int(split[LABEL_COLUMN].sum())
            if n_pos < 2:
                raise ValueError(
                    f"{path.name} {name} split has {n_pos} fraud cases — too few to train/"
                    "evaluate; partition is degenerate"
                )

        x_train = self._featurize(train_df)
        x_val = self._featurize(val_df)

        # Fit on LOCAL TRAIN ONLY — see module docstring.
        self._scaler = RobustScaler().fit(x_train)
        x_train = self._scaler.transform(x_train)
        x_val = self._scaler.transform(x_val)

        y_train = train_df[LABEL_COLUMN].to_numpy(dtype=np.float32)
        y_val = val_df[LABEL_COLUMN].to_numpy(dtype=np.float32)

        n_pos = float(y_train.sum())
        self._pos_weight = (len(y_train) - n_pos) / n_pos

        self._train_ds = TensorDataset(
            torch.from_numpy(x_train.astype(np.float32)),
            torch.from_numpy(y_train).unsqueeze(1),
        )
        self._val_ds = TensorDataset(
            torch.from_numpy(x_val.astype(np.float32)),
            torch.from_numpy(y_val).unsqueeze(1),
        )
        logger.info(
            "%s: train=%d (frauds=%d) val=%d (frauds=%d) pos_weight=%.1f",
            path.name, len(y_train), int(y_train.sum()), len(y_val), int(y_val.sum()),
            self._pos_weight,
        )

    @staticmethod
    def _featurize(df: pd.DataFrame) -> np.ndarray:
        """Assemble the (n, 30) raw feature matrix with log-transformed Amount."""
        features = df.loc[:, list(FEATURE_COLUMNS)].copy()
        features["Amount"] = np.log1p(features["Amount"])
        return features.to_numpy(dtype=np.float64)

    # ------------------------------------------------------------------ #
    @property
    def pos_weight(self) -> torch.Tensor:
        """``BCEWithLogitsLoss`` pos_weight from this bank's train split."""
        self._require_setup()
        return torch.tensor([self._pos_weight], dtype=torch.float32)

    @property
    def train_size(self) -> int:
        """Number of local training samples (= n_k in FedAvg weighting)."""
        self._require_setup()
        return len(self._train_ds)  # type: ignore[arg-type]

    def train_loader(self) -> DataLoader:
        """Shuffled training loader.

        Note: when Opacus wraps training (Phase B), it replaces this
        loader's sampler with Poisson sampling — required for the DP
        guarantee, since the privacy analysis assumes each sample is
        included in a batch independently with probability q.
        """
        self._require_setup()
        gen = torch.Generator().manual_seed(self.config.seed)
        return DataLoader(
            self._train_ds,  # type: ignore[arg-type]
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=gen,
            num_workers=0,  # in-memory tensors: workers add IPC cost, no benefit
            pin_memory=self.config.device.startswith("cuda"),
            drop_last=False,
        )

    def val_loader(self) -> DataLoader:
        """Deterministic validation loader (no shuffling)."""
        self._require_setup()
        return DataLoader(
            self._val_ds,  # type: ignore[arg-type]
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.config.device.startswith("cuda"),
        )

    def _require_setup(self) -> None:
        if self._train_ds is None:
            raise RuntimeError("call setup() before using the data module")
