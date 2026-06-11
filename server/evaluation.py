"""Server-side evaluation of the global model on the held-out test set.

The scaling problem (an honest FL wrinkle):
    Each bank scaled its features with a *local* RobustScaler — there is
    no shared scaler, because sharing exact normalization statistics
    across institutions would itself leak data. So what scaling does the
    server apply at evaluation time?

    We fit a RobustScaler on the test set's own features (labels are
    never touched). This is mildly transductive but label-free, and
    because V1–V28 are PCA components with near-identical distributions
    everywhere, the test scaler closely approximates each bank's local
    mapping; log1p tames the one genuinely shifted feature (Amount).

    The production-grade alternative — banks agree on global
    normalization constants via a DP-protected secure aggregation of
    their quantiles — is documented here as future work and would be a
    strong talking point in the write-up.

Threshold choice:
    Precision/recall/F1 are reported at the threshold that maximizes F1
    *on the test scores of the previous evaluation*? No — simpler and
    honest: AUC-ROC is the primary, threshold-free metric (spec G6).
    F1/precision/recall are reported at 0.5 for continuity with client
    logs, plus ``best_f1`` over the PR curve as the achievable operating
    point, so DP's threshold-shifting (seen in Week 2) stays visible
    without us quietly optimizing the headline number.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import RobustScaler

from client.data_loader import FEATURE_COLUMNS, LABEL_COLUMN
from model.fraud_mlp import FraudDetectorMLP

__all__ = ["GlobalTestMetrics", "GlobalEvaluator"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GlobalTestMetrics:
    """Evaluation of one global model on the held-out test set."""

    auc_roc: float
    precision: float
    recall: float
    f1: float
    best_f1: float
    best_threshold: float
    test_loss: float
    n_samples: int
    n_positives: int

    def as_dict(self) -> dict[str, float]:
        return {
            "auc_roc": self.auc_roc,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "best_f1": self.best_f1,
            "best_threshold": self.best_threshold,
            "test_loss": self.test_loss,
        }


class GlobalEvaluator:
    """Loads the global test set once; evaluates any global state dict.

    CPU-only by design: evaluation runs inside the server process whose
    job is coordination, and 42k rows through a 100k-param MLP takes
    well under a second on CPU. Keeping the server CUDA-free also lets
    it run in a slim Docker image (Week 7).
    """

    def __init__(self, test_csv: str | Path) -> None:
        path = Path(test_csv)
        if not path.is_file():
            raise FileNotFoundError(
                f"global test set not found: {path} — run: "
                "python -m data.prepare_dataset partition"
            )
        df = pd.read_csv(path)
        missing = [c for c in (*FEATURE_COLUMNS, LABEL_COLUMN) if c not in df.columns]
        if missing:
            raise ValueError(f"global test set missing columns: {missing}")

        features = df.loc[:, list(FEATURE_COLUMNS)].copy()
        features["Amount"] = np.log1p(features["Amount"])
        x = RobustScaler().fit_transform(features.to_numpy(dtype=np.float64))
        self._x = torch.from_numpy(x.astype(np.float32))
        self._y = torch.from_numpy(df[LABEL_COLUMN].to_numpy(dtype=np.float32)).unsqueeze(1)
        self._n_pos = int(self._y.sum().item())
        if self._n_pos < 10:
            logger.warning(
                "global test set has only %d positives — AUC will be high-variance",
                self._n_pos,
            )
        logger.info(
            "evaluator ready: %d test rows, %d frauds", len(self._y), self._n_pos
        )

    @torch.no_grad()
    def evaluate(self, state_dict: dict[str, torch.Tensor]) -> GlobalTestMetrics:
        """Score a global model.

        Args:
            state_dict: Clean (unprefixed) FraudDetectorMLP weights.

        Returns:
            Metrics on the held-out global test set.
        """
        model = FraudDetectorMLP()
        model.load_state_dict(state_dict)
        model.eval()

        logits = model(self._x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, self._y
        ).item()
        probs = torch.sigmoid(logits).squeeze(1).numpy()
        labels = self._y.squeeze(1).numpy()

        auc = float(roc_auc_score(labels, probs))
        preds = (probs >= 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        # Best achievable F1 over the PR curve (diagnostic, not headline).
        pr, rc, thresholds = precision_recall_curve(labels, probs)
        f1_curve = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
        best_idx = int(np.argmax(f1_curve))

        return GlobalTestMetrics(
            auc_roc=auc,
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            best_f1=float(f1_curve[best_idx]),
            best_threshold=float(thresholds[best_idx]),
            test_loss=loss,
            n_samples=int(len(labels)),
            n_positives=self._n_pos,
        )
