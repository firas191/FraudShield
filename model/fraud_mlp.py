"""Fraud detection MLP — the global model trained federatedly across banks.

Architecture (spec §6.4):
    Input(30) → Linear(256) → GroupNorm → GELU → Dropout(0.3)
              → Linear(128) → GroupNorm → GELU → Dropout(0.3)
              → Linear(64)  → GroupNorm → GELU
              → Linear(1)   [raw logit]

Design decisions (and why):

* **Raw logits, no Sigmoid layer.** The spec pairs the model with
  ``BCEWithLogitsLoss``, which applies the sigmoid internally using the
  log-sum-exp trick. Adding our own Sigmoid would (a) apply it twice and
  (b) lose numerical stability for extreme logits. Callers who need
  probabilities use :meth:`FraudDetectorMLP.predict_proba`.

* **GroupNorm, never BatchNorm.** Opacus computes *per-sample* gradients;
  BatchNorm mixes statistics across samples in a batch, which makes a
  sample's gradient depend on every other sample — undefined under
  per-sample clipping and rejected by ``opacus.validators.ModuleValidator``.
  GroupNorm normalizes within a single sample, so it is DP-compatible.
  It also works with batch size 1, which matters for the DLG attack demo
  (Week 5) where single-sample gradients are intercepted.

* **GELU over ReLU.** Smooth activation gives slightly better-behaved
  gradients under DP noise; cost is negligible at this scale.

* **Weights initialized with Kaiming-uniform (PyTorch default) but biases
  of the final layer set to the negative log prior odds.** With 0.17%
  positives, a zero-initialized output bias makes the model start out
  predicting ~50% fraud probability, producing huge initial losses and —
  under DP-SGD — wasting privacy budget on correcting an avoidable offset.
  Initializing the bias to ``log(p/(1-p))`` starts the model at the base
  rate. This is the standard trick from focal-loss literature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

__all__ = ["ModelConfig", "FraudDetectorMLP"]


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters of :class:`FraudDetectorMLP`.

    Frozen so a config attached to a checkpoint can't drift from the
    weights it describes.

    Attributes:
        input_dim: Number of input features. 30 for the Kaggle dataset
            (V1–V28 + scaled Amount + scaled Time).
        hidden_dims: Width of each hidden layer.
        dropout: Dropout probability applied after the first two hidden
            blocks (matches spec §6.4 — no dropout before the head).
        group_norm_groups: Number of groups for ``nn.GroupNorm``. Must
            divide every entry of ``hidden_dims``.
        fraud_prior: Expected positive-class rate, used to initialize the
            output bias at the base rate. 0.0017 = Kaggle dataset prior.
    """

    input_dim: int = 30
    hidden_dims: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.3
    group_norm_groups: int = 8
    fraud_prior: float = 0.0017

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        if not self.hidden_dims:
            raise ValueError("hidden_dims must not be empty")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not 0.0 < self.fraud_prior < 1.0:
            raise ValueError(f"fraud_prior must be in (0, 1), got {self.fraud_prior}")
        for width in self.hidden_dims:
            if width % self.group_norm_groups != 0:
                raise ValueError(
                    f"group_norm_groups={self.group_norm_groups} must divide "
                    f"every hidden width; {width} is not divisible"
                )


class FraudDetectorMLP(nn.Module):
    """MLP binary classifier over tabular transaction features.

    Outputs **raw logits** of shape ``(batch, 1)``. Pair with
    ``torch.nn.BCEWithLogitsLoss(pos_weight=...)``.

    The module is Opacus-compatible by construction: no BatchNorm, no
    parameter sharing, no buffers that depend on batch statistics.
    """

    config: ModelConfig

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config

        layers: list[nn.Module] = []
        in_dim = cfg.input_dim
        for i, width in enumerate(cfg.hidden_dims):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.GroupNorm(cfg.group_norm_groups, width))
            layers.append(nn.GELU())
            # Spec §6.4: dropout after the first two blocks only.
            if i < len(cfg.hidden_dims) - 1:
                layers.append(nn.Dropout(cfg.dropout))
            in_dim = width
        self.backbone = nn.Sequential(*layers)

        self.head = nn.Linear(in_dim, 1)
        with torch.no_grad():
            prior = cfg.fraud_prior
            self.head.bias.fill_(math.log(prior / (1.0 - prior)))

    def forward(self, x: Tensor) -> Tensor:
        """Compute fraud logits.

        Args:
            x: Float tensor of shape ``(batch, input_dim)``.

        Returns:
            Raw logits of shape ``(batch, 1)``. Apply sigmoid for
            probabilities; do NOT feed to ``BCELoss`` (use
            ``BCEWithLogitsLoss`` on these logits instead).

        Raises:
            ValueError: If ``x`` is not 2-D or has the wrong feature count.
        """
        if x.dim() != 2 or x.size(1) != self.config.input_dim:
            raise ValueError(
                f"expected input of shape (batch, {self.config.input_dim}), "
                f"got {tuple(x.shape)}"
            )
        return self.head(self.backbone(x))

    @torch.no_grad()
    def predict_proba(self, x: Tensor) -> Tensor:
        """Fraud probabilities in ``[0, 1]``, shape ``(batch,)``.

        Switches to eval mode (disables dropout) and restores the previous
        mode afterwards, so it is safe to call mid-training for metrics.
        """
        was_training = self.training
        self.eval()
        try:
            return torch.sigmoid(self.forward(x)).squeeze(1)
        finally:
            self.train(was_training)

    def parameter_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
