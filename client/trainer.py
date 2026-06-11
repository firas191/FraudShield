"""Local training engine: standard SGD and DP-SGD (Opacus) behind one interface.

Design decisions:

* **One trainer, a DP switch.** The FL comparison experiments (Weeks 4–5)
  need DP and non-DP runs that differ in *nothing* except the privacy
  mechanism. Two separate trainer classes would inevitably drift apart;
  a single class with ``enable_dp`` guarantees the only delta is what
  ``PrivacyEngine.make_private`` injects: per-sample gradient hooks,
  clipping, Gaussian noise, and Poisson sampling.

* **SGD with momentum, not Adam** (spec §6.4). Adam's per-parameter
  adaptive scaling renormalizes the carefully calibrated DP noise
  per-coordinate, which both weakens the utility analysis and makes the
  noise's effect on training dynamics harder to reason about. SGD keeps
  the mechanism faithful to the DP-SGD paper (Abadi et al., 2016).

* **RDP accountant, explicitly.** Opacus's default accountant is PRV
  (tighter), but the spec — and the privacy server component in Week 4 —
  standardize on Rényi DP composition (Mironov, 2017). Pinning
  ``accountant="rdp"`` makes our reported ε reproducible against the
  theoretical calculation in :func:`theoretical_epsilon_from_history`.

* **Poisson sampling is not optional.** The (ε, δ) analysis of DP-SGD
  assumes each sample enters a batch independently with probability
  q = B/n (privacy amplification by subsampling). Sequential shuffled
  batches violate that assumption and the reported ε would be wrong.
  ``make_private`` swaps the loader accordingly; batch sizes then vary
  stochastically around B.

* **pos_weight × DP interaction (known trade-off).** Weighting fraud
  samples by ~578 multiplies their per-sample gradient norms, so they
  hit the clipping threshold C far more often than legit samples —
  clipping partially cancels the class weighting. We keep pos_weight
  (it still helps: direction survives clipping, only magnitude is
  capped) but expose ``pos_weight_scale`` so Week 4 tuning can trade
  recall against clipping distortion, e.g. scale=0.1 → effective weight
  ~58. This asymmetry of DP-SGD on minority classes is well documented
  (Bagdasaryan et al., 2019) and worth mentioning in your write-up.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from client.data_loader import BankDataModule

__all__ = [
    "TrainerConfig",
    "EpochResult",
    "EvalMetrics",
    "LocalTrainer",
    "theoretical_epsilon_from_history",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainerConfig:
    """Hyperparameters for local (DP-)training.

    Attributes:
        epochs: Local epochs per :meth:`LocalTrainer.fit` call. In FL
            rounds this is E (typically 1–5); for Week 2 convergence
            checks it can be larger.
        lr: SGD learning rate.
        momentum: SGD momentum (compatible with Opacus).
        device: ``"auto"`` resolves to CUDA when available.
        enable_dp: Wrap training with Opacus DP-SGD.
        max_grad_norm: Per-sample L2 clipping threshold C (spec §6.2).
        noise_multiplier: σ — Gaussian noise std as a multiple of C.
        delta: Target δ for ε reporting. Must be < 1/n; 1e-5 is safe for
            partitions of 36k–145k rows.
        pos_weight_scale: Multiplier applied to the data-derived
            pos_weight (see module docstring on the DP interaction).
        secure_mode: Use a cryptographically secure RNG for DP noise.
            False in development (reproducible experiments); MUST be True
            in any real deployment, otherwise the Gaussian noise is
            predictable from torch's Mersenne state.
    """

    epochs: int = 5
    lr: float = 0.05
    momentum: float = 0.9
    device: str = "auto"
    enable_dp: bool = False
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.1
    delta: float = 1e-5
    pos_weight_scale: float = 1.0
    secure_mode: bool = False

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.enable_dp:
            if self.max_grad_norm <= 0:
                raise ValueError(f"max_grad_norm must be positive, got {self.max_grad_norm}")
            if self.noise_multiplier <= 0:
                raise ValueError(
                    f"noise_multiplier must be positive, got {self.noise_multiplier}"
                )
            if not 0 < self.delta < 1:
                raise ValueError(f"delta must be in (0, 1), got {self.delta}")
        if self.pos_weight_scale <= 0:
            raise ValueError(f"pos_weight_scale must be positive, got {self.pos_weight_scale}")

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


@dataclass(frozen=True)
class EvalMetrics:
    """Evaluation metrics on a validation/test loader."""

    loss: float
    auc_roc: float
    precision: float
    recall: float
    f1: float
    n_samples: int
    n_positives: int


@dataclass(frozen=True)
class EpochResult:
    """Outcome of one local training epoch."""

    epoch: int
    train_loss: float
    val: EvalMetrics
    epsilon: float | None = None  # cumulative ε after this epoch (DP only)


class LocalTrainer:
    """Trains a model on one bank's local data, optionally under DP-SGD.

    Usage::

        trainer = LocalTrainer(model, data_module, TrainerConfig(enable_dp=True))
        history = trainer.fit()
        weights = trainer.state_dict()        # clean keys, FL-ready
        eps = trainer.epsilon                  # cumulative privacy spent
    """

    def __init__(
        self,
        model: nn.Module,
        data: BankDataModule,
        config: TrainerConfig,
    ) -> None:
        self.config = config
        self.device = config.resolve_device()
        self.data = data

        self.model: nn.Module = model.to(self.device)
        pos_weight = (data.pos_weight * config.pos_weight_scale).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer: torch.optim.Optimizer = torch.optim.SGD(
            self.model.parameters(), lr=config.lr, momentum=config.momentum
        )
        self.train_loader: DataLoader = data.train_loader()
        self.privacy_engine: Any | None = None

        if config.enable_dp:
            from opacus import PrivacyEngine

            self.privacy_engine = PrivacyEngine(
                accountant="rdp", secure_mode=config.secure_mode
            )
            # Replaces: model → GradSampleModule (per-sample grad hooks),
            # optimizer → DPOptimizer (clip + noise), loader → Poisson sampling.
            self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                noise_multiplier=config.noise_multiplier,
                max_grad_norm=config.max_grad_norm,
                poisson_sampling=True,
            )
            # DPDataLoader exposes sample_rate directly; batch_size is None
            # under Poisson sampling (batch sizes are stochastic).
            sample_rate = getattr(self.train_loader, "sample_rate", float("nan"))
            logger.info(
                "DP-SGD enabled: C=%.2f σ=%.2f q=%.5f accountant=rdp",
                config.max_grad_norm,
                config.noise_multiplier,
                sample_rate,
            )

    # ------------------------------------------------------------------ #
    def fit(self) -> list[EpochResult]:
        """Run ``config.epochs`` local epochs, evaluating after each.

        Returns:
            Per-epoch results including cumulative ε when DP is enabled.
        """
        val_loader = self.data.val_loader()
        history: list[EpochResult] = []
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_epoch()
            val = self.evaluate(val_loader)
            eps = self.epsilon
            history.append(
                EpochResult(epoch=epoch, train_loss=train_loss, val=val, epsilon=eps)
            )
            logger.info(
                "epoch %d/%d train_loss=%.4f val_loss=%.4f auc=%.4f f1=%.4f%s",
                epoch, self.config.epochs, train_loss, val.loss, val.auc_roc, val.f1,
                f" ε={eps:.3f}" if eps is not None else "",
            )
        return history

    def _train_epoch(self) -> float:
        """One pass over the (possibly Poisson-sampled) training loader.

        Returns:
            Sample-weighted mean training loss.

        Raises:
            RuntimeError: If the loss goes non-finite — fail fast rather
                than aggregate NaN weights into a global model later.
        """
        self.model.train()
        total_loss = 0.0
        total_n = 0
        for x, y in self.train_loader:
            if x.numel() == 0:
                continue  # Poisson sampling can produce empty batches
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.criterion(self.model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "non-finite training loss — lower lr or check input scaling"
                )
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(x)
            total_n += len(x)
        if total_n == 0:
            raise RuntimeError("empty training epoch — loader produced no samples")
        return total_loss / total_n

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> EvalMetrics:
        """Compute loss/AUC/precision/recall/F1 over a loader.

        Decision threshold is 0.5 on the sigmoid probability; AUC-ROC is
        threshold-free and remains the primary metric (spec §6.4).
        """
        self.model.eval()
        logits_all: list[torch.Tensor] = []
        labels_all: list[torch.Tensor] = []
        total_loss, total_n = 0.0, 0
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            logits = self.model(x)
            total_loss += self.criterion(logits, y).item() * len(x)
            total_n += len(x)
            logits_all.append(logits.cpu())
            labels_all.append(y.cpu())

        logits_cat = torch.cat(logits_all).squeeze(1)
        labels = torch.cat(labels_all).squeeze(1).numpy()
        probs = torch.sigmoid(logits_cat).numpy()
        preds = (probs >= 0.5).astype(int)

        n_pos = int(labels.sum())
        auc = float(roc_auc_score(labels, probs)) if 0 < n_pos < len(labels) else math.nan
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        return EvalMetrics(
            loss=total_loss / total_n,
            auc_roc=auc,
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            n_samples=int(len(labels)),
            n_positives=n_pos,
        )

    # ------------------------------------------------------------------ #
    @property
    def epsilon(self) -> float | None:
        """Cumulative (ε, δ)-DP spent so far, or None when DP is off."""
        if self.privacy_engine is None:
            return None
        return float(self.privacy_engine.get_epsilon(delta=self.config.delta))

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Model weights with clean (architecture-native) parameter names.

        Opacus wraps the model in ``GradSampleModule``, which prefixes
        every key with ``_module.``. The FL server aggregates weights by
        name, so clients must export the unprefixed form regardless of
        whether they trained with DP.
        """
        raw = self.model.state_dict()
        return {k.removeprefix("_module."): v.detach().cpu() for k, v in raw.items()}

    def accountant_history(self) -> list[tuple[float, float, int]]:
        """Raw ``(noise_multiplier, sample_rate, num_steps)`` accounting log.

        Raises:
            RuntimeError: If DP is not enabled.
        """
        if self.privacy_engine is None:
            raise RuntimeError("accountant history requires enable_dp=True")
        return list(self.privacy_engine.accountant.history)


def theoretical_epsilon_from_history(
    history: list[tuple[float, float, int]],
    delta: float,
    orders: list[float] | None = None,
) -> float:
    """Compute ε directly from RDP composition theory (Mironov, 2017).

    Independent re-derivation of what the Opacus RDP accountant reports:
    for each (σ, q, steps) segment, compute the RDP of the subsampled
    Gaussian mechanism at a grid of orders α, sum across segments
    (RDP composes additively), then convert to (ε, δ)-DP via
    ``ε = min_α [ ε_RDP(α) + log(1/δ)/(α-1) ]`` (spec §6.1).

    Week 2's validation requirement is that this matches
    ``LocalTrainer.epsilon`` — proving our reported privacy budget is
    real accounting, not a number we trust blindly from a library.

    Args:
        history: From :meth:`LocalTrainer.accountant_history`.
        delta: Target δ.
        orders: RDP orders α to optimize over; defaults to Opacus's grid.

    Returns:
        Theoretical ε.
    """
    from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent

    if not history:
        raise ValueError("empty accounting history")
    if orders is None:
        orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

    total_rdp = None
    for noise_multiplier, sample_rate, steps in history:
        rdp = compute_rdp(
            q=sample_rate, noise_multiplier=noise_multiplier, steps=steps, orders=orders
        )
        total_rdp = rdp if total_rdp is None else total_rdp + rdp
    eps, _best_alpha = get_privacy_spent(orders=orders, rdp=total_rdp, delta=delta)
    return float(eps)
