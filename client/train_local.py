"""Week 2 verification CLI: train locally on one bank's partition.

Usage (from repo root, venv active):

    python -m client.train_local --bank a --epochs 5
    python -m client.train_local --bank a --epochs 5 --dp
    python -m client.train_local --bank c --epochs 5 --dp --noise-multiplier 1.5

Purpose: prove each partition converges with plain SGD *before* the FL
server exists, then observe what DP-SGD costs in utility and what it
buys in ε. This isolates variables — when the federated runs in Week 3
behave oddly, local convergence is already established and the suspect
list shrinks to the aggregation layer.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from client.data_loader import BankDataConfig, BankDataModule
from client.trainer import LocalTrainer, TrainerConfig, theoretical_epsilon_from_history
from model.fraud_mlp import FraudDetectorMLP

REPO_ROOT = Path(__file__).resolve().parent.parent
PARTITIONS_DIR = REPO_ROOT / "data" / "partitions"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", choices=("a", "b", "c"), required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dp", action="store_true", help="enable DP-SGD (Opacus)")
    parser.add_argument("--noise-multiplier", type=float, default=1.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument(
        "--pos-weight-scale", type=float, default=1.0,
        help="scale factor on class weighting (see trainer docs re: DP clipping)",
    )
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)

    csv_path = PARTITIONS_DIR / f"bank_{args.bank}.csv"
    data = BankDataModule(
        BankDataConfig(
            csv_path=csv_path,
            batch_size=args.batch_size,
            seed=args.seed,
            device=str(TrainerConfig(device=args.device).resolve_device()),
        )
    )
    data.setup()

    config = TrainerConfig(
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        enable_dp=args.dp,
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.max_grad_norm,
        delta=args.delta,
        pos_weight_scale=args.pos_weight_scale,
    )
    trainer = LocalTrainer(FraudDetectorMLP(), data, config)

    mode = "DP-SGD" if args.dp else "SGD"
    print(f"\n=== bank_{args.bank} | {mode} | device={trainer.device} "
          f"| train={data.train_size:,} ===")
    history = trainer.fit()

    last = history[-1]
    print(f"\nfinal: val AUC-ROC={last.val.auc_roc:.4f}  precision={last.val.precision:.4f}  "
          f"recall={last.val.recall:.4f}  f1={last.val.f1:.4f}")
    if args.dp:
        engine_eps = trainer.epsilon
        theory_eps = theoretical_epsilon_from_history(
            trainer.accountant_history(), delta=args.delta
        )
        assert engine_eps is not None
        rel_err = abs(engine_eps - theory_eps) / theory_eps
        print(f"privacy: ε(engine)={engine_eps:.4f}  ε(theory)={theory_eps:.4f}  "
              f"rel.err={rel_err:.2e}  (δ={args.delta:g})")
        if rel_err > 0.01:
            print("WARNING: accountant deviates >1% from theoretical RDP composition")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
