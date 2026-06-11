"""FL bank client: poll → download global → train locally → submit → repeat.

Run one bank (server must be reachable):

    python -m client.client --bank a
    python -m client.client --bank b --dp

Privacy accounting across rounds (important subtlety):
    A fresh Opacus ``PrivacyEngine`` is created for each round's local
    training, so the engine's own ε only covers that round. The *real*
    privacy cost is cumulative across all rounds — RDP composes
    additively, so this client concatenates every round's accounting
    history and reports ε over the whole concatenated history via
    :func:`theoretical_epsilon_from_history`. Week 4 moves this duty to
    the server's Privacy Accountant component (per the spec) so the
    budget is enforced, not just observed; until then the client logs it
    every round and warns when the target budget is exceeded.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from client.communicator import ServerCommunicator
from client.data_loader import BankDataConfig, BankDataModule
from client.trainer import LocalTrainer, TrainerConfig, theoretical_epsilon_from_history
from model.fraud_mlp import FraudDetectorMLP

__all__ = ["FLClientConfig", "FLClient"]

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class FLClientConfig:
    """One bank's federated-client configuration."""

    client_id: str
    partition_csv: str | Path
    local_epochs: int = 1
    batch_size: int = 512
    lr: float = 0.05
    device: str = "auto"
    enable_dp: bool = False
    noise_multiplier: float = 1.1
    max_grad_norm: float = 1.0
    delta: float = 1e-5
    target_epsilon: float = 5.0
    poll_interval: float = 1.0
    seed: int = 42


class FLClient:
    """Drives one bank through the federated protocol.

    ``run_once`` does at most one unit of work (train+submit for a new
    round, or nothing if waiting) — this makes the loop unit-testable
    without threads. ``run`` loops until the server reports FINISHED.
    """

    def __init__(self, config: FLClientConfig, comm: ServerCommunicator) -> None:
        self.config = config
        self.comm = comm
        self.data = BankDataModule(
            BankDataConfig(
                csv_path=config.partition_csv,
                batch_size=config.batch_size,
                seed=config.seed,
                device=str(TrainerConfig(device=config.device).resolve_device()),
            )
        )
        self.data.setup()
        self.rounds_completed = 0
        self._dp_history: list[tuple[float, float, int]] = []

    # ------------------------------------------------------------------ #
    @property
    def cumulative_epsilon(self) -> float | None:
        """Total ε spent across ALL rounds (RDP composition), None if non-DP."""
        if not self._dp_history:
            return None
        return theoretical_epsilon_from_history(self._dp_history, self.config.delta)

    def run_once(self) -> bool:
        """One protocol step.

        Returns:
            True if a round was trained and submitted; False if there was
            nothing to do (waiting on other clients, or training finished).
        """
        status = self.comm.get_status()
        if status.phase == "finished":
            return False
        if self.config.client_id in status.received_clients:
            return False  # already submitted this round; wait for the others

        round_num, global_sd = self.comm.download_global_model()
        model = FraudDetectorMLP()
        model.load_state_dict(global_sd)

        trainer = LocalTrainer(
            model,
            self.data,
            TrainerConfig(
                epochs=self.config.local_epochs,
                lr=self.config.lr,
                device=self.config.device,
                enable_dp=self.config.enable_dp,
                noise_multiplier=self.config.noise_multiplier,
                max_grad_norm=self.config.max_grad_norm,
                delta=self.config.delta,
            ),
        )
        history = trainer.fit()
        if self.config.enable_dp:
            self._dp_history.extend(trainer.accountant_history())
            eps = self.cumulative_epsilon
            assert eps is not None
            if eps > self.config.target_epsilon:
                logger.warning(
                    "%s: cumulative ε=%.3f EXCEEDS target %.1f — in production "
                    "this client must STOP participating",
                    self.config.client_id, eps, self.config.target_epsilon,
                )

        resp = self.comm.submit_update(
            self.config.client_id, self.data.train_size, trainer.state_dict()
        )
        self.rounds_completed += 1
        eps_str = (
            f" cumulative_ε={self.cumulative_epsilon:.3f}" if self.config.enable_dp else ""
        )
        logger.info(
            "%s: trained round %d (global r%d, local_loss=%.4f)%s — %s",
            self.config.client_id, status.current_round, round_num,
            history[-1].train_loss, eps_str, resp.message,
        )
        return True

    def run(self) -> int:
        """Participate until the server reports training finished.

        Returns:
            Number of rounds this client trained.
        """
        self.comm.wait_for_server()
        logger.info("%s: connected, train_size=%d", self.config.client_id, self.data.train_size)
        while True:
            status = self.comm.get_status()
            if status.phase == "finished":
                logger.info(
                    "%s: training finished after %d rounds%s",
                    self.config.client_id, self.rounds_completed,
                    f", total ε={self.cumulative_epsilon:.3f}"
                    if self.config.enable_dp else "",
                )
                return self.rounds_completed
            if not self.run_once():
                time.sleep(self.config.poll_interval)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for one bank client."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", choices=("a", "b", "c"), required=True)
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dp", action="store_true")
    parser.add_argument("--noise-multiplier", type=float, default=1.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--target-epsilon", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    config = FLClientConfig(
        client_id=f"bank_{args.bank}",
        partition_csv=REPO_ROOT / "data" / "partitions" / f"bank_{args.bank}.csv",
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        enable_dp=args.dp,
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.max_grad_norm,
        target_epsilon=args.target_epsilon,
        seed=args.seed,
    )
    with ServerCommunicator(base_url=args.server_url) as comm:
        FLClient(config, comm).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
