"""One-command federated training: spawns the server and all three banks.

    python scripts/run_federation.py --rounds 10
    python scripts/run_federation.py --rounds 10 --dp

Each component is a real OS process talking real HTTP over localhost —
the same topology Docker Compose will reproduce in Week 7, minus the
containers. Logs from all processes interleave on stdout, prefixed by
uvicorn/client loggers.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_URL = "http://localhost:8000"
BANKS = ("a", "b", "c")


def wait_health(deadline: float = 60.0) -> None:
    start = time.monotonic()
    while time.monotonic() - start < deadline:
        try:
            if httpx.get(f"{SERVER_URL}/health", timeout=2.0).status_code == 200:
                return
        except httpx.TransportError:
            time.sleep(0.5)
    raise RuntimeError("server failed to become healthy")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--dp", action="store_true")
    parser.add_argument("--noise-multiplier", type=float, default=1.1)
    parser.add_argument("--fresh", action="store_true",
                        help="delete checkpoints/ first (restart from round 0)")
    args = parser.parse_args(argv)

    if args.fresh:
        import shutil

        shutil.rmtree(REPO_ROOT / "checkpoints", ignore_errors=True)
        print("cleared checkpoints/ — starting from round 0")

    env = os.environ.copy()
    env["NUM_ROUNDS"] = str(args.rounds)

    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.main:app",
         "--host", "127.0.0.1", "--port", "8000", "--log-level", "warning"],
        cwd=REPO_ROOT,
        env=env,
    )
    clients: list[subprocess.Popen] = []
    try:
        wait_health()
        print(f"server up — launching {len(BANKS)} bank clients "
              f"({'DP-SGD' if args.dp else 'plain SGD'}, {args.rounds} rounds)")
        for bank in BANKS:
            cmd = [
                sys.executable, "-m", "client.client",
                "--bank", bank,
                "--server-url", SERVER_URL,
                "--local-epochs", str(args.local_epochs),
            ]
            if args.dp:
                cmd += ["--dp", "--noise-multiplier", str(args.noise_multiplier)]
            clients.append(subprocess.Popen(cmd, cwd=REPO_ROOT, env=env))

        exit_codes = [c.wait() for c in clients]
        if any(code != 0 for code in exit_codes):
            print(f"WARNING: client exit codes: {exit_codes}")
            return 1

        history = httpx.get(f"{SERVER_URL}/metrics/history", timeout=10.0).json()
        print("\n=== federated training summary (global test set) ===")
        print(f"{'round':>5} {'AUC-ROC':>9} {'loss':>9} {'f1@0.5':>8}")
        for m in history:
            print(f"{m['round']:>5} {m['auc_roc']:>9.4f} {m['test_loss']:>9.4f} "
                  f"{m['f1']:>8.4f}")
        if history:
            best = max(history, key=lambda m: m["auc_roc"])
            print(f"\nbest: round {best['round']} AUC-ROC={best['auc_roc']:.4f} "
                  f"(target ≥ 0.90)")
        return 0
    finally:
        server.send_signal(signal.SIGINT)
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    sys.exit(main())
