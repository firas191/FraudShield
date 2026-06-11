"""End-to-end FL tests: real client logic against the real server, in-process.

The FastAPI TestClient is an httpx.Client over an ASGI transport, so the
ServerCommunicator runs unmodified against the actual server code — full
rounds of download → local training → submit → aggregation, no sockets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from client.client import FLClient, FLClientConfig
from client.communicator import ServerCommunicator

N_ROWS = 3_000
N_TEST = 800
ROUNDS = 2


def synthetic_frame(rng: np.random.Generator, n: int) -> pd.DataFrame:
    y = (rng.uniform(size=n) < 0.05).astype(np.int64)
    df = pd.DataFrame({f"V{i}": rng.normal(size=n) for i in range(1, 29)})
    for i in range(1, 7):
        df[f"V{i}"] += 2.0 * y
    df.insert(0, "Time", rng.uniform(0, 172_800, size=n))
    df["Amount"] = rng.lognormal(3.0, 1.5, size=n)
    df["Class"] = y
    return df


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Server TestClient + two FLClients on synthetic partitions."""
    rng = np.random.default_rng(11)
    paths = {}
    for bank in ("bank_a", "bank_b"):
        df = synthetic_frame(rng, N_ROWS)
        paths[bank] = tmp_path / f"{bank}.csv"
        df.to_csv(paths[bank], index=False)
    test_csv = tmp_path / "global_test.csv"
    synthetic_frame(rng, N_TEST).to_csv(test_csv, index=False)

    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("GLOBAL_TEST_CSV", str(test_csv))
    monkeypatch.setenv("EXPECTED_CLIENTS", "bank_a,bank_b")
    monkeypatch.setenv("NUM_ROUNDS", str(ROUNDS))
    monkeypatch.setenv("MLFLOW_ENABLED", "false")

    from server.main import app, get_settings

    get_settings.cache_clear()
    with TestClient(app) as tc:
        def make_client(bank: str, dp: bool = False) -> FLClient:
            cfg = FLClientConfig(
                client_id=bank,
                partition_csv=paths[bank],
                local_epochs=1,
                batch_size=256,
                device="cpu",
                enable_dp=dp,
                poll_interval=0.0,
            )
            return FLClient(cfg, ServerCommunicator(http=tc))

        yield tc, make_client
    get_settings.cache_clear()


class TestFederatedRounds:
    def test_two_clients_complete_all_rounds(self, env) -> None:
        tc, make_client = env
        a, b = make_client("bank_a"), make_client("bank_b")

        for _ in range(ROUNDS):
            assert a.run_once() is True   # trains + submits
            assert a.run_once() is False  # waiting for b
            assert b.run_once() is True   # completes the round

        status = tc.get("/status").json()
        assert status["phase"] == "finished"
        assert a.rounds_completed == ROUNDS
        assert b.rounds_completed == ROUNDS

        hist = tc.get("/metrics/history").json()
        assert len(hist) == ROUNDS
        assert hist[-1]["auc_roc"] > 0.9  # separable synthetic data

    def test_clients_do_nothing_after_finish(self, env) -> None:
        _, make_client = env
        a, b = make_client("bank_a"), make_client("bank_b")
        for _ in range(ROUNDS):
            a.run_once()
            b.run_once()
        assert a.run_once() is False
        assert b.run_once() is False

    def test_global_model_changes_each_round(self, env) -> None:
        _, make_client = env
        a, b = make_client("bank_a"), make_client("bank_b")

        _, sd0 = a.comm.download_global_model()
        a.run_once()
        b.run_once()
        r1, sd1 = a.comm.download_global_model()
        assert r1 == 1
        assert any((sd1[k] - sd0[k]).abs().max() > 0 for k in sd0)

    def test_dp_client_accumulates_epsilon_across_rounds(self, env) -> None:
        _, make_client = env
        a, b = make_client("bank_a", dp=True), make_client("bank_b")

        a.run_once()
        b.run_once()
        eps_after_1 = a.cumulative_epsilon
        assert eps_after_1 is not None and eps_after_1 > 0

        a.run_once()
        b.run_once()
        eps_after_2 = a.cumulative_epsilon
        assert eps_after_2 is not None
        assert eps_after_2 > eps_after_1, "ε must compose across rounds"

    def test_non_dp_client_reports_no_epsilon(self, env) -> None:
        _, make_client = env
        assert make_client("bank_a").cumulative_epsilon is None
