"""Integration tests for the FastAPI coordination server.

Drives full federated rounds through the real HTTP layer (TestClient):
download global model → perturb → multipart submit → aggregation →
checkpoint → next round. MLflow disabled; tiny synthetic test set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save

N_TEST_ROWS = 800


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Server wired to a temp checkpoint dir and synthetic test CSV."""
    rng = np.random.default_rng(5)
    y = (rng.uniform(size=N_TEST_ROWS) < 0.05).astype(np.int64)
    df = pd.DataFrame({f"V{i}": rng.normal(size=N_TEST_ROWS) for i in range(1, 29)})
    for i in range(1, 7):
        df[f"V{i}"] += 2.0 * y
    df.insert(0, "Time", rng.uniform(0, 172_800, size=N_TEST_ROWS))
    df["Amount"] = rng.lognormal(3.0, 1.5, size=N_TEST_ROWS)
    df["Class"] = y
    test_csv = tmp_path / "global_test.csv"
    df.to_csv(test_csv, index=False)

    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("GLOBAL_TEST_CSV", str(test_csv))
    monkeypatch.setenv("EXPECTED_CLIENTS", "bank_a,bank_b")
    monkeypatch.setenv("NUM_ROUNDS", "2")
    monkeypatch.setenv("MLFLOW_ENABLED", "false")

    from server.main import app, get_settings

    get_settings.cache_clear()
    with TestClient(app) as tc:
        yield tc
    get_settings.cache_clear()


def download_model(tc: TestClient) -> tuple[int, dict[str, torch.Tensor]]:
    resp = tc.get("/model/current")
    assert resp.status_code == 200
    return int(resp.headers["X-Round"]), st_load(resp.content)


def submit(
    tc: TestClient,
    client_id: str,
    sd: dict[str, torch.Tensor],
    num_samples: int = 100,
):
    blob = st_save({k: v.contiguous() for k, v in sd.items()})
    return tc.post(
        "/round/submit",
        data={"client_id": client_id, "num_samples": str(num_samples)},
        files={"weights": ("w.safetensors", blob, "application/octet-stream")},
    )


def perturbed(sd: dict[str, torch.Tensor], scale: float) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(int(scale * 1000))
    return {
        k: v + scale * torch.randn(v.shape, generator=g) for k, v in sd.items()
    }


class TestBasics:
    def test_health(self, client: TestClient) -> None:
        assert client.get("/health").json() == {"status": "ok"}

    def test_initial_status(self, client: TestClient) -> None:
        s = client.get("/status").json()
        assert s["current_round"] == 1
        assert s["phase"] == "collecting"
        assert s["expected_clients"] == ["bank_a", "bank_b"]
        assert s["received_clients"] == []
        assert s["latest_checkpoint_round"] == 0

    def test_model_download_round_zero(self, client: TestClient) -> None:
        round_num, sd = download_model(client)
        assert round_num == 0
        from model.fraud_mlp import FraudDetectorMLP

        FraudDetectorMLP().load_state_dict(sd)  # structurally valid


class TestRoundFlow:
    def test_full_round_aggregates_and_advances(self, client: TestClient) -> None:
        _, sd = download_model(client)

        r1 = submit(client, "bank_a", perturbed(sd, 0.01), num_samples=300)
        assert r1.status_code == 200
        assert r1.json()["round_completed"] is False

        r2 = submit(client, "bank_b", perturbed(sd, 0.02), num_samples=100)
        assert r2.status_code == 200
        body = r2.json()
        assert body["round_completed"] is True
        assert body["round"] == 1

        s = client.get("/status").json()
        assert s["current_round"] == 2
        assert s["received_clients"] == []
        assert s["latest_checkpoint_round"] == 1

        new_round, new_sd = download_model(client)
        assert new_round == 1
        # Aggregated model must differ from round 0.
        diffs = [(new_sd[k] - sd[k]).abs().max().item() for k in sd]
        assert max(diffs) > 0

    def test_metrics_history_populated(self, client: TestClient) -> None:
        _, sd = download_model(client)
        submit(client, "bank_a", perturbed(sd, 0.01), 300)
        submit(client, "bank_b", perturbed(sd, 0.02), 100)

        hist = client.get("/metrics/history").json()
        assert len(hist) == 1
        m = hist[0]
        assert m["round"] == 1
        assert 0.0 <= m["auc_roc"] <= 1.0
        assert m["client_weights"]["bank_a"] == pytest.approx(0.75)
        assert m["client_weights"]["bank_b"] == pytest.approx(0.25)

    def test_training_finishes_after_total_rounds(self, client: TestClient) -> None:
        for _ in range(2):  # NUM_ROUNDS=2
            _, sd = download_model(client)
            submit(client, "bank_a", perturbed(sd, 0.01), 300)
            submit(client, "bank_b", perturbed(sd, 0.02), 100)

        assert client.get("/status").json()["phase"] == "finished"
        _, sd = download_model(client)
        resp = submit(client, "bank_a", sd, 300)
        assert resp.json()["accepted"] is False


class TestRejection:
    def test_unknown_client(self, client: TestClient) -> None:
        _, sd = download_model(client)
        assert submit(client, "bank_evil", sd).status_code == 422

    def test_duplicate_submission(self, client: TestClient) -> None:
        _, sd = download_model(client)
        assert submit(client, "bank_a", perturbed(sd, 0.01)).status_code == 200
        assert submit(client, "bank_a", perturbed(sd, 0.01)).status_code == 422
        # round must still be salvageable by the other client
        s = client.get("/status").json()
        assert s["received_clients"] == ["bank_a"]

    def test_nan_payload_rejected(self, client: TestClient) -> None:
        _, sd = download_model(client)
        bad = {k: v.clone() for k, v in sd.items()}
        first = next(iter(bad))
        bad[first][..., 0] = float("nan")
        resp = submit(client, "bank_a", bad)
        assert resp.status_code == 422
        assert "non-finite" in resp.json()["detail"]
        # NaN client must NOT count as received
        assert client.get("/status").json()["received_clients"] == []

    def test_garbage_payload_rejected(self, client: TestClient) -> None:
        resp = client.post(
            "/round/submit",
            data={"client_id": "bank_a", "num_samples": "10"},
            files={"weights": ("w.safetensors", b"not safetensors", "application/octet-stream")},
        )
        assert resp.status_code == 422

    def test_wrong_architecture_rejected(self, client: TestClient) -> None:
        bad = {"some.weight": torch.zeros(3, 3)}
        assert submit(client, "bank_a", bad).status_code == 422


class TestWebSocket:
    def test_round_metrics_pushed(self, client: TestClient) -> None:
        _, sd = download_model(client)
        with client.websocket_connect("/ws/metrics") as ws:
            submit(client, "bank_a", perturbed(sd, 0.01), 300)
            submit(client, "bank_b", perturbed(sd, 0.02), 100)
            msg = ws.receive_json()
        assert msg["round"] == 1
        assert "auc_roc" in msg
