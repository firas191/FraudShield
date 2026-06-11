"""Tests for FedAvg aggregation: math, validation, adversarial inputs."""

from __future__ import annotations

import pytest
import torch

from model.fraud_mlp import FraudDetectorMLP
from model.utils import flatten_state_dict
from server.aggregation import AggregationError, ClientUpdate, fedavg, validate_update


@pytest.fixture()
def reference():
    torch.manual_seed(0)
    model = FraudDetectorMLP()
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    _, spec = flatten_state_dict(sd)
    return sd, spec


def constant_update(client_id: str, n: int, sd: dict, value: float) -> ClientUpdate:
    return ClientUpdate(
        client_id=client_id,
        num_samples=n,
        state_dict={k: torch.full_like(v, value) for k, v in sd.items()},
    )


class TestFedAvgMath:
    def test_weighted_average_exact(self, reference) -> None:
        """3 clients with constant weights: result must be Σ (n_k/N)·v_k."""
        sd, spec = reference
        updates = [
            constant_update("bank_a", 60, sd, 1.0),
            constant_update("bank_b", 25, sd, 2.0),
            constant_update("bank_c", 15, sd, 3.0),
        ]
        result = fedavg(updates, spec)
        expected = (60 * 1.0 + 25 * 2.0 + 15 * 3.0) / 100  # = 1.55
        for tensor in result.values():
            assert torch.allclose(tensor, torch.full_like(tensor, expected), atol=1e-6)

    def test_identical_updates_are_fixed_point(self, reference) -> None:
        sd, spec = reference
        updates = [
            ClientUpdate("bank_a", 100, {k: v.clone() for k, v in sd.items()}),
            ClientUpdate("bank_b", 50, {k: v.clone() for k, v in sd.items()}),
        ]
        result = fedavg(updates, spec)
        for k in sd:
            assert torch.allclose(result[k], sd[k], atol=1e-6)

    def test_single_client_passthrough(self, reference) -> None:
        sd, spec = reference
        result = fedavg([ClientUpdate("bank_a", 10, sd)], spec)
        for k in sd:
            assert torch.allclose(result[k], sd[k], atol=1e-6)

    def test_weighting_dominated_by_larger_client(self, reference) -> None:
        sd, spec = reference
        updates = [
            constant_update("big", 999_999, sd, 1.0),
            constant_update("tiny", 1, sd, 100.0),
        ]
        result = fedavg(updates, spec)
        for tensor in result.values():
            assert tensor.max().item() < 1.01  # tiny client ~no influence

    def test_result_loadable_into_model(self, reference) -> None:
        sd, spec = reference
        result = fedavg(
            [constant_update("a", 1, sd, 0.5), constant_update("b", 1, sd, 1.5)], spec
        )
        FraudDetectorMLP().load_state_dict(result)


class TestValidation:
    def test_rejects_empty_round(self, reference) -> None:
        _, spec = reference
        with pytest.raises(AggregationError, match="no client updates"):
            fedavg([], spec)

    def test_rejects_duplicate_clients(self, reference) -> None:
        sd, spec = reference
        with pytest.raises(AggregationError, match="duplicate"):
            fedavg([ClientUpdate("a", 1, sd), ClientUpdate("a", 1, sd)], spec)

    def test_rejects_nan_bomb(self, reference) -> None:
        """A single client submitting NaNs must abort, not poison silently."""
        sd, spec = reference
        poisoned = {k: v.clone() for k, v in sd.items()}
        first = next(iter(poisoned))
        poisoned[first][..., 0] = float("nan")
        with pytest.raises(AggregationError, match="non-finite"):
            fedavg(
                [ClientUpdate("honest", 99, sd), ClientUpdate("evil", 1, poisoned)],
                spec,
            )

    def test_rejects_missing_parameter(self, reference) -> None:
        sd, spec = reference
        partial = dict(sd)
        partial.pop(next(iter(partial)))
        with pytest.raises(AggregationError, match="parameter set mismatch"):
            validate_update(ClientUpdate("a", 1, partial), spec)

    def test_rejects_shape_mismatch(self, reference) -> None:
        sd, spec = reference
        bad = {k: v.clone() for k, v in sd.items()}
        first = next(iter(bad))
        bad[first] = torch.zeros(7, 7)
        with pytest.raises(AggregationError, match="shape mismatch"):
            validate_update(ClientUpdate("a", 1, bad), spec)

    def test_rejects_nonpositive_samples(self, reference) -> None:
        sd, _ = reference
        with pytest.raises(AggregationError, match="num_samples"):
            ClientUpdate("a", 0, sd)

    def test_key_order_does_not_matter(self, reference) -> None:
        """A client sending the same params in different dict order must
        aggregate identically — networks don't guarantee key order."""
        sd, spec = reference
        reordered = dict(reversed(list(sd.items())))
        r1 = fedavg([ClientUpdate("a", 1, sd)], spec)
        r2 = fedavg([ClientUpdate("a", 1, reordered)], spec)
        for k in r1:
            assert torch.equal(r1[k], r2[k])
