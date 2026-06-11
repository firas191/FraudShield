"""Federated aggregation: FedAvg (McMahan et al., 2017, Algorithm 1).

    w_{t+1} = Σ_k (n_k / N) · w_k        over selected clients k

Design decisions:

* **Aggregation happens on flat vectors.** Client weights are flattened
  via ``model.utils.flatten_state_dict`` and averaged as single 1-D
  tensors. One ``torch.stack`` + matrix-vector product replaces a
  per-parameter loop, and the same representation feeds Krum's pairwise
  distances (Week 5) unchanged.

* **Weights vs. deltas.** We aggregate full weight tensors w_k, not
  deltas Δw_k. With full client participation (our 3 banks) the two are
  mathematically identical: Σ(n_k/N)(w_t + Δw_k) = w_t + Σ(n_k/N)Δw_k.
  Full weights make each round stateless (the server needs no memory of
  w_t to verify an update) and rollback trivially safe.

* **Validation is the server's first line of defense.** Every update is
  checked against the reference spec (parameter names + shapes from the
  current global model) and rejected on mismatch or non-finite values.
  In Week 5 one client becomes actively malicious; structural validation
  is the cheap filter that runs before the expensive statistical one
  (Krum). A NaN bomb from a single client would otherwise corrupt the
  global model irreversibly.

* **FedProx note.** FedProx (Week 4) changes the *client* objective
  (adds the proximal term μ/2·||w − w_t||²); the server-side aggregation
  step is the same weighted average. So this module already serves both
  algorithms — the Week 4 work lands in ``client/trainer.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor

from model.utils import flatten_state_dict, unflatten_to_state_dict

__all__ = ["ClientUpdate", "AggregationError", "validate_update", "fedavg"]

logger = logging.getLogger(__name__)


class AggregationError(ValueError):
    """An update failed validation or aggregation preconditions."""


@dataclass(frozen=True)
class ClientUpdate:
    """One client's contribution to a round.

    Attributes:
        client_id: Stable identifier (e.g. ``"bank_a"``).
        num_samples: Local training set size n_k — the FedAvg weight.
            Trust note: in our simulation n_k is honest; in a real
            deployment a client could inflate n_k to dominate the
            average. Mitigations (server-side caps, proof-of-data) are
            out of scope but worth mentioning in the write-up.
        state_dict: Full model weights after local training.
    """

    client_id: str
    num_samples: int
    state_dict: dict[str, Tensor]

    def __post_init__(self) -> None:
        if self.num_samples <= 0:
            raise AggregationError(
                f"{self.client_id}: num_samples must be positive, got {self.num_samples}"
            )
        if not self.state_dict:
            raise AggregationError(f"{self.client_id}: empty state dict")


def validate_update(
    update: ClientUpdate, reference_spec: list[tuple[str, torch.Size]]
) -> None:
    """Check one update's structure and numeric sanity against the global spec.

    Args:
        update: The client's submission.
        reference_spec: ``(name, shape)`` list from flattening the
            current global model — the single source of structural truth.

    Raises:
        AggregationError: On missing/extra parameters, shape mismatch,
            or non-finite values. The message names the client so the
            round manager can exclude and log it.
    """
    expected = dict(reference_spec)
    actual_keys = set(update.state_dict.keys())
    expected_keys = set(expected.keys())
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise AggregationError(
            f"{update.client_id}: parameter set mismatch (missing={missing}, extra={extra})"
        )
    for name, tensor in update.state_dict.items():
        if tensor.shape != expected[name]:
            raise AggregationError(
                f"{update.client_id}: shape mismatch for '{name}': "
                f"{tuple(tensor.shape)} vs {tuple(expected[name])}"
            )
        if not torch.isfinite(tensor).all():
            raise AggregationError(
                f"{update.client_id}: non-finite values in '{name}' — update rejected"
            )


def fedavg(
    updates: list[ClientUpdate],
    reference_spec: list[tuple[str, torch.Size]],
) -> dict[str, Tensor]:
    """Sample-weighted average of validated client updates.

    Args:
        updates: Client submissions for this round (≥ 1).
        reference_spec: Structural spec of the current global model.

    Returns:
        New global state dict w_{t+1}.

    Raises:
        AggregationError: If no updates were provided or any update
            fails validation. All-or-nothing per call: the round manager
            decides which updates to pass in (e.g. after Byzantine
            filtering); this function refuses to silently skip bad ones.
    """
    if not updates:
        raise AggregationError("no client updates to aggregate")
    ids = [u.client_id for u in updates]
    if len(set(ids)) != len(ids):
        raise AggregationError(f"duplicate client ids in round: {ids}")

    for update in updates:
        validate_update(update, reference_spec)

    total = sum(u.num_samples for u in updates)
    weights = torch.tensor(
        [u.num_samples / total for u in updates], dtype=torch.float64
    )

    vectors = []
    for update in updates:
        vec, spec = flatten_state_dict(update.state_dict)
        # flatten order is dict insertion order; enforce the reference order
        if [name for name, _ in spec] != [name for name, _ in reference_spec]:
            ordered = {name: update.state_dict[name] for name, _ in reference_spec}
            vec, _ = flatten_state_dict(ordered)
        vectors.append(vec.to(torch.float64))

    stacked = torch.stack(vectors)                      # (K, P)
    averaged = (weights.unsqueeze(1) * stacked).sum(0)  # (P,)

    logger.info(
        "fedavg: aggregated %d clients (N=%d, weights=%s)",
        len(updates), total, [f"{w:.3f}" for w in weights.tolist()],
    )
    return dict(unflatten_to_state_dict(averaged.to(torch.float32), reference_spec))
