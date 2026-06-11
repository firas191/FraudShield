"""Weight serialization and manipulation helpers.

Why safetensors instead of ``torch.save`` (pickle):
    Pickle deserialization executes arbitrary code. In an FL system the
    server deserializes payloads received over the network from clients —
    one of which (Bank C) we deliberately make malicious in Week 5. A
    pickle-based protocol would let a Byzantine client achieve remote code
    execution on the coordinator, which is a far worse failure than a
    poisoned model. safetensors is a pure-data format: worst case is a
    corrupted tensor, which Krum/median then handles.

Why flatten/unflatten:
    FedAvg/FedProx and Krum operate on whole-model weight *vectors*
    (weighted means, pairwise L2 distances). Doing that per-tensor is
    error-prone; flattening to a single 1-D tensor with a recorded spec
    makes the math one line and trivially testable.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

__all__ = [
    "save_weights",
    "load_weights",
    "flatten_state_dict",
    "unflatten_to_state_dict",
    "state_dict_delta",
]

logger = logging.getLogger(__name__)


def save_weights(state_dict: dict[str, Tensor], path: str | Path) -> None:
    """Serialize a model state dict to a ``.safetensors`` file.

    Tensors are moved to CPU and made contiguous first — safetensors
    requires contiguous tensors, and checkpoints must be loadable on
    machines without the GPU they were trained on.

    Args:
        state_dict: Mapping of parameter names to tensors
            (``model.state_dict()``).
        path: Destination file path; parent directories are created.

    Raises:
        ValueError: If the state dict is empty.
        OSError: If the file cannot be written.
    """
    if not state_dict:
        raise ValueError("refusing to save an empty state dict")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_dict = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
    save_file(cpu_dict, str(path))
    logger.debug("saved %d tensors to %s", len(cpu_dict), path)


def load_weights(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Tensor]:
    """Load a state dict from a ``.safetensors`` file.

    Args:
        path: Source file path.
        device: Device to map tensors onto (e.g. ``"cuda"``).

    Returns:
        Parameter-name → tensor mapping, loadable via
        ``model.load_state_dict``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"weights file not found: {path}")
    return load_file(str(path), device=str(device))


def flatten_state_dict(state_dict: dict[str, Tensor]) -> tuple[Tensor, list[tuple[str, torch.Size]]]:
    """Flatten a state dict into a single 1-D float32 vector.

    Iteration follows the dict's own (insertion) order, which for
    ``model.state_dict()`` is the module registration order — stable
    across processes for the same architecture. The returned spec makes
    the operation invertible and lets the receiver *verify* structure
    instead of trusting the sender.

    Args:
        state_dict: Parameter-name → tensor mapping.

    Returns:
        ``(vector, spec)`` where ``spec`` is a list of
        ``(parameter_name, shape)`` in flattening order.

    Raises:
        ValueError: If the state dict is empty.
    """
    if not state_dict:
        raise ValueError("cannot flatten an empty state dict")
    spec: list[tuple[str, torch.Size]] = []
    chunks: list[Tensor] = []
    for name, tensor in state_dict.items():
        spec.append((name, tensor.shape))
        chunks.append(tensor.detach().reshape(-1).to(torch.float32))
    return torch.cat(chunks), spec


def unflatten_to_state_dict(
    vector: Tensor, spec: list[tuple[str, torch.Size]]
) -> "OrderedDict[str, Tensor]":
    """Inverse of :func:`flatten_state_dict`.

    Args:
        vector: 1-D tensor whose length equals the total element count
            described by ``spec``.
        spec: ``(parameter_name, shape)`` list from
            :func:`flatten_state_dict`.

    Returns:
        Ordered state dict with tensors reshaped per the spec.

    Raises:
        ValueError: If ``vector`` is not 1-D or its length does not match
            the spec — the first integrity check against malformed client
            payloads.
    """
    if vector.dim() != 1:
        raise ValueError(f"expected a 1-D vector, got shape {tuple(vector.shape)}")
    expected = sum(int(torch.Size(shape).numel()) for _, shape in spec)
    if vector.numel() != expected:
        raise ValueError(
            f"vector has {vector.numel()} elements but spec describes {expected}"
        )
    out: OrderedDict[str, Tensor] = OrderedDict()
    offset = 0
    for name, shape in spec:
        n = int(torch.Size(shape).numel())
        out[name] = vector[offset : offset + n].reshape(shape).clone()
        offset += n
    return out


def state_dict_delta(
    new: dict[str, Tensor], old: dict[str, Tensor]
) -> "OrderedDict[str, Tensor]":
    """Compute ``new - old`` per parameter (the Δw_k a client transmits).

    Args:
        new: Locally trained weights.
        old: Global weights the client started the round from.

    Returns:
        Ordered mapping of parameter name to difference tensor.

    Raises:
        ValueError: If the two dicts do not share identical keys and
            shapes — catches architecture mismatches before they corrupt
            aggregation.
    """
    if new.keys() != old.keys():
        missing = set(new.keys()) ^ set(old.keys())
        raise ValueError(f"state dict key mismatch: {sorted(missing)}")
    out: OrderedDict[str, Tensor] = OrderedDict()
    for name, new_t in new.items():
        old_t = old[name]
        if new_t.shape != old_t.shape:
            raise ValueError(
                f"shape mismatch for '{name}': {tuple(new_t.shape)} vs {tuple(old_t.shape)}"
            )
        out[name] = (new_t.detach().cpu() - old_t.detach().cpu()).to(torch.float32)
    return out
