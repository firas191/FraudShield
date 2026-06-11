"""FraudShield model package: architecture and weight-serialization utilities."""

from model.fraud_mlp import FraudDetectorMLP, ModelConfig
from model.utils import (
    flatten_state_dict,
    load_weights,
    save_weights,
    state_dict_delta,
    unflatten_to_state_dict,
)

__all__ = [
    "FraudDetectorMLP",
    "ModelConfig",
    "save_weights",
    "load_weights",
    "flatten_state_dict",
    "unflatten_to_state_dict",
    "state_dict_delta",
]
