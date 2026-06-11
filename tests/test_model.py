"""Week 1 unit tests: model forward pass, DP compatibility, serialization.

Run from repo root:  pytest tests/test_model.py -v
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from model.fraud_mlp import FraudDetectorMLP, ModelConfig
from model.utils import (
    flatten_state_dict,
    load_weights,
    save_weights,
    state_dict_delta,
    unflatten_to_state_dict,
)

BATCH = 32
INPUT_DIM = 30


@pytest.fixture()
def model() -> FraudDetectorMLP:
    torch.manual_seed(42)
    return FraudDetectorMLP()


@pytest.fixture()
def batch() -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randn(BATCH, INPUT_DIM)


class TestForwardPass:
    def test_output_shape(self, model: FraudDetectorMLP, batch: torch.Tensor) -> None:
        assert model(batch).shape == (BATCH, 1)

    def test_outputs_are_logits_not_probabilities(
        self, model: FraudDetectorMLP, batch: torch.Tensor
    ) -> None:
        """The head must emit raw logits. With the output bias initialized
        to log-prior-odds (~ -6.38 for 0.17% fraud), every initial logit
        sits far below 0 — impossible if a sigmoid were baked in."""
        out = model(batch)
        assert out.min().item() < 0.0, "no negative outputs — is a Sigmoid baked in?"

    def test_prior_bias_initialization(self, model: FraudDetectorMLP) -> None:
        expected = math.log(0.0017 / (1 - 0.0017))
        assert model.head.bias.item() == pytest.approx(expected, rel=1e-6)

    def test_predict_proba_range_and_shape(
        self, model: FraudDetectorMLP, batch: torch.Tensor
    ) -> None:
        proba = model.predict_proba(batch)
        assert proba.shape == (BATCH,)
        assert torch.all((proba >= 0) & (proba <= 1))

    def test_predict_proba_restores_training_mode(
        self, model: FraudDetectorMLP, batch: torch.Tensor
    ) -> None:
        model.train()
        model.predict_proba(batch)
        assert model.training

    def test_batch_size_one(self, model: FraudDetectorMLP) -> None:
        """GroupNorm (unlike BatchNorm) must handle a single sample —
        required for the Week 5 DLG attack on per-sample gradients."""
        out = model(torch.randn(1, INPUT_DIM))
        assert out.shape == (1, 1)

    def test_rejects_wrong_feature_count(self, model: FraudDetectorMLP) -> None:
        with pytest.raises(ValueError, match="expected input"):
            model(torch.randn(BATCH, INPUT_DIM + 1))

    def test_rejects_non_2d_input(self, model: FraudDetectorMLP) -> None:
        with pytest.raises(ValueError, match="expected input"):
            model(torch.randn(BATCH))

    def test_eval_mode_is_deterministic(
        self, model: FraudDetectorMLP, batch: torch.Tensor
    ) -> None:
        model.eval()
        assert torch.equal(model(batch), model(batch))

    def test_gradients_flow_to_all_parameters(
        self, model: FraudDetectorMLP, batch: torch.Tensor
    ) -> None:
        target = torch.zeros(BATCH, 1)
        loss = torch.nn.BCEWithLogitsLoss()(model(batch), target)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"no gradient for {name}"
            assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"


class TestOpacusCompatibility:
    def test_no_batchnorm_modules(self, model: FraudDetectorMLP) -> None:
        banned = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
        offenders = [n for n, m in model.named_modules() if isinstance(m, banned)]
        assert not offenders, f"BatchNorm breaks per-sample gradients: {offenders}"

    def test_opacus_module_validator(self, model: FraudDetectorMLP) -> None:
        """Authoritative check: Opacus's own validator must pass."""
        validators = pytest.importorskip("opacus.validators")
        errors = validators.ModuleValidator.validate(model, strict=False)
        assert errors == [], f"Opacus rejects the model: {errors}"


class TestConfigValidation:
    def test_default_config_values(self) -> None:
        cfg = ModelConfig()
        assert cfg.input_dim == 30
        assert cfg.hidden_dims == (256, 128, 64)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"input_dim": 0},
            {"hidden_dims": ()},
            {"dropout": 1.0},
            {"dropout": -0.1},
            {"fraud_prior": 0.0},
            {"hidden_dims": (250, 128, 64)},  # 250 not divisible by 8 groups
        ],
    )
    def test_invalid_configs_rejected(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            ModelConfig(**kwargs)

    def test_custom_architecture(self) -> None:
        cfg = ModelConfig(hidden_dims=(64, 32), group_norm_groups=4)
        m = FraudDetectorMLP(cfg)
        assert m(torch.randn(4, 30)).shape == (4, 1)


class TestSerialization:
    def test_safetensors_round_trip(
        self, model: FraudDetectorMLP, batch: torch.Tensor, tmp_path: Path
    ) -> None:
        path = tmp_path / "ckpt" / "global_round_0.safetensors"
        save_weights(model.state_dict(), path)
        restored = FraudDetectorMLP()
        restored.load_state_dict(load_weights(path))
        model.eval()
        restored.eval()
        assert torch.equal(model(batch), restored(batch))

    def test_save_empty_dict_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            save_weights({}, tmp_path / "x.safetensors")

    def test_load_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_weights(tmp_path / "nope.safetensors")

    def test_flatten_round_trip(self, model: FraudDetectorMLP) -> None:
        sd = model.state_dict()
        vec, spec = flatten_state_dict(sd)
        assert vec.dim() == 1
        assert vec.numel() == model.parameter_count()
        rebuilt = unflatten_to_state_dict(vec, spec)
        for name, t in sd.items():
            assert torch.allclose(t.float(), rebuilt[name]), name

    def test_unflatten_rejects_wrong_length(self, model: FraudDetectorMLP) -> None:
        vec, spec = flatten_state_dict(model.state_dict())
        with pytest.raises(ValueError, match="elements"):
            unflatten_to_state_dict(vec[:-1], spec)

    def test_state_dict_delta(self, model: FraudDetectorMLP) -> None:
        old = {k: v.clone() for k, v in model.state_dict().items()}
        new = {k: v + 1.0 for k, v in old.items()}
        delta = state_dict_delta(new, old)
        for t in delta.values():
            assert torch.allclose(t, torch.ones_like(t))

    def test_state_dict_delta_rejects_key_mismatch(self, model: FraudDetectorMLP) -> None:
        old = dict(model.state_dict())
        new = dict(old)
        new.pop(next(iter(new)))
        with pytest.raises(ValueError, match="key mismatch"):
            state_dict_delta(new, old)
