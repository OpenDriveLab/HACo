from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dexterity.models.pace.force_encoder import ForceEncoder
from dexterity.models.haco.action_head import HacoActionHead
from dexterity.models.haco.contract import HACO_ACTION_CONTRACTS
from dexterity.models.haco.model import (
    HACO_EXTENSION_INIT_SEED,
    Haco,
    reinitialize_haco_extension_parameters,
)
from scripts.train.haco.checkpoint import load_official_as_haco


class _PhysicalAdapter(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attention = nn.Sequential(nn.Linear(width, width), nn.Linear(width, width))
        self.gate = nn.Parameter(torch.empty(()))


class _PhysicalModel(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.shared_dit_projection = nn.Linear(width, width)
        self.physical_cross_adapters = nn.ModuleList([_PhysicalAdapter(width)])
        self.null_physical_token = nn.Parameter(torch.empty(1, 1, width))


class _ActionHead(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.shared_decoder = nn.Linear(config.input_embedding_dim, 4)
        self.force_encoder = ForceEncoder(config)
        self.sensor_to_vlm = nn.Sequential(
            nn.LayerNorm(config.input_embedding_dim),
            nn.Linear(config.input_embedding_dim, config.backbone_embedding_dim),
        )
        self.physical_type_embedding = nn.Parameter(
            torch.empty(1, 1, config.backbone_embedding_dim)
        )
        self.model = _PhysicalModel(config.input_embedding_dim)


class _MigrationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            force_history_length=9,
            force_joint_count=44,
            sensor_encoder_dim=16,
            joint_history_dim=8,
            tactile_finger_count=10,
            input_embedding_dim=16,
            backbone_embedding_dim=24,
            sensor_encoder_heads=4,
            sensor_encoder_layers=1,
            attn_dropout=0.0,
            physical_cross_gate_init=0.125,
            physical_type_embedding_std=0.02,
            model_type="Haco",
            experiment_id="hp_wo_torque",
            action_contract="joint_compliance_delta",
            action_target="q_compliance",
        )
        self.backbone_shared = nn.Linear(5, 7)
        self.action_head = _ActionHead(self.config)


def _extension_keys(model: nn.Module) -> list[str]:
    prefixes = (
        "action_head.force_encoder.",
        "action_head.sensor_to_vlm.",
        "action_head.physical_type_embedding",
        "action_head.model.physical_cross_adapters.",
        "action_head.model.null_physical_token",
    )
    return sorted(
        name
        for name, _ in model.named_parameters()
        if name.startswith(prefixes)
    )


def test_extension_reinit_is_finite_deterministic_and_preserves_shared() -> None:
    model = _MigrationModel()
    missing = _extension_keys(model)
    parameters = dict(model.named_parameters())
    shared_before = {
        name: value.detach().clone()
        for name, value in parameters.items()
        if name not in missing
    }
    with torch.no_grad():
        for key in missing:
            parameters[key].fill_(float("inf"))

    report = reinitialize_haco_extension_parameters(model, missing)
    first = {key: parameters[key].detach().clone() for key in missing}

    assert report == {
        "seed": HACO_EXTENSION_INIT_SEED,
        "reinitialized_keys": missing,
        "all_finite": True,
    }
    assert all(torch.isfinite(parameters[key]).all() for key in missing)
    for name, expected in shared_before.items():
        torch.testing.assert_close(parameters[name], expected, rtol=0, atol=0)

    modality = parameters[
        "action_head.force_encoder.contextualizer.modality_embedding"
    ]
    hand = parameters["action_head.force_encoder.contextualizer.hand_embedding.weight"]
    digit = parameters[
        "action_head.force_encoder.contextualizer.digit_embedding.weight"
    ]
    type_embedding = parameters["action_head.physical_type_embedding"]
    assert 0.001 < float(modality.detach().std()) < 0.08
    assert 0.001 < float(hand.detach().std()) < 0.08
    assert 0.001 < float(digit.detach().std()) < 0.08
    assert 0.001 < float(type_embedding.detach().std()) < 0.08
    assert parameters[
        "action_head.model.physical_cross_adapters.0.gate"
    ].item() == pytest.approx(0.125)
    assert torch.count_nonzero(
        parameters["action_head.model.null_physical_token"]
    ) == 0

    with torch.no_grad():
        for key in missing:
            parameters[key].fill_(float("nan"))
    reinitialize_haco_extension_parameters(model, missing)
    for key in missing:
        torch.testing.assert_close(parameters[key], first[key], rtol=0, atol=0)


def test_extension_reinit_refuses_shared_or_partial_leaf() -> None:
    model = _MigrationModel()
    with pytest.raises(RuntimeError, match="non-HACO"):
        reinitialize_haco_extension_parameters(model, ["backbone_shared.weight"])
    with pytest.raises(RuntimeError, match="overwrite checkpoint tensors"):
        reinitialize_haco_extension_parameters(
            model, ["action_head.sensor_to_vlm.1.weight"]
        )


def test_training_loss_guard_stops_nan_with_variant_identity() -> None:
    head = SimpleNamespace(
        config=SimpleNamespace(experiment_id="hp_wo_torque"),
        sensor_encoder_mode="tactile_only",
        physical_integration="physcross_gated",
        contract=HACO_ACTION_CONTRACTS["joint_compliance_delta"],
    )
    HacoActionHead._assert_finite_training_loss(head, torch.tensor(1.0))
    with pytest.raises(FloatingPointError) as error:
        HacoActionHead._assert_finite_training_loss(
            head, torch.tensor(float("nan"))
        )
    message = str(error.value)
    assert "experiment=hp_wo_torque" in message
    assert "sensor=tactile_only" in message
    assert "integration=physcross_gated" in message
    assert "contract=joint_compliance_delta" in message


def test_official_loader_records_exact_reinitialization_manifest(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "official"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "Gr00tN1d7",
                "action_horizon": 40,
                "max_action_dim": 132,
            }
        ),
        encoding="utf-8",
    )
    (source / "processor_config.json").write_text("{}", encoding="utf-8")
    model = _MigrationModel()
    missing = _extension_keys(model)
    shared = model.backbone_shared.weight.detach().clone()
    with torch.no_grad():
        for key in missing:
            dict(model.named_parameters())[key].fill_(float("inf"))

    def fake_from_pretrained(cls, *args, **kwargs):
        return model, {
            "missing_keys": missing,
            "unexpected_keys": [],
            "mismatched_keys": [],
        }

    monkeypatch.setattr(Haco, "from_pretrained", classmethod(fake_from_pretrained))
    loaded, info = Haco.from_official_checkpoint(
        source,
        config=model.config,
        output_loading_info=True,
    )

    assert loaded is model
    assert info["reinitialized_haco_extension_keys"] == missing
    assert info["haco_extension_init_seed"] == HACO_EXTENSION_INIT_SEED
    assert info["haco_extension_parameters_finite"] is True
    assert all(
        torch.isfinite(dict(model.named_parameters())[key]).all() for key in missing
    )
    torch.testing.assert_close(model.backbone_shared.weight, shared, rtol=0, atol=0)

    loaded, manifest = load_official_as_haco(
        source,
        config=model.config,
        transformers_loading_kwargs={},
    )
    assert loaded is model
    assert manifest["schema"] == "haco.initialization.v1"
    assert manifest["missing_haco_extension_keys"] == missing
    assert manifest["reinitialized_haco_extension_keys"] == missing
    assert manifest["haco_extension_init_seed"] == HACO_EXTENSION_INIT_SEED
    assert manifest["haco_extension_parameters_finite"] is True
    assert manifest["shared_pretrained_keys_preserved"] is True
