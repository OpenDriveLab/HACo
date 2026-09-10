from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dexterity.models.pace.masked_dit import MaskedAlternateVLDiT
from dexterity.models.haco.config import HacoConfig
from dexterity.models.haco.contract import HACO_ACTION_CONTRACTS
from dexterity.models.haco.physical_integration import HacoPhysicalCrossDiT
from dexterity.models.haco.rtc import (
    clamp_prefix,
    default_prefix_weights,
    postfix_loss_mask,
)


def _components(batch: int = 2, horizon: int = 40):
    wrist = torch.randn(batch, horizon, 18)
    compliance = torch.randn(batch, horizon, 44)
    nominal = torch.randn(batch, horizon, 44)
    delta = compliance - nominal
    return wrist, compliance, nominal, delta


def test_joint_contract_is_compliance_plus_aux_delta_but_executes_compliance() -> None:
    contract = HACO_ACTION_CONTRACTS["joint_compliance_delta"]
    wrist, compliance, _, delta = _components()
    carrier = contract.pack(
        wrist, q_compliance=compliance, delta_q=delta
    )

    assert carrier.shape == (2, 40, 132)
    assert contract.semantic_dim == 106
    assert torch.count_nonzero(carrier[..., 106:]) == 0
    torch.testing.assert_close(contract.executable(carrier)[..., :18], wrist)
    torch.testing.assert_close(contract.executable_q(carrier), compliance)
    assert not torch.equal(contract.executable_q(carrier), compliance + delta)


@pytest.mark.parametrize(
    ("name", "target"),
    (("compliance_only", "q_compliance"), ("nominal_only", "q_nominal")),
)
def test_ac_contracts_are_true_62d_vocabularies_without_delta(
    name: str, target: str
) -> None:
    contract = HACO_ACTION_CONTRACTS[name]
    wrist, compliance, nominal, delta = _components()
    q_kwargs = {target: compliance if target == "q_compliance" else nominal}
    carrier = contract.pack(wrist, **q_kwargs)

    assert contract.semantic_dim == 62
    assert contract.component_names == ("wrist", target)
    assert "delta_q" not in contract.unpack(carrier)
    assert torch.count_nonzero(carrier[..., 62:]) == 0
    with pytest.raises(ValueError, match="must not receive delta_q"):
        contract.pack(wrist, delta_q=delta, **q_kwargs)


def test_rtc_prefix_locks_all_joint_components_and_loss_is_postfix_only() -> None:
    contract = HACO_ACTION_CONTRACTS["joint_compliance_delta"]
    wrist, compliance, _, delta = _components(batch=2, horizon=10)
    prefix = contract.pack(wrist, q_compliance=compliance, delta_q=delta)[:, :3]
    generated = torch.randn(2, 10, 132)
    clamped = clamp_prefix(generated, prefix, contract)

    torch.testing.assert_close(clamped[:, :3, :106], prefix[..., :106])
    assert torch.count_nonzero(clamped[:, :3, 106:]) == 0
    action_mask = contract.active_mask().view(1, 1, -1).expand(2, 10, -1)
    loss_mask = postfix_loss_mask(action_mask, torch.tensor([3, 1]), contract)
    assert not loss_mask[0, :3].any()
    assert loss_mask[0, 3:, :106].all()
    assert not loss_mask[..., 106:].any()
    assert not loss_mask[1, :1].any()


def test_rtc_and_config_matrix_are_frozen() -> None:
    assert default_prefix_weights() == [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        2.0,
        1.0,
        1.0,
    ]
    joint = HacoConfig(action_contract="joint_compliance_delta")
    ac2 = HacoConfig(action_contract="compliance_only")
    ac3 = HacoConfig(action_contract="nominal_only")
    assert joint.action_target == ac2.action_target == "q_compliance"
    assert ac3.action_target == "q_nominal"
    assert joint.delta_q_flow_weight == 0.5
    assert ac2.delta_q_flow_weight == ac3.delta_q_flow_weight == 0.0
    assert joint.action_horizon == 40
    assert joint.rtc_training_max_prefix_steps == 12
    assert joint.rtc_inference_prefix_steps == 10


def test_empty_physical_memory_is_a_strict_hp_wo_haptic_noop(monkeypatch) -> None:
    sentinel = object()
    calls = []

    def fake_base_forward(self, **kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(MaskedAlternateVLDiT, "forward", fake_base_forward)
    batch, suffix, width = 2, 4, 8
    output = HacoPhysicalCrossDiT.forward(
        SimpleNamespace(),
        hidden_states=torch.zeros(batch, suffix, width),
        encoder_hidden_states=torch.zeros(batch, 3, width),
        timestep=torch.zeros(batch, dtype=torch.long),
        image_mask=torch.ones(batch, 3, dtype=torch.bool),
        backbone_attention_mask=torch.ones(batch, 3, dtype=torch.bool),
        suffix_attention_mask=torch.ones(batch, suffix, dtype=torch.bool),
        physical_hidden_states=torch.zeros(batch, 0, width),
        physical_attention_mask=torch.zeros(batch, 0, dtype=torch.bool),
    )

    assert output is sentinel
    assert len(calls) == 1
