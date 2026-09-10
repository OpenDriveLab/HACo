"""Trained real-time chunking primitives shared by all HACO variants."""

from __future__ import annotations

import math
from typing import Sequence

import torch

from .contract import HacoActionContract


RTC_TRAINING_MAX_PREFIX_STEPS = 12
RTC_INFERENCE_PREFIX_STEPS = 10


def default_prefix_weights(
    max_prefix_steps: int = RTC_TRAINING_MAX_PREFIX_STEPS,
    preferred_steps: int = RTC_INFERENCE_PREFIX_STEPS,
) -> list[float]:
    weights = [1.0] * (int(max_prefix_steps) + 1)
    if not 0 <= int(preferred_steps) <= int(max_prefix_steps):
        raise ValueError("preferred RTC prefix is outside the trained range")
    weights[int(preferred_steps)] = 2.0
    return weights


def validate_rtc_configuration(
    *,
    horizon: int,
    max_prefix_steps: int,
    inference_prefix_steps: int,
    prefix_weights: Sequence[float],
) -> None:
    if int(horizon) != 40:
        raise ValueError(f"HACO RTC requires horizon 40, got {horizon}")
    if not 0 <= int(max_prefix_steps) < int(horizon):
        raise ValueError("RTC training prefix range must fit inside the horizon")
    if not 0 <= int(inference_prefix_steps) <= int(max_prefix_steps):
        raise ValueError("RTC inference prefix must be represented in training")
    if len(prefix_weights) != int(max_prefix_steps) + 1:
        raise ValueError("RTC prefix weights must cover every length from 0..max")
    values = [float(value) for value in prefix_weights]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("RTC prefix weights must be finite and nonnegative")
    if sum(values) <= 0:
        raise ValueError("at least one RTC prefix weight must be positive")


def sample_prefix_steps(
    action_mask: torch.Tensor,
    prefix_weights: Sequence[float],
) -> torch.Tensor:
    """Sample a valid clean-prefix length independently for each batch row."""

    if action_mask.ndim != 3:
        raise ValueError("action_mask must be [B,T,D]")
    weights = torch.as_tensor(
        prefix_weights, dtype=torch.float32, device=action_mask.device
    )[None].expand(action_mask.shape[0], -1).clone()
    valid_steps = action_mask.bool().any(dim=-1).sum(dim=-1)
    candidates = torch.arange(weights.shape[1], device=weights.device)[None]
    weights.masked_fill_(candidates >= valid_steps[:, None], 0.0)
    if torch.any(weights.sum(dim=-1) <= 0):
        raise ValueError("every RTC sample must contain at least one valid action")
    return torch.multinomial(weights, 1, replacement=True).squeeze(1)


def temporal_prefix_mask(
    prefix_steps: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    if prefix_steps.ndim != 1:
        raise ValueError("prefix_steps must be [B]")
    positions = torch.arange(int(horizon), device=prefix_steps.device)[None]
    return positions < prefix_steps[:, None]


def postfix_loss_mask(
    action_mask: torch.Tensor,
    prefix_steps: torch.Tensor,
    contract: HacoActionContract,
) -> torch.Tensor:
    """Select active, valid postfix elements and exclude every clean prefix."""

    if action_mask.ndim != 3 or action_mask.shape[-1] != contract.expert_dim:
        raise ValueError("action_mask does not match the HACO carrier")
    active = contract.active_mask(device=action_mask.device).view(1, 1, -1)
    prefix = temporal_prefix_mask(prefix_steps, action_mask.shape[1])
    return action_mask.bool() & active & (~prefix[..., None])


def clamp_prefix(
    actions: torch.Tensor,
    prefix: torch.Tensor,
    contract: HacoActionContract,
) -> torch.Tensor:
    """Hard-lock all and only active prefix components after a flow step."""

    contract.validate_carrier(actions)
    contract.validate_carrier(prefix)
    if actions.ndim != 3 or prefix.ndim != 3:
        raise ValueError("RTC action and prefix tensors must be [B,T,D]")
    if actions.shape[0] != prefix.shape[0]:
        raise ValueError("RTC action and prefix batch sizes differ")
    if prefix.shape[1] > actions.shape[1]:
        raise ValueError("RTC prefix exceeds generated action horizon")
    active = contract.active_mask(device=actions.device).view(1, 1, -1)
    clean = torch.where(active, prefix.to(actions), torch.zeros_like(prefix))
    output = actions.clone()
    output[:, : prefix.shape[1]] = clean
    return output


__all__ = [
    "RTC_INFERENCE_PREFIX_STEPS",
    "RTC_TRAINING_MAX_PREFIX_STEPS",
    "clamp_prefix",
    "default_prefix_weights",
    "postfix_loss_mask",
    "sample_prefix_steps",
    "temporal_prefix_mask",
    "validate_rtc_configuration",
]
