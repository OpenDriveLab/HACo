from __future__ import annotations

from collections.abc import Mapping
import os
from typing import Any

import numpy as np
import torch

from .open_loop_eval_callback import (
    OpenLoopEvalCallback,
    OpenLoopEvalConfig,
    PreparedRollout,
    _select_training_sample,
    _split_control,
    build_open_loop_eval_trainer,
)


HACO_CONTRACT_NAMES = (
    "joint_compliance_delta",
    "compliance_only",
    "nominal_only",
)
HACO_CARRIER_DIM = 132
HACO_CONTROL_DIM = 62
HACO_DELTA_Q_SLICE = slice(62, 106)


def _action_prediction(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, Mapping):
        action = outputs.get("action_pred")
    else:
        action = getattr(outputs, "action_pred", None)
    if not torch.is_tensor(action):
        raise TypeError("HACO get_action() must return tensor action_pred")
    if action.ndim != 3 or action.shape[-1] != HACO_CARRIER_DIM:
        raise ValueError(
            "HACO action prediction must have shape [B,H,132], got "
            f"{tuple(action.shape)}"
        )
    return action


def _contract_by_name(name: str):
    if name not in HACO_CONTRACT_NAMES:
        raise ValueError(
            f"HACO action contract must be one of {HACO_CONTRACT_NAMES}, "
            f"got {name!r}"
        )
    from dexterity.models.haco.contract import HACO_ACTION_CONTRACTS

    return HACO_ACTION_CONTRACTS[name]


def _clean_active_prefix(
    previous_action: torch.Tensor,
    *,
    prefix_steps: int,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """Return the complete active RTC prefix and zero every carrier pad dim."""
    if previous_action.ndim != 3 or previous_action.shape[-1] != HACO_CARRIER_DIM:
        raise ValueError("previous HACO action must have shape [B,H,132]")
    if prefix_steps <= 0 or prefix_steps >= previous_action.shape[1]:
        raise ValueError(
            "RTC prefix must be positive and shorter than the action horizon"
        )
    if active_mask.shape != (HACO_CARRIER_DIM,) or active_mask.dtype != torch.bool:
        raise ValueError("HACO active mask must be bool[132]")
    source = previous_action[:, -prefix_steps:].detach()
    prefix = torch.zeros_like(source)
    prefix[..., active_mask.to(device=source.device)] = source[
        ..., active_mask.to(device=source.device)
    ]
    return prefix


def _prediction_sequence(
    model,
    policy_inputs: Mapping[str, Any],
    *,
    chunk_count: int,
    horizon: int,
    prefix_steps: int,
    active_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Generate adjacent chunks in order, carrying prediction-only RTC state."""
    if chunk_count <= 0:
        raise ValueError("HACO RTC rollout requires at least one chunk")
    predictions = []
    rtc_prefix = None
    for chunk_index in range(chunk_count):
        chunk_inputs = _select_training_sample(policy_inputs, chunk_index)
        options = None
        if rtc_prefix is not None:
            options = {
                "rtc_prefix_steps": prefix_steps,
                "rtc_prefix_action": rtc_prefix,
            }
        outputs = model.get_action(chunk_inputs, options=options)
        action = _action_prediction(outputs)[:, :horizon]
        if action.shape[0] != 1 or action.shape[1] != horizon:
            raise ValueError(
                "HACO sequential rollout requires one complete action chunk; "
                f"got {tuple(action.shape)}"
            )
        predictions.append(action)
        if chunk_index + 1 < chunk_count:
            rtc_prefix = _clean_active_prefix(
                action,
                prefix_steps=prefix_steps,
                active_mask=active_mask,
            )
    return {
        "action_pred": torch.cat(predictions, dim=0),
        "rtc_prefix_steps": torch.as_tensor(
            prefix_steps, device=predictions[0].device, dtype=torch.int64
        ),
    }


def _decoded_value(
    decoded: Mapping[str, Any], candidates: tuple[str, ...]
) -> np.ndarray:
    for key in candidates:
        if key in decoded:
            return np.asarray(decoded[key], dtype=np.float32)
    raise KeyError(f"decoded HACO action is missing all of {candidates}")


def decode_haco_action(
    processor,
    normalized_action: torch.Tensor,
    embodiment,
    state_abs: np.ndarray,
    *,
    action_contract: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Decode direct main-q control and optional physical-rad delta_q."""
    if action_contract not in HACO_CONTRACT_NAMES:
        raise ValueError(f"unsupported HACO contract {action_contract!r}")
    decoded = processor.decode_action(
        normalized_action.detach().float().cpu().numpy(),
        embodiment,
        state=_split_control(state_abs),
    )
    left_wrist = _decoded_value(
        decoded, ("left_wrist_eef", "action.left_wrist_eef")
    )
    right_wrist = _decoded_value(
        decoded, ("right_wrist_eef", "action.right_wrist_eef")
    )
    if action_contract == "nominal_only":
        left_q_candidates = (
            "left_hand_joints",
            "action.left_hand_joints",
        )
        right_q_candidates = (
            "right_hand_joints",
            "action.right_hand_joints",
        )
    else:
        left_q_candidates = (
            "left_hand_q_teleop",
            "action.left_hand_q_teleop",
            "left_hand_joints",
            "action.left_hand_joints",
        )
        right_q_candidates = (
            "right_hand_q_teleop",
            "action.right_hand_q_teleop",
            "right_hand_joints",
            "action.right_hand_joints",
        )
    control = np.concatenate(
        (
            left_wrist,
            right_wrist,
            _decoded_value(decoded, left_q_candidates),
            _decoded_value(decoded, right_q_candidates),
        ),
        axis=-1,
    ).astype(np.float32, copy=False)
    delta_q = None
    if action_contract == "joint_compliance_delta":
        delta_q = np.concatenate(
            (
                _decoded_value(
                    decoded,
                    ("left_hand_delta_q", "action.left_hand_delta_q"),
                ),
                _decoded_value(
                    decoded,
                    ("right_hand_delta_q", "action.right_hand_delta_q"),
                ),
            ),
            axis=-1,
        ).astype(np.float32, copy=False)
    return control, delta_q


class HacoOpenLoopEvalCallback(OpenLoopEvalCallback):
    """HACO trained-RTC probe with direct main-q execution semantics."""

    def __init__(
        self,
        trainer,
        config: OpenLoopEvalConfig,
        *,
        action_contract: str,
        rtc_prefix_steps: int = 10,
    ) -> None:
        super().__init__(trainer, config)
        self.action_contract_name = action_contract
        self.action_contract = _contract_by_name(action_contract)
        self.rtc_prefix_steps = int(rtc_prefix_steps)
        if not 0 < self.rtc_prefix_steps < config.action_horizon:
            raise ValueError("rtc_prefix_steps must be in [1, action_horizon)")
        self.render_delta_q_slice = (
            HACO_DELTA_Q_SLICE
            if action_contract == "joint_compliance_delta"
            else None
        )

    @property
    def action_target(self) -> str:
        return (
            "q_nominal"
            if self.action_contract_name == "nominal_only"
            else "q_compliance"
        )

    def prediction_for_unapply(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape[-1] != HACO_CARRIER_DIM:
            raise ValueError("HACO action carrier must end in 132")
        return action

    def control_norm(self, action: torch.Tensor) -> torch.Tensor:
        # The main 62-D command is executable as-is. In particular, joint mode
        # never adds delta_q to the q_compliance branch.
        return action[..., :HACO_CONTROL_DIM]

    def target_control_norm(
        self,
        batch: Mapping,
        processor,
        target_control_abs,
        state_abs,
        tag_value: str,
        chunk_count: int,
        horizon: int,
    ):
        del processor, target_control_abs, state_abs, tag_value
        target = batch.get("action")
        if not torch.is_tensor(target):
            raise TypeError("HACO open-loop evaluation requires tensor action")
        if target.shape[-1] != HACO_CARRIER_DIM:
            raise ValueError("HACO training target must use the 132-D carrier")
        return (
            target[:chunk_count, :horizon, :HACO_CONTROL_DIM]
            .detach()
            .float()
            .cpu()
            .numpy()
            .reshape(-1, HACO_CONTROL_DIM)
        )

    def extra_rollout_metrics(
        self, action_pred: torch.Tensor, batch: Mapping
    ) -> Mapping[str, torch.Tensor]:
        # delta_q is reported only after processor decoding, in physical rad.
        del action_pred, batch
        return {}

    def rollout_outputs(self, model, prepared: PreparedRollout) -> Any:
        return _prediction_sequence(
            model,
            prepared.policy_inputs,
            chunk_count=prepared.chunk_count,
            horizon=prepared.horizon,
            prefix_steps=self.rtc_prefix_steps,
            active_mask=self.action_contract.active_mask(),
        )

    def rollout_metric_step_mask(
        self,
        *,
        chunk_count: int,
        horizon: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = torch.ones((chunk_count, horizon), dtype=torch.bool, device=device)
        if chunk_count > 1:
            mask[1:, : self.rtc_prefix_steps] = False
        return mask

    def _prepare_rollout(
        self,
        batch: Mapping[str, Any],
        processor,
        tag_value: str,
        model_device: torch.device,
    ) -> PreparedRollout:
        adapted = dict(batch)
        haco_action = adapted.get("viz_rollout_gt_haco_action")
        if self.action_contract_name == "joint_compliance_delta":
            if haco_action is None:
                raise KeyError(
                    "joint HACO rollout requires "
                    "viz_rollout_gt_haco_action"
                )
            adapted["viz_rollout_gt_component_major_action"] = haco_action
        return super()._prepare_rollout(
            adapted, processor, tag_value, model_device
        )

    def _decode_action(
        self,
        processor,
        normalized_action: torch.Tensor,
        embodiment,
        state_abs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        return decode_haco_action(
            processor,
            normalized_action,
            embodiment,
            state_abs,
            action_contract=self.action_contract_name,
        )

    def _unapply_control(
        self,
        processor,
        normalized_action: torch.Tensor,
        embodiment,
        state_abs: np.ndarray,
    ) -> np.ndarray:
        control, _ = self._decode_action(
            processor, normalized_action, embodiment, state_abs
        )
        return control

    def _prediction_action_for_render(
        self,
        processor,
        action_pred: torch.Tensor,
        embodiment,
        prepared: PreparedRollout,
        prediction_control_abs: np.ndarray,
    ) -> np.ndarray | None:
        if self.action_contract_name != "joint_compliance_delta":
            return None
        chunks = []
        for chunk_index in range(min(prepared.chunk_count, action_pred.shape[0])):
            control, delta_q = self._decode_action(
                processor,
                action_pred[chunk_index : chunk_index + 1],
                embodiment,
                prepared.state_abs[chunk_index : chunk_index + 1],
            )
            if delta_q is None:
                raise AssertionError("joint contract must decode delta_q")
            chunks.append(np.concatenate((control[0], delta_q[0]), axis=-1))
        if not chunks:
            raise ValueError("HACO component-major decode produced no chunks")
        return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)

    def extra_decoded_rollout_metrics(
        self,
        prediction_action_for_render: np.ndarray | None,
        prepared: PreparedRollout,
        metric_step_mask: np.ndarray,
    ) -> Mapping[str, float]:
        if self.action_contract_name != "joint_compliance_delta":
            return {}
        target = prepared.target_action_for_render
        if target is None or prediction_action_for_render is None:
            raise ValueError("joint HACO delta metric requires decoded GT/prediction")
        steps = min(len(metric_step_mask), len(target), len(prediction_action_for_render))
        include = np.asarray(metric_step_mask[:steps], dtype=bool)
        if not include.any():
            raise ValueError("HACO RTC metric mask excludes every step")
        error = np.abs(
            prediction_action_for_render[:steps, HACO_DELTA_Q_SLICE][include]
            - target[:steps, HACO_DELTA_Q_SLICE][include]
        )
        return {"open-loop-eval/delta_q_mae_rad": float(np.mean(error))}


def _trainer_contract_name(trainer) -> str:
    value = os.environ.get("HACO_ACTION_CONTRACT")
    if value:
        return value
    model = getattr(trainer, "model", None)
    config = getattr(model, "config", None)
    value = getattr(config, "action_contract", None)
    if value is None:
        value = getattr(config, "action_contract_name", None)
    if value not in HACO_CONTRACT_NAMES:
        raise ValueError(
            "HACO callback could not resolve action contract from "
            "HACO_ACTION_CONTRACT or model.config"
        )
    return str(value)


def install_haco_open_loop_eval_callback() -> None:
    if os.environ.get("OPEN_LOOP_EVAL_ENABLE", "1").lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return
    import gr00t.experiment.experiment as experiment_module

    config = OpenLoopEvalConfig(
        project_root=os.environ.get("HACO_ROOT", "."),
        embodiment_tag=os.environ.get(
            "HACO_EMBODIMENT_TAG", "real_r1_pro_sharpa_absolute_eef"
        ),
        action_horizon=int(os.environ.get("HACO_ACTION_HORIZON", "40")),
        every_n_steps=int(os.environ.get("OPEN_LOOP_EVAL_EVERY_N_STEPS", "10000")),
        first_step=os.environ.get("OPEN_LOOP_EVAL_FIRST_STEP", "1").lower()
        in {"1", "true", "yes", "on"},
        log_name="haco_rtc_open_loop_eval",
        sample_seed=int(os.environ.get("OPEN_LOOP_EVAL_SEED", "42")),
    )
    rtc_prefix_steps = int(os.environ.get("HACO_RTC_PREFIX_STEPS", "10"))
    experiment_module.Gr00tTrainer = build_open_loop_eval_trainer(
        experiment_module.Gr00tTrainer,
        lambda trainer: HacoOpenLoopEvalCallback(
            trainer,
            config,
            action_contract=_trainer_contract_name(trainer),
            rtc_prefix_steps=rtc_prefix_steps,
        ),
        callback_attribute="_haco_open_loop_eval_callback",
    )


__all__ = [
    "HACO_CONTRACT_NAMES",
    "HacoOpenLoopEvalCallback",
    "_clean_active_prefix",
    "_prediction_sequence",
    "decode_haco_action",
    "install_haco_open_loop_eval_callback",
]
