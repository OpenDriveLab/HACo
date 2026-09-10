"""Single configuration surface for the HACO experiment matrix."""

from __future__ import annotations

import math
from typing import Any

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config

from .contract import (
    ACTION_CONTRACT_NAMES,
    HACO_ACTION_HORIZON,
    HACO_EXPERT_ACTION_DIM,
    get_action_contract,
)
from .rtc import (
    RTC_INFERENCE_PREFIX_STEPS,
    RTC_TRAINING_MAX_PREFIX_STEPS,
    default_prefix_weights,
    validate_rtc_configuration,
)

HACO_MODEL_TYPE = "Haco"
HACO_CHECKPOINT_SCHEMA = "haco.checkpoint.v1"
HACO_OFFICIAL_BASE = "checkpoints/base_model"

# Checkpoints through 500k used the original internal labels.  Normalize them
# at load time so their weights remain usable with the public q_cmp/q_obs API.
LEGACY_ACTION_TARGETS = {"q_compliance": "q_cmp", "q_nominal": "q_obs"}

SENSOR_ENCODER_MODES = (
    "none",
    "tactile_only",
    "torque_only",
    "separate",
    "fused",
)
PHYSICAL_INTEGRATIONS = (
    "suffix",
    "imgmem",
    "physcross_ungated",
    "physcross_gated",
)
CAMERA_MODES = ("three", "ego")

HACO_CONFIG_FIELDS = (
    "haco_schema",
    "official_base_checkpoint",
    "experiment_id",
    "sensor_encoder_mode",
    "physical_integration",
    "action_contract",
    "action_target",
    "camera_mode",
    "force_history_length",
    "force_joint_count",
    "force_input_dim",
    "joint_history_dim",
    "tactile_history_length",
    "tactile_finger_count",
    "tactile_wrench_dim",
    "tactile_image_size",
    "sensor_encoder_dim",
    "sensor_encoder_heads",
    "sensor_encoder_layers",
    "tune_sensor_encoders",
    "control_flow_weight",
    "delta_q_flow_weight",
    "delta_q_normalized_clip",
    "physical_cross_gate_init",
    "physical_type_embedding_std",
    "rtc_training_max_prefix_steps",
    "rtc_inference_prefix_steps",
    "rtc_training_prefix_weights",
    "rtc_training_sampling",
)


class HacoConfig(Gr00tN1d7Config):
    """Configuration for one HACO experiment variant."""

    model_type = HACO_MODEL_TYPE

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("model_type", None)
        kwargs.pop("architectures", None)
        action_target = kwargs.pop("action_target", None)
        action_target = LEGACY_ACTION_TARGETS.get(action_target, action_target)
        kwargs["max_action_dim"] = HACO_EXPERT_ACTION_DIM
        kwargs["action_horizon"] = HACO_ACTION_HORIZON
        kwargs["use_relative_action"] = False

        custom: dict[str, Any] = {
            "haco_schema": kwargs.pop("haco_schema", HACO_CHECKPOINT_SCHEMA),
            "official_base_checkpoint": kwargs.pop(
                "official_base_checkpoint", HACO_OFFICIAL_BASE
            ),
            "experiment_id": kwargs.pop("experiment_id", "haco"),
            "sensor_encoder_mode": kwargs.pop("sensor_encoder_mode", "fused"),
            "physical_integration": kwargs.pop(
                "physical_integration", "physcross_gated"
            ),
            "action_contract": kwargs.pop("action_contract", "joint_compliance_delta"),
            "action_target": action_target,
            "camera_mode": kwargs.pop("camera_mode", "three"),
            "force_history_length": int(kwargs.pop("force_history_length", 9)),
            "force_joint_count": int(kwargs.pop("force_joint_count", 44)),
            "force_input_dim": int(kwargs.pop("force_input_dim", 1)),
            "joint_history_dim": int(kwargs.pop("joint_history_dim", 64)),
            "tactile_history_length": int(kwargs.pop("tactile_history_length", 9)),
            "tactile_finger_count": int(kwargs.pop("tactile_finger_count", 10)),
            "tactile_wrench_dim": int(kwargs.pop("tactile_wrench_dim", 6)),
            "tactile_image_size": int(kwargs.pop("tactile_image_size", 240)),
            "sensor_encoder_dim": int(kwargs.pop("sensor_encoder_dim", 256)),
            "sensor_encoder_heads": int(kwargs.pop("sensor_encoder_heads", 8)),
            "sensor_encoder_layers": int(kwargs.pop("sensor_encoder_layers", 2)),
            "tune_sensor_encoders": bool(kwargs.pop("tune_sensor_encoders", True)),
            "control_flow_weight": float(kwargs.pop("control_flow_weight", 1.0)),
            "delta_q_flow_weight": float(kwargs.pop("delta_q_flow_weight", 0.5)),
            "delta_q_normalized_clip": float(
                kwargs.pop("delta_q_normalized_clip", 5.0)
            ),
            "physical_cross_gate_init": float(
                kwargs.pop("physical_cross_gate_init", 0.0)
            ),
            "physical_type_embedding_std": float(
                kwargs.pop("physical_type_embedding_std", 0.02)
            ),
            "rtc_training_max_prefix_steps": int(
                kwargs.pop(
                    "rtc_training_max_prefix_steps",
                    RTC_TRAINING_MAX_PREFIX_STEPS,
                )
            ),
            "rtc_inference_prefix_steps": int(
                kwargs.pop("rtc_inference_prefix_steps", RTC_INFERENCE_PREFIX_STEPS)
            ),
            "rtc_training_prefix_weights": kwargs.pop(
                "rtc_training_prefix_weights", None
            ),
            "rtc_training_sampling": kwargs.pop("rtc_training_sampling", "categorical"),
        }
        if custom["rtc_training_prefix_weights"] is None:
            custom["rtc_training_prefix_weights"] = default_prefix_weights(
                custom["rtc_training_max_prefix_steps"],
                custom["rtc_inference_prefix_steps"],
            )
        else:
            custom["rtc_training_prefix_weights"] = [
                float(value) for value in custom["rtc_training_prefix_weights"]
            ]

        super().__init__(**kwargs)
        self.model_type = type(self).model_type
        self.architectures = ["Haco"]
        for field, value in custom.items():
            setattr(self, field, value)
        self._validate_haco()

    @property
    def action_contract_spec(self):
        return get_action_contract(self.action_contract)

    def _validate_haco(self) -> None:
        if self.haco_schema != HACO_CHECKPOINT_SCHEMA:
            raise ValueError(f"haco_schema must be {HACO_CHECKPOINT_SCHEMA!r}")
        if self.sensor_encoder_mode not in SENSOR_ENCODER_MODES:
            raise ValueError(
                f"sensor_encoder_mode must be one of {SENSOR_ENCODER_MODES}"
            )
        if self.physical_integration not in PHYSICAL_INTEGRATIONS:
            raise ValueError(
                f"physical_integration must be one of {PHYSICAL_INTEGRATIONS}"
            )
        if self.action_contract not in ACTION_CONTRACT_NAMES:
            raise ValueError(f"action_contract must be one of {ACTION_CONTRACT_NAMES}")
        contract = self.action_contract_spec
        if self.action_target is None:
            self.action_target = contract.action_target
        if self.action_target != contract.action_target:
            raise ValueError(
                f"{self.action_contract} requires action_target="
                f"{contract.action_target}, got {self.action_target}"
            )
        if self.camera_mode not in CAMERA_MODES:
            raise ValueError(f"camera_mode must be one of {CAMERA_MODES}")
        if self.force_input_dim != 1 or self.force_joint_count != 44:
            raise ValueError("HACO torque history is 44 joints x one value")
        if self.tactile_image_size != 240 or self.tactile_finger_count != 10:
            raise ValueError("HACO uses ten native 240x240 tactile maps")
        if self.rtc_training_sampling != "categorical":
            raise ValueError("rtc_training_sampling must be 'categorical'")
        validate_rtc_configuration(
            horizon=self.action_horizon,
            max_prefix_steps=self.rtc_training_max_prefix_steps,
            inference_prefix_steps=self.rtc_inference_prefix_steps,
            prefix_weights=self.rtc_training_prefix_weights,
        )
        for field in (
            "control_flow_weight",
            "delta_q_flow_weight",
            "delta_q_normalized_clip",
            "physical_type_embedding_std",
        ):
            value = float(getattr(self, field))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field} must be finite and nonnegative")
        if not contract.has_delta_q and self.delta_q_flow_weight != 0.5:
            # The field remains serialized for one config schema, but it is
            # not a delta-free-variant loss knob and cannot activate a target.
            self.delta_q_flow_weight = 0.0
        elif not contract.has_delta_q:
            self.delta_q_flow_weight = 0.0

    def to_filtered_dict(self, exclude_augment: bool = True) -> dict[str, Any]:
        config = super().to_filtered_dict(exclude_augment=exclude_augment)
        config.update({field: getattr(self, field) for field in HACO_CONFIG_FIELDS})
        config.update(
            {
                "model_type": self.model_type,
                "architectures": list(self.architectures),
                "action_horizon": HACO_ACTION_HORIZON,
                "max_action_dim": HACO_EXPERT_ACTION_DIM,
                "use_relative_action": False,
            }
        )
        return config


__all__ = [
    "CAMERA_MODES",
    "HACO_CHECKPOINT_SCHEMA",
    "HACO_CONFIG_FIELDS",
    "HACO_MODEL_TYPE",
    "HACO_OFFICIAL_BASE",
    "PHYSICAL_INTEGRATIONS",
    "SENSOR_ENCODER_MODES",
    "HacoConfig",
]
