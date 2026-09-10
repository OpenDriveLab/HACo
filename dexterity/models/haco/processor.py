"""Contract-aware processor for HACO.

The action modality points directly at ``q_cmp`` for the learned compliant
command or ``q_obs`` for the next observed hand state.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import (
    Gr00tN1d7DataCollator,
    Gr00tN1d7Processor,
)
from .local_model import resolve_local_model_path

from .config import CAMERA_MODES, PHYSICAL_INTEGRATIONS, SENSOR_ENCODER_MODES
from .contract import get_action_contract


ACTION_COMPONENT_VALID_KEY = "action_component_valid"
SENSOR_SHAPES = {
    "force_history": (44, 9),
    "force_history_valid": (44, 9),
    "tactile_wrench_history": (10, 9, 6),
    "tactile_wrench_valid": (10, 9),
    "tactile_deformation": (10, 240, 240),
    "tactile_deformation_valid": (10,),
}
TORQUE_KEYS = ("force_history", "force_history_valid")
TACTILE_KEYS = (
    "tactile_wrench_history",
    "tactile_wrench_valid",
    "tactile_deformation",
    "tactile_deformation_valid",
)


def required_sensor_keys(sensor_encoder_mode: str) -> tuple[str, ...]:
    if sensor_encoder_mode == "none":
        return ()
    if sensor_encoder_mode == "torque_only":
        return TORQUE_KEYS
    if sensor_encoder_mode == "tactile_only":
        return TACTILE_KEYS
    if sensor_encoder_mode in ("separate", "fused"):
        return (*TORQUE_KEYS, *TACTILE_KEYS)
    raise ValueError(f"unsupported sensor mode {sensor_encoder_mode!r}")


class HacoDataCollator(Gr00tN1d7DataCollator):
    """Validate the variant's physical input contract before base collation."""

    def __init__(self, *args, **kwargs) -> None:
        self.sensor_encoder_mode = str(
            kwargs.pop("sensor_encoder_mode", "fused")
        )
        for field in (
            "experiment_id",
            "action_contract",
            "camera_mode",
            "physical_integration",
        ):
            kwargs.pop(field, None)
        super().__init__(*args, **kwargs)

    def __call__(self, features):
        required = required_sensor_keys(self.sensor_encoder_mode)
        for index, feature in enumerate(features):
            missing = [key for key in required if key not in feature]
            if missing:
                raise KeyError(
                    f"HACO collator sample {index} is missing {missing}"
                )
        return super().__call__(features)


class HacoProcessor(Gr00tN1d7Processor):
    """GR00T processor boundary with explicit HACO action/sensor semantics."""

    data_collator_class = HacoDataCollator

    def __init__(self, *args, **kwargs) -> None:
        self.experiment_id = str(
            kwargs.pop("experiment_id", os.environ.get("HACO_EXPERIMENT_ID", "haco"))
        )
        self.sensor_encoder_mode = str(
            kwargs.pop(
                "sensor_encoder_mode",
                os.environ.get("HACO_SENSOR_ENCODER_MODE", "fused"),
            )
        )
        action_contract = kwargs.pop(
            "action_contract",
            os.environ.get("HACO_ACTION_CONTRACT", "joint_compliance_delta"),
        )
        self.action_contract = get_action_contract(str(action_contract))
        self.camera_mode = str(
            kwargs.pop("camera_mode", os.environ.get("HACO_CAMERA_MODE", "three"))
        )
        self.physical_integration = str(
            kwargs.pop(
                "physical_integration",
                os.environ.get("HACO_PHYSICAL_INTEGRATION", "physcross_gated"),
            )
        )
        if self.sensor_encoder_mode not in SENSOR_ENCODER_MODES:
            raise ValueError(f"invalid sensor mode {self.sensor_encoder_mode!r}")
        if self.camera_mode not in CAMERA_MODES:
            raise ValueError(f"invalid camera mode {self.camera_mode!r}")
        if self.physical_integration not in PHYSICAL_INTEGRATIONS:
            raise ValueError(
                f"invalid physical integration {self.physical_integration!r}"
            )
        kwargs["max_action_dim"] = self.action_contract.expert_dim
        kwargs["max_action_horizon"] = self.action_contract.horizon
        kwargs["use_relative_action"] = False
        kwargs["clip_outliers"] = False
        kwargs["state_dropout_prob"] = 0.0
        loading_kwargs = kwargs.get("transformers_loading_kwargs", {})
        model_name = kwargs.get("model_name", "nvidia/Cosmos-Reason2-2B")
        kwargs["model_name"] = resolve_local_model_path(model_name, loading_kwargs)
        super().__init__(*args, **kwargs)
        # The upstream processor constructs the collator without forwarding
        # subclass-specific kwargs.  Keep its required sensor set identical to
        # this processor's HACO experiment contract.
        self._collator.sensor_encoder_mode = self.sensor_encoder_mode

    def save_pretrained(self, save_directory: str | Path) -> list[Path]:
        """Persist the HACO contract alongside the upstream processor config."""

        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if distributed and torch.distributed.get_rank() != 0:
            torch.distributed.barrier()
            return []
        files = super().save_pretrained(save_directory)
        config_path = Path(save_directory) / "processor_config.json"
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        processor_kwargs = payload.setdefault("processor_kwargs", {})
        processor_kwargs.update(
            {
                "experiment_id": self.experiment_id,
                "sensor_encoder_mode": self.sensor_encoder_mode,
                "action_contract": self.action_contract.name,
                "camera_mode": self.camera_mode,
                "physical_integration": self.physical_integration,
            }
        )
        temporary = config_path.with_name(f".{config_path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, config_path)
        if distributed:
            torch.distributed.barrier()
        return files

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load both upstream fields and the serialized/explicit HACO contract.

        GR00T N1.7's loader only forwards a fixed list of override keys, so a
        plain inherited call silently discards ``action_contract`` and sensor
        mode overrides.  HACO uses local checkpoints only and reconstructs
        the same payload explicitly to make contract selection deterministic.
        """

        # ``AutoProcessor.from_pretrained`` adds this private dispatch marker.
        # It is not a processor constructor option and must not weaken the
        # explicit public override allowlist below.
        kwargs.pop("_from_auto", None)
        transformers_loading_kwargs = kwargs.pop(
            "transformers_loading_kwargs", {"trust_remote_code": True}
        )
        # These are Hugging Face resolution options duplicated by some call
        # sites next to ``transformers_loading_kwargs``.  All HACO sources
        # are local; retain them for the nested VLM processor and do not treat
        # them as processor-constructor fields.
        for loading_key in (
            "trust_remote_code",
            "local_files_only",
            "cache_dir",
            "token",
            "revision",
        ):
            if loading_key in kwargs:
                transformers_loading_kwargs = dict(transformers_loading_kwargs)
                transformers_loading_kwargs[loading_key] = kwargs.pop(loading_key)
        source = Path(pretrained_model_name_or_path).expanduser()
        if not source.is_dir():
            raise FileNotFoundError(
                "HACO processors must be loaded from a local checkpoint: "
                f"{source}"
            )
        payload = json.loads(
            (source / "processor_config.json").read_text(encoding="utf-8")
        )
        processor_kwargs = dict(payload["processor_kwargs"])
        processor_kwargs["statistics"] = json.loads(
            (source / "statistics.json").read_text(encoding="utf-8")
        )
        embodiment_path = source / "embodiment_id.json"
        processor_kwargs["embodiment_id_mapping"] = (
            json.loads(embodiment_path.read_text(encoding="utf-8"))
            if embodiment_path.is_file()
            else None
        )
        processor_kwargs.setdefault("model_name", "nvidia/Cosmos-Reason2-2B")
        processor_kwargs.setdefault("model_type", "qwen")
        processor_kwargs.setdefault("clip_outliers", True)

        modality_overrides = kwargs.pop("modality_configs", {})
        for embodiment_tag, modality_config in modality_overrides.items():
            processor_kwargs["modality_configs"][embodiment_tag] = modality_config
        allowed_overrides = {
            "random_rotation_angle",
            "color_jitter_params",
            "use_relative_action",
            "exclude_state",
            "state_dropout_prob",
            "use_mean_std",
            "model_name",
            "model_type",
            "image_crop_size",
            "image_target_size",
            "shortest_image_edge",
            "crop_fraction",
            "formalize_language",
            "apply_sincos_state_encoding",
            "use_albumentations",
            "extra_augmentation_config",
            "max_state_dim",
            "max_action_dim",
            "max_action_horizon",
            "experiment_id",
            "sensor_encoder_mode",
            "action_contract",
            "camera_mode",
            "physical_integration",
        }
        unknown = sorted(set(kwargs) - allowed_overrides)
        if unknown:
            raise TypeError(f"unsupported HACO processor overrides: {unknown}")
        for key, value in kwargs.items():
            if value is not None:
                processor_kwargs[key] = value
        return cls(
            **processor_kwargs,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

    @property
    def action_target(self) -> str:
        return self.action_contract.action_target

    def _validate_modality_action_groups(self, embodiment_tag) -> None:
        tag = getattr(embodiment_tag, "value", embodiment_tag)
        action_config = self.modality_configs[tag]["action"]
        keys = tuple(str(key) for key in action_config.modality_keys)
        has_delta = any("delta_q" in key for key in keys)
        if has_delta != self.action_contract.has_delta_q:
            requirement = "requires" if self.action_contract.has_delta_q else "forbids"
            raise ValueError(
                f"{self.action_contract.name} {requirement} delta-q modality groups; "
                f"got {keys}"
            )
        if self.action_target == "q_cmp":
            q_keys = [
                key for key in keys if "q_cmp" in key or "q_teleop" in key
            ]
            if len(q_keys) != 2:
                raise ValueError(
                    "q_cmp must be read directly from left/right q_cmp"
                )
        elif any("q_cmp" in key or "q_teleop" in key for key in keys):
            raise ValueError("q_obs must not use q_cmp action groups")

    @staticmethod
    def _as_tensor(value: Any) -> torch.Tensor:
        return value if torch.is_tensor(value) else torch.from_numpy(np.asarray(value))

    @classmethod
    def _prepare_sensor_inputs(
        cls,
        sensor: dict[str, Any],
        mode: str,
        *,
        batch_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        required = required_sensor_keys(mode)
        prepared = {key: cls._as_tensor(sensor[key]) for key in required}
        prefix = () if batch_size is None else (batch_size,)
        for key in required:
            expected = (*prefix, *SENSOR_SHAPES[key])
            if tuple(prepared[key].shape) != expected:
                raise ValueError(
                    f"{key} must have shape {expected}, got {tuple(prepared[key].shape)}"
                )
        if "force_history" in prepared:
            force = prepared["force_history"].float()
            valid = prepared["force_history_valid"].bool() & torch.isfinite(force)
            prepared["force_history"] = torch.where(
                valid, force, torch.zeros_like(force)
            )
            prepared["force_history_valid"] = valid
        if "tactile_wrench_history" in prepared:
            wrench = prepared["tactile_wrench_history"].float()
            valid = prepared["tactile_wrench_valid"].bool()
            valid &= torch.isfinite(wrench).all(dim=-1)
            prepared["tactile_wrench_history"] = torch.where(
                valid[..., None], wrench, torch.zeros_like(wrench)
            )
            prepared["tactile_wrench_valid"] = valid
            deformation = prepared["tactile_deformation"]
            deformation_valid = prepared["tactile_deformation_valid"].bool()
            if torch.is_floating_point(deformation):
                deformation_valid &= torch.isfinite(deformation).flatten(-2).all(-1)
                deformation = torch.nan_to_num(deformation)
            prepared["tactile_deformation"] = deformation
            prepared["tactile_deformation_valid"] = deformation_valid
        return prepared

    def _clip_action(self, action: torch.Tensor) -> torch.Tensor:
        self.action_contract.validate_carrier(action)
        output = action.clone()
        output[..., self.action_contract.wrist_slice] = output[
            ..., self.action_contract.wrist_slice
        ].clamp(-1.0, 1.0)
        output[..., self.action_contract.q_slice] = output[
            ..., self.action_contract.q_slice
        ].clamp(-1.0, 1.0)
        if self.action_contract.has_delta_q:
            output[..., self.action_contract.delta_q_slice] = output[
                ..., self.action_contract.delta_q_slice
            ].clamp(-5.0, 5.0)
        output[..., self.action_contract.padding_slice] = 0
        return output

    def _apply_component_valid(self, mask, component_valid):
        mask = self._as_tensor(mask).bool()
        valid = self._as_tensor(component_valid).bool().to(mask.device)
        semantic_shape = (*mask.shape[:-1], self.action_contract.semantic_dim)
        if tuple(valid.shape) == semantic_shape:
            carrier_valid = torch.zeros_like(mask)
            carrier_valid[..., : self.action_contract.semantic_dim] = valid
            valid = carrier_valid
        elif tuple(valid.shape) != tuple(mask.shape):
            raise ValueError(
                f"{ACTION_COMPONENT_VALID_KEY} must have shape {semantic_shape} "
                f"or {tuple(mask.shape)}"
            )
        return mask & valid

    def _validate_carrier(self, action, mask) -> None:
        if action.shape != mask.shape:
            raise ValueError("HACO action and mask shapes differ")
        if torch.count_nonzero(action[..., self.action_contract.padding_slice]):
            raise ValueError("HACO carrier padding must be zero")
        if torch.count_nonzero(mask[..., self.action_contract.padding_slice]):
            raise ValueError("HACO carrier padding mask must be false")

    def __call__(self, messages: list[dict[str, Any]]):
        if len(messages) != 1:
            raise ValueError("HACO processor expects one VLA step")
        content = messages[0]["content"]
        self._validate_modality_action_groups(content.embodiment)
        metadata = getattr(content, "metadata", None) or {}
        sensor = metadata.get("haco", metadata)
        required = required_sensor_keys(self.sensor_encoder_mode)
        missing = [key for key in required if key not in sensor]
        if missing:
            raise KeyError(f"HACO sample is missing sensor keys: {missing}")
        # Call the official processor directly. No alternate direct-target
        # mutation (q + delta-q) is reachable from this code path.
        transformed = super().__call__(messages)
        transformed["state"] = transformed["state"].clamp(-1.0, 1.0)
        if "action" in transformed:
            transformed["action"] = self._clip_action(transformed["action"])
            mask = transformed["action_mask"].bool()
            if ACTION_COMPONENT_VALID_KEY in sensor:
                mask = self._apply_component_valid(
                    mask, sensor[ACTION_COMPONENT_VALID_KEY]
                )
            mask &= self.action_contract.active_mask(device=mask.device)
            transformed["action_mask"] = mask
            self._validate_carrier(transformed["action"], mask)
        transformed.update(
            self._prepare_sensor_inputs(sensor, self.sensor_encoder_mode)
        )
        return transformed

    def process_observation(self, observation, embodiment_tag):
        self._validate_modality_action_groups(embodiment_tag)
        transformed = super().process_observation(observation, embodiment_tag)
        transformed["state"] = transformed["state"].clamp(-1.0, 1.0)
        sensor = observation.get("haco", observation)
        batch_size = int(transformed["state"].shape[0])
        missing = [
            key
            for key in required_sensor_keys(self.sensor_encoder_mode)
            if key not in sensor
        ]
        if missing:
            raise KeyError(f"HACO observation is missing sensor keys: {missing}")
        transformed.update(
            self._prepare_sensor_inputs(
                sensor, self.sensor_encoder_mode, batch_size=batch_size
            )
        )
        return BatchFeature(data=dict(transformed))

    def decode_action(self, action, embodiment_tag, state=None):
        """Decode the directly predicted target; never add delta-q."""

        return super().decode_action(action, embodiment_tag, state=state)

    def unapply(self, action, embodiment_tag, state=None, prev_action=None):
        return super().unapply(
            action,
            embodiment_tag,
            state=state,
            prev_action=prev_action,
        )


__all__ = [
    "HacoDataCollator",
    "HacoProcessor",
    "required_sensor_keys",
]
