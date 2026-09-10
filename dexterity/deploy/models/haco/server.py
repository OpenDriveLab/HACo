#!/usr/bin/env python3
"""Serve a HACO checkpoint through the unified SharpA WebSocket API.

The checkpoint owns the perception variant and action vocabulary.  The server
owns RTC overlap state: it retains the complete normalized 132-D carrier while
exposing only the directly executable ``wrist18 + q44`` action.  Delta-q is
never added to q and is available only as a diagnostic for joint-contract
checkpoints.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import dataclasses
import json
import logging
from pathlib import Path
import socket
import sys
import tempfile
import time
from typing import Any, Mapping, cast

import numpy as np
import torch

from dexterity.deploy.models.haco.runtime import (
    LANGUAGE_KEY,
    STATE_KEYS,
    STATE_SLICES,
    PosttrainContract,
    checkpoint_view_with_local_backbone,
    decode_jpeg_rgb,
    trim_or_pad,
    write_json,
)
from dexterity.deploy.models.haco.contract import EXPERIMENT_CONTRACTS
from dexterity.deploy.template.protocol import (
    ACTION_SCHEMA,
    METADATA_FORMAT_SCHEMA,
    BaseSharpAPolicyAdapter,
    SharpAMetadataFormat,
    SharpAObservation,
    SharpAPolicyAction,
)
from dexterity.deploy.template.server import SharpAPolicyServer
from dexterity.models.haco.contract import get_action_contract
from dexterity.runtime.sharpa62 import (
    DEPLOY_TO_MODEL_JOINT,
    direct_eef_interface_metadata,
    model_joints_to_wire,
)


LOGGER = logging.getLogger("haco_sharpa62_server")
CHECKPOINT_SCHEMA = "haco.checkpoint.v1"
MODEL_TYPE = "Haco"
EMBODIMENT = "real_r1_pro_sharpa_absolute_eef"
ACTION_HORIZON = 40
ACTION_DIM = 62
EXPERT_ACTION_DIM = 132
ACTION_HZ = 30.0
RTC_STEPS = 10
VIDEO_KEYS_THREE = ("ego_view", "left_wrist_view", "right_wrist_view")
SENSOR_ENCODER_MODES = frozenset(
    ("none", "torque_only", "tactile_only", "separate", "fused")
)
PHYSICAL_INTEGRATIONS = frozenset(
    ("suffix", "imgmem", "physcross_ungated", "physcross_gated")
)
ACTION_KEYS = {
    "joint_compliance_delta": (
        "left_wrist_eef",
        "right_wrist_eef",
        "left_hand_q_teleop",
        "right_hand_q_teleop",
        "left_hand_delta_q",
        "right_hand_delta_q",
    ),
    "compliance_only": (
        "left_wrist_eef",
        "right_wrist_eef",
        "left_hand_q_teleop",
        "right_hand_q_teleop",
    ),
    "nominal_only": STATE_KEYS,
}


@dataclasses.dataclass(frozen=True)
class HacoCheckpointSpec:
    experiment_id: str
    sensor_encoder_mode: str
    physical_integration: str
    action_contract: str
    action_target: str
    camera_mode: str
    rtc_steps: int
    action_horizon: int = ACTION_HORIZON

    @property
    def video_keys(self) -> tuple[str, ...]:
        return VIDEO_KEYS_THREE if self.camera_mode == "three" else ("ego_view",)

    @property
    def action_keys(self) -> tuple[str, ...]:
        return ACTION_KEYS[self.action_contract]

    @property
    def needs_tau(self) -> bool:
        return self.sensor_encoder_mode in ("torque_only", "separate", "fused")

    @property
    def needs_tactile(self) -> bool:
        return self.sensor_encoder_mode in ("tactile_only", "separate", "fused")

    @classmethod
    def from_checkpoint(cls, checkpoint: Path) -> "HacoCheckpointSpec":
        required = ("config.json", "processor_config.json", "statistics.json")
        missing = [name for name in required if not (checkpoint / name).is_file()]
        if missing:
            raise FileNotFoundError(f"HACO checkpoint is missing {missing}: {checkpoint}")
        config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        if config.get("model_type") != MODEL_TYPE:
            raise ValueError(f"checkpoint model_type must be {MODEL_TYPE!r}")
        if config.get("haco_schema") != CHECKPOINT_SCHEMA:
            raise ValueError(f"checkpoint haco_schema must be {CHECKPOINT_SCHEMA!r}")
        contract = get_action_contract(str(config.get("action_contract")))
        rtc_steps = int(config.get("rtc_inference_prefix_steps", -1))
        if int(config.get("action_horizon", -1)) != ACTION_HORIZON:
            raise ValueError(f"HACO deployment requires action_horizon={ACTION_HORIZON}")
        if int(config.get("max_action_dim", -1)) != EXPERT_ACTION_DIM:
            raise ValueError(
                f"HACO deployment requires max_action_dim={EXPERT_ACTION_DIM}"
            )
        if rtc_steps != RTC_STEPS:
            raise ValueError(
                f"HACO deployment requires rtc_inference_prefix_steps={RTC_STEPS}"
            )
        spec = cls(
            experiment_id=str(config.get("experiment_id", "")),
            sensor_encoder_mode=str(config.get("sensor_encoder_mode", "")),
            physical_integration=str(config.get("physical_integration", "")),
            action_contract=contract.name,
            action_target=str(config.get("action_target", "")),
            camera_mode=str(config.get("camera_mode", "")),
            rtc_steps=rtc_steps,
        )
        if not spec.experiment_id:
            raise ValueError("checkpoint experiment_id must be nonempty")
        if spec.action_target != contract.action_target:
            raise ValueError("checkpoint action target disagrees with action contract")
        if spec.sensor_encoder_mode not in SENSOR_ENCODER_MODES:
            raise ValueError("unsupported checkpoint sensor_encoder_mode")
        if spec.physical_integration not in PHYSICAL_INTEGRATIONS:
            raise ValueError("unsupported checkpoint physical_integration")
        if spec.camera_mode not in {"three", "ego"}:
            raise ValueError("unsupported checkpoint camera_mode")
        try:
            expected_fields = EXPERIMENT_CONTRACTS[spec.experiment_id]
        except KeyError as error:
            raise ValueError(
                f"unknown HACO experiment {spec.experiment_id!r}"
            ) from error
        mismatched = {
            field: (getattr(spec, field), expected)
            for field, expected in expected_fields.items()
            if getattr(spec, field) != expected
        }
        if mismatched:
            raise ValueError(
                f"checkpoint disagrees with frozen {spec.experiment_id} contract: "
                f"{mismatched}"
            )
        processor = json.loads(
            (checkpoint / "processor_config.json").read_text(encoding="utf-8")
        ).get("processor_kwargs", {})
        if int(processor.get("max_action_horizon", -1)) != ACTION_HORIZON:
            raise ValueError("checkpoint processor action horizon is not 40")
        if int(processor.get("max_action_dim", -1)) != EXPERT_ACTION_DIM:
            raise ValueError("checkpoint processor action carrier is not 132-D")
        for field in (
            "experiment_id", "sensor_encoder_mode", "physical_integration",
            "action_contract", "camera_mode",
        ):
            if str(processor.get(field)) != str(getattr(spec, field)):
                raise ValueError(f"checkpoint model/processor {field} mismatch")
        return spec


def metadata_format(spec: HacoCheckpointSpec) -> SharpAMetadataFormat:
    def temporal(history_len: int, current: bool) -> dict[str, Any]:
        return {"history_len": history_len, "current": current}

    return cast(
        SharpAMetadataFormat,
        {
            "schema": METADATA_FORMAT_SCHEMA,
            "format_id": f"haco_{spec.experiment_id}_v1",
            "image": {
                "ego_cam": temporal(0, True),
                "left_wrist_cam": temporal(0, spec.camera_mode == "three"),
                "right_wrist_cam": temporal(0, spec.camera_mode == "three"),
            },
            "state": {
                **temporal(0, True),
                "left_wrist": {"joint": False, "eef": True},
                "right_wrist": {"joint": False, "eef": True},
                "hand_joint": {"left": True, "right": True},
            },
            "sensor": {
                "tau": temporal(8 if spec.needs_tau else 0, spec.needs_tau),
                "wrench": temporal(8 if spec.needs_tactile else 0, spec.needs_tactile),
                "deformation": temporal(0, spec.needs_tactile),
            },
        },
    )


def _state_62(obs: SharpAObservation) -> np.ndarray:
    state = obs["state"]["current"]
    if state is None or not state["valid"]:
        raise ValueError("HACO requires a valid current robot state")
    values = (
        state["left_wrist"]["eef"], state["right_wrist"]["eef"],
        state["hand_joint"]["left"], state["hand_joint"]["right"],
    )
    if any(value is None for value in values):
        raise ValueError("HACO state requires both wrist EEFs and both hands")
    result = np.concatenate(cast(tuple[np.ndarray, ...], values)).astype(np.float32)
    if result.shape != (ACTION_DIM,) or not np.all(np.isfinite(result)):
        raise ValueError("HACO state must be finite float32[62]")
    return result


def _sensor_rows(
    obs: SharpAObservation, name: str, *, right_first: bool
) -> tuple[np.ndarray, np.ndarray]:
    sensor = obs["sensor"][name]
    sides = ("right", "left") if right_first else ("left", "right")
    values: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    history = sensor["history"]
    if history is not None:
        values.append(np.concatenate(tuple(history[side] for side in sides), axis=1))
        valid.append(
            np.concatenate(tuple(history["valid"][side] for side in sides), axis=1)
        )
    current = sensor["current"]
    if current is not None:
        values.append(np.concatenate(tuple(current[side] for side in sides))[None])
        valid.append(
            np.concatenate(tuple(current["valid"][side] for side in sides))[None]
        )
    if not values:
        raise ValueError(f"HACO requires sensor {name!r}")
    return np.concatenate(values), np.concatenate(valid)


class HacoRequestBuilder:
    """Validate execution feedback and build one variant-specific request."""

    def __init__(self, spec: HacoCheckpointSpec) -> None:
        self.spec = spec
        self.reset()

    def reset(self) -> None:
        self._last_action: np.ndarray | None = None
        self._last_action_id: str | None = None
        self._execute_start = 0
        self._execute_stop = 0

    def _rtc_steps(self, obs: SharpAObservation) -> int:
        feedback = obs["execution_feedback"]
        if self._last_action is None:
            if feedback["last_action_id"] is not None or feedback["executed_steps"] != 0:
                raise ValueError("first HACO request after reset requires empty feedback")
            return 0
        if not feedback["success"]:
            raise ValueError("cannot continue RTC after failed action execution")
        if feedback["last_action_id"] != self._last_action_id:
            raise ValueError("execution_feedback.last_action_id does not match cached chunk")
        expected = self._execute_stop - self._execute_start - self.spec.rtc_steps
        if feedback["executed_steps"] != expected:
            raise ValueError(
                f"HACO request expected executed_steps={expected}, "
                f"got {feedback['executed_steps']}"
            )
        return self.spec.rtc_steps

    def build(self, obs: SharpAObservation) -> dict[str, Any]:
        request: dict[str, Any] = {
            "schema": "haco.observation.sharpa62.v1",
            "session_id": obs["session_id"],
            "prompt": obs["prompt"],
            "observation/hand_pose_62d": _state_62(obs),
            "rtc/prefix_steps": np.asarray(self._rtc_steps(obs), dtype=np.int64),
        }
        camera_names = {
            "ego_view": "ego_cam",
            "left_wrist_view": "left_wrist_cam",
            "right_wrist_view": "right_wrist_cam",
        }
        for model_name in self.spec.video_keys:
            camera = obs["image"][camera_names[model_name]]["current"]
            if camera is None or not camera["valid"]:
                raise ValueError(f"HACO requires valid {model_name}")
            request[f"observation/{model_name}_jpeg"] = camera["data"]
        if self.spec.needs_tau:
            tau, valid = _sensor_rows(obs, "tau", right_first=False)
            if tau.shape != (9, 44):
                raise ValueError(f"HACO tau history must be (9,44), got {tau.shape}")
            request["force_history"] = tau[:, DEPLOY_TO_MODEL_JOINT].T.astype(np.float32)
            request["force_history_valid"] = valid[:, DEPLOY_TO_MODEL_JOINT].T.astype(bool)
        if self.spec.needs_tactile:
            wrench, valid = _sensor_rows(obs, "wrench", right_first=True)
            if wrench.shape != (9, 10, 6):
                raise ValueError(
                    f"HACO wrench history must be (9,10,6), got {wrench.shape}"
                )
            deformation = obs["sensor"]["deformation"]["current"]
            if deformation is None:
                raise ValueError("HACO requires current tactile deformation")
            deformation_value = np.concatenate(
                (deformation["right"], deformation["left"]), axis=0
            )
            deformation_valid = np.concatenate(
                (deformation["valid"]["right"], deformation["valid"]["left"])
            ).astype(bool)
            if deformation_value.shape != (10, 240, 240):
                raise ValueError(
                    "HACO tactile deformation must be (10,240,240), "
                    f"got {deformation_value.shape}"
                )
            if deformation_valid.shape != (10,):
                raise ValueError(
                    "HACO tactile deformation validity must be (10,), "
                    f"got {deformation_valid.shape}"
                )
            request.update(
                {
                    "tactile_wrench_history": wrench.transpose(1, 0, 2).astype(np.float32),
                    "tactile_wrench_valid": valid.T.astype(bool),
                    "tactile_deformation": deformation_value,
                    "tactile_deformation_valid": deformation_valid,
                }
            )
        return request

    def record_result(
        self, result: Mapping[str, Any], action: np.ndarray, action_id: str
    ) -> None:
        start = int(np.asarray(result["execute_start"]).item())
        stop = int(np.asarray(result["execute_stop"]).item())
        rtc = int(np.asarray(result["rtc"]).item())
        if not 0 <= start < stop <= len(action) or rtc != self.spec.rtc_steps:
            raise ValueError("HACO returned an invalid execution slice")
        self._last_action = action.copy()
        self._last_action_id = action_id
        self._execute_start = start
        self._execute_stop = stop

    def next_metadata_format(self, current_format_id: str) -> None:
        return None


class HacoInferenceMixin:
    """Attach physical metadata and pass a normalized clean prefix to HACO."""

    def _unbatch_observation(self, value):
        observations = super()._unbatch_observation(value)
        sensor = value.get("haco", {})
        for index, observation in enumerate(observations):
            observation["haco"] = {key: item[index] for key, item in sensor.items()}
        return observations

    def _to_vla_step_data(self, observation):
        from gr00t.data.types import VLAStepData

        return VLAStepData(
            images=observation["video"],
            states=observation["state"],
            actions={},
            text=observation["language"][self.language_key][0],
            embodiment=self.embodiment_tag,
            metadata={"haco": observation["haco"]},
        )

    def _get_action(self, observation, options=None):
        from gr00t.data.types import MessageType
        from gr00t.policy.gr00t_policy import _rec_to_dtype

        processed = []
        states = []
        for item in self._unbatch_observation(observation):
            step = self._to_vla_step_data(item)
            states.append(step.states)
            processed.append(
                self.processor(
                    [{"type": MessageType.EPISODE_STEP.value, "content": step}]
                )
            )
        collated = _rec_to_dtype(self.collate_fn(processed), torch.bfloat16)
        model_options = dict(options or {})
        prefix = model_options.get("rtc_prefix_action")
        if prefix is not None and not torch.is_tensor(prefix):
            model_options["rtc_prefix_action"] = torch.as_tensor(prefix)
        with torch.inference_mode():
            prediction = self.model.get_action(**collated, options=model_options)
        normalized = prediction["action_pred"].float()
        state_batch = {
            key: np.stack([state[key] for state in states], axis=0)
            for key in self.modality_configs["state"].modality_keys
        }
        decoded = self.processor.decode_action(
            normalized.cpu().numpy(), self.embodiment_tag, state_batch
        )
        return (
            {key: value.astype(np.float32) for key, value in decoded.items()},
            {"normalized_action": normalized.detach().cpu()},
        )


def _policy_class():
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    class HacoInferencePolicy(HacoInferenceMixin, Gr00tPolicy):
        pass

    return HacoInferencePolicy


def load_policy(
    checkpoint: Path,
    reference_repo: Path,
    backbone_model: Path,
    embodiment_tag: str,
    device: str,
) -> tuple[Any, HacoCheckpointSpec, tempfile.TemporaryDirectory[str]]:
    if str(reference_repo) not in sys.path:
        sys.path.insert(0, str(reference_repo))
    if embodiment_tag != EMBODIMENT:
        raise ValueError(f"HACO deployment supports only {EMBODIMENT!r}")
    from scripts.train.haco.embodiment import register_sharpa_absolute_eef_embodiment

    register_sharpa_absolute_eef_embodiment()
    import dexterity.models.haco.model  # noqa: F401

    spec = HacoCheckpointSpec.from_checkpoint(checkpoint)
    view, owner = checkpoint_view_with_local_backbone(checkpoint, backbone_model)
    try:
        policy = _policy_class()(
            embodiment_tag=embodiment_tag,
            model_path=str(view),
            device=device,
            strict=True,
        )
    except Exception:
        owner.cleanup()
        raise
    modality = policy.get_modality_config()
    failures = []
    if tuple(modality["video"].modality_keys) != spec.video_keys:
        failures.append(f"video={modality['video'].modality_keys}")
    if tuple(modality["state"].modality_keys) != STATE_KEYS:
        failures.append(f"state={modality['state'].modality_keys}")
    if tuple(modality["action"].modality_keys) != spec.action_keys:
        failures.append(f"action={modality['action'].modality_keys}")
    if len(modality["action"].delta_indices) != ACTION_HORIZON:
        failures.append("action horizon is not 40")
    for field in (
        "experiment_id", "sensor_encoder_mode", "physical_integration",
        "action_contract", "action_target", "camera_mode",
    ):
        if str(getattr(policy.model.config, field)) != str(getattr(spec, field)):
            failures.append(f"runtime {field} mismatch")
    if failures:
        owner.cleanup()
        raise ValueError("HACO runtime contract mismatch: " + "; ".join(failures))
    return policy, spec, owner


def _decoded_group(action: Mapping[str, Any], key: str) -> np.ndarray:
    value = np.asarray(action[key], dtype=np.float32)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2 or value.shape[0] != ACTION_HORIZON:
        raise ValueError(f"decoded action {key!r} has invalid shape {value.shape}")
    return value


class HacoPolicy:
    """Stateful model-private boundary retaining the complete RTC carrier."""

    def __init__(
        self,
        policy: Any,
        spec: HacoCheckpointSpec,
        checkpoint_view_owner: tempfile.TemporaryDirectory[str] | None = None,
    ) -> None:
        self.policy = policy
        self.spec = spec
        self.contract = PosttrainContract()
        self.checkpoint_view_owner = checkpoint_view_owner
        self.reset({"session_id": "default"})

    def reset(self, info: dict[str, Any]) -> str:
        self.session_id = str(info.get("session_id", "default"))
        self.prompt = ""
        self.request_index = 0
        self.last_normalized: torch.Tensor | None = None
        self.last_action: np.ndarray | None = None
        self.policy.reset(info)
        return "reset successful"

    def infer(self, request: Mapping[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        session_id = str(request["session_id"])
        if session_id != self.session_id:
            self.reset({"session_id": session_id})
        if request.get("prompt"):
            self.prompt = str(request["prompt"])
        state_wire = np.asarray(request["observation/hand_pose_62d"], dtype=np.float32)
        state_model = self.contract.state_to_model(state_wire)
        video = {
            key: decode_jpeg_rgb(request[f"observation/{key}_jpeg"], key)[None, None]
            for key in self.spec.video_keys
        }
        observation: dict[str, Any] = {
            "video": video,
            "state": {
                key: state_model[index][None, None].astype(np.float32)
                for key, index in STATE_SLICES.items()
            },
            "language": {LANGUAGE_KEY: [[self.prompt]]},
            "haco": {},
        }
        for key in (
            "force_history", "force_history_valid", "tactile_wrench_history",
            "tactile_wrench_valid", "tactile_deformation",
            "tactile_deformation_valid",
        ):
            if key in request:
                observation["haco"][key] = np.asarray(request[key])[None]
        prefix_steps = int(np.asarray(request["rtc/prefix_steps"]).item())
        options: dict[str, Any] = {"rtc_prefix_steps": prefix_steps}
        if prefix_steps:
            if self.last_normalized is None:
                raise ValueError("HACO RTC prefix requested without cached carrier")
            options["rtc_prefix_action"] = self.last_normalized[
                :, ACTION_HORIZON - prefix_steps : ACTION_HORIZON
            ]
        decoded, info = self.policy.get_action(observation, options=options)
        normalized = info.pop("normalized_action")
        if not torch.is_tensor(normalized) or tuple(normalized.shape) != (
            1, ACTION_HORIZON, EXPERT_ACTION_DIM
        ):
            raise ValueError("HACO normalized prediction must be [1,40,132]")
        q_keys = self.spec.action_keys[2:4]
        action_model = np.concatenate(
            tuple(_decoded_group(decoded, key) for key in (*STATE_KEYS[:2], *q_keys)),
            axis=-1,
        )
        action_wire = self.contract.action_to_deploy(trim_or_pad(action_model, ACTION_HORIZON))
        if prefix_steps and self.last_action is not None:
            expected = self.last_action[-prefix_steps:]
            if not np.allclose(action_wire[:prefix_steps], expected, rtol=1e-4, atol=1e-5):
                raise RuntimeError("decoded RTC prefix changed across HACO chunks")
        diagnostics: dict[str, Any] = {
            "model": "haco",
            "experiment_id": self.spec.experiment_id,
            "action_contract": self.spec.action_contract,
            "action_target": self.spec.action_target,
            "execution_semantics": "direct_main_q",
            "delta_q_execution": "never_added",
            "rtc_prefix_steps": np.asarray(prefix_steps, dtype=np.int64),
            "elapsed_s": np.asarray(time.perf_counter() - started, dtype=np.float32),
        }
        if self.spec.action_contract == "joint_compliance_delta":
            delta_model = np.concatenate(
                tuple(_decoded_group(decoded, key) for key in self.spec.action_keys[4:6]),
                axis=-1,
            )
            diagnostics["delta_q_diagnostic_rad_40x44"] = model_joints_to_wire(delta_model)
        self.last_normalized = normalized.detach().cpu().clone()
        self.last_action = action_wire.copy()
        chunk_id = self.request_index
        self.request_index += 1
        return {
            "action_chunk_62d": action_wire,
            "action_hz": np.asarray(ACTION_HZ, dtype=np.float32),
            "execute_start": np.asarray(prefix_steps, dtype=np.int64),
            "execute_stop": np.asarray(ACTION_HORIZON, dtype=np.int64),
            "rtc": np.asarray(self.spec.rtc_steps, dtype=np.int64),
            "chunk_id": np.asarray(chunk_id, dtype=np.int64),
            "diagnostics": diagnostics,
        }


class HacoAdapter(BaseSharpAPolicyAdapter):
    def __init__(self, policy: HacoPolicy, spec: HacoCheckpointSpec) -> None:
        self.policy = policy
        self.model_kind = "haco"
        self.policy_family = "haco"
        self.spec = spec
        self.request_builder = HacoRequestBuilder(spec)

    def reset(self, session_id: str) -> None:
        self.request_builder.reset()
        self.policy.reset({"session_id": session_id})

    def infer(self, obs: SharpAObservation) -> SharpAPolicyAction:
        result = self.policy.infer(self.request_builder.build(obs))
        action = np.asarray(result["action_chunk_62d"], dtype=np.float32)
        if action.ndim != 2 or action.shape[1] != ACTION_DIM or not len(action):
            raise ValueError(f"HACO action must have shape (T,62), got {action.shape}")
        action = np.ascontiguousarray(action)
        frequency_hz = float(np.asarray(result.get("action_hz", ACTION_HZ)).item())
        execute_start = int(np.asarray(result.get("execute_start", 0)).item())
        execute_stop = int(np.asarray(result.get("execute_stop", len(action))).item())
        rtc = int(np.asarray(result.get("rtc", 0)).item())
        chunk_id = int(np.asarray(result.get("chunk_id", obs["request_id"])).item())
        action_id = f"{obs['session_id']}:chunk:{chunk_id}"
        diagnostics = result.get("diagnostics", {})
        if not isinstance(diagnostics, Mapping):
            diagnostics = {}
        self.request_builder.record_result(result, action, action_id)
        return {
            "schema": ACTION_SCHEMA,
            "session_id": obs["session_id"],
            "request_id": obs["request_id"],
            "action_id": action_id,
            "revision": 0,
            "timestamp_ns": time.time_ns(),
            "execution": {
                "frequency_hz": frequency_hz,
                "action_length": len(action),
                "execute_start": execute_start,
                "execute_length": execute_stop - execute_start,
                "rtc": rtc,
            },
            "action": {
                "left_wrist": {
                    "joint": None,
                    "eef": action[:, :9],
                    "eef_def": "absolute",
                },
                "right_wrist": {
                    "joint": None,
                    "eef": action[:, 9:18],
                    "eef_def": "absolute",
                },
                "hand_joint": {
                    "left": action[:, 18:40],
                    "right": action[:, 40:62],
                },
            },
            "auxiliary": {
                "video": {"ego": None, "left_wrist": None, "right_wrist": None},
                "tactile": {
                    "deformation": None,
                    "wrench": None,
                    "hand_tau": None,
                },
            },
            "diagnostics": {
                **dict(diagnostics),
                "policy_family": "haco",
                "checkpoint_id": "pending-server-injection",
                "checkpoint_path": "pending-server-injection",
                "inference_latency_ms": 0.0,
            },
            "next_metadata_format": None,
        }

    def initial_metadata_format(self) -> SharpAMetadataFormat:
        return deepcopy(metadata_format(self.spec))

    def interface_metadata(self) -> Mapping[str, Any]:
        return {
            "eef_contract": direct_eef_interface_metadata(),
            "experiment_id": self.spec.experiment_id,
            "sensor_encoder_mode": self.spec.sensor_encoder_mode,
            "physical_integration": self.spec.physical_integration,
            "action_contract": self.spec.action_contract,
            "action_target": self.spec.action_target,
            "execution_semantics": "direct_main_q",
            "delta_q_execution": "never_added",
            "execution": {
                "frequency_hz": ACTION_HZ,
                "action_length": ACTION_HORIZON,
                "execute_start": 0,
                "execute_length": ACTION_HORIZON,
                "rtc": self.spec.rtc_steps,
            },
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a trained HACO checkpoint.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5500)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-repo", required=True)
    parser.add_argument("--backbone-model", required=True)
    parser.add_argument("--embodiment-tag", default=EMBODIMENT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="load and validate the complete checkpoint, then exit without serving",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    checkpoint = Path(args.checkpoint).expanduser().absolute()
    reference_repo = Path(args.reference_repo).expanduser().absolute()
    backbone_model = Path(args.backbone_model).expanduser().absolute()
    if not reference_repo.is_dir():
        raise FileNotFoundError(f"Isaac-GR00T repository not found: {reference_repo}")
    if not (backbone_model / "config.json").is_file():
        raise FileNotFoundError(f"HACO backbone not found: {backbone_model}")
    spec = HacoCheckpointSpec.from_checkpoint(checkpoint)
    policy, runtime_spec, owner = load_policy(
        checkpoint, reference_repo, backbone_model, args.embodiment_tag, args.device
    )
    if runtime_spec != spec:
        raise RuntimeError("HACO checkpoint changed while loading")
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint / "deploy_haco"
    output_dir = output_dir.expanduser().absolute()
    write_json(
        output_dir / "server_metadata.json",
        {
            "schema": "haco.server.sharpa62.v1",
            **dataclasses.asdict(spec),
            "checkpoint": str(checkpoint),
            "reference_repo": str(reference_repo),
            "backbone_model": str(backbone_model),
            "device": args.device,
            "host": socket.gethostname(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "execution_semantics": "direct_main_q",
            "delta_q_execution": "never_added",
        },
    )
    adapter: BaseSharpAPolicyAdapter = HacoAdapter(
        HacoPolicy(policy, spec, owner), spec
    )
    if args.validate_only:
        LOGGER.info(
            "validated HACO experiment=%s contract=%s target=%s rtc=%d",
            spec.experiment_id,
            spec.action_contract,
            spec.action_target,
            spec.rtc_steps,
        )
        owner.cleanup()
        return 0
    SharpAPolicyServer(
        adapter,
        policy_family="haco",
        model_name="haco",
        checkpoint_path=checkpoint,
        host=args.host,
        port=args.port,
    ).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
