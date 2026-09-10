"""Adapt the unified SharpA observation to model-private request formats."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
import io
import time
from typing import Any, Mapping, cast

import numpy as np
from PIL import Image

from dexterity.deploy.template.protocol import (
    ACTION_SCHEMA,
    METADATA_FORMAT_SCHEMA,
    BaseSharpAPolicyAdapter,
    SharpAMetadataFormat,
    SharpAObservation,
    SharpAPolicyAction,
)
from dexterity.runtime.sharpa62 import (
    DEPLOY_JOINT_LAYOUT,
    DEPLOY_JOINT_ORDER,
    DEPLOY_TACTILE_LAYOUT,
    DEPLOY_TACTILE_ORDER,
    direct_eef_interface_metadata,
)

MODEL_KINDS = (
    "cgp",
    "deco",
    "dreamzero",
    "ftp1",
    "gcc",
    "groot",
    "haco",
    "pace",
    "pi05",
    "trex",
    "vitacformer",
)

MULTIVIEW_MODEL_KINDS = frozenset(
    ("deco", "ftp1", "groot", "haco", "pace", "pi05", "trex", "vitacformer")
)


def _temporal(history_len: int, current: bool) -> dict[str, Any]:
    return {"history_len": int(history_len), "current": bool(current)}


def _metadata_format(
    format_id: str,
    *,
    ego_current: bool,
    left_wrist_camera_current: bool = False,
    right_wrist_camera_current: bool = False,
    state_current: bool,
    state_history_len: int = 0,
    tau_history_len: int = 0,
    tau_current: bool = False,
    wrench_history_len: int = 0,
    wrench_current: bool = False,
    deformation_history_len: int = 0,
    deformation_current: bool = False,
) -> SharpAMetadataFormat:
    return cast(
        SharpAMetadataFormat,
        {
            "schema": METADATA_FORMAT_SCHEMA,
            "format_id": format_id,
            "image": {
                "ego_cam": _temporal(0, ego_current),
                "left_wrist_cam": _temporal(0, left_wrist_camera_current),
                "right_wrist_cam": _temporal(0, right_wrist_camera_current),
            },
            "state": {
                **_temporal(state_history_len, state_current),
                "left_wrist": {"joint": False, "eef": True},
                "right_wrist": {"joint": False, "eef": True},
                "hand_joint": {"left": True, "right": True},
            },
            "sensor": {
                "tau": _temporal(tau_history_len, tau_current),
                "wrench": _temporal(wrench_history_len, wrench_current),
                "deformation": _temporal(deformation_history_len, deformation_current),
            },
        },
    )


MODEL_INITIAL_METADATA_FORMATS: dict[str, SharpAMetadataFormat] = {
    "cgp": _metadata_format(
        "cgp_default_v1",
        ego_current=True,
        state_current=True,
        deformation_history_len=1,
        deformation_current=True,
    ),
    "dreamzero": _metadata_format(
        "dreamzero_default_v1",
        ego_current=True,
        state_current=True,
    ),
    "ftp1": _metadata_format(
        "ftp1_multiview_state_history_v2",
        ego_current=True,
        left_wrist_camera_current=True,
        right_wrist_camera_current=True,
        state_current=True,
        state_history_len=15,
        deformation_current=True,
    ),
    "deco": _metadata_format(
        "deco_multiview_v1",
        ego_current=True,
        left_wrist_camera_current=True,
        right_wrist_camera_current=True,
        state_current=True,
        wrench_history_len=8,
        wrench_current=True,
        deformation_current=True,
    ),
    "gcc": _metadata_format(
        "gcc_default_v1",
        ego_current=True,
        state_current=True,
        tau_history_len=8,
        tau_current=True,
        wrench_history_len=8,
        wrench_current=True,
        deformation_current=True,
    ),
    "groot": _metadata_format(
        "groot_multiview_v1",
        ego_current=True,
        left_wrist_camera_current=True,
        right_wrist_camera_current=True,
        state_current=True,
    ),
    "haco": _metadata_format(
        "haco_multiview_v1",
        ego_current=True,
        left_wrist_camera_current=True,
        right_wrist_camera_current=True,
        state_current=True,
        tau_history_len=8,
        tau_current=True,
        wrench_history_len=8,
        wrench_current=True,
        deformation_current=True,
    ),
    "pace": _metadata_format(
        "pace_multiview_v1",
        ego_current=True,
        left_wrist_camera_current=True,
        right_wrist_camera_current=True,
        state_current=True,
        tau_history_len=8,
        tau_current=True,
        wrench_history_len=8,
        wrench_current=True,
        deformation_current=True,
    ),
    "pi05": _metadata_format(
        "pi05_multiview_v1",
        ego_current=True,
        left_wrist_camera_current=True,
        right_wrist_camera_current=True,
        state_current=True,
    ),
    "trex": _metadata_format(
        "trex_slow_v1",
        ego_current=True,
        left_wrist_camera_current=True,
        right_wrist_camera_current=True,
        state_current=True,
        wrench_history_len=15,
        wrench_current=True,
        deformation_current=True,
    ),
    "vitacformer": _metadata_format(
        "vitacformer_multiview_v1",
        ego_current=True,
        left_wrist_camera_current=True,
        right_wrist_camera_current=True,
        state_current=True,
        state_history_len=15,
        wrench_history_len=17,
        wrench_current=True,
    ),
}

TREX_FAST_METADATA_FORMAT = _metadata_format(
    "trex_fast_v1",
    ego_current=False,
    state_current=False,
    wrench_history_len=15,
    wrench_current=True,
    deformation_current=True,
)


def model_initial_metadata_format(model_kind: str) -> SharpAMetadataFormat:
    try:
        return deepcopy(MODEL_INITIAL_METADATA_FORMATS[model_kind])
    except KeyError as error:
        raise ValueError(f"unsupported model adapter kind: {model_kind}") from error


def _current_state(obs: SharpAObservation) -> Mapping[str, Any]:
    state = obs["state"]["current"]
    if state is None:
        raise ValueError("current robot state is required for this model request")
    if not state["valid"]:
        raise ValueError("current robot state is marked invalid")
    return state


def _state_62(obs: SharpAObservation) -> np.ndarray:
    state = _current_state(obs)
    left_eef = state["left_wrist"]["eef"]
    right_eef = state["right_wrist"]["eef"]
    left_hand = state["hand_joint"]["left"]
    right_hand = state["hand_joint"]["right"]
    if any(value is None for value in (left_eef, right_eef, left_hand, right_hand)):
        raise ValueError("current state must contain both wrist EEFs and both hands")
    return np.concatenate(
        (
            cast(np.ndarray, left_eef),
            cast(np.ndarray, right_eef),
            cast(np.ndarray, left_hand),
            cast(np.ndarray, right_hand),
        )
    ).astype(np.float32, copy=False)


def _state_history_62(obs: SharpAObservation, history_len: int) -> np.ndarray:
    """Return exact chronological state history followed by current state."""

    history = obs["state"]["history"]
    if history is None:
        raise ValueError(f"model requires {history_len} historical state rows")
    if len(history["timestamp_ns"]) != history_len:
        raise ValueError(
            f"state history must contain {history_len} rows, "
            f"got {len(history['timestamp_ns'])}"
        )
    if not np.asarray(history["valid"], dtype=bool).all():
        raise ValueError("model requires every historical state row to be valid")
    components = (
        history["left_wrist"]["eef"],
        history["right_wrist"]["eef"],
        history["hand_joint"]["left"],
        history["hand_joint"]["right"],
    )
    if any(value is None for value in components):
        raise ValueError("state history must contain both wrists and both hands")
    rows = np.concatenate(
        tuple(np.asarray(value, dtype=np.float32) for value in components),
        axis=1,
    )
    return np.concatenate((rows, _state_62(obs)[None]), axis=0)


def _sensor_window(
    obs: SharpAObservation,
    name: str,
    *,
    side_order: tuple[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sensor = obs["sensor"][name]
    first_side, second_side = side_order
    values: list[np.ndarray] = []
    validity: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    history = sensor["history"]
    if history is not None:
        values.append(
            np.concatenate((history[first_side], history[second_side]), axis=1)
        )
        validity.append(
            np.concatenate(
                (
                    history["valid"][first_side],
                    history["valid"][second_side],
                ),
                axis=1,
            )
        )
        timestamps.append(history["timestamp_ns"])
    current = sensor["current"]
    if current is not None:
        values.append(
            np.concatenate((current[first_side], current[second_side]), axis=0)[None]
        )
        validity.append(
            np.concatenate(
                (
                    current["valid"][first_side],
                    current["valid"][second_side],
                ),
                axis=0,
            )[None]
        )
        timestamps.append(np.asarray([current["timestamp_ns"]], dtype=np.int64))
    if not values:
        raise ValueError(f"sensor {name!r} has no requested history or current value")
    return (
        np.concatenate(values),
        np.concatenate(validity),
        np.concatenate(timestamps),
    )


def _deformation_current(obs: SharpAObservation) -> tuple[np.ndarray, np.ndarray]:
    sensor = obs["sensor"]["deformation"]["current"]
    if sensor is None:
        raise ValueError("current deformation is required for this model request")
    value = np.concatenate((sensor["right"], sensor["left"]), axis=0).copy()
    valid = np.concatenate((sensor["valid"]["right"], sensor["valid"]["left"]))
    value[~valid] = 0
    return value, valid


def _current_camera(
    obs: SharpAObservation,
    camera_name: str,
) -> Mapping[str, Any]:
    camera = obs["image"][camera_name]["current"]
    if camera is None or not camera["valid"]:
        raise ValueError(f"current {camera_name} must be valid for policy inference")
    return camera


def _ego_camera(obs: SharpAObservation) -> Mapping[str, Any]:
    return _current_camera(obs, "ego_cam")


def _decode_ego(obs: SharpAObservation) -> np.ndarray:
    camera = _ego_camera(obs)
    with Image.open(io.BytesIO(camera["data"])) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _pad_window(
    values: list[np.ndarray], valid: list[np.ndarray], length: int
) -> tuple[np.ndarray, np.ndarray]:
    if not values:
        raise ValueError("cannot build an empty model history")
    values = values[-length:]
    valid = valid[-length:]
    missing = length - len(values)
    if missing:
        values = [values[0].copy() for _ in range(missing)] + values
        valid = [np.zeros_like(valid[0], dtype=bool) for _ in range(missing)] + valid
    return np.stack(values), np.stack(valid)


class ModelRequestBuilder:
    """Stateful conversion from robot facts to one model's private input."""

    def __init__(self, model_kind: str) -> None:
        if model_kind not in MODEL_KINDS:
            raise ValueError(f"unsupported model adapter kind: {model_kind}")
        self.model_kind = model_kind
        self.reset()

    def reset(self) -> None:
        self._states: deque[np.ndarray] = deque(maxlen=18)
        self._q_commands: deque[np.ndarray] = deque(maxlen=9)
        self._state_stamps: deque[int] = deque(maxlen=18)
        self._state_sequences: deque[int] = deque(maxlen=18)
        self._sequence = 0
        self._last_action: np.ndarray | None = None
        self._last_action_id: str | None = None
        self._trex_next_offset = 0

    def _append_state(self, obs: SharpAObservation) -> None:
        state = _state_62(obs)
        state_current = _current_state(obs)
        stamp = int(state_current["timestamp_ns"])
        if self._state_stamps and stamp <= self._state_stamps[-1]:
            stamp = self._state_stamps[-1] + 1
        self._sequence += 1
        self._states.append(state)
        self._state_stamps.append(stamp)
        self._state_sequences.append(self._sequence)

        current_q = state[18:].copy()
        feedback = obs["execution_feedback"]
        if (
            self._last_action is not None
            and feedback["last_action_id"] == self._last_action_id
            and feedback["executed_steps"] > 0
        ):
            index = min(feedback["executed_steps"] - 1, len(self._last_action) - 1)
            current_q_command = self._last_action[index, 18:].copy()
        else:
            current_q_command = current_q
        self._q_commands.append(current_q_command.astype(np.float32))

    def _state_history(self, length: int) -> tuple[np.ndarray, np.ndarray]:
        values = list(self._states)
        valid = [np.asarray(True, dtype=bool) for _ in values]
        return _pad_window(values, valid, length)

    def _joint_history(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        states = [value[18:] for value in self._states][-9:]
        commands = list(self._q_commands)[-9:]
        real_count = len(states)
        missing = 9 - real_count
        states = [states[0].copy() for _ in range(missing)] + states
        commands = [commands[0].copy() for _ in range(missing)] + commands
        valid = np.concatenate(
            (np.zeros((missing, 44), dtype=bool), np.ones((real_count, 44), dtype=bool))
        )
        history_real = np.concatenate(
            (np.zeros(missing, dtype=bool), np.ones(real_count, dtype=bool))
        )
        return np.stack(states), np.stack(commands), valid, history_real

    def _history_identifiers(self) -> tuple[np.ndarray, np.ndarray]:
        count = len(self._state_sequences)
        missing = 9 - min(count, 9)
        sequences = list(self._state_sequences)[-9:]
        stamps = list(self._state_stamps)[-9:]
        return (
            np.asarray(([0] * missing) + sequences, dtype=np.int64),
            np.asarray(([0] * missing) + stamps, dtype=np.int64),
        )

    def build(self, obs: SharpAObservation) -> dict[str, Any]:
        if self.model_kind == "trex":
            if self._trex_next_offset == 0:
                self._append_state(obs)
            return self._trex(obs)

        self._append_state(obs)
        if self.model_kind in ("gcc", "haco", "pace"):
            return self._gcc_pace(obs)
        if self.model_kind == "vitacformer":
            return self._vitacformer(obs)
        if self.model_kind == "ftp1":
            return self._ftp1(obs)
        if self.model_kind == "deco":
            return self._deco(obs)
        if self.model_kind == "dreamzero":
            return self._dreamzero(obs)
        if self.model_kind == "cgp":
            return self._cgp(obs)
        return self._groot(obs)

    def _common(self, obs: SharpAObservation) -> dict[str, Any]:
        request = {
            "schema": "sharpa62_model_observation.v1",
            "session_id": obs["session_id"],
            "prompt": obs["prompt"],
            "observation/hand_pose_62d": _state_62(obs),
            "observation/timestamp_unix_s": obs["timestamp_ns"] / 1e9,
        }
        request.update(self._camera_jpegs(obs))
        return request

    def _camera_jpegs(self, obs: SharpAObservation) -> dict[str, bytes]:
        cameras = {
            "observation/ego_view_jpeg": _current_camera(obs, "ego_cam")["data"],
        }
        if self.model_kind in MULTIVIEW_MODEL_KINDS:
            cameras.update(
                {
                    "observation/left_wrist_view_jpeg": _current_camera(
                        obs, "left_wrist_cam"
                    )["data"],
                    "observation/right_wrist_view_jpeg": _current_camera(
                        obs, "right_wrist_cam"
                    )["data"],
                }
            )
        return cameras

    def _groot(self, obs: SharpAObservation) -> dict[str, Any]:
        return self._common(obs)

    def _dreamzero(self, obs: SharpAObservation) -> dict[str, Any]:
        request = self._common(obs)
        request["observation/ego_view"] = _decode_ego(obs)
        return request

    def _cgp(self, obs: SharpAObservation) -> dict[str, Any]:
        deformation = obs["sensor"]["deformation"]
        history = deformation["history"]
        current = deformation["current"]
        values: list[tuple[np.ndarray, np.ndarray, int]] = []
        if history is not None:
            values.extend(
                (
                    np.stack((left[::-1], right[::-1]), axis=0),
                    np.stack((left_valid[::-1], right_valid[::-1]), axis=0),
                    int(timestamp),
                )
                for left, right, left_valid, right_valid, timestamp in zip(
                    history["left"],
                    history["right"],
                    history["valid"]["left"],
                    history["valid"]["right"],
                    history["timestamp_ns"],
                )
            )
        if current is not None:
            values.append(
                (
                    np.stack(
                        (current["left"][::-1], current["right"][::-1]),
                        axis=0,
                    ),
                    np.stack(
                        (
                            current["valid"]["left"][::-1],
                            current["valid"]["right"][::-1],
                        ),
                        axis=0,
                    ),
                    int(current["timestamp_ns"]),
                )
            )
        if len(values) < 2:
            raise ValueError(
                "CGP requires one previous and one current deformation frame"
            )
        values = values[-2:]
        tactile_history = np.stack([item[0] for item in values], axis=0)
        tactile_history_valid = np.stack([item[1] for item in values], axis=0)
        tactile_history = tactile_history[..., None, :, :].astype(np.uint8, copy=False)
        tactile_history = np.where(
            tactile_history_valid[..., None, None, None],
            tactile_history,
            0,
        ).astype(np.uint8, copy=False)
        return {
            "schema": "cgp_n17_sharpa62_observation.v1",
            "session_id": obs["session_id"],
            "prompt": obs["prompt"],
            "observation/ego_view_jpeg": _ego_camera(obs)["data"],
            "observation/hand_pose_62d": _state_62(obs),
            "observation/tactile_history_2x2x5x1x240x240": tactile_history,
            "observation/tactile_history_valid_2x2x5": tactile_history_valid,
            "history_stamp_ns_2": np.asarray(
                [item[2] for item in values], dtype=np.int64
            ),
        }

    def _gcc_pace(self, obs: SharpAObservation) -> dict[str, Any]:
        q_exe, q_cmd, q_valid, history_real = self._joint_history()
        tau, tau_valid, _ = _sensor_window(obs, "tau", side_order=("left", "right"))
        wrench, wrench_valid, _ = _sensor_window(
            obs, "wrench", side_order=("right", "left")
        )
        deformation, deformation_valid = _deformation_current(obs)
        sequences, stamps = self._history_identifiers()
        if tau.shape != (9, 44):
            raise ValueError(f"PACE/GCC tau window must be (9,44), got {tau.shape}")
        if wrench.shape != (9, 10, 6):
            raise ValueError(
                f"PACE/GCC wrench window must be (9,10,6), got {wrench.shape}"
            )
        if self.model_kind == "gcc":
            indices = np.rint(np.linspace(0, deformation.shape[-1] - 1, 64)).astype(int)
            deformation = deformation[:, indices][:, :, indices]
            schema = "gcc_n17_sharpa62_observation.v1"
            deformation_key = "observation/tactile_deformation_10x64x64"
        else:
            schema = "pace_n17_sharpa62_observation.v2"
            deformation_key = "observation/tactile_deformation_10x240x240"
        request = {
            "schema": schema,
            "session_id": obs["session_id"],
            "prompt": obs["prompt"],
            "joint_layout": DEPLOY_JOINT_LAYOUT,
            "tactile_layout": DEPLOY_TACTILE_LAYOUT,
            "joint_order": list(DEPLOY_JOINT_ORDER),
            "tactile_order": list(DEPLOY_TACTILE_ORDER),
            "observation/hand_pose_62d": _state_62(obs),
            "observation/q_exe_history_9x44": q_exe,
            "observation/q_cmd_history_9x44": q_cmd,
            "observation/tau_history_9x44": tau,
            "observation/q_exe_valid_history_9x44": q_valid,
            "observation/q_cmd_valid_history_9x44": q_valid.copy(),
            "observation/tau_valid_history_9x44": tau_valid,
            "observation/tactile_wrench_history_9x10x6": wrench,
            "observation/tactile_wrench_valid_history_9x10": wrench_valid,
            deformation_key: deformation,
            "observation/tactile_deformation_valid_10": deformation_valid,
            "history_is_real_9": history_real,
            "history_real_count": int(history_real.sum()),
            "history_obs_seq_9": sequences,
            "history_stamp_ns_9": stamps,
        }
        request.update(self._camera_jpegs(obs))
        return request

    def _trex(self, obs: SharpAObservation) -> dict[str, Any]:
        slow_tick = self._trex_next_offset == 0
        request = (
            self._common(obs)
            if slow_tick
            else {
                "schema": "sharpa_trex_observation.v1",
                "session_id": obs["session_id"],
                "prompt": obs["prompt"],
            }
        )
        wrench, wrench_valid, _ = _sensor_window(
            obs, "wrench", side_order=("right", "left")
        )
        deformation, deformation_valid = _deformation_current(obs)
        if wrench.shape != (16, 10, 6):
            raise ValueError(
                f"T-Rex wrench window must be (16,10,6), got {wrench.shape}"
            )
        request.update(
            {
                "schema": "sharpa_trex_observation.v1",
                "mode": "slow_and_fast" if slow_tick else "fast",
                "refine_offset": self._trex_next_offset,
                "observation/tactile_wrench_history_16x10x6": wrench,
                "observation/tactile_wrench_valid_history_16x10": wrench_valid,
                "observation/tactile_deformation_10x240x240": deformation,
                "observation/tactile_deformation_valid_10": deformation_valid,
            }
        )
        return request

    def _vitacformer(self, obs: SharpAObservation) -> dict[str, Any]:
        request = self._common(obs)
        state = _state_history_62(obs, 15)
        wrench, wrench_valid, _ = _sensor_window(
            obs, "wrench", side_order=("right", "left")
        )
        if wrench.shape != (18, 10, 6):
            raise ValueError(
                f"ViTacFormer wrench window must be (18,10,6), got {wrench.shape}"
            )
        request.update(
            {
                "observation/hand_pose_history_16x62": state,
                "observation/tactile_wrench_history_18x10x6": wrench,
                "observation/tactile_wrench_valid_history_18x10": wrench_valid,
            }
        )
        return request

    def _ftp1(self, obs: SharpAObservation) -> dict[str, Any]:
        request = self._common(obs)
        state = _state_history_62(obs, 15)
        deformation, deformation_valid = _deformation_current(obs)
        if deformation.shape != (10, 240, 240):
            raise ValueError(
                f"FTP-1 deformation must be (10,240,240), got {deformation.shape}"
            )
        request.update(
            {
                "observation/hand_pose_history_16x62": state,
                "observation/tactile_deformation_10x240x240": deformation,
                "observation/tactile_deformation_valid_10": deformation_valid,
            }
        )
        return request

    def _deco(self, obs: SharpAObservation) -> dict[str, Any]:
        request = self._common(obs)
        wrench, wrench_valid, _ = _sensor_window(
            obs, "wrench", side_order=("right", "left")
        )
        deformation, deformation_valid = _deformation_current(obs)
        if wrench.shape != (9, 10, 6):
            raise ValueError(f"DECO wrench window must be (9,10,6), got {wrench.shape}")
        if deformation.shape != (10, 240, 240):
            raise ValueError(
                f"DECO deformation must be (10,240,240), got {deformation.shape}"
            )
        request.update(
            {
                "observation/tactile_wrench_history_9x10x6": wrench,
                "observation/tactile_wrench_valid_history_9x10": wrench_valid,
                "observation/tactile_deformation_10x240x240": deformation,
                "observation/tactile_deformation_valid_10": deformation_valid,
            }
        )
        return request

    def record_result(
        self, result: Mapping[str, Any], action: np.ndarray, action_id: str
    ) -> None:
        self._last_action = action.copy()
        self._last_action_id = action_id
        if self.model_kind == "trex":
            stop = int(result.get("execute_stop", len(action)))
            self._trex_next_offset = 0 if stop >= len(action) else stop

    def next_metadata_format(
        self, current_format_id: str
    ) -> SharpAMetadataFormat | None:
        if self.model_kind != "trex":
            return None
        desired = (
            MODEL_INITIAL_METADATA_FORMATS["trex"]
            if self._trex_next_offset == 0
            else TREX_FAST_METADATA_FORMAT
        )
        if desired["format_id"] == current_format_id:
            return None
        return deepcopy(desired)


class SharpAModelAdapter(BaseSharpAPolicyAdapter):
    """Common adapter inherited by every model deployment entry point."""

    def __init__(self, policy: Any, *, model_kind: str, policy_family: str) -> None:
        self.policy = policy
        self.model_kind = model_kind
        self.policy_family = policy_family
        self.request_builder = ModelRequestBuilder(model_kind)

    def initial_metadata_format(self) -> SharpAMetadataFormat:
        return model_initial_metadata_format(self.model_kind)

    def interface_metadata(self) -> Mapping[str, Any]:
        if self.model_kind not in MULTIVIEW_MODEL_KINDS:
            return {}
        return {"eef_contract": direct_eef_interface_metadata()}

    def reset(self, session_id: str) -> None:
        self.request_builder.reset()
        self.policy.reset({"session_id": session_id})

    def infer(self, obs: SharpAObservation) -> SharpAPolicyAction:
        result = self.policy.infer(self.request_builder.build(obs))
        if not isinstance(result, Mapping):
            raise TypeError("model policy must return a mapping")
        action_key = (
            "action_chunk_62d"
            if "action_chunk_62d" in result
            else "action_hand_pose_62d"
        )
        action = np.asarray(result[action_key], dtype=np.float32)
        if action.ndim != 2 or action.shape[1] != 62 or not len(action):
            raise ValueError(f"model action must have shape (T,62), got {action.shape}")
        action = np.ascontiguousarray(action)
        frequency_hz = float(np.asarray(result.get("action_hz", 30.0)).item())
        execute_start = int(np.asarray(result.get("execute_start", 0)).item())
        execute_stop = int(np.asarray(result.get("execute_stop", len(action))).item())
        rtc = int(np.asarray(result.get("rtc", 0)).item())
        chunk_id = int(np.asarray(result.get("chunk_id", obs["request_id"])).item())
        revision = int(np.asarray(result.get("refine_offset", 0)).item())
        action_id = (
            f"{obs['session_id']}:chunk:{chunk_id}"
            if action_key == "action_chunk_62d"
            else f"{obs['session_id']}:request:{obs['request_id']}"
        )
        diagnostics = result.get("diagnostics", result.get("debug", {}))
        if not isinstance(diagnostics, Mapping):
            diagnostics = {}
        self.request_builder.record_result(result, action, action_id)
        response: SharpAPolicyAction = {
            "schema": ACTION_SCHEMA,
            "session_id": obs["session_id"],
            "request_id": obs["request_id"],
            "action_id": action_id,
            "revision": revision,
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
                "video": {
                    "ego": None,
                    "left_wrist": None,
                    "right_wrist": None,
                },
                "tactile": {
                    "deformation": None,
                    "wrench": None,
                    "hand_tau": None,
                },
            },
            "diagnostics": {
                **dict(diagnostics),
                "policy_family": self.policy_family,
                "checkpoint_id": "pending-server-injection",
                "checkpoint_path": "pending-server-injection",
                "inference_latency_ms": 0.0,
            },
            "next_metadata_format": self.request_builder.next_metadata_format(
                obs["metadata_format_id"]
            ),
        }
        return response
