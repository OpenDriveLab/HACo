"""Shared input validation and canonical SharpA62 baseline response helpers."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

REQUEST_SCHEMA = "sharpa62_model_observation.v1"
ACTION_SCHEMA = "sharpa62_policy_action.v1"
ACTION_HORIZON = 40
ACTION_DIM = 62
ACTION_HZ = 30.0
NATIVE_ACTION_HZ = 30.0
ACTION_SPACE = "sharpa_dexretarget_position_62d"
ACTION_LAYOUT = "left_wrist9,right_wrist9,sharpa_q44"
WRIST_FRAME = "absolute_current_hip"
TACTILE_FINGERS = 10
TACTILE_CHANNELS = 6
DEFAULT_DEFORMATION_SIZE = 64


@dataclass(frozen=True)
class PreparedBaselineRequest:
    session_id: str
    prompt: str
    image_rgb: np.ndarray
    image_transport: str
    state_history_deploy: np.ndarray
    tactile_wrench_history: np.ndarray
    tactile_wrench_valid: np.ndarray
    tactile_deformation: np.ndarray | None
    tactile_deformation_valid: np.ndarray | None
    used_sensor_fallback: bool

    @property
    def state_deploy(self) -> np.ndarray:
        return self.state_history_deploy[-1]


class StartupProgress:
    """Small stage progress bar for long, monolithic checkpoint loads."""

    def __init__(self, name: str, total: int) -> None:
        self.name = str(name)
        self.bar = tqdm(
            total=int(total),
            unit="stage",
            dynamic_ncols=True,
            leave=True,
            desc=f"{self.name} | starting",
        )

    def __enter__(self) -> "StartupProgress":
        return self

    def begin(self, label: str) -> None:
        self.bar.set_description_str(f"{self.name} | {label}", refresh=True)

    def complete(self) -> None:
        self.bar.update(1)

    def __exit__(self, exc_type, exc, traceback) -> None:
        status = "ready" if exc_type is None else "failed"
        self.bar.set_description_str(f"{self.name} | {status}", refresh=True)
        self.bar.close()


def _finite_numeric(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{label} must be numeric, got {array.dtype}")
    array = array.astype(np.float32, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains NaN or Inf")
    return array


def _lookup(
    payload: Mapping[str, Any], keys: tuple[str, ...]
) -> tuple[Any, str] | tuple[None, None]:
    for key in keys:
        if key in payload:
            return payload[key], key
    return None, None


def ensure_uint8_rgb(value: Any) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim == 4:
        if image.shape[0] < 1:
            raise ValueError("observation/ego_view has an empty frame history")
        image = image[-1]
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(
            "observation/ego_view must have shape (H,W,3) or (T,H,W,3), "
            f"got {image.shape}"
        )
    if image.dtype != np.uint8:
        if not np.issubdtype(image.dtype, np.number):
            raise TypeError(f"observation/ego_view must be numeric, got {image.dtype}")
        image = image.astype(np.float32, copy=False)
        if (
            image.size
            and float(np.nanmin(image)) >= 0.0
            and float(np.nanmax(image)) <= 1.0
        ):
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(image)


def decode_jpeg_rgb(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        encoded = value.astype(np.uint8, copy=False).tobytes()
    elif isinstance(value, (bytes, bytearray, memoryview)):
        encoded = bytes(value)
    else:
        raise TypeError("observation/ego_view_jpeg must be bytes or uint8 ndarray")
    with Image.open(io.BytesIO(encoded)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _left_pad(array: np.ndarray, length: int) -> np.ndarray:
    if len(array) >= length:
        return array[-length:].copy()
    return np.concatenate(
        (np.repeat(array[:1], length - len(array), axis=0), array), axis=0
    )


def _state_history(
    payload: Mapping[str, Any], required_length: int, allow_sensor_fallback: bool
) -> tuple[np.ndarray, bool]:
    value, key = _lookup(
        payload,
        (
            "observation/hand_pose_history_16x62",
            "observation/hand_pose_history",
            "observation/hand_pose_62d",
        ),
    )
    if value is None:
        raise KeyError("request is missing observation/hand_pose_62d")
    state = _finite_numeric(value, str(key))
    if state.ndim == 1:
        state = state[None]
    if state.ndim != 2 or state.shape[1] != ACTION_DIM or len(state) < 1:
        raise ValueError(f"{key} must have shape (62,) or (T,62), got {state.shape}")
    used_fallback = False
    if len(state) < required_length:
        if not allow_sensor_fallback:
            raise ValueError(
                f"{key} supplies {len(state)} state frames; this model requires "
                f"{required_length} chronological frames"
            )
        state = _left_pad(state, required_length)
        used_fallback = True
    else:
        state = state[-required_length:].copy()
    return state.astype(np.float32), used_fallback


def _wrench_history(
    payload: Mapping[str, Any], required_length: int, allow_sensor_fallback: bool
) -> tuple[np.ndarray, np.ndarray, bool]:
    value, key = _lookup(
        payload,
        (
            f"observation/tactile_wrench_history_{required_length}x10x6",
            "observation/tactile_wrench_history_18x10x6",
            "observation/tactile_wrench_history_16x10x6",
            "observation/tactile_wrench_history_9x10x6",
            "observation/tactile_wrench_history",
        ),
    )
    if value is None:
        if not allow_sensor_fallback:
            raise KeyError(
                "request is missing tactile wrench history; expected "
                f"observation/tactile_wrench_history_{required_length}x10x6"
            )
        return (
            np.zeros(
                (required_length, TACTILE_FINGERS, TACTILE_CHANNELS), dtype=np.float32
            ),
            np.zeros((required_length, TACTILE_FINGERS), dtype=bool),
            True,
        )
    wrench = _finite_numeric(value, str(key))
    if wrench.ndim != 3 or wrench.shape[1:] != (TACTILE_FINGERS, TACTILE_CHANNELS):
        raise ValueError(f"{key} must have shape (T,10,6), got {wrench.shape}")
    valid_value, valid_key = _lookup(
        payload,
        (
            f"observation/tactile_wrench_valid_history_{len(wrench)}x10",
            "observation/tactile_wrench_valid_history_18x10",
            "observation/tactile_wrench_valid_history_16x10",
            "observation/tactile_wrench_valid_history_9x10",
            "observation/tactile_wrench_valid_history",
        ),
    )
    if valid_value is None:
        valid = np.ones((len(wrench), TACTILE_FINGERS), dtype=bool)
    else:
        valid = np.asarray(valid_value)
        if valid.dtype != np.bool_:
            raise TypeError(f"{valid_key} must have dtype bool, got {valid.dtype}")
        if valid.shape != (len(wrench), TACTILE_FINGERS):
            raise ValueError(
                f"{valid_key} must have shape ({len(wrench)},10), got {valid.shape}"
            )
        valid = valid.copy()
    used_fallback = False
    if len(wrench) < required_length:
        if not allow_sensor_fallback:
            raise ValueError(
                f"{key} supplies {len(wrench)} wrench frames; this model requires "
                f"{required_length} chronological frames"
            )
        wrench = _left_pad(wrench, required_length)
        valid = _left_pad(valid, required_length)
        used_fallback = True
    else:
        wrench = wrench[-required_length:].copy()
        valid = valid[-required_length:].copy()
    wrench[~valid] = 0.0
    return wrench.astype(np.float32), valid, used_fallback


def _deformation(
    payload: Mapping[str, Any], required: bool, allow_sensor_fallback: bool
) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
    if not required:
        return None, None, False
    value, key = _lookup(
        payload,
        (
            "observation/tactile_deformation_10x64x64",
            "observation/tactile_deformation_10x240x240",
            "observation/tactile_deformation",
        ),
    )
    if value is None:
        if not allow_sensor_fallback:
            raise KeyError(
                "request is missing observation/tactile_deformation_10x64x64"
            )
        return (
            np.zeros(
                (TACTILE_FINGERS, DEFAULT_DEFORMATION_SIZE, DEFAULT_DEFORMATION_SIZE),
                dtype=np.uint8,
            ),
            np.zeros(TACTILE_FINGERS, dtype=bool),
            True,
        )
    deformation = np.asarray(value)
    if deformation.ndim != 3 or deformation.shape[0] != TACTILE_FINGERS:
        raise ValueError(f"{key} must have shape (10,H,W), got {deformation.shape}")
    if deformation.shape[1] < 2 or deformation.shape[2] < 2:
        raise ValueError(f"{key} spatial dimensions are too small: {deformation.shape}")
    if deformation.dtype != np.uint8:
        if not np.issubdtype(deformation.dtype, np.number):
            raise TypeError(f"{key} must be numeric, got {deformation.dtype}")
        deformation = deformation.astype(np.float32, copy=False)
        if (
            deformation.size
            and float(np.nanmin(deformation)) >= 0.0
            and float(np.nanmax(deformation)) <= 1.0
        ):
            deformation = deformation * 255.0
        deformation = np.clip(deformation, 0.0, 255.0).astype(np.uint8)
    valid_value, valid_key = _lookup(
        payload,
        (
            "observation/tactile_deformation_valid_10",
            "observation/tactile_deformation_valid",
        ),
    )
    if valid_value is None:
        valid = np.ones(TACTILE_FINGERS, dtype=bool)
    else:
        valid = np.asarray(valid_value)
        if valid.dtype != np.bool_ or valid.shape != (TACTILE_FINGERS,):
            raise ValueError(
                f"{valid_key} must be bool[10], got {valid.dtype}{valid.shape}"
            )
        valid = valid.copy()
    deformation = deformation.copy()
    deformation[~valid] = 0
    return deformation, valid, False


def prepare_request(
    payload: Mapping[str, Any],
    *,
    state_history_length: int,
    wrench_history_length: int,
    require_deformation: bool,
    allow_sensor_fallback: bool,
    default_prompt: str,
    expected_schema: str = REQUEST_SCHEMA,
) -> PreparedBaselineRequest:
    schema = payload.get("schema")
    if schema != expected_schema:
        raise ValueError(f"schema must be {expected_schema!r}, got {schema!r}")
    jpeg = payload.get("observation/ego_view_jpeg")
    if jpeg is not None:
        image = decode_jpeg_rgb(jpeg)
        image_transport = "jpeg"
    elif "observation/ego_view" in payload:
        image = ensure_uint8_rgb(payload["observation/ego_view"])
        image_transport = "raw_numpy"
    else:
        raise KeyError(
            "request is missing observation/ego_view or observation/ego_view_jpeg"
        )
    states, state_fallback = _state_history(
        payload, state_history_length, allow_sensor_fallback
    )
    wrench, wrench_valid, wrench_fallback = _wrench_history(
        payload, wrench_history_length, allow_sensor_fallback
    )
    deformation, deformation_valid, deformation_fallback = _deformation(
        payload, require_deformation, allow_sensor_fallback
    )
    return PreparedBaselineRequest(
        session_id=str(payload.get("session_id", "default")),
        prompt=str(payload.get("prompt") or default_prompt),
        image_rgb=image,
        image_transport=image_transport,
        state_history_deploy=states,
        tactile_wrench_history=wrench,
        tactile_wrench_valid=wrench_valid,
        tactile_deformation=deformation,
        tactile_deformation_valid=deformation_valid,
        used_sensor_fallback=bool(
            state_fallback or wrench_fallback or deformation_fallback
        ),
    )


def fit_action_horizon(
    action: Any, horizon: int = ACTION_HORIZON
) -> tuple[np.ndarray, str]:
    """Fit a native 30 Hz chunk to the canonical 40-step 30 Hz horizon."""

    value = _finite_numeric(action, "model action")
    if value.ndim != 2 or value.shape[1] != ACTION_DIM or len(value) < 1:
        raise ValueError(f"model action must have shape (T,62), got {value.shape}")
    stride = int(round(NATIVE_ACTION_HZ / ACTION_HZ))
    if not np.isclose(NATIVE_ACTION_HZ / ACTION_HZ, stride) or stride < 1:
        raise ValueError(
            f"native/output action rates require an integer stride, got "
            f"{NATIVE_ACTION_HZ}/{ACTION_HZ}"
        )
    value = value[::stride]
    if len(value) < horizon:
        value = np.concatenate(
            (value, np.repeat(value[-1:], horizon - len(value), axis=0)), axis=0
        )
        operation = f"stride_{stride}_then_repeat_last"
    elif len(value) > horizon:
        value = value[:horizon]
        operation = f"stride_{stride}_then_truncate"
    else:
        value = value.copy()
        operation = f"stride_{stride}"
    return value.astype(np.float32), operation


def build_action_response(
    action: Any,
    *,
    session_id: str,
    request_index: int,
    policy_family: str,
    execute_joint_source: str,
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = np.asarray(action)
    if value.shape != (ACTION_HORIZON, ACTION_DIM) or value.dtype != np.float32:
        raise ValueError(
            f"action_hand_pose_62d must be float32[{ACTION_HORIZON},{ACTION_DIM}], "
            f"got {value.dtype}{value.shape}"
        )
    if not np.all(np.isfinite(value)):
        raise ValueError("action_hand_pose_62d contains NaN or Inf")
    return {
        "schema": ACTION_SCHEMA,
        "session_id": str(session_id),
        "request_index": np.asarray(request_index, dtype=np.int64),
        "is_real_policy": np.asarray(True, dtype=bool),
        "action_hand_pose_62d": value,
        "action_horizon": np.asarray(ACTION_HORIZON, dtype=np.int64),
        "action_hz": np.asarray(ACTION_HZ, dtype=np.float32),
        "action_space": ACTION_SPACE,
        "wrist_frame": WRIST_FRAME,
        "layout": ACTION_LAYOUT,
        "metadata": {
            "policy_family": str(policy_family),
            "execute_joint_source": str(execute_joint_source),
        },
        "diagnostics": dict(diagnostics or {}),
    }


def common_metadata(
    *, policy_family: str, native_action_horizon: int, execute_joint_source: str
) -> dict[str, Any]:
    return {
        "schema": "sharpa62_policy_server.v1",
        "model_name": policy_family,
        "policy_family": policy_family,
        "robot": "sharpa62",
        "request_schema": REQUEST_SCHEMA,
        "response_schema": ACTION_SCHEMA,
        "transport": "websocket+msgpack_numpy",
        "image_key": "observation/ego_view",
        "state_key": "observation/hand_pose_62d",
        "action_dim": ACTION_DIM,
        "action_horizon": ACTION_HORIZON,
        "native_action_horizon": int(native_action_horizon),
        "native_action_hz": NATIVE_ACTION_HZ,
        "action_hz": ACTION_HZ,
        "action_space": ACTION_SPACE,
        "output_wrist_frame": WRIST_FRAME,
        "layout": ACTION_LAYOUT,
        "execute_joint_source": execute_joint_source,
    }
