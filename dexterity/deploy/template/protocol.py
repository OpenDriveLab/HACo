"""Typed contracts for the unified SharpA deployment server."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, Mapping, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from dexterity.runtime.sharpa_kinematics import relative_eef_to_absolute


OBSERVATION_SCHEMA = "sharpa_policy_observation.v3"
ACTION_SCHEMA = "sharpa_policy_action.v5"
METADATA_FORMAT_SCHEMA = "sharpa_policy_metadata_format.v1"
ERROR_SCHEMA = "sharpa_policy_error.v1"

Float32Array = NDArray[np.float32]
Int64Array = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
UInt8Array = NDArray[np.uint8]
WristActionType = Literal["joint", "relative_eef", "eef"]


# Hand joint order for state, tau, and action hand_joint values.
# Action columns [0:22] are left and [22:44] are right.
#
# Left and right use the same 22-DoF order:
# thumb_CMC_FE, thumb_CMC_AA, thumb_MCP_FE, thumb_MCP_AA, thumb_IP,
# index_MCP_FE, index_MCP_AA, index_PIP, index_DIP,
# middle_MCP_FE, middle_MCP_AA, middle_PIP, middle_DIP,
# ring_MCP_FE, ring_MCP_AA, ring_PIP, ring_DIP,
# pinky_CMC, pinky_MCP_FE, pinky_MCP_AA, pinky_PIP, pinky_DIP


class TemporalRequirement(TypedDict):
    history_len: int
    current: bool


class WristStateRequirement(TypedDict):
    joint: bool
    eef: bool


class HandStateRequirement(TypedDict):
    left: bool
    right: bool


class StateRequirement(TemporalRequirement):
    left_wrist: WristStateRequirement
    right_wrist: WristStateRequirement
    hand_joint: HandStateRequirement


class ImageRequirement(TypedDict):
    ego_cam: TemporalRequirement
    left_wrist_cam: TemporalRequirement
    right_wrist_cam: TemporalRequirement


class SensorRequirement(TypedDict):
    tau: TemporalRequirement
    wrench: TemporalRequirement
    deformation: TemporalRequirement


class SharpAMetadataFormat(TypedDict):
    schema: Literal["sharpa_policy_metadata_format.v1"]
    format_id: str
    image: ImageRequirement
    state: StateRequirement
    sensor: SensorRequirement


class CameraFrame(TypedDict):
    encoding: Literal["jpeg"]
    data: bytes
    timestamp_ns: int
    valid: bool


class CameraObservation(TypedDict):
    history: list[CameraFrame]
    current: CameraFrame | None


class ImageObservation(TypedDict):
    ego_cam: CameraObservation
    left_wrist_cam: CameraObservation
    right_wrist_cam: CameraObservation


class WristState(TypedDict):
    joint: Float32Array | None
    eef: Float32Array | None
    eef_def: Literal["absolute", "relative"] | None


class HandPair(TypedDict):
    left: Float32Array | None
    right: Float32Array | None


class StateHistory(TypedDict):
    timestamp_ns: Int64Array
    left_wrist: WristState
    right_wrist: WristState
    hand_joint: HandPair
    valid: BoolArray


class StateCurrent(TypedDict):
    timestamp_ns: int
    left_wrist: WristState
    right_wrist: WristState
    hand_joint: HandPair
    valid: bool


class RobotState(TypedDict):
    history: StateHistory | None
    current: StateCurrent | None


class SideValidity(TypedDict):
    left: BoolArray
    right: BoolArray


class SensorHistory(TypedDict):
    left: NDArray[Any]
    right: NDArray[Any]
    timestamp_ns: Int64Array
    valid: SideValidity


class SensorCurrent(TypedDict):
    left: NDArray[Any]
    right: NDArray[Any]
    timestamp_ns: int
    valid: SideValidity


class HistoricalSensor(TypedDict):
    history: SensorHistory | None
    current: SensorCurrent | None


class SensorObservation(TypedDict):
    tau: HistoricalSensor
    wrench: HistoricalSensor
    deformation: HistoricalSensor


class ExecutionFeedback(TypedDict):
    last_action_id: str | None
    executed_steps: int
    success: bool


class SharpAObservation(TypedDict):
    schema: Literal["sharpa_policy_observation.v3"]
    metadata_format_id: str
    session_id: str
    request_id: int
    timestamp_ns: int
    prompt: str
    image: ImageObservation
    state: RobotState
    sensor: SensorObservation
    execution_feedback: ExecutionFeedback


class ActionExecution(TypedDict):
    frequency_hz: float
    action_length: int
    execute_start: int
    execute_length: int
    rtc: int


class WristActionTypes(TypedDict):
    left: WristActionType
    right: WristActionType


class WristAction(TypedDict):
    joint: Float32Array | None
    eef: Float32Array | None
    eef_def: Literal["absolute", "relative"] | None


# Kept as a type alias for existing template imports. Public actions now use
# WristAction with eef_def="absolute".
AbsoluteWristAction = WristAction


class PolicyActionData(TypedDict):
    left_wrist: WristAction
    right_wrist: WristAction
    hand_joint: HandPair


class AuxiliaryVideo(TypedDict):
    ego: Any | None
    left_wrist: Any | None
    right_wrist: Any | None


class AuxiliaryTactile(TypedDict):
    deformation: Any | None
    wrench: Any | None
    hand_tau: Any | None


class PolicyAuxiliary(TypedDict):
    video: AuxiliaryVideo
    tactile: AuxiliaryTactile


class ActionDiagnostics(TypedDict):
    policy_family: str
    checkpoint_id: str
    checkpoint_path: str
    inference_latency_ms: float


class SharpAPolicyAction(TypedDict):
    schema: Literal["sharpa_policy_action.v5"]
    session_id: str
    request_id: int
    action_id: str
    revision: int
    timestamp_ns: int
    execution: ActionExecution
    action: PolicyActionData
    auxiliary: PolicyAuxiliary
    diagnostics: ActionDiagnostics
    next_metadata_format: SharpAMetadataFormat | None


class PolicyProtocolError(ValueError):
    """Raised when a deployment message violates the common contract."""


def default_metadata_format() -> SharpAMetadataFormat:
    """Return a safe generic format for simple adapters and protocol tests."""

    return {
        "schema": METADATA_FORMAT_SCHEMA,
        "format_id": "sharpa_default_current_v1",
        "image": {
            "ego_cam": {"history_len": 0, "current": True},
            "left_wrist_cam": {"history_len": 0, "current": True},
            "right_wrist_cam": {"history_len": 0, "current": True},
        },
        "state": {
            "history_len": 0,
            "current": True,
            "left_wrist": {"joint": False, "eef": True},
            "right_wrist": {"joint": False, "eef": True},
            "hand_joint": {"left": True, "right": True},
        },
        "sensor": {
            "tau": {"history_len": 0, "current": True},
            "wrench": {"history_len": 0, "current": True},
            "deformation": {"history_len": 0, "current": True},
        },
    }


class BaseSharpAPolicyAdapter(ABC):
    """Model adapter consumed by the unified SharpA policy server."""

    def initial_metadata_format(self) -> SharpAMetadataFormat:
        """Describe the first observation required for every new session."""

        return default_metadata_format()

    def interface_metadata(self) -> Mapping[str, Any]:
        """Advertise model-specific public boundary semantics."""

        return {}

    @staticmethod
    def relative_eef_to_eef(
        obs: SharpAObservation,
        *,
        left_wrist: Float32Array,
        right_wrist: Float32Array,
    ) -> dict[str, WristAction]:
        """Convert model-relative wrist actions to public absolute EEF."""

        state = obs["state"]["current"]
        if state is None:
            raise PolicyProtocolError(
                "relative EEF conversion requires obs.state.current"
            )
        left_eef = state["left_wrist"]["eef"]
        right_eef = state["right_wrist"]["eef"]
        if left_eef is None or right_eef is None:
            raise PolicyProtocolError(
                "relative EEF conversion requires both current wrist EEF states"
            )
        return {
            "left_wrist": {
                "joint": None,
                "eef": _relative_wrist_action_to_eef(
                    left_wrist, left_eef, "left_wrist"
                ),
                "eef_def": "absolute",
            },
            "right_wrist": {
                "joint": None,
                "eef": _relative_wrist_action_to_eef(
                    right_wrist, right_eef, "right_wrist"
                ),
                "eef_def": "absolute",
            },
        }

    @abstractmethod
    def reset(self, session_id: str) -> None:
        """Reset all model state associated with a robot session."""

    @abstractmethod
    def infer(self, obs: SharpAObservation) -> SharpAPolicyAction:
        """Convert an observation into an executable SharpA action."""


def _relative_wrist_action_to_eef(
    relative_wrist: Any,
    current_eef: Any,
    label: str,
) -> Float32Array:
    if not isinstance(relative_wrist, np.ndarray) or relative_wrist.dtype != np.float32:
        raise PolicyProtocolError(f"{label} must be a float32 numpy.ndarray")
    if relative_wrist.ndim != 2 or relative_wrist.shape[1] != 9:
        raise PolicyProtocolError(
            f"{label} must have shape (T, 9), got {relative_wrist.shape}"
        )
    if not len(relative_wrist):
        raise PolicyProtocolError(f"{label} must contain at least one action")
    if not isinstance(current_eef, np.ndarray) or current_eef.dtype != np.float32:
        raise PolicyProtocolError(f"current {label} EEF must be float32")
    try:
        return cast(
            Float32Array,
            relative_eef_to_absolute(relative_wrist, current_eef),
        )
    except ValueError as error:
        raise PolicyProtocolError(f"cannot convert {label}: {error}") from error


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyProtocolError(f"{label} must be a mapping")
    return value


def _require_keys(
    value: Mapping[str, Any],
    label: str,
    keys: tuple[str, ...],
) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise PolicyProtocolError(f"{label} missing fields: {', '.join(missing)}")


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PolicyProtocolError(f"{label} must be a nonempty string")
    return value


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise PolicyProtocolError(f"{label} must be an integer")
    integer = int(value)
    if minimum is not None and integer < minimum:
        raise PolicyProtocolError(f"{label} must be >= {minimum}")
    return integer


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise PolicyProtocolError(f"{label} must be a boolean")
    return bool(value)


def _array(
    value: Any,
    label: str,
    *,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
    finite: bool = False,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise PolicyProtocolError(f"{label} must be a numpy.ndarray")
    if value.dtype != dtype:
        raise PolicyProtocolError(f"{label} must have dtype {dtype}, got {value.dtype}")
    if value.shape != shape:
        raise PolicyProtocolError(f"{label} must have shape {shape}, got {value.shape}")
    if finite and not np.all(np.isfinite(value)):
        raise PolicyProtocolError(f"{label} contains NaN or Inf")
    return value


def _temporal_requirement(value: Any, label: str) -> None:
    requirement = _mapping(value, label)
    _integer(requirement.get("history_len"), f"{label}.history_len", minimum=0)
    _boolean(requirement.get("current"), f"{label}.current")


def validate_metadata_format(value: Any) -> SharpAMetadataFormat:
    """Validate a complete observation requirement format."""

    metadata_format = _mapping(value, "metadata_format")
    if metadata_format.get("schema") != METADATA_FORMAT_SCHEMA:
        raise PolicyProtocolError(
            f"metadata_format.schema must be {METADATA_FORMAT_SCHEMA!r}"
        )
    _nonempty_string(metadata_format.get("format_id"), "metadata_format.format_id")

    image = _mapping(metadata_format.get("image"), "metadata_format.image")
    for camera_name in ("ego_cam", "left_wrist_cam", "right_wrist_cam"):
        _temporal_requirement(
            image.get(camera_name), f"metadata_format.image.{camera_name}"
        )

    state = _mapping(metadata_format.get("state"), "metadata_format.state")
    _temporal_requirement(state, "metadata_format.state")
    for side in ("left_wrist", "right_wrist"):
        wrist = _mapping(state.get(side), f"metadata_format.state.{side}")
        _boolean(wrist.get("joint"), f"metadata_format.state.{side}.joint")
        _boolean(wrist.get("eef"), f"metadata_format.state.{side}.eef")
    hands = _mapping(
        state.get("hand_joint"), "metadata_format.state.hand_joint"
    )
    for side in ("left", "right"):
        _boolean(hands.get(side), f"metadata_format.state.hand_joint.{side}")

    sensor = _mapping(metadata_format.get("sensor"), "metadata_format.sensor")
    for sensor_name in ("tau", "wrench", "deformation"):
        _temporal_requirement(
            sensor.get(sensor_name), f"metadata_format.sensor.{sensor_name}"
        )
    return cast(SharpAMetadataFormat, value)


def _validate_camera_frame(value: Any, label: str) -> int:
    camera = _mapping(value, label)
    if camera.get("encoding") != "jpeg":
        raise PolicyProtocolError(f"{label}.encoding must be 'jpeg'")
    data = camera.get("data")
    if not isinstance(data, bytes):
        raise PolicyProtocolError(f"{label}.data must be JPEG bytes")
    valid = _boolean(camera.get("valid"), f"{label}.valid")
    if valid and not data:
        raise PolicyProtocolError(f"{label}.data must be nonempty when valid is true")
    return _integer(camera.get("timestamp_ns"), f"{label}.timestamp_ns", minimum=0)


def _validate_camera_stream(
    value: Any,
    requirement: Mapping[str, Any],
    label: str,
) -> None:
    stream = _mapping(value, label)
    history = stream.get("history")
    if not isinstance(history, list):
        raise PolicyProtocolError(f"{label}.history must be a list")
    history_len = int(requirement["history_len"])
    if len(history) != history_len:
        raise PolicyProtocolError(
            f"{label}.history must contain {history_len} frames, got {len(history)}"
        )
    timestamps = [
        _validate_camera_frame(frame, f"{label}.history[{index}]")
        for index, frame in enumerate(history)
    ]
    if any(current < previous for previous, current in zip(timestamps, timestamps[1:])):
        raise PolicyProtocolError(f"{label}.history must be chronological")
    current = stream.get("current")
    if requirement["current"]:
        current_timestamp = _validate_camera_frame(current, f"{label}.current")
        if timestamps and current_timestamp < timestamps[-1]:
            raise PolicyProtocolError(f"{label}.current must not precede history")
    elif current is not None:
        raise PolicyProtocolError(f"{label}.current must be None when not requested")


def _validate_optional_state_array(
    value: Any,
    label: str,
    *,
    required: bool,
    leading_shape: tuple[int, ...],
    width: int | None,
) -> None:
    if not required:
        if value is not None:
            raise PolicyProtocolError(f"{label} must be None when not requested")
        return
    if not isinstance(value, np.ndarray) or value.dtype != np.float32:
        raise PolicyProtocolError(f"{label} must be a float32 numpy.ndarray")
    expected_ndim = len(leading_shape) + 1
    if value.ndim != expected_ndim or value.shape[: len(leading_shape)] != leading_shape:
        suffix = "D" if width is None else str(width)
        raise PolicyProtocolError(
            f"{label} must have shape {leading_shape + (suffix,)}, got {value.shape}"
        )
    if width is None:
        if value.shape[-1] < 1:
            raise PolicyProtocolError(f"{label} must contain at least one joint")
    elif value.shape[-1] != width:
        raise PolicyProtocolError(
            f"{label} must end in dimension {width}, got {value.shape}"
        )
    if not np.all(np.isfinite(value)):
        raise PolicyProtocolError(f"{label} contains NaN or Inf")


def _validate_wrist_state(
    value: Any,
    requirement: Mapping[str, Any],
    label: str,
    *,
    leading_shape: tuple[int, ...],
) -> None:
    wrist = _mapping(value, label)
    _require_keys(wrist, label, ("joint", "eef", "eef_def"))
    _validate_optional_state_array(
        wrist.get("joint"),
        f"{label}.joint",
        required=bool(requirement["joint"]),
        leading_shape=leading_shape,
        width=None,
    )
    eef_required = bool(requirement["eef"])
    _validate_optional_state_array(
        wrist.get("eef"),
        f"{label}.eef",
        required=eef_required,
        leading_shape=leading_shape,
        width=9,
    )
    eef_def = wrist["eef_def"]
    if eef_required:
        if eef_def not in (None, "absolute"):
            raise PolicyProtocolError(f"{label}.eef_def must be absolute or None")
    elif eef_def is not None:
        raise PolicyProtocolError(f"{label}.eef_def must be None when EEF is unused")


def _validate_hand_state(
    value: Any,
    requirement: Mapping[str, Any],
    label: str,
    *,
    leading_shape: tuple[int, ...],
) -> None:
    hands = _mapping(value, label)
    for side in ("left", "right"):
        _validate_optional_state_array(
            hands.get(side),
            f"{label}.{side}",
            required=bool(requirement[side]),
            leading_shape=leading_shape,
            width=22,
        )


def _validate_state(value: Any, requirement: Mapping[str, Any]) -> None:
    state = _mapping(value, "obs.state")
    history_len = int(requirement["history_len"])
    history = state.get("history")
    history_timestamp: int | None = None
    if history_len:
        history_value = _mapping(history, "obs.state.history")
        timestamps = _array(
            history_value.get("timestamp_ns"),
            "obs.state.history.timestamp_ns",
            dtype=np.dtype(np.int64),
            shape=(history_len,),
        )
        if np.any(timestamps < 0) or np.any(np.diff(timestamps) < 0):
            raise PolicyProtocolError(
                "obs.state.history.timestamp_ns must be nonnegative and chronological"
            )
        history_timestamp = int(timestamps[-1])
        _validate_wrist_state(
            history_value.get("left_wrist"),
            _mapping(requirement["left_wrist"], "metadata_format.state.left_wrist"),
            "obs.state.history.left_wrist",
            leading_shape=(history_len,),
        )
        _validate_wrist_state(
            history_value.get("right_wrist"),
            _mapping(requirement["right_wrist"], "metadata_format.state.right_wrist"),
            "obs.state.history.right_wrist",
            leading_shape=(history_len,),
        )
        _validate_hand_state(
            history_value.get("hand_joint"),
            _mapping(requirement["hand_joint"], "metadata_format.state.hand_joint"),
            "obs.state.history.hand_joint",
            leading_shape=(history_len,),
        )
        _array(
            history_value.get("valid"),
            "obs.state.history.valid",
            dtype=np.dtype(np.bool_),
            shape=(history_len,),
        )
    elif history is not None:
        raise PolicyProtocolError("obs.state.history must be None when history_len is 0")

    current = state.get("current")
    if requirement["current"]:
        current_value = _mapping(current, "obs.state.current")
        timestamp = _integer(
            current_value.get("timestamp_ns"),
            "obs.state.current.timestamp_ns",
            minimum=0,
        )
        if history_timestamp is not None and timestamp < history_timestamp:
            raise PolicyProtocolError("obs.state.current must not precede history")
        _validate_wrist_state(
            current_value.get("left_wrist"),
            _mapping(requirement["left_wrist"], "metadata_format.state.left_wrist"),
            "obs.state.current.left_wrist",
            leading_shape=(),
        )
        _validate_wrist_state(
            current_value.get("right_wrist"),
            _mapping(requirement["right_wrist"], "metadata_format.state.right_wrist"),
            "obs.state.current.right_wrist",
            leading_shape=(),
        )
        _validate_hand_state(
            current_value.get("hand_joint"),
            _mapping(requirement["hand_joint"], "metadata_format.state.hand_joint"),
            "obs.state.current.hand_joint",
            leading_shape=(),
        )
        _boolean(current_value.get("valid"), "obs.state.current.valid")
    elif current is not None:
        raise PolicyProtocolError("obs.state.current must be None when not requested")


def _validate_pair(
    value: Any,
    label: str,
    *,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
    finite: bool = False,
) -> None:
    pair = _mapping(value, label)
    for side in ("left", "right"):
        _array(
            pair.get(side),
            f"{label}.{side}",
            dtype=dtype,
            shape=shape,
            finite=finite,
        )


def _validate_sensor(
    value: Any,
    requirement: Mapping[str, Any],
    label: str,
    *,
    dtype: np.dtype[Any],
    sample_shape: tuple[int, ...],
    valid_sample_shape: tuple[int, ...],
    finite: bool,
) -> None:
    sensor = _mapping(value, label)
    history_len = int(requirement["history_len"])
    history = sensor.get("history")
    history_timestamp: int | None = None
    if history_len:
        history_value = _mapping(history, f"{label}.history")
        _validate_pair(
            history_value,
            f"{label}.history",
            dtype=dtype,
            shape=(history_len, *sample_shape),
            finite=finite,
        )
        timestamps = _array(
            history_value.get("timestamp_ns"),
            f"{label}.history.timestamp_ns",
            dtype=np.dtype(np.int64),
            shape=(history_len,),
        )
        if np.any(timestamps < 0) or np.any(np.diff(timestamps) < 0):
            raise PolicyProtocolError(
                f"{label}.history.timestamp_ns must be nonnegative and chronological"
            )
        history_timestamp = int(timestamps[-1])
        _validate_pair(
            history_value.get("valid"),
            f"{label}.history.valid",
            dtype=np.dtype(np.bool_),
            shape=(history_len, *valid_sample_shape),
        )
    elif history is not None:
        raise PolicyProtocolError(f"{label}.history must be None when history_len is 0")

    current = sensor.get("current")
    if requirement["current"]:
        current_value = _mapping(current, f"{label}.current")
        _validate_pair(
            current_value,
            f"{label}.current",
            dtype=dtype,
            shape=sample_shape,
            finite=finite,
        )
        timestamp = _integer(
            current_value.get("timestamp_ns"),
            f"{label}.current.timestamp_ns",
            minimum=0,
        )
        if history_timestamp is not None and timestamp < history_timestamp:
            raise PolicyProtocolError(f"{label}.current must not precede history")
        _validate_pair(
            current_value.get("valid"),
            f"{label}.current.valid",
            dtype=np.dtype(np.bool_),
            shape=valid_sample_shape,
        )
    elif current is not None:
        raise PolicyProtocolError(f"{label}.current must be None when not requested")


def validate_observation(
    value: Any,
    metadata_format: SharpAMetadataFormat | Mapping[str, Any],
) -> SharpAObservation:
    """Validate one observation against the server's active metadata format."""

    validated_format = validate_metadata_format(metadata_format)
    obs = _mapping(value, "obs")
    if obs.get("schema") != OBSERVATION_SCHEMA:
        raise PolicyProtocolError(f"obs.schema must be {OBSERVATION_SCHEMA!r}")
    if obs.get("metadata_format_id") != validated_format["format_id"]:
        raise PolicyProtocolError(
            "obs.metadata_format_id does not match the active metadata format"
        )
    _nonempty_string(obs.get("session_id"), "obs.session_id")
    _integer(obs.get("request_id"), "obs.request_id", minimum=0)
    _integer(obs.get("timestamp_ns"), "obs.timestamp_ns", minimum=0)
    _nonempty_string(obs.get("prompt"), "obs.prompt")

    image = _mapping(obs.get("image"), "obs.image")
    image_format = _mapping(validated_format["image"], "metadata_format.image")
    for camera_name in ("ego_cam", "left_wrist_cam", "right_wrist_cam"):
        _validate_camera_stream(
            image.get(camera_name),
            _mapping(
                image_format[camera_name],
                f"metadata_format.image.{camera_name}",
            ),
            f"obs.image.{camera_name}",
        )

    _validate_state(
        obs.get("state"),
        _mapping(validated_format["state"], "metadata_format.state"),
    )

    sensor = _mapping(obs.get("sensor"), "obs.sensor")
    sensor_format = _mapping(validated_format["sensor"], "metadata_format.sensor")
    _validate_sensor(
        sensor.get("tau"),
        _mapping(sensor_format["tau"], "metadata_format.sensor.tau"),
        "obs.sensor.tau",
        dtype=np.dtype(np.float32),
        sample_shape=(22,),
        valid_sample_shape=(22,),
        finite=True,
    )
    _validate_sensor(
        sensor.get("wrench"),
        _mapping(sensor_format["wrench"], "metadata_format.sensor.wrench"),
        "obs.sensor.wrench",
        dtype=np.dtype(np.float32),
        sample_shape=(5, 6),
        valid_sample_shape=(5,),
        finite=True,
    )
    _validate_sensor(
        sensor.get("deformation"),
        _mapping(
            sensor_format["deformation"],
            "metadata_format.sensor.deformation",
        ),
        "obs.sensor.deformation",
        dtype=np.dtype(np.uint8),
        sample_shape=(5, 240, 240),
        valid_sample_shape=(5,),
        finite=False,
    )

    feedback = _mapping(obs.get("execution_feedback"), "obs.execution_feedback")
    last_action_id = feedback.get("last_action_id")
    if last_action_id is not None:
        _nonempty_string(last_action_id, "obs.execution_feedback.last_action_id")
    _integer(
        feedback.get("executed_steps"),
        "obs.execution_feedback.executed_steps",
        minimum=0,
    )
    _boolean(feedback.get("success"), "obs.execution_feedback.success")
    return cast(SharpAObservation, value)


def _validate_wrist_action(
    value: Any,
    action_type: WristActionType,
    action_length: int,
    label: str,
) -> None:
    if not isinstance(value, np.ndarray) or value.dtype != np.float32:
        raise PolicyProtocolError(f"{label} must be a float32 numpy.ndarray")
    expected_width = 9 if action_type in ("eef", "relative_eef") else None
    if value.ndim != 2 or value.shape[0] != action_length:
        raise PolicyProtocolError(
            f"{label} must have shape ({action_length}, D), got {value.shape}"
        )
    if expected_width is not None and value.shape[1] != expected_width:
        raise PolicyProtocolError(
            f"{label} must have shape ({action_length}, {expected_width})"
        )
    if expected_width is None and value.shape[1] < 1:
        raise PolicyProtocolError(f"{label} joint action must have at least one joint")
    if not np.all(np.isfinite(value)):
        raise PolicyProtocolError(f"{label} contains NaN or Inf")


def validate_action(value: Any) -> SharpAPolicyAction:
    """Validate the policy action returned to the robot."""

    result = _mapping(value, "action_result")
    _require_keys(
        result,
        "action_result",
        (
            "schema",
            "session_id",
            "request_id",
            "action_id",
            "revision",
            "timestamp_ns",
            "execution",
            "action",
            "auxiliary",
            "diagnostics",
            "next_metadata_format",
        ),
    )
    if result.get("schema") != ACTION_SCHEMA:
        raise PolicyProtocolError(f"action_result.schema must be {ACTION_SCHEMA!r}")
    _nonempty_string(result.get("session_id"), "action_result.session_id")
    _integer(result.get("request_id"), "action_result.request_id", minimum=0)
    _nonempty_string(result.get("action_id"), "action_result.action_id")
    _integer(result.get("revision"), "action_result.revision", minimum=0)
    _integer(result.get("timestamp_ns"), "action_result.timestamp_ns", minimum=0)

    execution = _mapping(result.get("execution"), "action_result.execution")
    _require_keys(
        execution,
        "action_result.execution",
        ("frequency_hz", "action_length", "execute_start", "execute_length", "rtc"),
    )
    frequency = execution.get("frequency_hz")
    if isinstance(frequency, (bool, np.bool_)) or not isinstance(
        frequency, (int, float, np.integer, np.floating)
    ):
        raise PolicyProtocolError("action_result.execution.frequency_hz must be numeric")
    if not np.isfinite(float(frequency)) or float(frequency) <= 0:
        raise PolicyProtocolError("action_result.execution.frequency_hz must be positive")
    action_length = _integer(
        execution.get("action_length"),
        "action_result.execution.action_length",
        minimum=1,
    )
    execute_start = _integer(
        execution.get("execute_start"),
        "action_result.execution.execute_start",
        minimum=0,
    )
    execute_length = _integer(
        execution.get("execute_length"),
        "action_result.execution.execute_length",
        minimum=1,
    )
    if execute_start + execute_length > action_length:
        raise PolicyProtocolError("action_result execution slice exceeds action_length")
    rtc = _integer(
        execution.get("rtc"),
        "action_result.execution.rtc",
        minimum=0,
    )
    if rtc > 0 and rtc >= execute_length:
        raise PolicyProtocolError(
            "action_result.execution.rtc must be smaller than execute_length"
        )

    action = _mapping(result.get("action"), "action_result.action")
    _require_keys(
        action,
        "action_result.action",
        ("left_wrist", "right_wrist", "hand_joint"),
    )
    for side in ("left", "right"):
        label = f"action_result.action.{side}_wrist"
        wrist = _mapping(action[f"{side}_wrist"], label)
        _require_keys(wrist, label, ("joint", "eef", "eef_def"))
        joint = wrist["joint"]
        eef = wrist["eef"]
        if joint is not None:
            _validate_wrist_action(joint, "joint", action_length, f"{label}.joint")
        if eef is not None:
            _validate_wrist_action(eef, "eef", action_length, f"{label}.eef")
        if wrist["eef_def"] not in (None, "absolute"):
            raise PolicyProtocolError(
                f"{label}.eef_def must be absolute or None at the public boundary"
            )

    hands = _mapping(
        action["hand_joint"], "action_result.action.hand_joint"
    )
    _require_keys(hands, "action_result.action.hand_joint", ("left", "right"))
    for side in ("left", "right"):
        hand = hands[side]
        if hand is not None:
            _array(
                hand,
                f"action_result.action.hand_joint.{side}",
                dtype=np.dtype(np.float32),
                shape=(action_length, 22),
                finite=True,
            )

    auxiliary = _mapping(result["auxiliary"], "action_result.auxiliary")
    _require_keys(auxiliary, "action_result.auxiliary", ("video", "tactile"))
    video = _mapping(auxiliary["video"], "action_result.auxiliary.video")
    _require_keys(
        video,
        "action_result.auxiliary.video",
        ("ego", "left_wrist", "right_wrist"),
    )
    tactile = _mapping(auxiliary["tactile"], "action_result.auxiliary.tactile")
    _require_keys(
        tactile,
        "action_result.auxiliary.tactile",
        ("deformation", "wrench", "hand_tau"),
    )

    diagnostics = _mapping(result.get("diagnostics"), "action_result.diagnostics")
    _nonempty_string(
        diagnostics.get("policy_family"), "action_result.diagnostics.policy_family"
    )
    _nonempty_string(
        diagnostics.get("checkpoint_id"), "action_result.diagnostics.checkpoint_id"
    )
    _nonempty_string(
        diagnostics.get("checkpoint_path"),
        "action_result.diagnostics.checkpoint_path",
    )
    latency = diagnostics.get("inference_latency_ms")
    if isinstance(latency, (bool, np.bool_)) or not isinstance(
        latency, (int, float, np.integer, np.floating)
    ):
        raise PolicyProtocolError(
            "action_result.diagnostics.inference_latency_ms must be numeric"
        )
    if not np.isfinite(float(latency)) or float(latency) < 0:
        raise PolicyProtocolError(
            "action_result.diagnostics.inference_latency_ms must be nonnegative"
        )

    next_metadata_format = result.get("next_metadata_format")
    if next_metadata_format is not None:
        validate_metadata_format(next_metadata_format)
    return cast(SharpAPolicyAction, value)


def build_error_response(
    *,
    code: str,
    message: str,
    request_id: int | None,
    retryable: bool,
) -> dict[str, Any]:
    return {
        "schema": ERROR_SCHEMA,
        "request_id": request_id,
        "error": {
            "code": str(code),
            "message": str(message),
            "retryable": bool(retryable),
        },
    }
