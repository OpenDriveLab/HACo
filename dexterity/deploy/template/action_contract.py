"""Legacy SharpA62 deployment response contract.

Both GR00T and GCC servers cross the workstation boundary with this exact
contract.  Model-specific values belong in ``metadata`` or ``diagnostics``;
the executable path only consumes ``action_hand_pose_62d`` and the common
contract fields.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


ACTION_SCHEMA = "sharpa62_policy_action.v1"
ACTION_HORIZON = 40
ACTION_HZ = 30.0
ACTION_DIM = 62
ACTION_LAYOUT = "left_wrist9,right_wrist9,sharpa_q44"
ACTION_SPACE = "sharpa_dexretarget_position_62d"
WRIST_FRAME = "absolute_current_hip"


def finite_action(value: Any) -> np.ndarray:
    """Return a canonical float32 action or reject the response."""

    action = np.asarray(value)
    if action.shape != (ACTION_HORIZON, ACTION_DIM):
        raise ValueError(
            "action_hand_pose_62d must have shape "
            f"({ACTION_HORIZON},{ACTION_DIM}), got {action.shape}"
        )
    if action.dtype != np.float32:
        raise ValueError(
            f"action_hand_pose_62d must have dtype float32, got {action.dtype}"
        )
    if not np.all(np.isfinite(action)):
        raise ValueError("action_hand_pose_62d contains NaN or Inf")
    return action


def build_policy_action(
    action_hand_pose_62d: Any,
    *,
    session_id: str,
    request_index: int,
    policy_family: str,
    execute_joint_source: str,
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and validate the response shared by every SharpA62 policy."""

    payload: dict[str, Any] = {
        "schema": ACTION_SCHEMA,
        "session_id": str(session_id),
        "request_index": np.asarray(request_index, dtype=np.int64),
        "is_real_policy": np.asarray(True, dtype=bool),
        "action_hand_pose_62d": finite_action(action_hand_pose_62d),
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
    validate_policy_action(payload)
    return payload


def validate_policy_action(payload: Mapping[str, Any]) -> np.ndarray:
    """Strictly validate a common response and return its executable action."""

    expected = {
        "schema": ACTION_SCHEMA,
        "action_space": ACTION_SPACE,
        "wrist_frame": WRIST_FRAME,
        "layout": ACTION_LAYOUT,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{key} must be {value!r}, got {payload.get(key)!r}")
    if int(payload.get("action_horizon", -1)) != ACTION_HORIZON:
        raise ValueError(f"action_horizon must be {ACTION_HORIZON}")
    if not np.isclose(float(payload.get("action_hz", -1.0)), ACTION_HZ):
        raise ValueError(f"action_hz must be {ACTION_HZ}")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    for key in ("policy_family", "execute_joint_source"):
        if not isinstance(metadata.get(key), str) or not metadata[key]:
            raise ValueError(f"metadata.{key} must be a nonempty string")
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("diagnostics must be a mapping")
    return finite_action(payload.get("action_hand_pose_62d"))


def common_action_metadata(
    *,
    policy_family: str,
    execute_joint_source: str,
) -> dict[str, Any]:
    """Metadata fragment advertised on WebSocket connect."""

    return {
        "response_schema": ACTION_SCHEMA,
        "action_dim": ACTION_DIM,
        "action_horizon": ACTION_HORIZON,
        "action_hz": ACTION_HZ,
        "layout": ACTION_LAYOUT,
        "action_space": ACTION_SPACE,
        "output_wrist_frame": WRIST_FRAME,
        "policy_family": str(policy_family),
        "execute_joint_source": str(execute_joint_source),
    }
