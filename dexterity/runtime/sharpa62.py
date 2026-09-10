"""SharpA62 joint layouts and coordinate conversions.

The constants mirror ``ws_core.kinematics`` so data conversion, evaluation,
offline inference, and deployment all use the same robot convention without a
ROS dependency.
"""

from __future__ import annotations

from typing import Any

import numpy as np


GEOMETRY_SCHEMA = "sharpa62_policy_geometry.v1"
DIRECT_EEF_INTERFACE_SCHEMA = "sharpa62_direct_absolute_eef.v1"
DEPLOY_JOINT_LAYOUT = "sharpa_joint_order.v1"
DEPLOY_TACTILE_LAYOUT = "sharpa_tactile_right_then_left_pinky_to_thumb.v1"
STATE_DIM = 62
JOINT_COUNT = 44
TACTILE_COUNT = 10

DEPLOY_JOINT_ORDER = (
    "left_thumb_CMC_FE",
    "left_thumb_CMC_AA",
    "left_thumb_MCP_FE",
    "left_thumb_MCP_AA",
    "left_thumb_IP",
    "left_index_MCP_FE",
    "left_index_MCP_AA",
    "left_index_PIP",
    "left_index_DIP",
    "left_middle_MCP_FE",
    "left_middle_MCP_AA",
    "left_middle_PIP",
    "left_middle_DIP",
    "left_ring_MCP_FE",
    "left_ring_MCP_AA",
    "left_ring_PIP",
    "left_ring_DIP",
    "left_pinky_CMC",
    "left_pinky_MCP_FE",
    "left_pinky_MCP_AA",
    "left_pinky_PIP",
    "left_pinky_DIP",
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
)

MODEL_JOINT_ORDER = (
    "left_index_MCP_FE",
    "left_index_MCP_AA",
    "left_index_PIP",
    "left_index_DIP",
    "left_middle_MCP_FE",
    "left_middle_MCP_AA",
    "left_middle_PIP",
    "left_middle_DIP",
    "left_pinky_CMC",
    "left_pinky_MCP_FE",
    "left_pinky_MCP_AA",
    "left_pinky_PIP",
    "left_pinky_DIP",
    "left_ring_MCP_FE",
    "left_ring_MCP_AA",
    "left_ring_PIP",
    "left_ring_DIP",
    "left_thumb_CMC_FE",
    "left_thumb_CMC_AA",
    "left_thumb_MCP_FE",
    "left_thumb_MCP_AA",
    "left_thumb_IP",
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
)

DEPLOY_TACTILE_ORDER = (
    "right_pinky",
    "right_ring",
    "right_middle",
    "right_index",
    "right_thumb",
    "left_pinky",
    "left_ring",
    "left_middle",
    "left_index",
    "left_thumb",
)
MODEL_TACTILE_ORDER = DEPLOY_TACTILE_ORDER

DEPLOY_TO_MODEL_JOINT = np.asarray(
    [DEPLOY_JOINT_ORDER.index(name) for name in MODEL_JOINT_ORDER],
    dtype=np.int64,
)
MODEL_TO_DEPLOY_JOINT = np.asarray(
    [MODEL_JOINT_ORDER.index(name) for name in DEPLOY_JOINT_ORDER],
    dtype=np.int64,
)
DEPLOY_TO_MODEL_TACTILE = np.asarray(
    [DEPLOY_TACTILE_ORDER.index(name) for name in MODEL_TACTILE_ORDER],
    dtype=np.int64,
)

PND_ROOT_TO_PRETRAIN = np.asarray(
    (
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
    ),
    dtype=np.float32,
)

POSTTRAIN_RAW2HAND_BASE = {
    "left": (
        np.asarray(
            (
                (-0.05063197761774063, -0.9985009431838989, -0.020789900794625282),
                (-0.9941514730453491, 0.04840133339166641, 0.09654103964567184),
                (-0.09539006650447845, 0.025556374341249466, -0.995111882686615),
            ),
            dtype=np.float32,
        ),
        np.asarray(
            (0.006119123660027981, -0.004442085511982441, -0.03515041619539261),
            dtype=np.float32,
        ),
    ),
    "right": (
        np.asarray(
            (
                (-0.037729986011981964, 0.9990273714065552, -0.022819984704256058),
                (0.9962800741195679, 0.035836100578308105, -0.07836932688951492),
                (-0.07747532427310944, -0.02569197118282318, -0.9966631531715393),
            ),
            dtype=np.float32,
        ),
        np.asarray(
            (0.003884613746777177, 0.002192644402384758, -0.035935886204242706),
            dtype=np.float32,
        ),
    ),
}

def _rotation_matrix(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be a finite 3x3 matrix")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-5):
        raise ValueError(f"{label} must be orthonormal")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-5):
        raise ValueError(f"{label} must have determinant +1")
    return matrix


def column_rot6d_to_matrix(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (6,) or not np.all(np.isfinite(vector)):
        raise ValueError("column Rot6D must be a finite (6,) vector")
    first, second = vector[:3], vector[3:]
    first_norm = float(np.linalg.norm(first))
    if first_norm < 1e-8:
        raise ValueError("column Rot6D first axis is degenerate")
    first = first / first_norm
    second = second - float(np.dot(first, second)) * first
    second_norm = float(np.linalg.norm(second))
    if second_norm < 1e-8:
        raise ValueError("column Rot6D second axis is degenerate")
    second = second / second_norm
    return np.stack((first, second, np.cross(first, second)), axis=1)


def matrix_to_column_rot6d(value: Any) -> np.ndarray:
    matrix = _rotation_matrix(value, "rotation")
    return np.concatenate((matrix[:, 0], matrix[:, 1])).astype(np.float32)


def row_rot6d_to_matrix(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (6,) or not np.all(np.isfinite(vector)):
        raise ValueError("row Rot6D must be a finite (6,) vector")
    first, second = vector[:3], vector[3:]
    first_norm = float(np.linalg.norm(first))
    if first_norm < 1e-8:
        raise ValueError("row Rot6D first axis is degenerate")
    first = first / first_norm
    second = second - float(np.dot(first, second)) * first
    second_norm = float(np.linalg.norm(second))
    if second_norm < 1e-8:
        raise ValueError("row Rot6D second axis is degenerate")
    second = second / second_norm
    return np.stack((first, second, np.cross(first, second)), axis=0)


def matrix_to_row_rot6d(value: Any) -> np.ndarray:
    matrix = _rotation_matrix(value, "rotation")
    return matrix[:2, :].reshape(6).astype(np.float32)


def wire_state_to_model_layout(state_wire: Any) -> np.ndarray:
    """Keep public EEF values unchanged and reorder only the hand joints.

    The robot-facing process owns all world/root/axis-direction transforms. A
    deployment server receives the already canonical absolute wrist EEF and
    must not apply ``PND_ROOT_TO_PRETRAIN`` or either hand-base transform.
    """

    state = np.asarray(state_wire, dtype=np.float32)
    if state.shape != (STATE_DIM,) or not np.all(np.isfinite(state)):
        raise ValueError("observation/hand_pose_62d must be finite float[62]")
    return np.concatenate(
        (state[:18], state[18:][DEPLOY_TO_MODEL_JOINT])
    ).astype(np.float32)


def model_joints_to_wire(value: Any) -> np.ndarray:
    """Return model-order hand joints in the public SharpA joint order."""

    joints = np.asarray(value, dtype=np.float32)
    if joints.shape[-1] != JOINT_COUNT:
        raise ValueError(f"model joints must end in 44, got {joints.shape}")
    if not np.all(np.isfinite(joints)):
        raise ValueError("model joints contain NaN or Inf")
    return joints[..., MODEL_TO_DEPLOY_JOINT].astype(np.float32)


def model_action_to_wire_layout(action_model: Any) -> np.ndarray:
    """Expose absolute model EEF unchanged and reorder only hand joints."""

    action = np.asarray(action_model, dtype=np.float32)
    if action.ndim != 2 or action.shape[1] != STATE_DIM:
        raise ValueError(f"model action must have shape (T,62), got {action.shape}")
    if not np.all(np.isfinite(action)):
        raise ValueError("model action contains NaN or Inf")
    return np.concatenate(
        (action[:, :18], model_joints_to_wire(action[:, 18:])), axis=-1
    ).astype(np.float32)


def direct_eef_interface_metadata() -> dict[str, Any]:
    """Describe the robot/model boundary used by GR00T, PACE and pi0.5."""

    return {
        "schema": DIRECT_EEF_INTERFACE_SCHEMA,
        "observation_eef_def": "absolute",
        "action_eef_def": "absolute",
        "eef_pose_layout": "xyz,r00,r10,r20,r01,r11,r21",
        "server_xyz_direction_transform": "none",
        "xyz_direction_transform_owner": "robot",
        "relative_to_absolute_owner": "model_processor_or_deploy_adapter",
        "wire_joint_layout": DEPLOY_JOINT_LAYOUT,
        "wire_joint_order": list(DEPLOY_JOINT_ORDER),
        "model_joint_order": list(MODEL_JOINT_ORDER),
    }


class SharpA62PolicyGeometry:
    """Fixed physical SharpA62 ↔ posttrain model adapter."""

    def __init__(self) -> None:
        self.root_transform = _rotation_matrix(
            PND_ROOT_TO_PRETRAIN, "PND_ROOT_TO_PRETRAIN"
        )
        self.base: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for side in ("left", "right"):
            rotation, translation = POSTTRAIN_RAW2HAND_BASE[side]
            self.base[side] = (
                _rotation_matrix(rotation, f"{side} POSTTRAIN_RAW2HAND_BASE rotation"),
                np.asarray(translation, dtype=np.float64),
            )
        self.deploy_joint_layout = DEPLOY_JOINT_LAYOUT
        self.deploy_tactile_layout = DEPLOY_TACTILE_LAYOUT
        self.deploy_joint_order = DEPLOY_JOINT_ORDER
        self.model_joint_order = MODEL_JOINT_ORDER
        self.deploy_tactile_order = DEPLOY_TACTILE_ORDER
        self.model_tactile_order = MODEL_TACTILE_ORDER
        self.deploy_to_model_joint = DEPLOY_TO_MODEL_JOINT
        self.model_to_deploy_joint = MODEL_TO_DEPLOY_JOINT
        self.deploy_to_model_tactile = DEPLOY_TO_MODEL_TACTILE

    def deploy_pose_to_model(
        self,
        pose: Any,
        side: str,
    ) -> np.ndarray:
        value = np.asarray(pose, dtype=np.float64)
        if value.shape != (9,) or not np.all(np.isfinite(value)):
            raise ValueError("deploy wrist pose must be finite (9,)")
        physical_rotation = column_rot6d_to_matrix(value[3:9])
        base_rotation, base_translation = self.base[side]
        baked_position = value[:3] + physical_rotation @ base_translation
        baked_rotation = physical_rotation @ base_rotation
        model_rotation = self.root_transform @ baked_rotation
        return np.concatenate(
            (
                self.root_transform @ baked_position,
                matrix_to_row_rot6d(model_rotation),
            )
        ).astype(np.float32)

    def model_pose_to_deploy(
        self,
        pose: Any,
        side: str,
    ) -> np.ndarray:
        value = np.asarray(pose, dtype=np.float64)
        if value.shape != (9,) or not np.all(np.isfinite(value)):
            raise ValueError("model wrist pose must be finite (9,)")
        baked_position = self.root_transform.T @ value[:3]
        baked_rotation = self.root_transform.T @ row_rot6d_to_matrix(
            value[3:9]
        )
        base_rotation, base_translation = self.base[side]
        physical_rotation = baked_rotation @ base_rotation.T
        physical_position = baked_position - physical_rotation @ base_translation
        return np.concatenate(
            (physical_position, matrix_to_column_rot6d(physical_rotation))
        ).astype(np.float32)

    def state_to_model(
        self,
        state_deploy: Any,
    ) -> np.ndarray:
        state = np.asarray(state_deploy, dtype=np.float32)
        if state.shape != (STATE_DIM,) or not np.all(np.isfinite(state)):
            raise ValueError("observation/hand_pose_62d must be finite float[62]")
        return np.concatenate(
            (
                self.deploy_pose_to_model(state[:9], "left"),
                self.deploy_pose_to_model(state[9:18], "right"),
                state[18:][self.deploy_to_model_joint],
            )
        ).astype(np.float32)

    def wrists_to_deploy(
        self,
        wrist_model: Any,
    ) -> np.ndarray:
        wrist = np.asarray(wrist_model, dtype=np.float32)
        if wrist.ndim != 2 or wrist.shape[1] != 18:
            raise ValueError(f"model wrist action must have shape (T,18), got {wrist.shape}")
        result = np.empty_like(wrist)
        for index, value in enumerate(wrist):
            result[index, :9] = self.model_pose_to_deploy(value[:9], "left")
            result[index, 9:] = self.model_pose_to_deploy(value[9:], "right")
        return result

    def joints_to_deploy(self, value: Any) -> np.ndarray:
        joints = np.asarray(value, dtype=np.float32)
        if joints.shape[-1] != JOINT_COUNT:
            raise ValueError(f"model joints must end in 44, got {joints.shape}")
        return joints[..., self.model_to_deploy_joint].astype(np.float32)

    def action_to_deploy(
        self,
        action_model: Any,
    ) -> np.ndarray:
        action = np.asarray(action_model, dtype=np.float32)
        if action.ndim != 2 or action.shape[1] != STATE_DIM:
            raise ValueError(f"model action must have shape (T,62), got {action.shape}")
        output = np.concatenate(
            (
                self.wrists_to_deploy(action[:, :18]),
                self.joints_to_deploy(action[:, 18:]),
            ),
            axis=-1,
        )
        if not np.all(np.isfinite(output)):
            raise ValueError("converted action contains NaN or Inf")
        return output.astype(np.float32)


def geometry_metadata() -> dict[str, Any]:
    return {
        "schema": GEOMETRY_SCHEMA,
        "robot_wire_wrist_rotation": "matrix_columns_0_and_1",
        "model_wrist_rotation": "matrix_rows_0_and_1",
        "deploy_joint_layout": DEPLOY_JOINT_LAYOUT,
        "deploy_tactile_layout": DEPLOY_TACTILE_LAYOUT,
        "deploy_joint_order": list(DEPLOY_JOINT_ORDER),
        "model_joint_order": list(MODEL_JOINT_ORDER),
        "deploy_tactile_order": list(DEPLOY_TACTILE_ORDER),
        "model_tactile_order": list(MODEL_TACTILE_ORDER),
    }
