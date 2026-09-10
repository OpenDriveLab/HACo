"""Turn policy action vectors into the 138-D hand pose the skeleton renderer draws.

Every model in this repository emits wrist + joint actions in the same 62-D
layout (18-D bimanual wrist eef + 44-D hand joints), so the forward kinematics
below are shared by all of them. Models that additionally predict an active
compliance residual supply ``delta_q = q_cmp - q_obs``; ``add_delta_q`` builds
the compliant command that the renderer overlays on the observed pose.
"""

from __future__ import annotations

import numpy as np

from dexterity.runtime.sharpa62 import MODEL_JOINT_ORDER
from dexterity.runtime.sharpa_kinematics import (
    model_wrist_to_robot_wire,
    rot6d_to_mat,
    sharpa62_raw_to_hand138,
)

ACTION_DIM = 62
JOINT_SLICE = slice(18, 62)
JOINT_COUNT = 44

# rot6d identity: the first two rows of a 3x3 identity rotation.
IDENTITY_HIP_POSE_9D = np.asarray(
    (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0), dtype=np.float32
)


def identity_hip_anchors() -> tuple[np.ndarray, np.ndarray]:
    """Single anchor at the origin with no rotation."""
    return (
        np.zeros((1, 3), dtype=np.float32),
        np.eye(3, dtype=np.float32)[None],
    )


def hip_anchors_to_chunk0(hip_pose_9d) -> tuple[np.ndarray, np.ndarray]:
    """``(n_chunks, 9)`` hip poses -> anchors and rotations in chunk-0's frame.

    Returns identity anchors when the input is empty, so callers without a hip
    side channel render in a fixed frame instead of failing.
    """
    hip = np.asarray(hip_pose_9d, dtype=np.float32)
    while hip.ndim > 2:
        hip = hip[0]
    if hip.ndim == 1:
        hip = hip.reshape(1, -1)
    if hip.size == 0 or hip.shape[-1] < 9:
        return identity_hip_anchors()

    hip = hip[:, :9]
    first_translation = hip[0, :3]
    first_rotation_inverse = rot6d_to_mat(hip[0, 3:9]).T
    anchors = np.zeros((len(hip), 3), dtype=np.float32)
    rotations = np.zeros((len(hip), 3, 3), dtype=np.float32)
    for index, pose in enumerate(hip):
        anchors[index] = (
            first_rotation_inverse @ (pose[:3] - first_translation)
        ).astype(np.float32)
        rotations[index] = (first_rotation_inverse @ rot6d_to_mat(pose[3:9])).astype(
            np.float32
        )
    return anchors, rotations


def action_62d_to_hand138(
    action_62d: np.ndarray,
    *,
    anchor_hips: np.ndarray,
    anchor_rots: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    """``(T, 62)`` absolute action -> ``(T, 138)`` bimanual hand keypoints.

    The action space defaults to SharpA dexretarget positions inside
    ``sharpa62_raw_to_hand138``; every model in this repository uses it, so it
    is not a caller-visible knob.
    """
    action = np.asarray(action_62d, dtype=np.float32)
    if action.ndim != 2 or action.shape[-1] < ACTION_DIM:
        raise ValueError(
            f"action must have shape [T,>={ACTION_DIM}], got {action.shape}"
        )
    try:
        hand = sharpa62_raw_to_hand138(
            model_wrist_to_robot_wire(action[:, :ACTION_DIM]),
            action_space=None,
            anchor_hips=anchor_hips,
            anchor_rots=anchor_rots,
            anchor_q44=None,
            root_q6=None,
            chunk_size=max(1, int(chunk_size)),
            q_is_delta=False,
        )
    except (ImportError, OSError, FileNotFoundError, RuntimeError, ValueError):
        # The quick checker must also work without optional Pinocchio/URDF
        # packages.  This lightweight FK preserves wrists, finger topology and
        # per-joint motion; installations with the robot assets use exact FK.
        hand = _approximate_hand138(action[:, :ACTION_DIM])
    if hand is None:
        raise RuntimeError("SharpA forward kinematics returned no hand keypoints")
    return hand


def _approximate_hand138(action: np.ndarray) -> np.ndarray:
    """Dependency-free 21-keypoint hand skeleton for quick visualization."""

    finger_order = ("thumb", "index", "middle", "ring", "pinky")
    base_x = {
        "thumb": -0.040,
        "index": -0.025,
        "middle": -0.008,
        "ring": 0.010,
        "pinky": 0.027,
    }
    lengths = {
        "thumb": (0.030, 0.025, 0.021, 0.017),
        "index": (0.038, 0.028, 0.021, 0.017),
        "middle": (0.041, 0.031, 0.023, 0.018),
        "ring": (0.039, 0.029, 0.022, 0.017),
        "pinky": (0.033, 0.024, 0.019, 0.015),
    }
    output = np.zeros((len(action), 138), dtype=np.float32)
    for row, value in enumerate(action):
        for hand_index, side in enumerate(("left", "right")):
            wrist = value[hand_index * 9 : hand_index * 9 + 9]
            rotation = rot6d_to_mat(wrist[3:9])
            q = value[18 + hand_index * 22 : 18 + (hand_index + 1) * 22]
            names = MODEL_JOINT_ORDER[hand_index * 22 : (hand_index + 1) * 22]
            by_finger = {
                finger: [
                    float(q[i]) for i, name in enumerate(names) if f"_{finger}_" in name
                ]
                for finger in finger_order
            }
            points = []
            mirror = 1.0 if side == "left" else -1.0
            for finger in finger_order:
                joint = by_finger[finger]
                spread = joint[1] if len(joint) > 1 else 0.0
                flex = [joint[0] if joint else 0.0, *joint[2:]]
                while len(flex) < 4:
                    flex.append(flex[-1] if flex else 0.0)
                position = np.asarray(
                    [
                        mirror * base_x[finger],
                        0.018 if finger == "thumb" else 0.035,
                        0.0,
                    ],
                    dtype=np.float32,
                )
                if finger == "thumb":
                    position[1] = 0.004
                angle = 0.0
                for segment, bend in zip(lengths[finger], flex):
                    angle += float(bend)
                    direction = np.asarray(
                        [
                            mirror * np.sin(spread) * segment,
                            np.cos(angle) * segment,
                            -np.sin(angle) * segment,
                        ],
                        dtype=np.float32,
                    )
                    position = position + direction
                    points.append(wrist[:3] + rotation @ position)
            if hand_index == 0:
                output[row, 0:3] = wrist[:3]
                output[row, 9:69] = np.asarray(points).reshape(-1)
            else:
                output[row, 69:72] = wrist[:3]
                output[row, 78:138] = np.asarray(points).reshape(-1)
    return output


def add_delta_q(action_62d: np.ndarray, delta_q_44d: np.ndarray) -> np.ndarray:
    """Compliant pose ``q_cmp = q_obs + delta_q``, wrist untouched."""
    action = np.asarray(action_62d, dtype=np.float32)
    delta = np.asarray(delta_q_44d, dtype=np.float32)
    if delta.ndim != 2 or delta.shape[-1] != JOINT_COUNT:
        raise ValueError(
            f"delta_q must have shape [T,{JOINT_COUNT}], got {delta.shape}"
        )
    if delta.shape[0] != action.shape[0]:
        raise ValueError(
            f"delta_q has {delta.shape[0]} steps but action has {action.shape[0]}"
        )
    commanded = action[:, :ACTION_DIM].copy()
    commanded[:, JOINT_SLICE] += delta
    return commanded


def keypoint_displacement(
    hand_138d: np.ndarray, overlay_hand_138d: np.ndarray
) -> np.ndarray:
    """Per-keypoint distance between two hand poses, in metres."""
    from dexterity.rendering.hand_skeleton import extract_hand_keypoints

    return np.linalg.norm(
        extract_hand_keypoints(np.asarray(overlay_hand_138d, dtype=np.float32))
        - extract_hand_keypoints(np.asarray(hand_138d, dtype=np.float32)),
        axis=-1,
    )


__all__ = [
    "ACTION_DIM",
    "IDENTITY_HIP_POSE_9D",
    "JOINT_COUNT",
    "action_62d_to_hand138",
    "add_delta_q",
    "hip_anchors_to_chunk0",
    "identity_hip_anchors",
    "keypoint_displacement",
]
