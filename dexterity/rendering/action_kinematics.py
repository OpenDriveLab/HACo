"""Turn policy action vectors into the 138-D hand pose the skeleton renderer draws.

Every model in this repository emits wrist + joint actions in the same 62-D
layout (18-D bimanual wrist eef + 44-D hand joints), so the forward kinematics
below are shared by all of them. Models that additionally predict an active
compliance residual supply ``delta_q = q_cmp - q_obs``; ``add_delta_q`` builds
the compliant command that the renderer overlays on the observed pose.
"""

from __future__ import annotations

import numpy as np

from dexterity.runtime.sharpa_kinematics import (
    rot6d_to_mat,
    sharpa62_raw_to_hand138,
)

ACTION_DIM = 62
JOINT_SLICE = slice(18, 62)
JOINT_COUNT = 44

# Canonical Rot6D identity: the first two columns of a 3x3 identity rotation.
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
    """Canonical absolute EEF actions -> exact URDF hand keypoints.

    Dataset wrists and decoded HACO actions already store rotation columns
    0 and 1. Pass them directly to FK; treating them as model rotation rows
    would transpose the wrist rotation. Missing robot assets or dependencies
    must raise rather than silently drawing a different hand model.
    Set SHARPA_URDF_DIR to the asset directory with dex_retarget_configs and
    wave_01 when those assets are outside the repository.
    """
    action = np.asarray(action_62d, dtype=np.float32)
    if action.ndim != 2 or action.shape[-1] < ACTION_DIM:
        raise ValueError(
            f"action must have shape [T,>={ACTION_DIM}], got {action.shape}"
        )
    hand = sharpa62_raw_to_hand138(
        action[:, :ACTION_DIM],
        action_space=None,
        anchor_hips=anchor_hips,
        anchor_rots=anchor_rots,
        anchor_q44=None,
        root_q6=None,
        chunk_size=max(1, int(chunk_size)),
        q_is_delta=False,
    )
    if hand is None:
        raise RuntimeError("SharpA forward kinematics returned no hand keypoints")
    return hand


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
