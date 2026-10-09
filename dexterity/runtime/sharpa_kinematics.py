"""SharpA hand kinematics and retargeting helpers."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import numpy as np

SHARPA_URDF_DIR = Path(
    os.environ.get(
        "SHARPA_URDF_DIR",
        Path(__file__).resolve().parents[2] / "third_party/sharpa-urdf",
    )
)
SHARPA_CONFIG_DIR = SHARPA_URDF_DIR / "dex_retarget_configs"
SHARPA_DOF_PER_HAND = 22
SHARPA_TOTAL_DOF = 44

SHARPA_TARGET_LINKS = {
    "left": [
        "left_hand_C_MC",
        "left_thumb_MC",
        "left_thumb_PP",
        "left_thumb_DP",
        "left_thumb_fingertip",
        "left_index_PP",
        "left_index_MP",
        "left_index_DP",
        "left_index_fingertip",
        "left_middle_PP",
        "left_middle_MP",
        "left_middle_DP",
        "left_middle_fingertip",
        "left_ring_PP",
        "left_ring_MP",
        "left_ring_DP",
        "left_ring_fingertip",
        "left_pinky_PP",
        "left_pinky_MP",
        "left_pinky_DP",
        "left_pinky_fingertip",
    ],
    "right": [
        "right_hand_C_MC",
        "right_thumb_MC",
        "right_thumb_PP",
        "right_thumb_DP",
        "right_thumb_fingertip",
        "right_index_PP",
        "right_index_MP",
        "right_index_DP",
        "right_index_fingertip",
        "right_middle_PP",
        "right_middle_MP",
        "right_middle_DP",
        "right_middle_fingertip",
        "right_ring_PP",
        "right_ring_MP",
        "right_ring_DP",
        "right_ring_fingertip",
        "right_pinky_PP",
        "right_pinky_MP",
        "right_pinky_DP",
        "right_pinky_fingertip",
    ],
}

DEXRETARGET_METHOD_TYPES = {
    "dexretarget_position": "position",
    "dexretarget_vector": "vector",
    "dexretarget_dexpilot": "dexpilot",
}


def action_space_to_sharpa_method(action_space: str | None) -> str:
    space = (action_space or "sharpa_dexretarget_position_62d").lower()
    if "vector" in space:
        return "dexretarget_vector"
    if "dexpilot" in space:
        return "dexretarget_dexpilot"
    if "egoscale" in space:
        return "egoscale"
    return "dexretarget_position"


def rot6d_to_mat(r6d: np.ndarray) -> np.ndarray:
    r6d = np.asarray(r6d, dtype=np.float32).reshape(6)
    a1, a2 = r6d[:3], r6d[3:]
    n1 = np.linalg.norm(a1)
    if n1 < 1e-8:
        return np.eye(3, dtype=np.float32)
    b1 = a1 / n1
    b2 = a2 - np.dot(b1, a2) * b1
    n2 = np.linalg.norm(b2)
    if n2 < 1e-8:
        seed = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(seed, b1))) > 0.95:
            seed = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b2 = seed - np.dot(b1, seed) * b1
        n2 = np.linalg.norm(b2)
    b2 = b2 / max(n2, 1e-8)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=1).astype(np.float32)


def mat_to_rot6d(rot: np.ndarray) -> np.ndarray:
    rot = np.asarray(rot, dtype=np.float32).reshape(3, 3)
    return np.concatenate([rot[:, 0], rot[:, 1]]).astype(np.float32)


def model_wrist_to_robot_wire(action_62d: np.ndarray) -> np.ndarray:
    """Convert current model wrist rotations to the fixed robot wire format."""
    action = np.asarray(action_62d, dtype=np.float32)
    if action.ndim != 2 or action.shape[-1] < 62:
        raise ValueError(f"action must have shape [T,>=62], got {action.shape}")
    output = action.copy()
    for start in (3, 12):
        rows = action[:, start : start + 6].reshape(-1, 2, 3)
        first = rows[:, 0]
        second = rows[:, 1]
        first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-8)
        second = second - np.sum(first * second, axis=-1, keepdims=True) * first
        second = second / np.maximum(
            np.linalg.norm(second, axis=-1, keepdims=True), 1e-8
        )
        rotation = np.stack((first, second, np.cross(first, second)), axis=-2)
        output[:, start : start + 6] = np.concatenate(
            (rotation[:, :, 0], rotation[:, :, 1]), axis=-1
        )
    return output


def relative_eef_to_absolute(
    relative_eef: np.ndarray,
    reference_eef: np.ndarray,
) -> np.ndarray:
    """Compose relative XYZ+Rot6D poses with one reference EEF pose."""

    relative = np.asarray(relative_eef)
    reference = np.asarray(reference_eef)
    if relative.dtype != np.float32:
        raise ValueError(f"relative_eef must have dtype float32, got {relative.dtype}")
    if reference.dtype != np.float32:
        raise ValueError(
            f"reference_eef must have dtype float32, got {reference.dtype}"
        )
    if relative.ndim not in (1, 2) or relative.shape[-1] != 9:
        raise ValueError(
            f"relative_eef must have shape (9,) or (T,9), got {relative.shape}"
        )
    if reference.shape != (9,):
        raise ValueError(f"reference_eef must have shape (9,), got {reference.shape}")
    if not np.all(np.isfinite(relative)) or not np.all(np.isfinite(reference)):
        raise ValueError("relative_eef and reference_eef must contain finite values")

    relative_rows = relative.reshape(-1, 9)
    absolute_rows = np.empty_like(relative_rows)
    reference_position = reference[:3]
    reference_rotation = rot6d_to_mat(reference[3:9])
    for index, relative_pose in enumerate(relative_rows):
        relative_position = relative_pose[:3]
        relative_rotation = rot6d_to_mat(relative_pose[3:9])
        absolute_rows[index, :3] = (
            reference_position + reference_rotation @ relative_position
        )
        absolute_rows[index, 3:9] = mat_to_rot6d(reference_rotation @ relative_rotation)
    return absolute_rows.reshape(relative.shape)


def absolute_eef_to_relative(
    absolute_eef: np.ndarray,
    reference_eef: np.ndarray,
) -> np.ndarray:
    """Express absolute XYZ+Rot6D poses in one reference EEF frame."""

    absolute = np.asarray(absolute_eef)
    reference = np.asarray(reference_eef)
    if absolute.dtype != np.float32:
        raise ValueError(f"absolute_eef must have dtype float32, got {absolute.dtype}")
    if reference.dtype != np.float32:
        raise ValueError(
            f"reference_eef must have dtype float32, got {reference.dtype}"
        )
    if absolute.ndim not in (1, 2) or absolute.shape[-1] != 9:
        raise ValueError(
            f"absolute_eef must have shape (9,) or (T,9), got {absolute.shape}"
        )
    if reference.shape != (9,):
        raise ValueError(f"reference_eef must have shape (9,), got {reference.shape}")
    if not np.all(np.isfinite(absolute)) or not np.all(np.isfinite(reference)):
        raise ValueError("absolute_eef and reference_eef must contain finite values")

    absolute_rows = absolute.reshape(-1, 9)
    relative_rows = np.empty_like(absolute_rows)
    reference_position = reference[:3]
    reference_rotation = rot6d_to_mat(reference[3:9])
    for index, absolute_pose in enumerate(absolute_rows):
        absolute_position = absolute_pose[:3]
        absolute_rotation = rot6d_to_mat(absolute_pose[3:9])
        relative_rows[index, :3] = reference_rotation.T @ (
            absolute_position - reference_position
        )
        relative_rows[index, 3:9] = mat_to_rot6d(
            reference_rotation.T @ absolute_rotation
        )
    return relative_rows.reshape(absolute.shape)


def sharpa62_wrist_rel_eef_to_abs(
    raw_action_62d: np.ndarray,
    anchor_state_62d: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    """Convert SharpA62 wrist18 from relative EEF deltas to anchor-frame poses.

    The two wrist 9D groups follow Isaac-GR00T's RELATIVE EEF / XYZ_ROT6D
    convention. The SharpA q44 slice remains absolute.
    """
    action = np.asarray(raw_action_62d, dtype=np.float32).copy()
    while action.ndim > 2:
        action = action[0]
    refs = np.asarray(anchor_state_62d, dtype=np.float32)
    while refs.ndim > 2:
        refs = refs[0]
    if (
        action.ndim != 2
        or action.shape[-1] < 62
        or refs.ndim != 2
        or refs.shape[-1] < 18
    ):
        return action
    for t in range(len(action)):
        ci = min(t // max(1, int(chunk_size)), len(refs) - 1)
        for offset in (0, 9):
            action[t, offset : offset + 9] = relative_eef_to_absolute(
                action[t, offset : offset + 9],
                refs[ci, offset : offset + 9],
            )
    return action


def _sharpa_fingertip_links(side: str) -> list[str]:
    return [
        f"{side}_thumb_fingertip",
        f"{side}_index_fingertip",
        f"{side}_middle_fingertip",
        f"{side}_ring_fingertip",
        f"{side}_pinky_fingertip",
    ]


def _sharpa_dexpilot_human_pairs() -> np.ndarray:
    human_tip_indices = np.asarray([4, 8, 12, 16, 20], dtype=np.int64)
    origin_idx: list[int] = []
    task_idx: list[int] = []
    for i in range(1, 6):
        for j in range(i + 1, 6):
            origin_idx.append(j)
            task_idx.append(i)
    for i in range(1, 6):
        origin_idx.append(0)
        task_idx.append(i)
    return np.stack(
        [human_tip_indices[origin_idx], human_tip_indices[task_idx]], axis=0
    )


@lru_cache(maxsize=8)
def _build_dex_retargeter(side: str, method: str):
    from dex_retargeting.retargeting_config import RetargetingConfig

    RetargetingConfig.set_default_urdf_dir(str(SHARPA_URDF_DIR))
    retarget_type = DEXRETARGET_METHOD_TYPES.get(method, method)
    override: dict[str, Any] = {"type": retarget_type}
    if retarget_type == "vector":
        tips = _sharpa_fingertip_links(side)
        override.update(
            {
                "target_link_names": None,
                "target_origin_link_names": [f"{side}_hand_C_MC"] * len(tips),
                "target_task_link_names": tips,
                "target_link_human_indices": np.stack(
                    [
                        np.zeros(len(tips), dtype=np.int64),
                        np.asarray([4, 8, 12, 16, 20], dtype=np.int64),
                    ],
                    axis=0,
                ),
                "scaling_factor": 1.0,
            }
        )
    elif retarget_type == "dexpilot":
        override.update(
            {
                "target_link_names": None,
                "target_link_human_indices": _sharpa_dexpilot_human_pairs(),
                "wrist_link_name": f"{side}_hand_C_MC",
                "finger_tip_link_names": _sharpa_fingertip_links(side),
                "scaling_factor": 1.0,
            }
        )
    config_path = SHARPA_CONFIG_DIR / f"sharpa_wave_{side}.yml"
    if config_path.is_file():
        cfg = RetargetingConfig.load_from_file(str(config_path), override=override)
    else:
        # The public SharpA assets contain the URDFs but no retargeting YAML.
        # Match the original 21-link configuration without changing the model.
        cfg = RetargetingConfig.from_dict(
            {
                "type": "position",
                "urdf_path": f"wave_01/{side}_sharpa_wave/{side}_sharpa_wave.urdf",
                "target_joint_names": None,
                "target_link_names": SHARPA_TARGET_LINKS[side],
                "target_link_human_indices": list(range(21)),
                "add_dummy_free_joint": True,
                "low_pass_alpha": 1,
            },
            override=override,
        )
    return cfg, cfg.build()


def _dex_reference(kp21_wrist: np.ndarray, cfg) -> np.ndarray:
    human_idx = np.asarray(cfg.target_link_human_indices)
    if cfg.type == "position":
        return kp21_wrist[human_idx]
    return kp21_wrist[human_idx[1]] - kp21_wrist[human_idx[0]]


def sharpa_root_q6_from_kp21(
    kp21_wrist: np.ndarray,
    *,
    side: str,
    method: str,
) -> np.ndarray:
    if method == "egoscale":
        return np.zeros(6, dtype=np.float32)
    cfg, retargeter = _build_dex_retargeter(side, method)
    q_full = retargeter.retarget(
        _dex_reference(np.asarray(kp21_wrist, dtype=np.float32), cfg)
    )
    return np.asarray(q_full[:6], dtype=np.float32)


def sharpa_fk_q22_in_wrist(
    q22: np.ndarray,
    *,
    side: str,
    method: str,
    root_q6: np.ndarray | None,
) -> np.ndarray:
    if method == "egoscale":
        raise NotImplementedError("EgoScale runtime FK needs a dedicated root model.")
    _cfg, retargeter = _build_dex_retargeter(side, method)
    q_full = np.zeros(28, dtype=np.float32)
    if root_q6 is not None:
        q_full[:6] = np.asarray(root_q6, dtype=np.float32).reshape(-1)[:6]
    q_full[-22:] = np.asarray(q22, dtype=np.float32).reshape(-1)[:22]
    robot = retargeter.optimizer.robot
    robot.compute_forward_kinematics(q_full)
    out = np.zeros((21, 3), dtype=np.float32)
    for i, link_name in enumerate(SHARPA_TARGET_LINKS[side]):
        pose = robot.get_link_pose(robot.get_link_index(link_name))
        out[i] = pose[:3, 3].astype(np.float32)
    return out


def kp21_in_wrist_frame_from_components(
    wrist18: np.ndarray,
    fingers120: np.ndarray,
    *,
    side: str,
) -> np.ndarray:
    wrist18 = np.asarray(wrist18, dtype=np.float32).reshape(18)
    fingers120 = np.asarray(fingers120, dtype=np.float32).reshape(120)
    if side == "left":
        wp = wrist18[0:3]
        wr = wrist18[3:9]
        fingers = fingers120[0:60].reshape(20, 3)
    else:
        wp = wrist18[9:12]
        wr = wrist18[12:18]
        fingers = fingers120[60:120].reshape(20, 3)
    rot = rot6d_to_mat(wr)
    kp = np.concatenate([wp[None], fingers], axis=0)
    return (rot.T @ (kp - wp[None]).T).T.astype(np.float32)


def hand_keypoints_to_legacy_138(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    T = left.shape[0]
    out = np.zeros((T, 138), dtype=np.float32)
    out[:, 0:3] = left[:, 0]
    out[:, 9:69] = left[:, 1:21].reshape(T, 60)
    out[:, 69:72] = right[:, 0]
    out[:, 78:138] = right[:, 1:21].reshape(T, 60)
    return out


def sharpa62_raw_to_hand138(
    raw_action_62d: np.ndarray,
    *,
    action_space: str | None,
    anchor_hips: np.ndarray | None,
    anchor_rots: np.ndarray | None,
    anchor_q44: np.ndarray | None,
    root_q6: np.ndarray | None,
    chunk_size: int,
    q_is_delta: bool,
) -> np.ndarray | None:
    action = np.asarray(raw_action_62d, dtype=np.float32)
    while action.ndim > 2:
        action = action[0]
    if action.ndim != 2 or action.shape[-1] < 62:
        return None
    action = action[:, :62]
    T = len(action)
    method = action_space_to_sharpa_method(action_space)
    if method == "egoscale":
        print("[sharpa_kinematics] runtime motion viz currently skips egoscale FK")
        return None

    hips = np.zeros((max(1, (T + chunk_size - 1) // chunk_size), 3), dtype=np.float32)
    if anchor_hips is not None:
        hips = np.asarray(anchor_hips, dtype=np.float32)
        while hips.ndim > 2:
            hips = hips[0]
    rots = np.repeat(np.eye(3, dtype=np.float32)[None], len(hips), axis=0)
    if anchor_rots is not None:
        rots = np.asarray(anchor_rots, dtype=np.float32)
        while rots.ndim > 3:
            rots = rots[0]
    q_anchor = None if anchor_q44 is None else np.asarray(anchor_q44, dtype=np.float32)
    if q_anchor is not None:
        while q_anchor.ndim > 2:
            q_anchor = q_anchor[0]
    roots = None if root_q6 is None else np.asarray(root_q6, dtype=np.float32)
    if roots is not None:
        while roots.ndim > 3:
            roots = roots[0]

    left = np.zeros((T, 21, 3), dtype=np.float32)
    right = np.zeros((T, 21, 3), dtype=np.float32)
    for t in range(T):
        ci = min(t // max(1, int(chunk_size)), len(hips) - 1, len(rots) - 1)
        q44 = action[t, 18:62].copy()
        if q_is_delta and q_anchor is not None and len(q_anchor) > 0:
            q44 += q_anchor[min(ci, len(q_anchor) - 1), :44]
        for hand_i, side, offset, q_slice, dst in (
            (0, "left", 0, slice(0, 22), left),
            (1, "right", 9, slice(22, 44), right),
        ):
            root = None
            if roots is not None and len(roots) > 0:
                root_idx = t if len(roots) >= T else ci
                root = roots[min(root_idx, len(roots) - 1), hand_i]
            wrist_pos = action[t, offset : offset + 3]
            wrist_rot = rot6d_to_mat(action[t, offset + 3 : offset + 9])
            kp_wrist = sharpa_fk_q22_in_wrist(
                q44[q_slice], side=side, method=method, root_q6=root
            )
            kp_chunk = (wrist_rot @ kp_wrist.T).T + wrist_pos[None, :]
            dst[t] = (rots[ci] @ kp_chunk.T).T + hips[ci][None, :]
    return hand_keypoints_to_legacy_138(left, right)


def sharpa62_absolute_to_hand138_in_hip0(
    abs_action_62d: np.ndarray,
    *,
    action_space: str | None,
    hip_pose_9d: np.ndarray | None,
    root_q6: np.ndarray | None,
) -> np.ndarray | None:
    """Convert absolute SharpA62 actions into legacy hand keypoints in hip-0."""

    action = np.asarray(abs_action_62d, dtype=np.float32)
    while action.ndim > 2:
        action = action[0]
    if action.ndim != 2 or action.shape[-1] < 62:
        return None
    action = action[:, :62].copy()
    if hip_pose_9d is not None:
        hip = np.asarray(hip_pose_9d, dtype=np.float32).reshape(-1)
        t_world_hip0 = hip[:3]
        r_hip0_world = rot6d_to_mat(hip[3:9]).T
        for pos_start, rot_start in ((0, 3), (9, 12)):
            action[:, pos_start : pos_start + 3] = (
                r_hip0_world
                @ (action[:, pos_start : pos_start + 3] - t_world_hip0[None]).T
            ).T
            for i in range(len(action)):
                r_rel = r_hip0_world @ rot6d_to_mat(
                    action[i, rot_start : rot_start + 6]
                )
                action[i, rot_start : rot_start + 6] = np.concatenate(
                    [r_rel[:, 0], r_rel[:, 1]]
                )
    return sharpa62_raw_to_hand138(
        action,
        action_space=action_space,
        anchor_hips=np.zeros((1, 3), dtype=np.float32),
        anchor_rots=np.eye(3, dtype=np.float32)[None],
        anchor_q44=None,
        root_q6=root_q6,
        chunk_size=max(1, len(action)),
        q_is_delta=False,
    )
