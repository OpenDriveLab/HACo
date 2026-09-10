#!/usr/bin/env python3
"""Shared SharpA62 conversion and rendering for baseline inference videos."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
from PIL import Image
import torch

from dexterity.data.posttrain import GCC_TO_TREX_HAND
from dexterity.runtime.sharpa_kinematics import (
    rot6d_to_mat,
    sharpa62_wrist_rel_eef_to_abs,
)
from dexterity.utils.evaluation import anchors_for_split, evenly_spaced


def select_video_anchors(store, split: str, horizon: int, count: int):
    eligible = [
        anchor
        for anchor in anchors_for_split(store, split)
        if anchor.frame_index + horizon <= store.episodes[anchor.episode_index].length
    ]
    if len(eligible) < count:
        raise ValueError(
            f"split={split} only has {len(eligible)} anchors with horizon={horizon}"
        )
    return evenly_spaced(eligible, count)


def denormalize_trex(values: np.ndarray, dataset) -> np.ndarray:
    return np.where(
        dataset.action_mask,
        0.5 * (values + 1.0) * (dataset.action_max - dataset.action_min)
        + dataset.action_min,
        values,
    ).astype(np.float32)


def denormalize_vitac(values: np.ndarray, dataset) -> np.ndarray:
    return (
        values * dataset.stats["action_std"] + dataset.stats["action_mean"]
    ).astype(np.float32)


def trex_relative_to_absolute(
    action_native: np.ndarray, anchor_state_gcc: np.ndarray
) -> np.ndarray:
    """Convert T-Rex [L wrist,L q22,R wrist,R q22] to absolute GCC order."""
    native = np.asarray(action_native, dtype=np.float32)
    native_q44 = np.concatenate((native[:, 9:31], native[:, 40:62]), axis=-1)
    gcc_q44 = np.empty_like(native_q44)
    gcc_q44[:, GCC_TO_TREX_HAND] = native_q44
    relative_gcc = np.concatenate(
        (native[:, :9], native[:, 31:40], gcc_q44), axis=-1
    )
    return sharpa62_wrist_rel_eef_to_abs(
        relative_gcc,
        np.asarray(anchor_state_gcc, dtype=np.float32)[None],
        chunk_size=len(relative_gcc),
    )


def vitac_relative_to_absolute(
    action_relative: np.ndarray, anchor_state_gcc: np.ndarray
) -> np.ndarray:
    relative = np.asarray(action_relative, dtype=np.float32)
    return sharpa62_wrist_rel_eef_to_abs(
        relative,
        np.asarray(anchor_state_gcc, dtype=np.float32)[None],
        chunk_size=len(relative),
    )


def save_input_image(image_tensor: torch.Tensor, path: Path) -> None:
    image = image_tensor.detach().float().cpu()
    while image.ndim > 3:
        image = image[0]
    array = (
        image.clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def rotation_error_degrees(prediction: np.ndarray, target: np.ndarray) -> float:
    angles = []
    for pred, gt in zip(prediction, target, strict=True):
        for offset in (3, 12):
            relative = rot6d_to_mat(pred[offset : offset + 6]) @ rot6d_to_mat(
                gt[offset : offset + 6]
            ).T
            cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
            angles.append(np.degrees(np.arccos(cosine)))
    return float(np.mean(angles))


def action_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    difference = np.asarray(prediction) - np.asarray(target)
    return {
        "action_mse_62d": float(np.mean(difference**2)),
        "wrist_xyz_mse": float(
            np.mean(np.concatenate((difference[:, :3], difference[:, 9:12]), axis=-1) ** 2)
        ),
        "joint_q44_mae": float(np.mean(np.abs(difference[:, 18:62]))),
        "wrist_rotation_mean_angle_deg": rotation_error_degrees(prediction, target),
    }


def render_motion_pair(
    *,
    output_dir: Path,
    prediction: np.ndarray,
    target: np.ndarray,
    renderer_python: str,
    wrench_gt: np.ndarray | None = None,
    wrench_valid_gt: np.ndarray | None = None,
    wrench_pred: np.ndarray | None = None,
    wrench_valid_pred: np.ndarray | None = None,
) -> dict[str, str]:
    """Render a GT/prediction action pair through the shared paired-motion CLI.

    Supplying a predicted wrench turns the tactile gauges into a
    GT-versus-prediction comparison, exactly as during training.
    """
    from dexterity.rendering.action_kinematics import IDENTITY_HIP_POSE_9D

    output_dir.mkdir(parents=True, exist_ok=True)
    motion_path = output_dir / "motion_pair.npz"
    payload: dict[str, np.ndarray] = {
        "action_62d_gt": np.asarray(target, dtype=np.float32)[:, :62],
        "action_62d_pred": np.asarray(prediction, dtype=np.float32)[:, :62],
        "hip_pose_9d": IDENTITY_HIP_POSE_9D[None],
        "chunk_size": np.asarray([len(prediction)], dtype=np.int32),
    }
    if wrench_gt is not None:
        payload["wrench_gt"] = np.asarray(wrench_gt, dtype=np.float32)
        payload["wrench_valid_gt"] = (
            np.ones(payload["wrench_gt"].shape[:2], dtype=bool)
            if wrench_valid_gt is None
            else np.asarray(wrench_valid_gt, dtype=bool)
        )
        if wrench_pred is not None:
            payload["wrench_pred"] = np.asarray(wrench_pred, dtype=np.float32)
            payload["wrench_valid_pred"] = (
                np.ones(payload["wrench_pred"].shape[:2], dtype=bool)
                if wrench_valid_pred is None
                else np.asarray(wrench_valid_pred, dtype=bool)
            )
    np.savez_compressed(motion_path, **payload)
    command = [
        renderer_python,
        "-m",
        "dexterity.rendering.cli.paired_motion_video",
        "--input-npz",
        str(motion_path),
        "--out-dir",
        str(output_dir),
        "--fps",
        "15",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}"
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (output_dir / "renderer.log").write_text(
        completed.stdout or "", encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"renderer failed with code {completed.returncode}; "
            f"see {output_dir / 'renderer.log'}"
        )
    return {
        "motion_pair": str(motion_path.resolve()),
        "gt_video": str((output_dir / "viz__gt_hand_motion.mp4").resolve()),
        "pred_video": str((output_dir / "viz__pred_hand_motion.mp4").resolve()),
        "render_report": str((output_dir / "render_report.json").resolve()),
        "renderer_log": str((output_dir / "renderer.log").resolve()),
    }


ROOT = Path(__file__).resolve().parents[2]


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
