"""Render a GT/prediction action pair to two mp4 files.

This is the single subprocess entry point for open-loop evaluation videos. The
layers it draws are decided by which arrays the caller put in the npz, never by
which model produced them:

    action_62d_gt / action_62d_pred    required   (T, 62)
    chunk_size                         required   int
    hip_pose_9d                        optional   (n_chunks, 9); absent -> identity
    delta_q_44d_gt / delta_q_44d_pred  optional   (T, 44)  -> L2
    wrench_gt / wrench_valid_gt        optional   (T,10,6)/(T,10) -> L3
    wrench_pred / wrench_valid_pred    optional   (T,10,6)/(T,10) -> L3 shows
                                                  the signed GT->pred delta

Writes ``viz__gt_hand_motion.mp4`` and ``viz__pred_hand_motion.mp4`` plus a
``render_report.json`` describing what was drawn.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio
import numpy as np

from dexterity.rendering.action_kinematics import (
    JOINT_COUNT,
    action_62d_to_hand138,
    add_delta_q,
    hip_anchors_to_chunk0,
    keypoint_displacement,
)
from dexterity.rendering.hand_skeleton import render_hand_skeleton
from dexterity.runtime.sharpa62 import (
    MODEL_JOINT_ORDER,
    MODEL_TACTILE_ORDER,
)

DELTA_Q_FIXED_LIMIT_RAD = 0.20
TACTILE_FORCE_FIXED_LIMIT_N = 25.0
TACTILE_TORQUE_FIXED_LIMIT_NM = 0.40

GT_TITLE = "GT hand motion"
PRED_TITLE = "Pred hand motion"
GT_FILENAME = "viz__gt_hand_motion.mp4"
PRED_FILENAME = "viz__pred_hand_motion.mp4"


def _optional(archive, key: str, dtype) -> np.ndarray | None:
    if key not in archive.files:
        return None
    return np.asarray(archive[key], dtype=dtype)


def _require_steps(name: str, array: np.ndarray | None, steps: int) -> None:
    if array is not None and array.shape[0] != steps:
        raise ValueError(
            f"{name} must have {steps} steps to match the action, got {array.shape[0]}"
        )


def _write_video(path: Path, frames: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), list(frames), fps=fps, codec="libx264")


def _render_side(
    *,
    action_62d: np.ndarray,
    delta_q_44d: np.ndarray | None,
    wrench: np.ndarray | None,
    wrench_valid: np.ndarray | None,
    wrench_pred: np.ndarray | None,
    wrench_pred_valid: np.ndarray | None,
    anchor_hips: np.ndarray,
    anchor_rots: np.ndarray,
    chunk_size: int,
    title_prefix: str,
) -> tuple[np.ndarray, dict]:
    hand = action_62d_to_hand138(
        action_62d,
        anchor_hips=anchor_hips,
        anchor_rots=anchor_rots,
        chunk_size=chunk_size,
    )
    overlay_hand = None
    report: dict = {}
    if delta_q_44d is not None:
        overlay_hand = action_62d_to_hand138(
            add_delta_q(action_62d, delta_q_44d),
            anchor_hips=anchor_hips,
            anchor_rots=anchor_rots,
            chunk_size=chunk_size,
        )
        displacement = keypoint_displacement(hand, overlay_hand)
        report["delta_q_abs"] = {
            "mean_rad": float(np.mean(np.abs(delta_q_44d))),
            "max_rad": float(np.max(np.abs(delta_q_44d))),
        }
        report["keypoint_displacement_m"] = {
            "mean": float(np.mean(displacement)),
            "p95": float(np.quantile(displacement, 0.95)),
            "max": float(np.max(displacement)),
        }
    if wrench is not None:
        report["wrench"] = {
            "valid_ratio": (
                1.0 if wrench_valid is None else float(np.mean(wrench_valid))
            ),
            "has_prediction": wrench_pred is not None,
        }
        if wrench_pred is not None and wrench_pred_valid is not None:
            report["wrench"]["pred_valid_ratio"] = float(np.mean(wrench_pred_valid))

    frames = render_hand_skeleton(
        hand,
        anchor_hips,
        chunk_size=chunk_size,
        title_prefix=title_prefix,
        units_label="m",
        overlay_hand_138d=overlay_hand,
        primary_label="q_obs (solid blue)",
        overlay_label="q_cmp = q_obs + delta_q (dashed orange)",
        joint_slider_values=delta_q_44d,
        joint_slider_names=MODEL_JOINT_ORDER if delta_q_44d is not None else None,
        joint_slider_fixed_limit=(
            DELTA_Q_FIXED_LIMIT_RAD if delta_q_44d is not None else None
        ),
        tactile_wrench_values=wrench,
        tactile_wrench_valid=wrench_valid,
        tactile_wrench_pred_values=wrench_pred,
        tactile_wrench_pred_valid=wrench_pred_valid,
        tactile_finger_names=MODEL_TACTILE_ORDER if wrench is not None else None,
        tactile_force_fixed_limit_n=TACTILE_FORCE_FIXED_LIMIT_N,
        tactile_torque_fixed_limit_nm=TACTILE_TORQUE_FIXED_LIMIT_NM,
    )
    if frames is None:
        raise RuntimeError(f"skeleton renderer returned no frames for {title_prefix}")
    return frames, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    with np.load(args.input_npz, allow_pickle=False) as archive:
        gt_action = np.asarray(archive["action_62d_gt"], dtype=np.float32)
        pred_action = np.asarray(archive["action_62d_pred"], dtype=np.float32)
        chunk_size = max(1, int(np.asarray(archive["chunk_size"]).reshape(-1)[0]))
        hip_pose = _optional(archive, "hip_pose_9d", np.float32)
        gt_delta_q = _optional(archive, "delta_q_44d_gt", np.float32)
        pred_delta_q = _optional(archive, "delta_q_44d_pred", np.float32)
        wrench_gt = _optional(archive, "wrench_gt", np.float32)
        wrench_valid_gt = _optional(archive, "wrench_valid_gt", bool)
        wrench_pred = _optional(archive, "wrench_pred", np.float32)
        wrench_valid_pred = _optional(archive, "wrench_valid_pred", bool)

    if gt_action.shape != pred_action.shape:
        raise ValueError(
            "GT and prediction actions must have the same shape, got "
            f"{gt_action.shape} and {pred_action.shape}"
        )
    if gt_action.ndim != 2 or gt_action.shape[-1] < 62:
        raise ValueError(f"actions must have shape [T,>=62], got {gt_action.shape}")
    steps = gt_action.shape[0]
    if (gt_delta_q is None) != (pred_delta_q is None):
        raise ValueError(
            "delta_q must be supplied for both GT and prediction or for neither"
        )
    for name, array in (
        ("delta_q_44d_gt", gt_delta_q),
        ("delta_q_44d_pred", pred_delta_q),
        ("wrench_gt", wrench_gt),
        ("wrench_valid_gt", wrench_valid_gt),
        ("wrench_pred", wrench_pred),
        ("wrench_valid_pred", wrench_valid_pred),
    ):
        _require_steps(name, array, steps)
    for name, array in (
        ("delta_q_44d_gt", gt_delta_q),
        ("delta_q_44d_pred", pred_delta_q),
    ):
        if array is not None and array.shape[-1] != JOINT_COUNT:
            raise ValueError(
                f"{name} must have shape [T,{JOINT_COUNT}], got {array.shape}"
            )
    if wrench_pred is not None and wrench_gt is None:
        raise ValueError("wrench_pred requires wrench_gt to compare against")

    anchor_hips, anchor_rots = hip_anchors_to_chunk0(
        np.zeros((0, 9), dtype=np.float32) if hip_pose is None else hip_pose
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "fps": args.fps,
        "frames": steps,
        "chunk_size": chunk_size,
        "layers": {
            "hand_skeleton": True,
            "delta_q": gt_delta_q is not None,
            "wrench": wrench_gt is not None,
            "wrench_prediction": wrench_pred is not None,
        },
        "rendered": {},
    }
    # GT keeps the observed wrench without a prediction overlay; only the
    # prediction video contrasts predicted wrench against ground truth.
    sides = (
        ("gt", GT_TITLE, GT_FILENAME, gt_action, gt_delta_q, None, None),
        (
            "pred",
            PRED_TITLE,
            PRED_FILENAME,
            pred_action,
            pred_delta_q,
            wrench_pred,
            wrench_valid_pred,
        ),
    )
    for name, title, filename, action, delta_q, side_pred, side_pred_valid in sides:
        frames, side_report = _render_side(
            action_62d=action,
            delta_q_44d=delta_q,
            wrench=wrench_gt,
            wrench_valid=wrench_valid_gt,
            wrench_pred=side_pred,
            wrench_pred_valid=side_pred_valid,
            anchor_hips=anchor_hips,
            anchor_rots=anchor_rots,
            chunk_size=chunk_size,
            title_prefix=title,
        )
        output = args.out_dir / filename
        _write_video(output, frames, args.fps)
        report["rendered"][name] = side_report
        print(f"wrote {output}")

    report_path = args.out_dir / "render_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
