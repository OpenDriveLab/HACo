"""Render ground-truth hand motion directly from a HACo LeRobot episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from dexterity.rendering.action_kinematics import identity_hip_anchors
from dexterity.rendering.cli.paired_motion_video import (
    GT_FILENAME,
    GT_TITLE,
    _render_side,
    _write_video,
)
from dexterity.runtime.sharpa_kinematics import SHARPA_URDF_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.episode < 0 or args.start_frame < 0:
        parser.error("--episode and --start-frame must be nonnegative")
    if args.frames is not None and args.frames < 1:
        parser.error("--frames must be positive")

    info = json.loads((args.dataset / "meta/info.json").read_text())
    path_args = {
        "episode_index": args.episode,
        "episode_chunk": args.episode // int(info["chunks_size"]),
    }
    parquet_path = args.dataset / info["data_path"].format(**path_args)
    sensor_path = args.dataset / info["sensor_path"].format(**path_args)
    action = np.asarray(
        pq.read_table(parquet_path, columns=["action"])["action"].to_pylist(),
        dtype=np.float32,
    )
    if action.ndim != 2 or action.shape[1] != 150:
        raise ValueError(f"Expected HACo actions [T,150], got {action.shape}")
    stop = len(action) if args.frames is None else args.start_frame + args.frames
    if not args.start_frame < stop <= len(action):
        raise ValueError(
            f"Frame range [{args.start_frame}, {stop}) exceeds episode length "
            f"{len(action)}"
        )
    selected = slice(args.start_frame, stop)
    with np.load(sensor_path, allow_pickle=False) as sensors:
        wrench = np.asarray(sensors["tactile_wrench"], dtype=np.float32)
        wrench_valid = np.asarray(sensors["tactile_wrench_valid_mask"], dtype=bool)
    if wrench.shape != (len(action), 10, 6):
        raise ValueError("Tactile wrench must be aligned with actions as [T,10,6]")
    if wrench_valid.shape != (len(action), 10):
        raise ValueError("Tactile wrench validity must have shape [T,10]")

    gt = action[selected]
    anchor_hips, anchor_rots = identity_hip_anchors()
    frames, details = _render_side(
        action_62d=gt[:, :62],
        delta_q_44d=gt[:, 106:150],
        wrench=wrench[selected],
        wrench_valid=wrench_valid[selected],
        wrench_pred=None,
        wrench_pred_valid=None,
        anchor_hips=anchor_hips,
        anchor_rots=anchor_rots,
        chunk_size=len(gt),
        title_prefix=GT_TITLE,
    )
    fps = int(info["fps"])
    output_path = args.out_dir / GT_FILENAME
    _write_video(output_path, frames, fps)
    report = {
        "dataset": str(args.dataset.resolve()),
        "episode": args.episode,
        "start_frame": args.start_frame,
        "frames": len(gt),
        "fps": fps,
        "kinematics": {
            "backend": "sharpa_urdf",
            "urdf_dir": str(SHARPA_URDF_DIR.resolve()),
            "wrist_rotation": "matrix_columns_0_and_1",
            "extra_wrist_transform": False,
        },
        "rendered": {"gt": details},
    }
    (args.out_dir / "render_report.json").write_text(json.dumps(report, indent=2))
    print(output_path)


if __name__ == "__main__":
    main()
