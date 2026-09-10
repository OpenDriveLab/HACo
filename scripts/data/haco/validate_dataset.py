"""Validate the public HACO LeRobot dataset contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

FPS = 30
STATE_DIM = 62
ACTION_DIM = 150
HISTORY = 8
HORIZON = 40
ACTION_ORDER = ["wrist_obs18", "q_obs44", "q_cmp44", "delta_q44"]


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _video_frames(path: Path) -> int:
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


def validate_dataset(root: Path) -> dict[str, int]:
    root = root.expanduser().resolve()
    info = _json(root / "meta/info.json")
    modality = _json(root / "meta/modality.json")
    required_meta = (
        "episodes.jsonl",
        "tasks.jsonl",
        "stats.json",
        "sensor_stats.json",
        "stats_provenance.json",
        "relative_stats.json",
    )
    for name in required_meta:
        if not (root / "meta" / name).is_file():
            raise FileNotFoundError(root / "meta" / name)
    if info.get("fps") != FPS:
        raise ValueError(f"fps must be {FPS}")
    if info.get("state_dim") != STATE_DIM or info.get("action_dim") != ACTION_DIM:
        raise ValueError("state/action dimensions must be 62/150")
    if info.get("action_order") != ACTION_ORDER:
        raise ValueError(f"action_order must be {ACTION_ORDER}")
    if info.get("delta_q_definition") != "q_cmp - q_obs":
        raise ValueError("delta_q_definition must be 'q_cmp - q_obs'")

    action_modality = modality.get("action", {})
    expected_slices = {
        "left_wrist_eef": (0, 9),
        "right_wrist_eef": (9, 18),
        "left_hand_q_obs": (18, 40),
        "right_hand_q_obs": (40, 62),
        "left_hand_q_cmp": (62, 84),
        "right_hand_q_cmp": (84, 106),
        "left_hand_delta_q": (106, 128),
        "right_hand_delta_q": (128, 150),
    }
    for name, (start, end) in expected_slices.items():
        item = action_modality.get(name)
        if item is None or (item.get("start"), item.get("end")) != (start, end):
            raise ValueError(f"bad modality slice for {name}")

    tasks = [
        json.loads(line)
        for line in (root / "meta/tasks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not tasks or any(not str(item.get("task", "")).strip() for item in tasks):
        raise ValueError("every task needs a non-empty language prompt")
    episodes = [
        json.loads(line)
        for line in (root / "meta/episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if len(episodes) != int(info.get("total_episodes", -1)):
        raise ValueError("episode count disagrees with info.json")

    total_frames = 0
    total_anchors = 0
    chunks_size = int(info.get("chunks_size", 1000))
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        chunk = episode_index // chunks_size
        stem = f"episode_{episode_index:06d}"
        table = pq.read_table(root / f"data/chunk-{chunk:03d}/{stem}.parquet")
        length = len(table)
        if length != int(episode["length"]):
            raise ValueError(f"episode {episode_index}: wrong declared length")
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if state.shape != (length, STATE_DIM) or action.shape != (length, ACTION_DIM):
            raise ValueError(f"episode {episode_index}: bad state/action shape")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"episode {episode_index}: state/action is not finite")
        q_obs, q_cmp, delta_q = action[:, 18:62], action[:, 62:106], action[:, 106:]
        if not np.allclose(q_cmp - q_obs, delta_q, rtol=1e-4, atol=1e-5):
            raise ValueError(f"episode {episode_index}: q_cmp-q_obs != delta_q")
        frame_index = table["frame_index"].to_numpy()
        timestamp = table["timestamp"].to_numpy()
        if not np.array_equal(frame_index, np.arange(length)):
            raise ValueError(f"episode {episode_index}: frame_index is not contiguous")
        if not np.allclose(timestamp, frame_index / FPS, atol=1e-8):
            raise ValueError(f"episode {episode_index}: timestamp is not 30 Hz")
        anchor_valid = table["anchor_valid"].to_numpy().astype(bool)
        strict = np.zeros(length, dtype=bool)
        strict[HISTORY : max(HISTORY, length - HORIZON)] = True
        valid_anchors = int(np.count_nonzero(anchor_valid & strict))
        if valid_anchors != int(episode["valid_anchor_count"]):
            raise ValueError(f"episode {episode_index}: wrong valid_anchor_count")

        sensor_path = root / f"sensors/episodes/{stem}.npz"
        with np.load(sensor_path, allow_pickle=False) as sensor:
            expected = {
                "tau": (length, 44),
                "tau_valid_mask": (length, 44),
                "tactile_wrench": (length, 10, 6),
                "tactile_wrench_valid_mask": (length, 10),
                "tactile_deformation": (length, 10, 240, 240),
                "tactile_deformation_valid_mask": (length, 10),
                "frame_index": (length,),
                "timestamp": (length,),
                "source_timeline_row": (length,),
            }
            for name, shape in expected.items():
                if name not in sensor or sensor[name].shape != shape:
                    raise ValueError(f"episode {episode_index}: bad sensor {name}")
            if not np.array_equal(sensor["frame_index"], frame_index):
                raise ValueError(f"episode {episode_index}: sensor frame mismatch")
            if not np.allclose(sensor["timestamp"], timestamp, atol=1e-8):
                raise ValueError(f"episode {episode_index}: sensor time mismatch")

        for video_key in (
            "observation.images.ego_view",
            "observation.images.left_wrist_view",
            "observation.images.right_wrist_view",
        ):
            path = root / f"videos/chunk-{chunk:03d}/{video_key}/{stem}.mp4"
            if _video_frames(path) != length:
                raise ValueError(
                    f"episode {episode_index}: wrong frames in {video_key}"
                )
        total_frames += length
        total_anchors += valid_anchors

    if total_frames != int(info.get("total_frames", -1)):
        raise ValueError("total frame count disagrees with info.json")
    return {"episodes": len(episodes), "frames": total_frames, "anchors": total_anchors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    result = validate_dataset(args.dataset)
    print(
        f"valid HACO dataset: {result['episodes']} episode(s), "
        f"{result['frames']} frames, {result['anchors']} training anchors"
    )


if __name__ == "__main__":
    main()
