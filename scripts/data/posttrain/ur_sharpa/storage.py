"""LeRobot artifact I/O for the timeline-based UR-SharpA converter."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dexterity.data.posttrain import (
    POSTTRAIN_JOINT_ORDER,
    POSTTRAIN_TACTILE_ORDER,
    relative_pose9_batch,
)
from scripts.data.posttrain.ur_sharpa.contract import (
    ACTION_DIM,
    CAMERA_NAMES,
    CAMERA_VIDEO_KEYS,
    DEFORMATION_SIZE,
    CanonicalEpisode,
    quantiles,
)

DATASET_SCHEMA = "sharpa.ur_sharpa_lerobot.v4_clean_30hz_three_camera"
EMBODIMENT_TAG = "real_r1_pro_sharpa_relative_eef"
RELATIVE_ACTION_HORIZON = 40
RELATIVE_STATS_SCHEMA = "sharpa.relative_action_stats.v2"
RELATIVE_STATS_FIELDS = ("mean", "std", "min", "max", "q01", "q99")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify_raw_checksums(episode: Path) -> dict[str, Any]:
    manifest = episode / "checksums.sha256"
    checked = 0
    mismatches: list[dict[str, str]] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("* ")
        path = episode / relative
        if not path.is_file():
            mismatches.append(
                {"path": relative, "expected": expected, "actual": "missing"}
            )
            continue
        actual = _sha256(path)
        checked += 1
        if actual != expected:
            mismatches.append(
                {"path": relative, "expected": expected, "actual": actual}
            )
    if mismatches:
        raise ValueError(f"{episode}: checksum failures: {mismatches}")
    return {"manifest": str(manifest), "checked_files": checked, "mismatches": []}


def fixed_list(values: np.ndarray, size: int):
    import pyarrow as pa

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != size:
        raise ValueError(f"expected [N,{size}], got {array.shape}")
    return pa.FixedSizeListArray.from_arrays(
        pa.array(array.reshape(-1), type=pa.float32()), size
    )


def probe_video(path: Path, *, count_frames: bool = False) -> dict[str, Any]:
    entries = "stream=codec_name,width,height,r_frame_rate,nb_frames"
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
    ]
    if count_frames:
        command.append("-count_frames")
        entries = "stream=codec_name,width,height,r_frame_rate,nb_read_frames,nb_frames"
    command.extend(("-show_entries", entries, "-of", "json", str(path)))
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(completed.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"{path}: expected one video stream")
    stream = streams[0]
    raw_count = (
        stream.get("nb_read_frames") if count_frames else stream.get("nb_frames")
    )
    if raw_count in (None, "N/A"):
        capture = cv2.VideoCapture(str(path))
        raw_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
    return {
        "codec": str(stream.get("codec_name", "")),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": str(stream.get("r_frame_rate", "")),
        "frames": int(raw_count),
    }


def encode_aligned_video(
    source: Path,
    target: Path,
    source_indices: np.ndarray,
    *,
    width: int,
    height: int,
    fps: int,
    crf: int,
) -> None:
    indices = np.asarray(source_indices, dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(np.diff(indices) < 0):
        raise ValueError(
            f"{source}: aligned source indices must be causal/nondecreasing"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        str(crf),
        "-g",
        str(fps),
        "-keyint_min",
        str(fps),
        "-sc_threshold",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(target),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    capture = cv2.VideoCapture(str(source))
    current_index = -1
    current_frame: np.ndarray | None = None
    try:
        assert process.stdin is not None
        for requested in indices:
            while current_index < int(requested):
                ok, current_frame = capture.read()
                current_index += 1
                if not ok:
                    raise ValueError(
                        f"{source}: video ended at {current_index}, requested {requested}"
                    )
            if current_frame is None or current_frame.shape[:2] != (height, width):
                raise ValueError(f"{source}: bad decoded frame {current_index}")
            process.stdin.write(np.ascontiguousarray(current_frame).tobytes())
        process.stdin.close()
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
    except BaseException:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        process.terminate()
        process.wait()
        raise
    finally:
        capture.release()


def _sensor_payload(episode: CanonicalEpisode) -> dict[str, np.ndarray]:
    count = episode.length
    frame_index = np.arange(count, dtype=np.int64)
    timestamp = frame_index.astype(np.float64) / float(episode.fps)
    payload: dict[str, np.ndarray] = {
        "tau": episode.tau,
        "tau_valid_mask": episode.tau_valid,
        "tactile_wrench": episode.tactile_wrench,
        "tactile_wrench_valid_mask": episode.tactile_wrench_valid,
        "tactile_deformation": episode.tactile_deformation,
        "tactile_deformation_valid_mask": episode.tactile_deformation_valid,
        "frame_index": frame_index,
        "timestamp": timestamp,
        "source_timeline_row": episode.source_timeline_row,
    }
    for name in CAMERA_NAMES:
        alignment = episode.camera_alignments[name]
        prefix = f"camera_{name}"
        payload[f"{prefix}_source_video_index"] = alignment.source_video_index
        payload[f"{prefix}_source_timestamp_ns"] = alignment.source_timestamp_ns
        payload[f"{prefix}_source_timeline_row"] = alignment.source_timeline_row
        payload[f"{prefix}_valid_mask"] = alignment.valid
        payload[f"{prefix}_frame_reused"] = alignment.frame_reused
        payload[f"{prefix}_timeline_row_gap"] = alignment.timeline_row_gap
        payload[f"{prefix}_recorder_age_rows"] = alignment.recorder_age_rows
        payload[f"{prefix}_alignment_latency_ns"] = alignment.alignment_latency_ns
    return payload


def _write_staged_artifacts(
    stage: Path,
    episode: CanonicalEpisode,
    *,
    task_index: int,
    video_crf: int,
) -> dict[str, Path]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    stage.mkdir(parents=True, exist_ok=False)
    count = episode.length
    frame_index = np.arange(count, dtype=np.int64)
    timestamp = frame_index.astype(np.float64) / float(episode.fps)
    parquet = stage / "episode.parquet"
    sensors = stage / "sensors.npz"
    table = pa.table(
        {
            "observation.state": fixed_list(episode.state, 62),
            "action": fixed_list(episode.action, ACTION_DIM),
            "anchor_valid": pa.array(episode.anchor_valid, type=pa.bool_()),
            "timestamp": pa.array(timestamp, type=pa.float64()),
            "frame_index": pa.array(frame_index, type=pa.int64()),
            "episode_index": pa.array(
                np.full(count, episode.source.output_episode_index), type=pa.int64()
            ),
            "task_index": pa.array(np.full(count, task_index), type=pa.int64()),
            "annotation.language.task_description": pa.array(
                np.full(count, task_index), type=pa.int64()
            ),
        }
    )
    pq.write_table(table, parquet, compression="zstd")
    np.savez_compressed(sensors, **_sensor_payload(episode))
    result = {"parquet": parquet, "sensors": sensors}
    for name in CAMERA_NAMES:
        height, width = episode.camera_dimensions[name]
        target = stage / f"{name}.mp4"
        encode_aligned_video(
            episode.source.path / "cameras" / name / "rgb.mp4",
            target,
            episode.camera_alignments[name].source_video_index,
            width=width,
            height=height,
            fps=episode.fps,
            crf=video_crf,
        )
        probe = probe_video(target, count_frames=True)
        if probe["frames"] != count:
            raise ValueError(
                f"{target}: encoded {probe['frames']} frames, expected {count}"
            )
        result[f"video_{name}"] = target
    return result


def episode_output_paths(
    output: Path, episode_index: int, chunks_size: int
) -> dict[str, Path]:
    chunk = episode_index // chunks_size
    paths = {
        "parquet": output
        / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet",
        "sensors": output / f"sensors/episodes/episode_{episode_index:06d}.npz",
        "report": output / f"meta/episode_reports/episode_{episode_index:06d}.json",
        "commit": output / f"meta/commits/episode_{episode_index:06d}.json",
    }
    for name in CAMERA_NAMES:
        key = CAMERA_VIDEO_KEYS[name]
        paths[f"video_{name}"] = (
            output / f"videos/chunk-{chunk:03d}/{key}/episode_{episode_index:06d}.mp4"
        )
    return paths


def publish_episode(
    output: Path,
    episode: CanonicalEpisode,
    summary: dict[str, Any],
    *,
    task_index: int,
    chunks_size: int,
    video_crf: int,
) -> None:
    index = episode.source.output_episode_index
    stage_root = output / ".staging"
    stage_root.mkdir(parents=True, exist_ok=True)
    stage = stage_root / f"episode_{index:06d}.{os.getpid()}"
    if stage.exists():
        shutil.rmtree(stage)
    final = episode_output_paths(output, index, chunks_size)
    try:
        staged = _write_staged_artifacts(
            stage, episode, task_index=task_index, video_crf=video_crf
        )
        for key in ("parquet", "sensors", *(f"video_{name}" for name in CAMERA_NAMES)):
            final[key].parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged[key], final[key])
        write_json(final["report"], summary)
        write_json(
            final["commit"],
            {
                "schema": DATASET_SCHEMA,
                "episode_index": index,
                "source_episode": episode.source.name,
                "length": episode.length,
                "artifacts": {
                    key: str(path.relative_to(output))
                    for key, path in final.items()
                    if key not in {"report", "commit"}
                },
            },
        )
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def build_modality(
    camera_dimensions: dict[str, tuple[int, int]], fps: int
) -> dict[str, Any]:
    state_groups = (
        ("left_wrist_eef", 0, 9),
        ("right_wrist_eef", 9, 18),
        ("left_hand_joints", 18, 40),
        ("right_hand_joints", 40, 62),
    )
    action_groups = state_groups + (
        ("left_hand_q_teleop", 62, 84),
        ("right_hand_q_teleop", 84, 106),
        ("left_hand_delta_q", 106, 128),
        ("right_hand_delta_q", 128, 150),
    )
    return {
        "state": {
            name: {"start": start, "end": end, "original_key": "observation.state"}
            for name, start, end in state_groups
        },
        "action": {
            name: {"start": start, "end": end, "original_key": "action"}
            for name, start, end in action_groups
        },
        "video": {
            f"{name}_view" if name != "ego" else "ego_view": {
                "original_key": CAMERA_VIDEO_KEYS[name],
                "fps": fps,
                "dataset_fps": fps,
                "height": camera_dimensions[name][0],
                "width": camera_dimensions[name][1],
            }
            for name in CAMERA_NAMES
        },
        "annotation": {
            "language.task_description": {
                "original_key": "annotation.language.task_description"
            }
        },
    }


def write_metadata(
    output: Path,
    source: Path,
    episodes: list[dict[str, Any]],
    *,
    task_text: str,
    chunks_size: int,
    fps: int,
) -> None:
    if not episodes:
        raise ValueError("cannot write metadata for zero episodes")
    camera_dimensions = {
        name: (
            int(episodes[0]["cameras"][name]["height"]),
            int(episodes[0]["cameras"][name]["width"]),
        )
        for name in CAMERA_NAMES
    }
    for episode in episodes:
        for name in CAMERA_NAMES:
            actual = (
                int(episode["cameras"][name]["height"]),
                int(episode["cameras"][name]["width"]),
            )
            if actual != camera_dimensions[name]:
                raise ValueError(
                    f"mixed {name} resolution: {actual} != {camera_dimensions[name]}"
                )
    features: dict[str, Any] = {}
    for name in CAMERA_NAMES:
        height, width = camera_dimensions[name]
        features[CAMERA_VIDEO_KEYS[name]] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": fps,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    features.update(
        {
            "observation.state": {"dtype": "float32", "shape": [62]},
            "action": {"dtype": "float32", "shape": [ACTION_DIM]},
            "anchor_valid": {"dtype": "bool", "shape": [1]},
            "timestamp": {"dtype": "float64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "annotation.language.task_description": {
                "dtype": "int64",
                "shape": [1],
            },
        }
    )
    info = {
        "codebase_version": "v2.0",
        "schema": DATASET_SCHEMA,
        "robot_type": "ur_sharpa",
        "embodiment_tag": EMBODIMENT_TAG,
        "state_dim": 62,
        "action_dim": ACTION_DIM,
        "action_order": [
            "wrist_exe_next18",
            "q_exe_next44",
            "q_teleop44",
            "delta_q44",
        ],
        "wrist_source": "ur/physical_12.npz:canonical_wrist_pose9",
        "wrist_transform_at_conversion": "none",
        "wrist_pose9": "[x,y,z,r00,r10,r20,r01,r11,r21] canonical root-from-hand_C_MC",
        "control_cleaning": {
            "wrist": "causal latest-valid hold",
            "hand_state_and_command": (
                "q and q_cmd are paired by raw timeline row and held together from "
                "one latest-valid pair; source timestamps are diagnostic only"
            ),
            "model_history_or_future_window_applied": False,
            "anchor_valid": "true for every cleaned row",
        },
        "q_exe_definition": "next cleaned executed state; terminal row repeats last",
        "q_teleop_definition": (
            "Manus q_cmd stored in the same raw timeline row as its paired q state"
        ),
        "delta_q_definition": "q_teleop - q_exe_next",
        "execution_identity": "q_exe_next + delta_q == q_teleop",
        "joint_order": list(POSTTRAIN_JOINT_ORDER),
        "finger_order": list(POSTTRAIN_TACTILE_ORDER),
        "tau_role": "optional measured sensor; never an action target",
        "camera_alignment": {
            "basis": "latest index.npz timeline_row <= master timeline row",
            "future_frames_allowed": False,
            "mp4_pts_used_for_capture_timing": False,
            "fixed_latency_validity_threshold": None,
            "views": list(CAMERA_NAMES),
        },
        "sensor_path": "sensors/episodes/episode_{episode_index:06d}.npz",
        "sensor_storage": {
            "row_alignment": "frame_index and timestamp exactly equal Parquet",
            "tactile_deformation": {
                "shape": [10, DEFORMATION_SIZE, DEFORMATION_SIZE],
                "dtype": "uint8",
                "spatial_storage": "original_240x240_no_resize",
            },
            "camera_field_prefixes": [f"camera_{name}" for name in CAMERA_NAMES],
        },
        "total_episodes": len(episodes),
        "total_frames": int(sum(int(episode["length"]) for episode in episodes)),
        "total_tasks": 1,
        "total_chunks": len(
            {int(episode["episode_index"]) // chunks_size for episode in episodes}
        ),
        "chunks_size": chunks_size,
        "fps": fps,
        "video_fps": fps,
        "video_codec": "h264",
        "splits": {"train": f"0:{len(episodes)}"},
        "split_counts": {"train": len(episodes), "val": 0, "test": 0},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    write_json(output / "meta/info.json", info)
    write_json(output / "meta/modality.json", build_modality(camera_dimensions, fps))
    (output / "meta").mkdir(parents=True, exist_ok=True)
    (output / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": task_text}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (output / "meta/episodes.jsonl").open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(
                json.dumps(
                    {
                        "episode_index": int(episode["episode_index"]),
                        "source_episode": episode["source_episode"],
                        "source_start_time_unix_ns": int(
                            episode["source_start_time_unix_ns"]
                        ),
                        "split": "train",
                        "tasks": [task_text],
                        "length": int(episode["length"]),
                        "valid_anchor_count": int(episode["valid_anchor_count"]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    write_json(
        output / "meta/source_manifest.json",
        {
            "schema": DATASET_SCHEMA,
            "source": str(source.resolve()),
            "episode_order": "timeline/clock.npz time_unix_ns[0], then directory name",
            "selected_input_count": len(episodes),
            "episodes": [
                {
                    "episode_index": int(item["episode_index"]),
                    "source_episode": item["source_episode"],
                    "source_timeline_rows": int(item["source_timeline_rows"]),
                    "crop_start_timeline_row": int(item["crop_start_timeline_row"]),
                    "output_rows": int(item["length"]),
                }
                for item in episodes
            ],
        },
    )


def relative_wrist_offset_rows(
    state: np.ndarray,
    action: np.ndarray,
    *,
    future_frame_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return all valid wrist pairs for one future-frame offset.

    Canonical UR-SharpA rows store the next executed wrist in ``action[t]``.
    Future frame ``k`` relative to current frame ``t`` is therefore stored at
    ``action[t + k - 1]``.  The repeated terminal action row is never a real
    future frame and is deliberately excluded.
    """

    states = np.asarray(state, dtype=np.float32)
    actions = np.asarray(action, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] < 18:
        raise ValueError(f"state must be [T,>=18], got {states.shape}")
    if actions.ndim != 2 or actions.shape[1] < 18:
        raise ValueError(f"action must be [T,>=18], got {actions.shape}")
    if len(states) != len(actions):
        raise ValueError(
            f"state/action row counts differ: {len(states)}/{len(actions)}"
        )
    offset = int(future_frame_offset)
    if offset <= 0:
        raise ValueError(f"future_frame_offset must be positive, got {offset}")

    usable = len(states) - offset
    if usable <= 0:
        empty = np.empty((0, 9), dtype=np.float32)
        return empty, empty.copy()

    def collect(wrist_slice: slice) -> np.ndarray:
        anchors = states[:usable, wrist_slice]
        targets = actions[offset - 1 : len(actions) - 1, wrist_slice]
        if len(targets) != usable:
            raise AssertionError(
                f"future offset {offset} produced {len(targets)}/{usable} targets"
            )
        relative = relative_pose9_batch(anchors, targets[:, None, :])[:, 0]
        return np.asarray(relative, dtype=np.float32)

    return collect(slice(0, 9)), collect(slice(9, 18))


def relative_wrist_horizon_rows(
    state: np.ndarray,
    action: np.ndarray,
    *,
    horizon: int = RELATIVE_ACTION_HORIZON,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return ragged relative wrist rows for future frames ``1..horizon``.

    Offset ``k`` has ``max(L-k, 0)`` rows for an episode with ``L`` frames.
    Keeping these lists separate is essential: GR00T uses one normalization
    parameter set per action-chunk position, with shape ``[horizon, 9]``.
    """

    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    for future_frame_offset in range(1, int(horizon) + 1):
        left_rows, right_rows = relative_wrist_offset_rows(
            state,
            action,
            future_frame_offset=future_frame_offset,
        )
        left.append(left_rows)
        right.append(right_rows)
    return left, right


def _stack_horizon_statistics(results: list[dict[str, Any]]) -> dict[str, Any]:
    if len(results) != RELATIVE_ACTION_HORIZON:
        raise ValueError(
            "relative wrist statistics must contain "
            f"{RELATIVE_ACTION_HORIZON} offsets, got {len(results)}"
        )
    return {
        field: [result[field] for result in results] for field in RELATIVE_STATS_FIELDS
    }


def validate_relative_stats_contract(
    dataset_root: str | Path,
    *,
    expected_horizon: int = RELATIVE_ACTION_HORIZON,
) -> dict[str, Any]:
    """Reject single-step statistics before a relative-EEF training run starts."""

    root = Path(dataset_root).resolve()
    provenance = json.loads(
        (root / "meta/stats_provenance.json").read_text(encoding="utf-8")
    )
    contract = provenance.get("relative_action")
    expected_indices = list(range(expected_horizon))
    expected_offsets = list(range(1, expected_horizon + 1))
    if (
        not isinstance(contract, dict)
        or contract.get("schema") != RELATIVE_STATS_SCHEMA
    ):
        raise ValueError(
            f"{root}: relative action statistics lack {RELATIVE_STATS_SCHEMA}; "
            "regenerate dataset statistics before relative-EEF training"
        )
    if (
        int(contract.get("horizon", -1)) != expected_horizon
        or contract.get("action_delta_indices") != expected_indices
        or contract.get("future_frame_offsets") != expected_offsets
        or contract.get("reference") != "current_observation_wrist"
        or contract.get("action_row_semantics") != "next_cleaned_executed_state"
        or contract.get("preserve_horizon_for_statistics") is not True
        or contract.get("flatten_horizon_for_statistics") is not False
        or contract.get("terminal_repeat_included") is not False
    ):
        raise ValueError(
            f"{root}: relative action statistics do not satisfy the per-horizon "
            f"future-frame contract: {contract}"
        )
    expected_window = {
        "relative_action_horizon": expected_horizon,
        "action_delta_indices": expected_indices,
        "future_frame_offsets": expected_offsets,
        "state_anchor_index": 0,
        "physical_frame_count_for_full_chunk": expected_horizon + 1,
        "terminal_suffix_policy": "use_each_available_future_offset_for_statistics",
    }
    actual_window = contract.get("sampling_window")
    if actual_window is None:
        # Compatibility with rowwise_stats_provenance.v1 files generated
        # before the derived relative-action window had its own namespace.
        actual_window = provenance.get("sampling", {}).get(
            "model_history_or_future_window"
        )
    if actual_window != expected_window:
        raise ValueError(
            f"{root}: relative action sampling window is inconsistent: {actual_window}"
        )

    episodes = [
        json.loads(line)
        for line in (root / "meta/episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    selected = [item for item in episodes if item.get("split") == "train"]
    expected_counts = [
        sum(max(int(item["length"]) - offset, 0) for item in selected)
        for offset in expected_offsets
    ]
    if contract.get("per_offset_sample_counts") != expected_counts:
        raise ValueError(
            f"{root}: relative action per-offset sample counts are inconsistent; "
            f"expected {expected_counts}, got {contract.get('per_offset_sample_counts')}"
        )

    stats = json.loads((root / "meta/relative_stats.json").read_text(encoding="utf-8"))
    for key in ("left_wrist_eef", "right_wrist_eef"):
        if key not in stats:
            raise KeyError(f"{root}: relative_stats.json is missing {key}")
        for field in RELATIVE_STATS_FIELDS:
            values = np.asarray(stats[key].get(field), dtype=np.float64)
            if values.shape != (expected_horizon, 9) or not np.isfinite(values).all():
                raise ValueError(
                    f"{root}: relative_stats.json {key}.{field} must be finite "
                    f"[{expected_horizon},9]"
                )
        if np.any(
            np.asarray(stats[key]["max"], dtype=np.float64)
            <= np.asarray(stats[key]["min"], dtype=np.float64)
        ):
            raise ValueError(
                f"{root}: relative_stats.json {key} has an empty min/max range"
            )
        if np.any(
            np.asarray(stats[key]["q99"], dtype=np.float64)
            <= np.asarray(stats[key]["q01"], dtype=np.float64)
        ):
            raise ValueError(
                f"{root}: relative_stats.json {key} has an empty q01/q99 range"
            )
    return contract


def write_rowwise_stats(dataset_root: str | Path) -> dict[str, Any]:
    """Write rowwise base stats and horizon-aware relative EEF stats."""

    import pyarrow.parquet as pq

    from dexterity.data.normalization import _RunningMoments, _StreamingStatistics

    root = Path(dataset_root).resolve()
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    episodes = [
        json.loads(line)
        for line in (root / "meta/episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    selected = [item for item in episodes if item.get("split") == "train"]
    if not selected:
        raise ValueError("dataset has no train episodes")
    total_rows = sum(int(item["length"]) for item in selected)
    state_stats = _StreamingStatistics(62, total_rows)
    action_stats = _StreamingStatistics(ACTION_DIM, total_rows)
    timestamp_stats = _StreamingStatistics(1, total_rows)
    future_frame_offsets = list(range(1, RELATIVE_ACTION_HORIZON + 1))
    relative_sample_counts = [
        sum(max(int(item["length"]) - offset, 0) for item in selected)
        for offset in future_frame_offsets
    ]
    if not relative_sample_counts[-1]:
        raise ValueError(
            f"dataset has no complete {RELATIVE_ACTION_HORIZON}-future-frame windows"
        )
    tau_stats = _RunningMoments(44)
    wrench_stats = _RunningMoments(60)
    sensor_digest = hashlib.sha256()
    train_indices: list[int] = []
    relative_episode_rows: list[tuple[np.ndarray, np.ndarray]] = []
    chunks_size = int(info["chunks_size"])

    for item in selected:
        episode_index = int(item["episode_index"])
        chunk = episode_index // chunks_size
        parquet_path = (
            root / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
        )
        table = pq.read_table(
            parquet_path,
            columns=["observation.state", "action", "anchor_valid", "timestamp"],
        )
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        anchor = np.asarray(table["anchor_valid"].to_numpy(), dtype=bool)
        timestamp = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)
        if not anchor.all() or len(state) != int(item["length"]):
            raise ValueError(
                f"episode {episode_index}: rowwise stats require all anchors"
            )
        state_stats.update(state)
        action_stats.update(action)
        timestamp_stats.update(timestamp[:, None])
        relative_episode_rows.append(
            (
                np.ascontiguousarray(state[:, :18]),
                np.ascontiguousarray(action[:, :18]),
            )
        )

        sensor_path = root / "sensors/episodes" / f"episode_{episode_index:06d}.npz"
        with np.load(sensor_path, allow_pickle=False) as sensors:
            tau = np.asarray(sensors["tau"], dtype=np.float32)
            tau_valid = np.asarray(sensors["tau_valid_mask"], dtype=bool)
            wrench = np.asarray(sensors["tactile_wrench"], dtype=np.float32)
            wrench_valid = np.asarray(sensors["tactile_wrench_valid_mask"], dtype=bool)
        tau_stats.update(tau, tau_valid)
        wrench_stats.update(
            wrench.reshape(-1, 60),
            np.broadcast_to(wrench_valid[..., None], (*wrench_valid.shape, 6)).reshape(
                -1, 60
            ),
        )
        sensor_digest.update(sensor_path.name.encode("utf-8"))
        sensor_digest.update(np.ascontiguousarray(tau).tobytes())
        sensor_digest.update(np.ascontiguousarray(tau_valid).tobytes())
        sensor_digest.update(np.ascontiguousarray(wrench).tobytes())
        sensor_digest.update(np.ascontiguousarray(wrench_valid).tobytes())
        train_indices.append(episode_index)

    relative_left_results: list[dict[str, Any]] = []
    relative_right_results: list[dict[str, Any]] = []
    for future_frame_offset, sample_count in zip(
        future_frame_offsets,
        relative_sample_counts,
        strict=True,
    ):
        relative_left = _StreamingStatistics(9, sample_count)
        relative_right = _StreamingStatistics(9, sample_count)
        for state_wrist, action_wrist in relative_episode_rows:
            left_rows, right_rows = relative_wrist_offset_rows(
                state_wrist,
                action_wrist,
                future_frame_offset=future_frame_offset,
            )
            if len(left_rows):
                relative_left.update(left_rows)
                relative_right.update(right_rows)
        relative_left_results.append(relative_left.result())
        relative_right_results.append(relative_right.result())

    wrench_result = wrench_stats.result(allow_empty=True)
    for key in ("count", "mean", "std", "min", "max"):
        wrench_result[key] = np.asarray(wrench_result[key]).reshape(10, 6).tolist()
    digest = sensor_digest.hexdigest()
    provenance = {
        "schema": "sharpa.control_sensors.rowwise_stats_provenance.v1",
        "split": "train",
        "train_episode_indices": train_indices,
        "train_episode_count": len(train_indices),
        "sensor_input_sha256": digest,
        "statistics_method": {
            "mean_std_min_max": "exact_streaming",
            "q01_q99": "deterministic_uniform_global_row_sample",
            "maximum_quantile_sample_rows": 200_000,
            "relative_action": {
                "mean_std_min_max": "exact_streaming_per_future_offset",
                "q01_q99": "deterministic_uniform_anchor_sample_per_future_offset",
                "maximum_quantile_sample_rows_per_offset": 200_000,
            },
        },
        "sampling": {
            "state": "all cleaned 30 Hz rows",
            "action": "all cleaned 30 Hz rows",
            "relative_action": (
                "for each future frame offset k=1..40, every in-episode state[t] "
                "is paired with action[t+k-1] when t+k<L; horizon is preserved "
                "and the terminal repeated action row is excluded"
            ),
            "sensor": "all rows subject to each sensor validity mask",
            # Base state/action statistics are rowwise.  The future window is
            # a property of the separately generated relative-action stats.
            "model_history_or_future_window": None,
        },
        "relative_action": {
            "schema": RELATIVE_STATS_SCHEMA,
            "horizon": RELATIVE_ACTION_HORIZON,
            "action_delta_indices": list(range(RELATIVE_ACTION_HORIZON)),
            "future_frame_offsets": future_frame_offsets,
            "sampling_window": {
                "relative_action_horizon": RELATIVE_ACTION_HORIZON,
                "action_delta_indices": list(range(RELATIVE_ACTION_HORIZON)),
                "future_frame_offsets": future_frame_offsets,
                "state_anchor_index": 0,
                "physical_frame_count_for_full_chunk": RELATIVE_ACTION_HORIZON + 1,
                "terminal_suffix_policy": (
                    "use_each_available_future_offset_for_statistics"
                ),
            },
            "reference": "current_observation_wrist",
            "action_row_semantics": "next_cleaned_executed_state",
            "per_offset_sample_counts": relative_sample_counts,
            "preserve_horizon_for_statistics": True,
            "flatten_horizon_for_statistics": False,
            "episode_boundaries_crossed": False,
            "terminal_repeat_included": False,
        },
    }
    values = {
        "stats.json": {
            "observation.state": state_stats.result(),
            "action": action_stats.result(),
            "timestamp": timestamp_stats.result(),
        },
        "relative_stats.json": {
            "left_wrist_eef": _stack_horizon_statistics(relative_left_results),
            "right_wrist_eef": _stack_horizon_statistics(relative_right_results),
        },
        "sensor_stats.json": {
            "schema": "sharpa.sensor_normalization.v2",
            "split": "train",
            "train_episode_indices": train_indices,
            "train_episode_count": len(train_indices),
            "joint_order": list(POSTTRAIN_JOINT_ORDER),
            "finger_order": list(POSTTRAIN_TACTILE_ORDER),
            "wrench_order": ["fx", "fy", "fz", "tx", "ty", "tz"],
            "tau": tau_stats.result(allow_empty=True),
            "wrench": wrench_result,
            "input_sha256": digest,
        },
        "stats_provenance.json": provenance,
    }
    for name, value in values.items():
        write_json(root / "meta" / name, value)
    return provenance


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def aggregate_report(
    *,
    source: Path,
    output: Path,
    selected_count: int,
    successes: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    checksum_results: dict[str, dict[str, Any]],
    verifier_report: str | None = None,
) -> dict[str, Any]:
    camera_summary: dict[str, Any] = {}
    for name in CAMERA_NAMES:
        values = [item["cameras"][name] for item in successes]
        weighted_rows = sum(int(item["length"]) for item in successes)
        gap_histogram: dict[int, int] = {}
        for value in values:
            for raw_gap, raw_count in value["timeline_row_gap_histogram"].items():
                gap = int(raw_gap)
                gap_histogram[gap] = gap_histogram.get(gap, 0) + int(raw_count)
        gaps = (
            np.concatenate(
                [
                    np.full(count, gap, dtype=np.int32)
                    for gap, count in gap_histogram.items()
                ]
            )
            if gap_histogram
            else np.empty(0, dtype=np.int32)
        )
        camera_summary[name] = {
            "valid_ratio": sum(
                float(value["valid_ratio"]) * int(item["length"])
                for item, value in zip(successes, values, strict=True)
            )
            / max(weighted_rows, 1),
            "reuse_ratio": sum(
                float(value["reuse_ratio"]) * max(int(item["length"]) - 1, 0)
                for item, value in zip(successes, values, strict=True)
            )
            / max(weighted_rows - len(successes), 1),
            "timeline_row_gap": quantiles(gaps),
            "timeline_row_gap_per_episode": {
                item["source_episode"]: value["timeline_row_gap"]
                for item, value in zip(successes, values, strict=True)
            },
        }

    def weighted_ratio(key: str) -> float:
        total = sum(int(item["length"]) for item in successes)
        return sum(float(item[key]) * int(item["length"]) for item in successes) / max(
            total, 1
        )

    cleaning_summary: dict[str, Any] = {}
    for name in ("wrist", "hand_pair"):
        values = [item["cleaning"][name] for item in successes]
        gap_histogram: dict[int, int] = {}
        for value in values:
            for raw_gap, raw_count in value["hold_gap_histogram"].items():
                gap = int(raw_gap)
                gap_histogram[gap] = gap_histogram.get(gap, 0) + int(raw_count)
        gaps = (
            np.concatenate(
                [
                    np.full(count, gap, dtype=np.int32)
                    for gap, count in gap_histogram.items()
                ]
            )
            if gap_histogram
            else np.empty(0, dtype=np.int32)
        )
        observed_rows = sum(int(value["observed_rows"]) for value in values)
        held_rows = sum(int(value["held_rows"]) for value in values)
        cleaning_summary[name] = {
            "method": (
                "raw_row_paired_latest_valid_hold"
                if name == "hand_pair"
                else "causal_latest_valid_hold"
            ),
            "observed_rows": observed_rows,
            "held_rows": held_rows,
            "held_ratio": held_rows / max(observed_rows + held_rows, 1),
            "hold_run_count": sum(int(value["hold_run_count"]) for value in values),
            "hold_gap_rows": quantiles(gaps),
            "max_source_age_rows": max(
                (int(value["max_source_age_rows"]) for value in values), default=0
            ),
        }

    return {
        "schema": DATASET_SCHEMA,
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "selected_input_episodes": selected_count,
        "successful_episodes": len(successes),
        "failed_episodes": len(failures),
        "failures": failures,
        "total_output_rows": int(sum(int(item["length"]) for item in successes)),
        "anchor_counts": {
            item["source_episode"]: int(item["valid_anchor_count"])
            for item in successes
        },
        "anchor_zero_episodes": [
            item["source_episode"]
            for item in successes
            if int(item["valid_anchor_count"]) == 0
        ],
        "all_anchor_valid": all(
            bool(item["all_anchor_valid"])
            and int(item["valid_anchor_count"]) == int(item["length"])
            for item in successes
        ),
        "validity": (
            {
                key: weighted_ratio(key)
                for key in (
                    "tau_valid_ratio",
                    "tactile_wrench_valid_ratio",
                    "tactile_deformation_valid_ratio",
                )
            }
            if successes
            else {}
        ),
        "control_cleaning": cleaning_summary,
        "cameras": camera_summary,
        "action_identity_max_error": max(
            (float(item["action_identity_max_error"]) for item in successes),
            default=0.0,
        ),
        "wrist_reconstruction_position_max_abs_m": max(
            (
                float(item["wrist_reconstruction_position_max_abs_m"])
                for item in successes
            ),
            default=0.0,
        ),
        "wrist_reconstruction_rotation_max_abs": max(
            (
                float(item["wrist_reconstruction_rotation_max_abs"])
                for item in successes
            ),
            default=0.0,
        ),
        "wrist_audit": {
            "output_source": "ur/physical_12.npz:canonical_wrist_pose9",
            "extra_root_or_hand_mount_transform_applied": False,
            "audit_method": "held_out_fixed_rigid_chain_reconstruction",
            "producer_geometry_dependency": (
                "ur_common.geometry is recorder-side and is not vendored in this repository"
            ),
            "evidence_scope": (
                "The output path is proven to use stored canonical Pose9 exactly once; "
                "raw TCP recomputation is an independent rigid-chain consistency audit, "
                "not a byte-identical execution of the unavailable producer package."
            ),
        },
        "checksum_results": checksum_results,
        "missing_or_checksum_issues": failures,
        "output_size_bytes": directory_size(output) if output.exists() else 0,
        "verifier_report": verifier_report,
        "episodes": successes,
    }


__all__ = [
    "DATASET_SCHEMA",
    "EMBODIMENT_TAG",
    "aggregate_report",
    "build_modality",
    "directory_size",
    "encode_aligned_video",
    "episode_output_paths",
    "fixed_list",
    "probe_video",
    "publish_episode",
    "verify_raw_checksums",
    "write_json",
    "write_metadata",
    "write_rowwise_stats",
]
