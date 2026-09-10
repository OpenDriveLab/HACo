#!/usr/bin/env python3
"""Verify every artifact in a timeline-based three-camera UR LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.posttrain.ur_sharpa.contract import (  # noqa: E402
    ACTION_DIM,
    CAMERA_NAMES,
    CAMERA_VIDEO_KEYS,
    DEFORMATION_SIZE,
)
from scripts.data.posttrain.ur_sharpa.storage import (  # noqa: E402
    DATASET_SCHEMA,
    RELATIVE_ACTION_HORIZON,
    probe_video,
    validate_relative_stats_contract,
    write_json,
)

PARQUET_COLUMNS = (
    "observation.state",
    "action",
    "anchor_valid",
    "timestamp",
    "frame_index",
    "episode_index",
    "task_index",
    "annotation.language.task_description",
)
BASE_SENSOR_FIELDS = (
    "tau",
    "tau_valid_mask",
    "tactile_wrench",
    "tactile_wrench_valid_mask",
    "tactile_deformation",
    "tactile_deformation_valid_mask",
    "frame_index",
    "timestamp",
    "source_timeline_row",
)
CAMERA_SENSOR_SUFFIXES = (
    "source_video_index",
    "source_timestamp_ns",
    "source_timeline_row",
    "valid_mask",
    "frame_reused",
    "timeline_row_gap",
    "recorder_age_rows",
    "alignment_latency_ns",
)
FORBIDDEN_SENSOR_FIELDS = (
    "source_video_index",
    "source_video_pts_s",
    "alignment_error_s",
    "transition_valid_mask",
    "ur_valid_mask",
    "q_valid_mask",
    "q_cmd_valid_mask",
    "q_cmd_teleop_mask",
    "q_observed_mask",
    "q_held_mask",
    "q_source_timeline_row",
    "q_age_rows",
    "q_cmd_observed_mask",
    "q_cmd_held_mask",
    "q_cmd_source_timeline_row",
    "q_cmd_age_rows",
    "wrist_observed_mask",
    "wrist_held_mask",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _fixed_list(table, name: str, width: int) -> np.ndarray:
    column = table[name].combine_chunks()
    if column.type.list_size != width:
        raise ValueError(
            f"{name}: expected fixed-list width {width}, got {column.type}"
        )
    return np.asarray(
        column.values.to_numpy(zero_copy_only=False), dtype=np.float32
    ).reshape(-1, width)


def _require_shape(name: str, value: np.ndarray, shape: tuple[int, ...]) -> None:
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")


def _require_invalid_zero(name: str, value: np.ndarray, valid: np.ndarray) -> None:
    expanded = valid
    while expanded.ndim < value.ndim:
        expanded = expanded[..., None]
    if np.any(value[~np.broadcast_to(expanded, value.shape)] != 0):
        raise ValueError(f"{name}: invalid entries must be zero-filled")


def verify_episode(
    dataset: Path,
    info: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    episode_index = int(metadata["episode_index"])
    chunks_size = int(info["chunks_size"])
    chunk = episode_index // chunks_size
    expected_length = int(metadata["length"])
    parquet_path = (
        dataset / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    sensor_path = dataset / f"sensors/episodes/episode_{episode_index:06d}.npz"
    table = pq.read_table(parquet_path)
    if tuple(table.column_names) != PARQUET_COLUMNS:
        raise ValueError(
            f"{parquet_path}: columns {table.column_names} != {list(PARQUET_COLUMNS)}"
        )
    if len(table) != expected_length:
        raise ValueError(
            f"{parquet_path}: {len(table)} rows != metadata {expected_length}"
        )
    state = _fixed_list(table, "observation.state", 62)
    action = _fixed_list(table, "action", ACTION_DIM)
    anchor_valid = np.asarray(table["anchor_valid"].to_numpy(), dtype=bool)
    timestamp = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)
    frame_index = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
    episode_column = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
    task_index = np.asarray(table["task_index"].to_numpy(), dtype=np.int64)
    annotation = np.asarray(
        table["annotation.language.task_description"].to_numpy(), dtype=np.int64
    )
    if not np.array_equal(frame_index, np.arange(expected_length, dtype=np.int64)):
        raise ValueError(f"{parquet_path}: frame_index is not contiguous from zero")
    expected_timestamp = frame_index.astype(np.float64) / float(info["fps"])
    if not np.array_equal(timestamp, expected_timestamp):
        raise ValueError(f"{parquet_path}: timestamp != frame_index/fps")
    if not np.all(episode_column == episode_index):
        raise ValueError(f"{parquet_path}: episode_index column mismatch")
    if not np.all(task_index == 0) or not np.array_equal(task_index, annotation):
        raise ValueError(f"{parquet_path}: task annotation columns mismatch")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError(f"{parquet_path}: state/action contains NaN or Inf")
    identity_error = float(
        np.max(np.abs(action[:, 18:62] + action[:, 106:150] - action[:, 62:106]))
    )
    if identity_error > 1e-6:
        raise ValueError(
            f"{parquet_path}: action identity error {identity_error} > 1e-6"
        )
    if not anchor_valid.all():
        raise ValueError(
            f"{parquet_path}: every cleaned row must have anchor_valid=true"
        )
    if np.any(np.all(state[:, 18:62] == 0.0, axis=1)):
        raise ValueError(
            f"{parquet_path}: cleaned SharpA state contains an all-zero row"
        )
    if np.any(np.all(action[:, 62:106] == 0.0, axis=1)):
        raise ValueError(
            f"{parquet_path}: cleaned Manus command contains an all-zero row"
        )

    required_sensor_fields = set(BASE_SENSOR_FIELDS)
    for name in CAMERA_NAMES:
        required_sensor_fields.update(
            f"camera_{name}_{suffix}" for suffix in CAMERA_SENSOR_SUFFIXES
        )
    with np.load(sensor_path, allow_pickle=False) as payload:
        fields = set(payload.files)
        missing = sorted(required_sensor_fields - fields)
        forbidden = sorted(set(FORBIDDEN_SENSOR_FIELDS) & fields)
        if missing:
            raise KeyError(f"{sensor_path}: missing sensor fields {missing}")
        if forbidden:
            raise ValueError(
                f"{sensor_path}: forbidden control provenance fields: {forbidden}"
            )
        tau = np.asarray(payload["tau"], dtype=np.float32)
        tau_valid = np.asarray(payload["tau_valid_mask"], dtype=bool)
        wrench = np.asarray(payload["tactile_wrench"], dtype=np.float32)
        wrench_valid = np.asarray(payload["tactile_wrench_valid_mask"], dtype=bool)
        deformation = np.asarray(payload["tactile_deformation"])
        deformation_valid = np.asarray(
            payload["tactile_deformation_valid_mask"], dtype=bool
        )
        sensor_frame_index = np.asarray(payload["frame_index"], dtype=np.int64)
        sensor_timestamp = np.asarray(payload["timestamp"], dtype=np.float64)
        source_timeline_row = np.asarray(payload["source_timeline_row"], dtype=np.int64)
        cameras = {
            name: {
                suffix: np.asarray(payload[f"camera_{name}_{suffix}"])
                for suffix in CAMERA_SENSOR_SUFFIXES
            }
            for name in CAMERA_NAMES
        }
    _require_shape("tau", tau, (expected_length, 44))
    _require_shape("tau_valid_mask", tau_valid, tau.shape)
    _require_shape("tactile_wrench", wrench, (expected_length, 10, 6))
    _require_shape("tactile_wrench_valid_mask", wrench_valid, (expected_length, 10))
    _require_shape(
        "tactile_deformation",
        deformation,
        (expected_length, 10, DEFORMATION_SIZE, DEFORMATION_SIZE),
    )
    if deformation.dtype != np.uint8:
        raise ValueError(
            f"{sensor_path}: tactile_deformation dtype is {deformation.dtype}"
        )
    _require_shape(
        "tactile_deformation_valid_mask", deformation_valid, (expected_length, 10)
    )
    if not np.isfinite(tau).all() or not np.isfinite(wrench).all():
        raise ValueError(f"{sensor_path}: tau/wrench contains NaN or Inf")
    _require_invalid_zero("tau", tau, tau_valid)
    _require_invalid_zero("tactile_wrench", wrench, wrench_valid)
    _require_invalid_zero("tactile_deformation", deformation, deformation_valid)
    if not np.array_equal(sensor_frame_index, frame_index):
        raise ValueError(f"{sensor_path}: frame_index differs from Parquet")
    if not np.array_equal(sensor_timestamp, timestamp):
        raise ValueError(f"{sensor_path}: timestamp differs from Parquet")
    if source_timeline_row.shape != (expected_length,) or np.any(
        np.diff(source_timeline_row) != 1
    ):
        raise ValueError(f"{sensor_path}: source_timeline_row is not contiguous")
    if int(anchor_valid.sum()) != int(metadata["valid_anchor_count"]):
        raise ValueError(f"{parquet_path}: anchor count differs from episodes.jsonl")
    if int(anchor_valid.sum()) != expected_length:
        raise ValueError(f"{parquet_path}: anchor count must equal episode length")

    video_results: dict[str, Any] = {}
    camera_results: dict[str, Any] = {}
    for name in CAMERA_NAMES:
        camera = cameras[name]
        for suffix, value in camera.items():
            _require_shape(f"camera_{name}_{suffix}", value, (expected_length,))
        source_video_index = camera["source_video_index"].astype(np.int64)
        camera_source_row = camera["source_timeline_row"].astype(np.int64)
        camera_valid = camera["valid_mask"].astype(bool)
        if not camera_valid.all():
            raise ValueError(f"{sensor_path}: {name} is not valid on every cleaned row")
        if np.any(source_video_index < 0) or np.any(np.diff(source_video_index) < 0):
            raise ValueError(f"{sensor_path}: {name} source video mapping is invalid")
        if np.any(camera_source_row > source_timeline_row):
            raise ValueError(f"{sensor_path}: {name} selected a future frame")
        if np.any(np.diff(camera_source_row) < 0):
            raise ValueError(f"{sensor_path}: {name} source timeline is not monotonic")
        expected_reuse = np.zeros(expected_length, dtype=bool)
        expected_gap = np.zeros(expected_length, dtype=np.int64)
        if expected_length > 1:
            expected_reuse[1:] = source_video_index[1:] == source_video_index[:-1]
            expected_gap[1:] = np.diff(camera_source_row)
        if not np.array_equal(camera["frame_reused"].astype(bool), expected_reuse):
            raise ValueError(f"{sensor_path}: {name} reuse diagnostics mismatch")
        if not np.array_equal(
            camera["timeline_row_gap"].astype(np.int64), expected_gap
        ):
            raise ValueError(f"{sensor_path}: {name} timeline gap diagnostics mismatch")
        expected_age = source_timeline_row - camera_source_row
        if not np.array_equal(
            camera["recorder_age_rows"].astype(np.int64), expected_age
        ):
            raise ValueError(f"{sensor_path}: {name} recorder age diagnostics mismatch")
        video_key = CAMERA_VIDEO_KEYS[name]
        video_path = (
            dataset
            / f"videos/chunk-{chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        )
        video = probe_video(video_path, count_frames=True)
        feature = info["features"][video_key]
        if video["frames"] != expected_length:
            raise ValueError(
                f"{video_path}: {video['frames']} frames != {expected_length} rows"
            )
        if [video["height"], video["width"], 3] != feature["shape"]:
            raise ValueError(f"{video_path}: dimensions differ from info.json")
        video_results[name] = video
        camera_results[name] = {
            "causal": True,
            "valid_ratio": float(camera_valid.mean()),
            "reuse_ratio": (
                float(expected_reuse[1:].mean()) if expected_length > 1 else 0.0
            ),
            "source_video_index_max": int(source_video_index.max()),
        }
    return {
        "episode_index": episode_index,
        "source_episode": metadata.get("source_episode"),
        "rows": expected_length,
        "anchor_count": int(anchor_valid.sum()),
        "action_identity_max_error": identity_error,
        "parquet_sensor_timestamp_exact": True,
        "videos": video_results,
        "cameras": camera_results,
        "tactile_deformation_dtype": str(deformation.dtype),
        "tactile_deformation_shape": list(deformation.shape),
    }


def _verify_stats(
    dataset: Path, info: dict[str, Any], episode_count: int
) -> dict[str, Any]:
    required = (
        "stats.json",
        "sensor_stats.json",
        "relative_stats.json",
        "stats_provenance.json",
    )
    values = {name: _read_json(dataset / "meta" / name) for name in required}
    stats = values["stats.json"]
    for field, width in (
        ("observation.state", 62),
        ("action", ACTION_DIM),
        ("timestamp", 1),
    ):
        if field not in stats:
            raise KeyError(f"stats.json is missing {field}")
        for key in ("mean", "std", "min", "max", "q01", "q99"):
            if np.asarray(stats[field][key]).shape != (width,):
                raise ValueError(f"stats.json {field}.{key} has wrong shape")
    provenance = values["stats_provenance.json"]
    if int(provenance["train_episode_count"]) != episode_count:
        raise ValueError("stats provenance episode count mismatch")
    validate_relative_stats_contract(
        dataset,
        expected_horizon=RELATIVE_ACTION_HORIZON,
    )
    sensor_stats = values["sensor_stats.json"]
    if sensor_stats.get("joint_order") != info.get("joint_order"):
        raise ValueError("sensor stats joint order mismatch")
    if sensor_stats.get("finger_order") != info.get("finger_order"):
        raise ValueError("sensor stats finger order mismatch")
    return {"files": list(required), "consistent": True}


def verify_dataset(
    dataset: str | Path,
    *,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(dataset).resolve()
    failures: list[dict[str, Any]] = []
    episodes_verified: list[dict[str, Any]] = []
    try:
        info = _read_json(root / "meta/info.json")
        modality = _read_json(root / "meta/modality.json")
        if info.get("schema") != DATASET_SCHEMA:
            raise ValueError(f"unsupported dataset schema {info.get('schema')!r}")
        forbidden_model_fields = sorted(
            set(info) & {"history_length", "history_past_frames", "action_horizon"}
        )
        if forbidden_model_fields:
            raise ValueError(
                f"info.json contains model-specific window fields: {forbidden_model_fields}"
            )
        video_features = {
            key
            for key, value in info.get("features", {}).items()
            if value.get("dtype") == "video"
        }
        expected_video_features = set(CAMERA_VIDEO_KEYS.values())
        if video_features != expected_video_features:
            raise ValueError(
                f"info.json video features {video_features} != {expected_video_features}"
            )
        modality_video_keys = {
            value["original_key"] for value in modality.get("video", {}).values()
        }
        if modality_video_keys != expected_video_features:
            raise ValueError(
                "modality.json does not register exactly three camera views"
            )
        episode_rows = [
            json.loads(line)
            for line in (root / "meta/episodes.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        tasks = [
            json.loads(line)
            for line in (root / "meta/tasks.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        if len(tasks) != 1 or int(tasks[0].get("task_index", -1)) != 0:
            raise ValueError("tasks.jsonl must contain exactly task_index=0")
        if len(episode_rows) != int(info["total_episodes"]):
            raise ValueError("episodes.jsonl count differs from info.json")
        if [int(item["episode_index"]) for item in episode_rows] != list(
            range(len(episode_rows))
        ):
            raise ValueError("episodes.jsonl indices are not contiguous")
        source_manifest = _read_json(root / "meta/source_manifest.json")
        if int(source_manifest["selected_input_count"]) != len(episode_rows):
            raise ValueError("source_manifest input count mismatch")
        for metadata in episode_rows:
            try:
                result = verify_episode(root, info, metadata)
            except BaseException as error:
                failures.append(
                    {
                        "episode_index": int(metadata["episode_index"]),
                        "source_episode": metadata.get("source_episode"),
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": "".join(
                            traceback.format_exception(
                                type(error), error, error.__traceback__
                            )
                        ),
                    }
                )
            else:
                episodes_verified.append(result)
        stats = _verify_stats(root, info, len(episode_rows)) if not failures else {}
        total_rows = sum(int(item["rows"]) for item in episodes_verified)
        if not failures and total_rows != int(info["total_frames"]):
            raise ValueError("verified row total differs from info.json")
    except BaseException as error:
        failures.append(
            {
                "episode_index": None,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": "".join(
                    traceback.format_exception(type(error), error, error.__traceback__)
                ),
            }
        )
        info = {}
        stats = {}
        total_rows = sum(int(item["rows"]) for item in episodes_verified)
    report = {
        "schema": DATASET_SCHEMA,
        "dataset": str(root),
        "ok": not failures,
        "expected_episodes": int(info.get("total_episodes", 0)),
        "verified_episodes": len(episodes_verified),
        "verified_rows": total_rows,
        "three_camera_frame_match": not failures,
        "failures": failures,
        "stats": stats,
        "episodes": episodes_verified,
    }
    target = (
        Path(report_path)
        if report_path is not None
        else root / "meta/verification_report.json"
    )
    write_json(target, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = verify_dataset(args.dataset, report_path=args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
