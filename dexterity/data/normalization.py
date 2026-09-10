"""Load and apply statistics owned by a converted LeRobot dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

DEFAULT_STATS_FILENAME = "stats.json"


def load_stats(
    dataset_root: str | Path,
    *,
    filename: str = DEFAULT_STATS_FILENAME,
) -> dict[str, Any]:
    """Load statistics from this dataset's ``meta`` directory.

    Model-specific physical statistics may select a different metadata filename;
    the path is always resolved relative to the selected dataset, never a model
    source directory or a global registry.
    """

    path = Path(dataset_root) / "meta" / filename
    if not path.is_file():
        raise FileNotFoundError(f"dataset normalization statistics not found: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"normalization statistics must be a mapping: {path}")
    return value


def require_stats(stats: Mapping[str, Any], fields: tuple[str, ...]) -> None:
    missing = [field for field in fields if field not in stats]
    if missing:
        raise KeyError(f"dataset is missing required normalization fields: {missing}")


def normalize(
    value,
    field_stats: Mapping[str, Any],
    eps: float = 1e-6,
    *,
    clip: float | None = None,
):
    mean = np.asarray(field_stats["mean"], dtype=np.float32)
    std = np.asarray(field_stats["std"], dtype=np.float32)
    result = (value - mean) / np.maximum(std, eps)
    if clip is not None:
        result = np.clip(result, -float(clip), float(clip))
    return np.asarray(result, dtype=np.float32)


def unnormalize(value, field_stats: Mapping[str, Any], eps: float = 1e-6):
    mean = np.asarray(field_stats["mean"], dtype=np.float32)
    std = np.asarray(field_stats["std"], dtype=np.float32)
    return value * np.maximum(std, eps) + mean


class _RunningMoments:
    def __init__(self, dimension: int) -> None:
        self.count = np.zeros(dimension, dtype=np.int64)
        self.total = np.zeros(dimension, dtype=np.float64)
        self.total_square = np.zeros(dimension, dtype=np.float64)
        self.minimum = np.full(dimension, np.inf, dtype=np.float64)
        self.maximum = np.full(dimension, -np.inf, dtype=np.float64)

    def update(self, values: np.ndarray, valid: np.ndarray | None = None) -> None:
        source = np.asarray(values, dtype=np.float64)
        if source.shape[-1] != self.count.size:
            raise ValueError(
                f"statistics values must end in {self.count.size}, got {source.shape}"
            )
        array = source.reshape(-1, self.count.size)
        mask = np.isfinite(array)
        if valid is not None:
            given = np.asarray(valid, dtype=bool)
            if given.shape == source.shape[:-1]:
                given = np.broadcast_to(given[..., None], source.shape)
            else:
                given = np.broadcast_to(given, source.shape)
            mask &= given.reshape(array.shape)
        self.count += mask.sum(axis=0)
        safe = np.where(mask, array, 0.0)
        self.total += safe.sum(axis=0)
        self.total_square += (safe * safe).sum(axis=0)
        self.minimum = np.minimum(
            self.minimum,
            np.min(np.where(mask, array, np.inf), axis=0),
        )
        self.maximum = np.maximum(
            self.maximum,
            np.max(np.where(mask, array, -np.inf), axis=0),
        )

    def result(self, *, allow_empty: bool = False) -> dict[str, Any]:
        empty = self.count == 0
        if np.any(empty) and not allow_empty:
            raise ValueError(
                "statistics dimensions have no valid values: "
                f"{np.flatnonzero(empty).tolist()}"
            )
        denominator = np.maximum(self.count, 1)
        mean = self.total / denominator
        variance = np.maximum(
            self.total_square / denominator - mean * mean,
            0.0,
        )
        std = np.maximum(np.sqrt(variance), 1e-6)
        if np.any(empty):
            mean[empty] = 0.0
            std[empty] = 1.0
            self.minimum[empty] = 0.0
            self.maximum[empty] = 0.0
        return {
            "count": self.count.tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
        }


class _StreamingStatistics:
    """Exact moments with a bounded deterministic sample for q01/q99."""

    def __init__(self, dimension: int, total_rows: int, sample_rows: int = 200_000):
        if total_rows <= 0:
            raise ValueError("statistics require at least one row")
        self.dimension = int(dimension)
        self.total_rows = int(total_rows)
        self.rows_seen = 0
        self.moments = _RunningMoments(self.dimension)
        count = min(self.total_rows, int(sample_rows))
        self.sample_positions = np.unique(
            np.rint(np.linspace(0, self.total_rows - 1, count)).astype(np.int64)
        )
        self.samples: list[np.ndarray] = []

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2 or array.shape[1] != self.dimension:
            raise ValueError(
                f"statistics block must be [N,{self.dimension}], got {array.shape}"
            )
        start = self.rows_seen
        stop = start + len(array)
        if stop > self.total_rows:
            raise ValueError("statistics received more rows than declared")
        self.moments.update(array)
        lo = int(np.searchsorted(self.sample_positions, start, side="left"))
        hi = int(np.searchsorted(self.sample_positions, stop, side="left"))
        if hi > lo:
            self.samples.append(array[self.sample_positions[lo:hi] - start].copy())
        self.rows_seen = stop

    def result(self) -> dict[str, Any]:
        if self.rows_seen != self.total_rows:
            raise ValueError(f"statistics saw {self.rows_seen}/{self.total_rows} rows")
        result = self.moments.result()
        sample = np.concatenate(self.samples, axis=0).astype(np.float64)
        q01, q99 = np.quantile(sample, (0.01, 0.99), axis=0)
        result["q01"] = q01.tolist()
        result["q99"] = q99.tolist()
        result.pop("count")
        return result


def write_train_only_stats(dataset_root: str | Path) -> dict[str, Any]:
    """Write source-independent statistics for the canonical posttrain contract."""

    import pyarrow.parquet as pq

    from dexterity.data.posttrain import (
        ACTION_HORIZON,
        DISK_ACTION_DIM,
        HISTORY_PAST_FRAMES,
        POSTTRAIN_JOINT_ORDER,
        POSTTRAIN_TACTILE_ORDER,
        exact_anchor_indices,
        load_anchor_valid,
        relative_pose9_batch,
    )

    root = Path(dataset_root).resolve()
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    chunks_size = int(info["chunks_size"])
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
    total_anchors = sum(int(item["valid_anchor_count"]) for item in selected)
    total_targets = total_anchors * ACTION_HORIZON
    state_stats = _StreamingStatistics(62, total_anchors)
    action_stats = _StreamingStatistics(DISK_ACTION_DIM, total_targets)
    timestamp_stats = _StreamingStatistics(1, total_anchors)
    relative_left = _StreamingStatistics(9, total_targets)
    relative_right = _StreamingStatistics(9, total_targets)
    tau_stats = _RunningMoments(44)
    wrench_stats = _RunningMoments(60)
    sensor_digest = hashlib.sha256()
    train_indices: list[int] = []

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
        timestamp = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)
        anchors = exact_anchor_indices(load_anchor_valid(parquet_path))
        if len(anchors) != int(item["valid_anchor_count"]):
            raise ValueError(f"episode {episode_index} anchor count mismatch")
        state_stats.update(state[anchors])
        timestamp_stats.update(timestamp[anchors, None])
        for start in range(0, len(anchors), 256):
            block = anchors[start : start + 256]
            future = block[:, None] + np.arange(ACTION_HORIZON)[None, :]
            action_stats.update(action[future].reshape(-1, DISK_ACTION_DIM))
            relative_left.update(
                relative_pose9_batch(state[block, 0:9], action[future, 0:9]).reshape(
                    -1, 9
                )
            )
            relative_right.update(
                relative_pose9_batch(state[block, 9:18], action[future, 9:18]).reshape(
                    -1, 9
                )
            )

        sensor_path = root / "sensors/episodes" / f"episode_{episode_index:06d}.npz"
        with np.load(sensor_path, allow_pickle=False) as sensors:
            tau = np.asarray(sensors["tau"], dtype=np.float32)
            tau_valid = np.asarray(sensors["tau_valid_mask"], dtype=bool)
            wrench = np.asarray(sensors["tactile_wrench"], dtype=np.float32)
            wrench_valid = np.asarray(sensors["tactile_wrench_valid_mask"], dtype=bool)
        history = (
            anchors[:, None]
            + np.arange(-HISTORY_PAST_FRAMES, 1, dtype=np.int64)[None, :]
        )
        tau_stats.update(tau[history], tau_valid[history])
        wrench_stats.update(
            wrench[history].reshape(-1, 60),
            np.broadcast_to(
                wrench_valid[history][..., None],
                (*wrench_valid[history].shape, 6),
            ).reshape(-1, 60),
        )
        sensor_digest.update(sensor_path.name.encode("utf-8"))
        sensor_digest.update(np.ascontiguousarray(tau).tobytes())
        sensor_digest.update(np.ascontiguousarray(tau_valid).tobytes())
        sensor_digest.update(np.ascontiguousarray(wrench).tobytes())
        sensor_digest.update(np.ascontiguousarray(wrench_valid).tobytes())
        train_indices.append(episode_index)

    wrench_result = wrench_stats.result(allow_empty=True)
    for key in ("count", "mean", "std", "min", "max"):
        wrench_result[key] = np.asarray(wrench_result[key]).reshape(10, 6).tolist()
    sensor = {
        "schema": "sharpa.sensor_normalization.v2",
        "split": "train",
        "train_episode_indices": train_indices,
        "train_episode_count": len(train_indices),
        "joint_order": list(POSTTRAIN_JOINT_ORDER),
        "finger_order": list(POSTTRAIN_TACTILE_ORDER),
        "wrench_order": ["fx", "fy", "fz", "tx", "ty", "tz"],
        "tau": tau_stats.result(allow_empty=True),
        "wrench": wrench_result,
        "input_sha256": sensor_digest.hexdigest(),
    }
    stats = {
        "observation.state": state_stats.result(),
        "action": action_stats.result(),
        "timestamp": timestamp_stats.result(),
    }
    relative = {
        "left_wrist_eef": relative_left.result(),
        "right_wrist_eef": relative_right.result(),
    }
    provenance = {
        "schema": "sharpa.control_sensors.train_only_stats_provenance.v2",
        "split": "train",
        "train_episode_indices": train_indices,
        "train_episode_count": len(train_indices),
        "sensor_input_sha256": sensor_digest.hexdigest(),
        "statistics_method": {
            "mean_std_min_max": "exact_streaming",
            "q01_q99": "deterministic_uniform_global_row_sample",
            "maximum_quantile_sample_rows": 200_000,
        },
        "sampling": {
            "state": "exact_anchor_rows",
            "action": "all_40_canonical_action_rows_per_exact_anchor",
            "relative_action": "relative_to_anchor_wrist",
            "sensor": "all_9_history_rows_per_exact_anchor",
        },
    }
    meta = root / "meta"
    for name, value in (
        ("sensor_stats.json", sensor),
        ("stats.json", stats),
        ("relative_stats.json", relative),
        ("stats_provenance.json", provenance),
    ):
        (meta / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return provenance


__all__ = [
    "load_stats",
    "normalize",
    "require_stats",
    "unnormalize",
    "write_train_only_stats",
]
