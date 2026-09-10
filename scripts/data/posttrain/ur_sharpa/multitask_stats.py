#!/usr/bin/env python3
"""Compute task-balanced HACO normalization from canonical SharpA datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from scripts.train.haco.config import validate_training_dataset


SCHEMA = "sharpa.haco_multitask_normalization.v1"
PROVENANCE_SCHEMA = "sharpa.haco_multitask_equal_stats_provenance.v1"
SENSOR_SCHEMA = "sharpa.sensor_normalization.multitask.v1"
QUANTILE_ROWS_PER_TASK = 40_000


class RunningMoments:
    """Per-dimension streaming moments with optional validity masks."""

    def __init__(self, dimension: int) -> None:
        self.count = np.zeros(dimension, dtype=np.int64)
        self.total = np.zeros(dimension, dtype=np.float64)
        self.total_square = np.zeros(dimension, dtype=np.float64)
        self.minimum = np.full(dimension, np.inf, dtype=np.float64)
        self.maximum = np.full(dimension, -np.inf, dtype=np.float64)

    def update(self, values: np.ndarray, valid: np.ndarray | None = None) -> None:
        source = np.asarray(values, dtype=np.float64)
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
            self.minimum, np.min(np.where(mask, array, np.inf), axis=0)
        )
        self.maximum = np.maximum(
            self.maximum, np.max(np.where(mask, array, -np.inf), axis=0)
        )

    def result(self, *, allow_empty: bool = False) -> dict[str, Any]:
        empty = self.count == 0
        if np.any(empty) and not allow_empty:
            raise ValueError("statistics dimensions have no valid values")
        denominator = np.maximum(self.count, 1)
        mean = self.total / denominator
        variance = np.maximum(self.total_square / denominator - mean * mean, 0.0)
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


class SampledMoments:
    """Exact moments plus deterministic uniform rows for quantiles."""

    def __init__(self, dimension: int, total_rows: int) -> None:
        if total_rows <= 0:
            raise ValueError("statistics require positive row counts")
        self.dimension = int(dimension)
        self.total_rows = int(total_rows)
        self.rows_seen = 0
        self.moments = RunningMoments(self.dimension)
        count = min(self.total_rows, QUANTILE_ROWS_PER_TASK)
        self.positions = np.unique(
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
        first = int(np.searchsorted(self.positions, start, side="left"))
        last = int(np.searchsorted(self.positions, stop, side="left"))
        if last > first:
            self.samples.append(array[self.positions[first:last] - start].copy())
        self.rows_seen = stop

    def result(self) -> tuple[dict[str, Any], np.ndarray]:
        if self.rows_seen != self.total_rows:
            raise ValueError(
                f"statistics saw {self.rows_seen}/{self.total_rows} rows"
            )
        return self.moments.result(), np.concatenate(self.samples, axis=0)


def _merge_equal_task_moments(
    results: list[dict[str, Any]],
    *,
    samples: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Merge moments with equal mass per task, not per source row."""

    counts = np.asarray([item["count"] for item in results], dtype=np.int64)
    means = np.asarray([item["mean"] for item in results], dtype=np.float64)
    stds = np.asarray([item["std"] for item in results], dtype=np.float64)
    valid = counts > 0
    task_weights = valid / np.maximum(valid.sum(axis=0, keepdims=True), 1)
    mean = (task_weights * means).sum(axis=0)
    second = (task_weights * (stds * stds + means * means)).sum(axis=0)
    variance = np.maximum(second - mean * mean, 0.0)
    minimum = np.min(
        np.where(valid, np.asarray([item["min"] for item in results]), np.inf),
        axis=0,
    )
    maximum = np.max(
        np.where(valid, np.asarray([item["max"] for item in results]), -np.inf),
        axis=0,
    )
    output = {
        "count": counts.sum(axis=0).tolist(),
        "mean": mean.tolist(),
        "std": np.maximum(np.sqrt(variance), 1e-6).tolist(),
        "min": minimum.tolist(),
        "max": maximum.tolist(),
    }
    if samples is not None:
        rows_per_task = {len(value) for value in samples}
        if len(rows_per_task) != 1:
            raise ValueError("quantile samples must contribute equally per task")
        sample = np.concatenate(samples, axis=0).astype(np.float64)
        q01, q99 = np.quantile(sample, (0.01, 0.99), axis=0)
        output["q01"] = q01.tolist()
        output["q99"] = q99.tolist()
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_records(dataset: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in (dataset / "meta/episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    selected = [item for item in records if item.get("split") == "train"]
    if not selected:
        raise ValueError(f"dataset has no train episodes: {dataset}")
    return selected


def _compute_task(dataset: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    info_path = dataset / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    episodes = _episode_records(dataset)
    total_rows = sum(int(item["length"]) for item in episodes)
    state = SampledMoments(62, total_rows)
    action = SampledMoments(150, total_rows)
    timestamp = SampledMoments(1, total_rows)
    tau = RunningMoments(44)
    wrench = RunningMoments(60)
    digest = hashlib.sha256()
    chunk_size = int(info["chunks_size"])

    for item in episodes:
        episode_index = int(item["episode_index"])
        chunk = episode_index // chunk_size
        parquet = dataset / (
            f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
        )
        table = pq.read_table(
            parquet,
            columns=["observation.state", "action", "timestamp"],
        )
        state_rows = np.asarray(
            table["observation.state"].to_pylist(), dtype=np.float32
        )
        action_rows = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        timestamp_rows = np.asarray(
            table["timestamp"].to_numpy(), dtype=np.float64
        )[:, None]
        expected = int(item["length"])
        if len(state_rows) != expected or len(action_rows) != expected:
            raise ValueError(f"episode {episode_index} length changed")
        state.update(state_rows)
        action.update(action_rows)
        timestamp.update(timestamp_rows)
        digest.update(np.ascontiguousarray(state_rows).tobytes())
        digest.update(np.ascontiguousarray(action_rows).tobytes())

        sensor_path = (
            dataset / "sensors/episodes" / f"episode_{episode_index:06d}.npz"
        )
        with np.load(sensor_path, allow_pickle=False) as sensors:
            tau_rows = np.asarray(sensors["tau"], dtype=np.float32)
            tau_valid = np.asarray(sensors["tau_valid_mask"], dtype=bool)
            wrench_rows = np.asarray(sensors["tactile_wrench"], dtype=np.float32)
            wrench_valid = np.asarray(
                sensors["tactile_wrench_valid_mask"], dtype=bool
            )
        tau.update(tau_rows, tau_valid)
        wrench.update(
            wrench_rows.reshape(-1, 60),
            np.broadcast_to(
                wrench_valid[..., None], (*wrench_valid.shape, 6)
            ).reshape(-1, 60),
        )
        for value in (tau_rows, tau_valid, wrench_rows, wrench_valid):
            digest.update(np.ascontiguousarray(value).tobytes())

    state_result, state_sample = state.result()
    action_result, action_sample = action.result()
    timestamp_result, timestamp_sample = timestamp.result()
    return {
        "dataset": dataset,
        "info": info,
        "episodes": episodes,
        "input_sha256": digest.hexdigest(),
        "metadata_sha256": {
            "info.json": _sha256(info_path),
            "episodes.jsonl": _sha256(dataset / "meta/episodes.jsonl"),
        },
        "state": (state_result, state_sample),
        "action": (action_result, action_sample),
        "timestamp": (timestamp_result, timestamp_sample),
        "tau": tau.result(allow_empty=True),
        "wrench": wrench.result(allow_empty=True),
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_multitask_stats(
    dataset_paths: list[str | Path], output_dir: str | Path
) -> dict[str, Any]:
    datasets = [validate_training_dataset(value) for value in dataset_paths]
    if len(datasets) != 5 or len(set(datasets)) != 5:
        raise ValueError("HACO MT5 normalization requires five distinct datasets")
    results = [_compute_task(dataset) for dataset in datasets]

    def merge_field(name: str) -> dict[str, Any]:
        return _merge_equal_task_moments(
            [item[name][0] for item in results],
            samples=[item[name][1] for item in results],
        )

    stats = {
        "observation.state": merge_field("state"),
        "action": merge_field("action"),
        "timestamp": merge_field("timestamp"),
    }
    for value in stats.values():
        value.pop("count")
    tau = _merge_equal_task_moments([item["tau"] for item in results])
    wrench_flat = _merge_equal_task_moments(
        [item["wrench"] for item in results]
    )
    wrench = {
        key: np.asarray(value).reshape(10, 6).tolist()
        for key, value in wrench_flat.items()
    }
    weight = 1.0 / len(results)
    dataset_manifest = [
        {
            "task_id": Path(item["dataset"]).name,
            "path": str(item["dataset"]),
            "weight": weight,
            "total_episodes": int(item["info"]["total_episodes"]),
            "total_frames": int(item["info"]["total_frames"]),
            "input_sha256": item["input_sha256"],
            "metadata_sha256": item["metadata_sha256"],
        }
        for item in results
    ]
    combined_digest = hashlib.sha256(
        json.dumps(dataset_manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
    sensor = {
        "schema": SENSOR_SCHEMA,
        "split": "train",
        "task_weighting": "equal",
        "datasets": dataset_manifest,
        "train_episode_count": sum(
            int(item["info"]["total_episodes"]) for item in results
        ),
        "joint_order": results[0]["info"]["joint_order"],
        "finger_order": results[0]["info"]["finger_order"],
        "wrench_order": ["fx", "fy", "fz", "tx", "ty", "tz"],
        "tau": tau,
        "wrench": wrench,
        "input_sha256": combined_digest,
    }
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "split": "train",
        "task_weighting": "equal",
        "datasets": dataset_manifest,
        "train_episode_count": sensor["train_episode_count"],
        "train_frame_count": sum(
            int(item["info"]["total_frames"]) for item in results
        ),
        "input_sha256": combined_digest,
        "statistics_method": {
            "mean_std_min_max": "exact_streaming_per_task_then_equal_task_mixture",
            "q01_q99": "deterministic_equal_rows_per_task",
            "quantile_rows_per_task": QUANTILE_ROWS_PER_TASK,
        },
        "sampling": {
            "state": "all cleaned train rows with equal task mass",
            "action": "all cleaned train rows with equal task mass",
            "sensor": "all valid train rows with equal task mass",
            "tactile_deformation": "uint8 image input; no scalar normalization",
        },
    }
    manifest = {
        "schema": SCHEMA,
        "split": "train",
        "task_weighting": "equal",
        "datasets": dataset_manifest,
        "normalization_files": [
            "stats.json",
            "sensor_stats.json",
            "stats_provenance.json",
        ],
        "input_sha256": combined_digest,
    }
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"normalization output is not empty: {output}")
    for name, value in (
        ("stats.json", stats),
        ("sensor_stats.json", sensor),
        ("stats_provenance.json", provenance),
        ("mixture_manifest.json", manifest),
    ):
        _atomic_json(output / name, value)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = write_multitask_stats(args.dataset_path, args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
