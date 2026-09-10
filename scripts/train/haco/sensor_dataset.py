from __future__ import annotations

import json
from collections import OrderedDict
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
from typing import Any
import zipfile

import numpy as np

from dexterity.data.normalization import normalize
from dexterity.data.posttrain import (
    SensorEpisode,
    action_indices,
    exact_anchor_indices,
    history_indices,
    load_anchor_valid,
)


def sensor_cache_root(dataset_root: str | Path) -> Path:
    root = Path(dataset_root).resolve()
    base = Path(
        os.environ.get(
            "HACO_SENSOR_CACHE_DIR",
            f"/tmp/haco-{os.getuid()}/sensor_cache",
        )
    )
    namespace = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:16]
    return base / f"{root.name}-{namespace}"


def _valid_tactile_cache(path: Path, source: Path) -> bool:
    if not path.is_file() or path.stat().st_mtime_ns < source.stat().st_mtime_ns:
        return False
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        valid = (
            value.dtype == np.uint8
            and value.ndim == 4
            and tuple(value.shape[1:]) == (10, 240, 240)
        )
        del value
        return valid
    except (OSError, ValueError):
        return False


def ensure_tactile_cache(dataset_root: str | Path, source: str | Path) -> Path:
    """Extract the large compressed tactile member into a local mmap-able NPY."""
    source = Path(source)
    target = sensor_cache_root(dataset_root) / f"{source.stem}.tactile.npy"
    target.parent.mkdir(parents=True, exist_ok=True)
    if _valid_tactile_cache(target, source):
        return target

    lock_path = target.with_suffix(target.suffix + ".lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if _valid_tactile_cache(target, source):
            return target
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        try:
            with zipfile.ZipFile(source) as archive:
                with archive.open("tactile_deformation.npy") as compressed:
                    with temporary.open("wb") as output:
                        shutil.copyfileobj(compressed, output, length=16 * 1024 * 1024)
                        output.flush()
                        os.fsync(output.fileno())
            value = np.load(temporary, mmap_mode="r", allow_pickle=False)
            if (
                value.dtype != np.uint8
                or value.ndim != 4
                or tuple(value.shape[1:]) != (10, 240, 240)
            ):
                raise ValueError(
                    f"bad tactile cache payload from {source}: "
                    f"shape={value.shape}, dtype={value.dtype}"
                )
            del value
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
    return target


def load_cached_sensor_episode(
    dataset_root: str | Path,
    source: str | Path,
) -> SensorEpisode:
    """Load small sensor arrays normally and tactile frames from local mmap."""
    source = Path(source)
    tactile_path = ensure_tactile_cache(dataset_root, source)
    with np.load(source, allow_pickle=False) as payload:
        sensors = SensorEpisode(
            tau=np.asarray(payload["tau"], dtype=np.float32),
            tau_valid_mask=np.asarray(payload["tau_valid_mask"], dtype=bool),
            tactile_wrench=np.asarray(payload["tactile_wrench"], dtype=np.float32),
            tactile_wrench_valid_mask=np.asarray(
                payload["tactile_wrench_valid_mask"], dtype=bool
            ),
            tactile_deformation=np.load(
                tactile_path, mmap_mode="r", allow_pickle=False
            ),
            tactile_deformation_valid_mask=np.asarray(
                payload["tactile_deformation_valid_mask"], dtype=bool
            ),
            frame_index=np.asarray(payload["frame_index"], dtype=np.int64),
            timestamp=np.asarray(payload["timestamp"], dtype=np.float64),
            source_timeline_row=np.asarray(
                payload["source_timeline_row"], dtype=np.int32
            ),
        )
    sensors.validate()
    return sensors


class HacoSensorStore:
    def __init__(
        self,
        dataset_root: str | Path,
        *,
        clip: float | None = 10.0,
        cache_size: int = 2,
    ) -> None:
        self.root = Path(dataset_root)
        self.clip = clip
        self.cache_size = int(cache_size)
        if self.cache_size <= 0:
            raise ValueError("cache_size must be positive")
        stats_path = self.root / "meta/sensor_stats.json"
        self.stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if self.stats.get("split") != "train":
            raise ValueError("HACO normalization statistics must be train-only")
        self._cache: OrderedDict[int, Any] = OrderedDict()
        info = json.loads((self.root / "meta/info.json").read_text(encoding="utf-8"))
        self.chunks_size = int(info["chunks_size"])
        if "anchor_valid" not in info.get("features", {}):
            raise ValueError("HACO requires the canonical posttrain v2 contract")
        self.sensor_dir = self.root / "sensors/episodes"
        self._anchor_cache: dict[int, np.ndarray] = {}

    def parquet_path(self, episode_index: int) -> Path:
        chunk = episode_index // self.chunks_size
        return self.root / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"

    def anchor_valid(self, episode_index: int) -> np.ndarray:
        if episode_index not in self._anchor_cache:
            self._anchor_cache[episode_index] = load_anchor_valid(
                self.parquet_path(episode_index)
            )
        return self._anchor_cache[episode_index]

    def episode(self, episode_index: int):
        if episode_index not in self._cache:
            path = self.sensor_dir / f"episode_{episode_index:06d}.npz"
            self._cache[episode_index] = load_cached_sensor_episode(self.root, path)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(episode_index)
        return self._cache[episode_index]

    def metadata(self, episode_index: int, anchor: int) -> dict[str, np.ndarray]:
        payload = self.episode(episode_index)
        anchor_valid = self.anchor_valid(episode_index)
        if anchor < 0 or anchor >= payload.length or not anchor_valid[anchor]:
            raise ValueError(f"episode {episode_index} row {anchor} is not a HACO anchor")
        history = history_indices(anchor)
        future = action_indices(anchor, payload.length)
        tau = normalize(payload.tau[history], self.stats["tau"], clip=self.clip)
        wrench = normalize(
            payload.tactile_wrench[history], self.stats["wrench"], clip=self.clip
        )
        tau_valid = payload.tau_valid_mask[history].copy()
        tactile_valid = payload.tactile_wrench_valid[history].copy()
        tau[~tau_valid] = 0.0
        wrench[~tactile_valid] = 0.0
        deformation = payload.tactile_deformation[anchor].copy()
        deformation_valid = payload.tactile_deformation_valid[anchor].copy()
        deformation[~deformation_valid] = 0
        return {
            "force_history": tau.T.astype(np.float32),
            "force_history_valid": tau_valid.T,
            "tactile_wrench_history": wrench.transpose(1, 0, 2),
            "tactile_wrench_valid": tactile_valid.T,
            "tactile_deformation": deformation,
            "tactile_deformation_valid": deformation_valid,
            # HACO predicts wrist/q_exe/delta_q (106-D) from the 150-D disk
            # action. Exact anchors guarantee that every transition is valid.
            "action_component_valid": np.ones((len(future), 106), dtype=bool),
        }


def exact_steps_for_episode(dataset_root: str | Path, episode_index: int) -> np.ndarray:
    root = Path(dataset_root)
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    chunk = episode_index // int(info["chunks_size"])
    path = root / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
    return exact_anchor_indices(load_anchor_valid(path))
