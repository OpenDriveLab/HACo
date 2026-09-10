from __future__ import annotations

import fcntl
import json
import os
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import Sampler

from dexterity.data.model_stats import model_stats_path
from dexterity.runtime.sharpa62 import MODEL_JOINT_ORDER, MODEL_TACTILE_ORDER

ACTION_DIM = 62
DISK_ACTION_DIM = 150
STATE_DIM = 62
WRIST_DIM = 18
JOINT_COUNT = 44
TACTILE_FINGERS = 10
WRENCH_DIM = 6
DEFORMATION_HEIGHT = 240
DEFORMATION_WIDTH = 240
HISTORY_LENGTH = 9
HISTORY_PAST_FRAMES = HISTORY_LENGTH - 1
ACTION_HORIZON = 40
Q_EXE_NEXT_SLICE = slice(0, 62)
Q_TELEOP_SLICE = slice(62, 106)
DELTA_Q_SLICE = slice(106, 150)
POSTTRAIN_JOINT_ORDER = MODEL_JOINT_ORDER
POSTTRAIN_TACTILE_ORDER = MODEL_TACTILE_ORDER
TREX_ACTION_CHUNK = 16
VITAC_ACTION_CHUNK = 100
VITAC_STATE_OFFSETS = np.asarray([-15, -12, -9, -6, -3, 0], dtype=np.int64)
VITAC_TACTILE_HISTORY = 18
VITAC_TACTILE_FUTURE = 18
TACTILE_DIM = 60

# GCC hand order -> T-Rex native order (left thumb-first, then right thumb-first).
GCC_TO_TREX_HAND = np.asarray(
    [
        17,
        18,
        19,
        20,
        21,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        13,
        14,
        15,
        16,
        8,
        9,
        10,
        11,
        12,
        39,
        40,
        41,
        42,
        43,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        35,
        36,
        37,
        38,
        30,
        31,
        32,
        33,
        34,
    ],
    dtype=np.int64,
)
GCC_TO_TREX_TACTILE = np.arange(9, -1, -1, dtype=np.int64)
# GCC order is pinky->thumb for each hand; ViTacFormer is right thumb->pinky,
# then left thumb->pinky.
GCC_TO_VITAC_TACTILE = np.asarray([4, 3, 2, 1, 0, 9, 8, 7, 6, 5], dtype=np.int64)


@dataclass(frozen=True)
class Episode:
    episode_index: int
    length: int
    split: str
    task: str


@dataclass(frozen=True)
class Anchor:
    episode_index: int
    frame_index: int


@dataclass(frozen=True)
class SensorEpisode:
    """Optional sensor payload aligned one-to-one with canonical Parquet rows."""

    tau: np.ndarray
    tau_valid_mask: np.ndarray
    tactile_wrench: np.ndarray
    tactile_wrench_valid_mask: np.ndarray
    tactile_deformation: np.ndarray
    tactile_deformation_valid_mask: np.ndarray
    frame_index: np.ndarray
    timestamp: np.ndarray
    source_timeline_row: np.ndarray

    @property
    def length(self) -> int:
        return int(self.tau.shape[0])

    def validate(self) -> None:
        expected = {
            "tau": (self.length, JOINT_COUNT),
            "tau_valid_mask": (self.length, JOINT_COUNT),
            "tactile_wrench": (self.length, TACTILE_FINGERS, WRENCH_DIM),
            "tactile_wrench_valid_mask": (self.length, TACTILE_FINGERS),
            "tactile_deformation": (
                self.length,
                TACTILE_FINGERS,
                DEFORMATION_HEIGHT,
                DEFORMATION_WIDTH,
            ),
            "tactile_deformation_valid_mask": (
                self.length,
                TACTILE_FINGERS,
            ),
            "frame_index": (self.length,),
            "timestamp": (self.length,),
            "source_timeline_row": (self.length,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
        for name in ("tau", "tactile_wrench", "timestamp"):
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError(f"{name} contains NaN or Inf")
        if self.tactile_deformation.dtype != np.uint8:
            raise ValueError("tactile_deformation must be uint8")
        if not np.array_equal(self.frame_index, np.arange(self.length)):
            raise ValueError("sensor frame_index must be contiguous from zero")

    @property
    def tactile_wrench_valid(self) -> np.ndarray:
        return self.tactile_wrench_valid_mask

    @property
    def tactile_deformation_valid(self) -> np.ndarray:
        return self.tactile_deformation_valid_mask


def load_sensors(path: str | Path, *, mmap_mode: str | None = None) -> SensorEpisode:
    with np.load(path, mmap_mode=mmap_mode, allow_pickle=False) as payload:
        sensors = SensorEpisode(
            tau=np.asarray(payload["tau"], dtype=np.float32),
            tau_valid_mask=np.asarray(payload["tau_valid_mask"], dtype=bool),
            tactile_wrench=np.asarray(payload["tactile_wrench"], dtype=np.float32),
            tactile_wrench_valid_mask=np.asarray(
                payload["tactile_wrench_valid_mask"], dtype=bool
            ),
            tactile_deformation=np.asarray(
                payload["tactile_deformation"], dtype=np.uint8
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


def load_anchor_valid(parquet_path: str | Path) -> np.ndarray:
    table = pq.read_table(parquet_path, columns=["anchor_valid"])
    result = np.asarray(table["anchor_valid"].to_numpy(), dtype=bool)
    if result.ndim != 1:
        raise ValueError("anchor_valid must be one-dimensional")
    return result


def strict_anchor_indices(length: int) -> np.ndarray:
    if length <= HISTORY_PAST_FRAMES + ACTION_HORIZON:
        return np.empty(0, dtype=np.int64)
    return np.arange(
        HISTORY_PAST_FRAMES,
        length - ACTION_HORIZON,
        dtype=np.int64,
    )


def exact_anchor_indices(anchor_valid: np.ndarray) -> np.ndarray:
    mask = np.asarray(anchor_valid, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("anchor_valid must be one-dimensional")
    return np.flatnonzero(mask).astype(np.int64)


def history_indices(anchor: int) -> np.ndarray:
    if anchor < HISTORY_PAST_FRAMES:
        raise ValueError(f"anchor must be >= {HISTORY_PAST_FRAMES}")
    return np.arange(anchor - HISTORY_PAST_FRAMES, anchor + 1, dtype=np.int64)


def action_indices(anchor: int, length: int) -> np.ndarray:
    indices = np.arange(anchor, anchor + ACTION_HORIZON, dtype=np.int64)
    if indices[-1] >= length - 1:
        raise ValueError(f"anchor {anchor} has no strict future in {length}")
    return indices


def _fixed_list_to_numpy(table, name: str, width: int) -> np.ndarray:
    column = table[name].combine_chunks()
    values = column.values.to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.float32).reshape(-1, width)


def pose9_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    c1 = pose[3:6]
    c1 = c1 / max(float(np.linalg.norm(c1)), 1e-8)
    c2 = pose[6:9] - c1 * float(np.dot(c1, pose[6:9]))
    c2 = c2 / max(float(np.linalg.norm(c2)), 1e-8)
    c3 = np.cross(c1, c2)
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = np.column_stack([c1, c2, c3])
    out[:3, 3] = pose[:3]
    return out


def relative_pose9(base: np.ndarray, target: np.ndarray) -> np.ndarray:
    base_m = pose9_to_matrix(base)
    target_m = pose9_to_matrix(target)
    rotation = base_m[:3, :3].T @ target_m[:3, :3]
    translation = base_m[:3, :3].T @ (target_m[:3, 3] - base_m[:3, 3])
    return np.concatenate([translation, rotation[:, 0], rotation[:, 1]]).astype(
        np.float32
    )


def relative_pose9_batch(base: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Vectorized relative pose for base [N,9] and targets [N,T,9]."""
    base = np.asarray(base, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    def rotations(pose: np.ndarray) -> np.ndarray:
        col1 = pose[..., 3:6]
        col1 = col1 / np.maximum(np.linalg.norm(col1, axis=-1, keepdims=True), 1e-8)
        col2 = pose[..., 6:9] - col1 * np.sum(
            col1 * pose[..., 6:9], axis=-1, keepdims=True
        )
        col2 = col2 / np.maximum(np.linalg.norm(col2, axis=-1, keepdims=True), 1e-8)
        col3 = np.cross(col1, col2)
        return np.stack([col1, col2, col3], axis=-1)

    base_rotation_t = np.swapaxes(rotations(base), -1, -2)
    target_rotation = rotations(target)
    rotation = np.einsum("nij,ntjk->ntik", base_rotation_t, target_rotation)
    delta = target[..., :3] - base[:, None, :3]
    translation = np.einsum("nij,ntj->nti", base_rotation_t, delta)
    return np.concatenate(
        [translation, rotation[..., :, 0], rotation[..., :, 1]], axis=-1
    ).astype(np.float32)


def trex_native_state(state: np.ndarray) -> np.ndarray:
    hand = np.asarray(state[18:62], dtype=np.float32)[GCC_TO_TREX_HAND]
    return np.concatenate([state[:9], hand[:22], state[9:18], hand[22:]]).astype(
        np.float32
    )


def trex_action_chunk(
    state: np.ndarray,
    action: np.ndarray,
    anchor: int,
) -> np.ndarray:
    """Build the official T-Rex task target from future executed actions."""

    # action[t] is the transition to physical frame t+1.  The terminal action
    # row repeats the last state, so a real 16-step target needs frame t+16 to
    # exist rather than clipping the last rows to the repeated terminal value.
    if anchor < 0 or anchor + TREX_ACTION_CHUNK >= len(action):
        raise ValueError(
            f"anchor {anchor} has no complete {TREX_ACTION_CHUNK}-step "
            f"T-Rex target in episode length {len(action)}"
        )

    base_left = state[anchor, :9]
    base_right = state[anchor, 9:18]
    chunk = np.empty((TREX_ACTION_CHUNK, ACTION_DIM), dtype=np.float32)
    for offset in range(TREX_ACTION_CHUNK):
        index = anchor + offset
        q_actual = action[index, 18:62][GCC_TO_TREX_HAND]
        chunk[offset] = np.concatenate(
            [
                relative_pose9(base_left, action[index, :9]),
                q_actual[:22],
                relative_pose9(base_right, action[index, 9:18]),
                q_actual[22:],
            ]
        )
    return chunk


def vitac_action_chunk(
    state: np.ndarray, action: np.ndarray, anchor: int
) -> tuple[np.ndarray, np.ndarray]:
    base_left = state[anchor, :9]
    base_right = state[anchor, 9:18]
    chunk = np.empty((VITAC_ACTION_CHUNK, ACTION_DIM), dtype=np.float32)
    is_pad = np.zeros(VITAC_ACTION_CHUNK, dtype=bool)
    for offset in range(VITAC_ACTION_CHUNK):
        raw_index = anchor + offset
        index = min(raw_index, len(action) - 1)
        # The last disk row repeats the terminal state and is not a real
        # transition target.
        is_pad[offset] = raw_index >= len(action) - 1
        chunk[offset] = np.concatenate(
            [
                relative_pose9(base_left, action[index, :9]),
                relative_pose9(base_right, action[index, 9:18]),
                action[index, 18:62],
            ]
        )
    return chunk, is_pad


class PickPlaceStore:
    """PACE episode access shared by T-REX and ViTacFormer adapters.

    RGB remains canonical LeRobot video. Decoders are process-local and kept in
    a small LRU so DataLoader workers can batch sparse frame requests without a
    persistent, model-specific frame cache.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        decoder_cache_size: int = 2,
        episode_cache_size: int = 2,
    ):
        self.root = Path(dataset_root).resolve()
        if decoder_cache_size < 1 or episode_cache_size < 1:
            raise ValueError("decoder and episode cache sizes must be positive")
        self.decoder_cache_size = int(decoder_cache_size)
        self.episode_cache_size = int(episode_cache_size)
        info = json.loads((self.root / "meta/info.json").read_text(encoding="utf-8"))
        self.fps = int(info["fps"])
        action_feature = info.get("features", {}).get("action", {})
        action_shape = action_feature.get("shape", [info.get("action_dim", 0)])
        if not isinstance(action_shape, list) or len(action_shape) != 1:
            raise ValueError(f"unsupported action feature shape: {action_shape!r}")
        self.source_action_dim = int(action_shape[0])
        if self.source_action_dim != DISK_ACTION_DIM:
            raise ValueError(
                f"canonical posttrain action dim must be {DISK_ACTION_DIM}, "
                f"got {self.source_action_dim}"
            )
        if "anchor_valid" not in info.get("features", {}):
            raise ValueError("dataset is not the canonical posttrain v2 contract")
        if not (self.root / "sensors/episodes").is_dir():
            raise FileNotFoundError(f"missing canonical sensors directory: {self.root}")
        self.episodes: dict[int, Episode] = {}
        for line in (
            (self.root / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines()
        ):
            item = json.loads(line)
            episode_index = int(item["episode_index"])
            tasks = item.get("tasks") or [""]
            self.episodes[episode_index] = Episode(
                episode_index=episode_index,
                length=int(item["length"]),
                split=str(item["split"]),
                task=str(tasks[0]),
            )
        self.train_anchors: list[Anchor] = []
        self.anchors_by_episode: dict[int, list[int]] = {}
        for episode in self.episodes.values():
            if episode.split != "train":
                continue
            table = pq.read_table(
                self.parquet_path(episode.episode_index),
                columns=["anchor_valid"],
            )
            anchor_valid = np.asarray(table["anchor_valid"].to_numpy(), dtype=bool)
            indices = np.flatnonzero(anchor_valid).astype(np.int64)
            self.anchors_by_episode[episode.episode_index] = indices.tolist()
            self.train_anchors.extend(
                Anchor(episode.episode_index, int(index)) for index in indices
            )
        if not self.train_anchors:
            raise ValueError(f"no train anchors in {self.root}")
        self._episode_cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self._video_decoders: OrderedDict[tuple[int, str], object] = OrderedDict()

    def parquet_path(self, episode_index: int) -> Path:
        return (
            self.root
            / "data"
            / f"chunk-{episode_index // 1000:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )

    def sidecar_path(self, episode_index: int) -> Path:
        return self.root / "sensors" / "episodes" / f"episode_{episode_index:06d}.npz"

    def video_path(
        self,
        episode_index: int,
        video_key: str = "observation.images.ego_view",
    ) -> Path:
        return (
            self.root
            / "videos"
            / f"chunk-{episode_index // 1000:03d}"
            / video_key
            / f"episode_{episode_index:06d}.mp4"
        )

    def load_episode(self, episode_index: int) -> dict[str, np.ndarray]:
        cached = self._episode_cache.pop(episode_index, None)
        if cached is not None:
            self._episode_cache[episode_index] = cached
            return cached
        table = pq.read_table(self.parquet_path(episode_index))
        data: dict[str, np.ndarray] = {
            "state": _fixed_list_to_numpy(table, "observation.state", 62),
            "action": _fixed_list_to_numpy(table, "action", self.source_action_dim),
        }
        with np.load(self.sidecar_path(episode_index)) as sidecar:
            data["delta_q"] = data["action"][:, DELTA_Q_SLICE].copy()
            data["q_teleop"] = data["action"][:, Q_TELEOP_SLICE].copy()
            data["tactile_wrench"] = sidecar["tactile_wrench"].copy()
            data["tactile_wrench_valid"] = sidecar["tactile_wrench_valid_mask"].copy()
            data["tactile_deformation"] = sidecar["tactile_deformation"].copy()
            data["tactile_deformation_valid"] = sidecar[
                "tactile_deformation_valid_mask"
            ].copy()
            data["anchor_valid"] = np.asarray(
                table["anchor_valid"].to_numpy(), dtype=bool
            )
        self._episode_cache[episode_index] = data
        while len(self._episode_cache) > self.episode_cache_size:
            self._episode_cache.popitem(last=False)
        return data

    def rgb(
        self,
        episode_index: int,
        indices: np.ndarray | list[int],
        *,
        video_key: str = "observation.images.ego_view",
    ) -> np.ndarray:
        rows = np.asarray(indices, dtype=np.int64)
        if rows.ndim != 1:
            raise ValueError(f"RGB indices must be one-dimensional, got {rows.shape}")
        episode = self.episodes[episode_index]
        if np.any((rows < 0) | (rows >= episode.length)):
            raise IndexError(f"RGB row is outside episode {episode_index}")

        decoder_key = (int(episode_index), str(video_key))
        decoder = self._video_decoders.pop(decoder_key, None)
        if decoder is None:
            from torchcodec.decoders import VideoDecoder

            decoder = VideoDecoder(
                str(self.video_path(episode_index, video_key)),
                device="cpu",
                dimension_order="NHWC",
                num_ffmpeg_threads=1,
            )
        self._video_decoders[decoder_key] = decoder
        while len(self._video_decoders) > self.decoder_cache_size:
            self._video_decoders.popitem(last=False)
        # Canonical LeRobot MP4s already contain exactly one aligned frame for
        # every Parquet row; raw camera provenance remains in the sensor sidecar
        # diagnostics and must not be applied a second time here.
        frames = decoder.get_frames_at(indices=rows.tolist()).data
        return frames.numpy()


def _stats(values: np.ndarray, mask_size: int) -> dict:
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.maximum(np.std(values, axis=0), 1e-6).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
        "mask": [True] * mask_size,
    }


def prepare_trex_stats(
    store: PickPlaceStore,
    output: str | Path | None = None,
    anchors: Sequence[Anchor] | None = None,
) -> Path:
    output = (
        Path(output)
        if output is not None
        else model_stats_path("t_rex", store.root, ".json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_name(f".{output.name}.lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if output.is_file():
            return output
        selected_anchors = list(store.train_anchors if anchors is None else anchors)
        if not selected_anchors:
            raise ValueError("cannot compute T-Rex statistics without anchors")
        count = len(selected_anchors)
        actions = np.empty((count, TREX_ACTION_CHUNK, ACTION_DIM), dtype=np.float32)
        states = np.empty((count, ACTION_DIM), dtype=np.float32)
        tactile = np.empty((count, TACTILE_DIM), dtype=np.float32)
        for index, anchor in enumerate(selected_anchors):
            episode = store.load_episode(anchor.episode_index)
            actions[index] = trex_action_chunk(
                episode["state"], episode["action"], anchor.frame_index
            )
            states[index] = trex_native_state(episode["state"][anchor.frame_index])
            wrench = episode["tactile_wrench"][anchor.frame_index][
                GCC_TO_TREX_TACTILE
            ].copy()
            valid = episode["tactile_wrench_valid"][anchor.frame_index][
                GCC_TO_TREX_TACTILE
            ]
            wrench[~valid] = 0
            tactile[index] = wrench.reshape(-1)
        block = {
            "action": _stats(actions, ACTION_DIM),
            "state": _stats(states, ACTION_DIM),
            "tactile_f6": _stats(tactile, TACTILE_DIM),
            "tracking_error": {
                "mean": [0.0] * 56,
                "std": [0.0] * 56,
                "mean_abs": [0.0] * 56,
                "mask": [True] * 56,
            },
            "num_transitions": count,
            "num_trajectories": sum(
                ep.split == "train" for ep in store.episodes.values()
            ),
            "action_target": "q_actual",
        }
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps({"rlbench": block}, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
    return output


class RunningStats:
    def __init__(self, width: int):
        self.count = np.zeros(width, dtype=np.float64)
        self.total = np.zeros(width, dtype=np.float64)
        self.total_sq = np.zeros(width, dtype=np.float64)

    def add(self, values: np.ndarray, valid: np.ndarray | None = None) -> None:
        source = np.asarray(values, dtype=np.float64)
        if valid is None:
            valid = np.ones_like(source, dtype=bool)
        else:
            valid = np.broadcast_to(np.asarray(valid, dtype=bool), source.shape)
        values = source.reshape(-1, self.count.size)
        valid = valid.reshape(-1, self.count.size)
        self.count += valid.sum(axis=0)
        safe = np.where(valid, values, 0.0)
        self.total += safe.sum(axis=0)
        self.total_sq += (safe * safe).sum(axis=0)

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        count = np.maximum(self.count, 1.0)
        mean = self.total / count
        variance = np.maximum(self.total_sq / count - mean * mean, 0.0)
        return mean.astype(np.float32), np.maximum(np.sqrt(variance), 1e-2).astype(
            np.float32
        )


def tactile_features(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32).reshape(len(raw), TACTILE_DIM)
    return np.concatenate([raw, raw - raw[:1]], axis=-1)


def tactile_feature_validity(valid: np.ndarray) -> np.ndarray:
    """Expand per-finger validity to raw and baseline-relative F/T features."""

    valid = np.asarray(valid, dtype=bool)
    if valid.ndim < 2 or valid.shape[-1] != TACTILE_FINGERS:
        raise ValueError(
            f"tactile validity must end in {TACTILE_FINGERS} fingers, got {valid.shape}"
        )
    raw_valid = np.repeat(valid, WRENCH_DIM, axis=-1)
    relative_valid = np.repeat(valid & valid[..., :1, :], WRENCH_DIM, axis=-1)
    return np.concatenate([raw_valid, relative_valid], axis=-1)


def prepare_vitac_stats(
    store: PickPlaceStore,
    output: str | Path | None = None,
    anchors: Sequence[Anchor] | None = None,
) -> Path:
    output = (
        Path(output)
        if output is not None
        else model_stats_path("vitacformer", store.root, ".npz")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_name(f".{output.name}.lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if output.is_file():
            return output
        selected_anchors = list(store.train_anchors if anchors is None else anchors)
        if not selected_anchors:
            raise ValueError("cannot compute ViTacFormer statistics without anchors")
        anchors_by_episode: dict[int, list[int]] = {}
        for anchor in selected_anchors:
            anchors_by_episode.setdefault(anchor.episode_index, []).append(
                anchor.frame_index
            )

        state_stats = RunningStats(ACTION_DIM)
        action_stats = RunningStats(ACTION_DIM)
        tactile_stats = RunningStats(120)
        tactile_next_stats = RunningStats(120)
        for episode_meta in store.episodes.values():
            if episode_meta.split != "train":
                continue
            episode = store.load_episode(episode_meta.episode_index)
            anchors = np.asarray(
                anchors_by_episode.get(episode_meta.episode_index, []), dtype=np.int64
            )
            if anchors.size == 0:
                continue
            length = len(episode["state"])

            state_indices = anchors[:, None] + VITAC_STATE_OFFSETS[None]
            if np.any(state_indices < 0):
                raise ValueError("ViTacFormer statistics received incomplete history")
            state_stats.add(episode["state"][state_indices])

            action_raw = anchors[:, None] + np.arange(VITAC_ACTION_CHUNK)[None]
            action_pad = action_raw >= length - 1
            action_indices = np.clip(action_raw, 0, length - 1)
            action = episode["action"][action_indices]
            action_chunk = np.concatenate(
                [
                    relative_pose9_batch(
                        episode["state"][anchors, :9], action[..., :9]
                    ),
                    relative_pose9_batch(
                        episode["state"][anchors, 9:18], action[..., 9:18]
                    ),
                    action[..., 18:62],
                ],
                axis=-1,
            )
            action_stats.add(action_chunk, (~action_pad)[..., None])

            hist_indices = (
                anchors[:, None] + np.arange(-VITAC_TACTILE_HISTORY + 1, 1)[None]
            )
            if np.any(hist_indices < 0):
                raise ValueError(
                    "ViTacFormer statistics received incomplete tactile history"
                )
            hist = episode["tactile_wrench"][hist_indices][
                ..., GCC_TO_VITAC_TACTILE, :
            ].copy()
            hist_valid = episode["tactile_wrench_valid"][hist_indices][
                ..., GCC_TO_VITAC_TACTILE
            ]
            hist[~hist_valid] = 0
            hist_flat = hist.reshape(len(anchors), 18, TACTILE_DIM)
            hist_features = np.concatenate(
                [hist_flat, hist_flat - hist_flat[:, :1]], axis=-1
            )
            hist_feature_valid = tactile_feature_validity(hist_valid)
            tactile_stats.add(hist_features, hist_feature_valid)

            future_raw = anchors[:, None] + np.arange(VITAC_TACTILE_FUTURE)[None]
            future_pad = future_raw >= length
            future_indices = np.clip(future_raw, 0, length - 1)
            future = episode["tactile_wrench"][future_indices][
                ..., GCC_TO_VITAC_TACTILE, :
            ].copy()
            future_valid = episode["tactile_wrench_valid"][future_indices][
                ..., GCC_TO_VITAC_TACTILE
            ]
            future_valid[future_pad] = False
            future[~future_valid] = 0
            future_flat = future.reshape(
                len(anchors), VITAC_TACTILE_FUTURE, TACTILE_DIM
            )
            future_features = np.concatenate(
                [future_flat, future_flat - future_flat[:, :1]], axis=-1
            )
            future_feature_valid = tactile_feature_validity(future_valid)
            tactile_next_stats.add(future_features, future_feature_valid)
        state_mean, state_std = state_stats.finish()
        action_mean, action_std = action_stats.finish()
        tactile_mean, tactile_std = tactile_stats.finish()
        tactile_next_mean, tactile_next_std = tactile_next_stats.finish()
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            np.savez(
                handle,
                state_mean=state_mean,
                state_std=state_std,
                action_mean=action_mean,
                action_std=action_std,
                tactile_mean=tactile_mean,
                tactile_std=tactile_std,
                tactile_next_mean=tactile_next_mean,
                tactile_next_std=tactile_next_std,
            )
        os.replace(temporary, output)
    return output


class EpisodeGroupedSampler(Sampler[int]):
    """Deterministic episode-local stream used before Accelerate shards batches."""

    def __init__(
        self,
        store: PickPlaceStore,
        total_samples: int,
        seed: int = 42,
        anchors: Sequence[Anchor] | None = None,
    ):
        self.store = store
        self.total_samples = int(total_samples)
        self.seed = int(seed)
        selected_anchors = list(store.train_anchors if anchors is None else anchors)
        self.ref_lookup = {
            (anchor.episode_index, anchor.frame_index): index
            for index, anchor in enumerate(selected_anchors)
        }
        self.anchors_by_episode: dict[int, list[int]] = {}
        for anchor in selected_anchors:
            self.anchors_by_episode.setdefault(anchor.episode_index, []).append(
                anchor.frame_index
            )

    def __len__(self) -> int:
        return self.total_samples

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed)
        produced = 0
        episode_indices = list(self.anchors_by_episode)
        while produced < self.total_samples:
            rng.shuffle(episode_indices)
            for episode_index in episode_indices:
                frames = list(self.anchors_by_episode[episode_index])
                rng.shuffle(frames)
                for frame_index in frames:
                    if produced >= self.total_samples:
                        return
                    yield self.ref_lookup[(episode_index, frame_index)]
                    produced += 1


class DistributedEpisodeBatchSampler(Sampler[int]):
    """Return the local-rank slice of episode-grouped global batches."""

    def __init__(
        self,
        store: PickPlaceStore,
        max_steps: int,
        per_device_batch: int,
        rank: int,
        world_size: int,
        seed: int = 42,
        anchors: Sequence[Anchor] | None = None,
    ):
        self.local_samples = int(max_steps) * int(per_device_batch)
        global_samples = self.local_samples * int(world_size)
        source = EpisodeGroupedSampler(store, global_samples, seed, anchors)
        sequence = list(source)
        global_batch = int(per_device_batch) * int(world_size)
        local: list[int] = []
        for start in range(0, global_samples, global_batch):
            batch = sequence[start : start + global_batch]
            lo = int(rank) * int(per_device_batch)
            hi = lo + int(per_device_batch)
            local.extend(batch[lo:hi])
        self.indices = local

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)
