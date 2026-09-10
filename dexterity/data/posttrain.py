"""Canonical HACO post-training data contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from dexterity.runtime.sharpa62 import MODEL_JOINT_ORDER, MODEL_TACTILE_ORDER


DISK_ACTION_DIM = 150
JOINT_COUNT = 44
TACTILE_FINGERS = 10
WRENCH_DIM = 6
DEFORMATION_HEIGHT = 240
DEFORMATION_WIDTH = 240
HISTORY_LENGTH = 9
HISTORY_PAST_FRAMES = HISTORY_LENGTH - 1
ACTION_HORIZON = 40
POSTTRAIN_JOINT_ORDER = MODEL_JOINT_ORDER
POSTTRAIN_TACTILE_ORDER = MODEL_TACTILE_ORDER


@dataclass(frozen=True)
class SensorEpisode:
    """Sensor payload aligned one-to-one with canonical dataset rows."""

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

    @property
    def tactile_wrench_valid(self) -> np.ndarray:
        return self.tactile_wrench_valid_mask

    @property
    def tactile_deformation_valid(self) -> np.ndarray:
        return self.tactile_deformation_valid_mask

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
            "tactile_deformation_valid_mask": (self.length, TACTILE_FINGERS),
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


def relative_pose9_batch(base: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Vectorized relative pose for batched bases and target sequences."""

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


__all__ = [
    "ACTION_HORIZON",
    "DISK_ACTION_DIM",
    "HISTORY_PAST_FRAMES",
    "POSTTRAIN_JOINT_ORDER",
    "POSTTRAIN_TACTILE_ORDER",
    "SensorEpisode",
    "action_indices",
    "exact_anchor_indices",
    "history_indices",
    "load_anchor_valid",
    "relative_pose9_batch",
    "strict_anchor_indices",
]
