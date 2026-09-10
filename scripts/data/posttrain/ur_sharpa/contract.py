"""Raw UR-SharpA v1 loading and canonical row construction.

This module intentionally supports only the timeline-based recorder contract.  The
old ``data.json`` frame list is not a timing source and is never read here.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dexterity.data.posttrain import (
    DISK_ACTION_DIM,
    POSTTRAIN_JOINT_ORDER,
    POSTTRAIN_TACTILE_ORDER,
)
from dexterity.runtime.sharpa62 import DEPLOY_JOINT_ORDER

FPS = 30
ACTION_DIM = DISK_ACTION_DIM
DEFORMATION_SIZE = 240
CAMERA_NAMES = ("ego", "left_wrist", "right_wrist")
CAMERA_VIDEO_KEYS = {
    "ego": "observation.images.ego_view",
    "left_wrist": "observation.images.left_wrist_view",
    "right_wrist": "observation.images.right_wrist_view",
}
REQUIRED_RAW_FILES = (
    "COMPLETE",
    "checksums.sha256",
    "timeline/clock.npz",
    "timeline/source_quality.npz",
    "ur/physical_12.npz",
    "sharpa/joints_44.npz",
    "sharpa/tactile/force6d.npz",
    "sharpa/tactile/deform_index.npz",
    "sharpa/tactile/deform_u8.zst",
    "cameras/ego/index.npz",
    "cameras/ego/rgb.mp4",
    "cameras/left_wrist/index.npz",
    "cameras/left_wrist/rgb.mp4",
    "cameras/right_wrist/index.npz",
    "cameras/right_wrist/rgb.mp4",
)

RAW_TO_POSTTRAIN_JOINT_ORDER = np.asarray(
    [DEPLOY_JOINT_ORDER.index(name) for name in POSTTRAIN_JOINT_ORDER],
    dtype=np.int64,
)

# Raw force6d hand order is left, right; both hands are pinky -> thumb.  The
# canonical project order is right then left, also pinky -> thumb.
RAW_WRENCH_FINGER_ORDER = (
    "left_pinky",
    "left_ring",
    "left_middle",
    "left_index",
    "left_thumb",
    "right_pinky",
    "right_ring",
    "right_middle",
    "right_index",
    "right_thumb",
)
RAW_WRENCH_INDEX = {name: index for index, name in enumerate(RAW_WRENCH_FINGER_ORDER)}
RAW_TO_POSTTRAIN_TACTILE_ORDER = np.asarray(
    [RAW_WRENCH_INDEX[name] for name in POSTTRAIN_TACTILE_ORDER], dtype=np.int64
)


@dataclass(frozen=True)
class EpisodeSource:
    path: Path
    name: str
    start_time_unix_ns: int
    timeline_rows: int
    output_episode_index: int


@dataclass(frozen=True)
class CameraIndex:
    name: str
    video_path: Path
    frame_index: np.ndarray
    timeline_row: np.ndarray
    source_timestamp_ns: np.ndarray
    time_unix_ns: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class CameraAlignment:
    name: str
    source_video_index: np.ndarray
    source_timestamp_ns: np.ndarray
    source_timeline_row: np.ndarray
    valid: np.ndarray
    frame_reused: np.ndarray
    timeline_row_gap: np.ndarray
    recorder_age_rows: np.ndarray
    alignment_latency_ns: np.ndarray
    source_gap_rows: np.ndarray

    @property
    def length(self) -> int:
        return int(self.source_video_index.shape[0])

    def sliced(self, start: int) -> "CameraAlignment":
        return CameraAlignment(
            name=self.name,
            source_video_index=self.source_video_index[start:].copy(),
            source_timestamp_ns=self.source_timestamp_ns[start:].copy(),
            source_timeline_row=self.source_timeline_row[start:].copy(),
            valid=self.valid[start:].copy(),
            frame_reused=_reuse_mask(self.source_video_index[start:]),
            timeline_row_gap=_selected_gap(self.source_timeline_row[start:]),
            recorder_age_rows=self.recorder_age_rows[start:].copy(),
            alignment_latency_ns=self.alignment_latency_ns[start:].copy(),
            source_gap_rows=self.source_gap_rows.copy(),
        )


@dataclass
class CanonicalEpisode:
    source: EpisodeSource
    fps: int
    crop_start_row: int
    state: np.ndarray
    action: np.ndarray
    anchor_valid: np.ndarray
    tau: np.ndarray
    tau_valid: np.ndarray
    tactile_wrench: np.ndarray
    tactile_wrench_valid: np.ndarray
    tactile_deformation: np.ndarray
    tactile_deformation_valid: np.ndarray
    source_timeline_row: np.ndarray
    camera_alignments: dict[str, CameraAlignment]
    camera_dimensions: dict[str, tuple[int, int]]
    cleaning: dict[str, Any]
    wrist_audit: dict[str, Any]
    action_identity_max_error: float

    @property
    def length(self) -> int:
        return int(self.state.shape[0])


class ZstdDecoder:
    """Small libzstd wrapper without an additional Python dependency."""

    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libzstd.so.1")
        self.lib.ZSTD_getFrameContentSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.lib.ZSTD_getFrameContentSize.restype = ctypes.c_ulonglong
        self.lib.ZSTD_decompress.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self.lib.ZSTD_decompress.restype = ctypes.c_size_t
        self.lib.ZSTD_isError.argtypes = [ctypes.c_size_t]
        self.lib.ZSTD_isError.restype = ctypes.c_uint

    def content_size(self, path: Path) -> int:
        compressed = path.read_bytes()
        source = ctypes.create_string_buffer(compressed)
        size = int(self.lib.ZSTD_getFrameContentSize(source, len(compressed)))
        if size in (0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFE):
            raise ValueError(f"unknown zstd content size: {path}")
        return size

    def decode(self, path: Path) -> bytes:
        compressed = path.read_bytes()
        source = ctypes.create_string_buffer(compressed)
        size = int(self.lib.ZSTD_getFrameContentSize(source, len(compressed)))
        if size in (0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFE):
            raise ValueError(f"unknown zstd content size: {path}")
        output = ctypes.create_string_buffer(size)
        decoded = int(self.lib.ZSTD_decompress(output, size, source, len(compressed)))
        if self.lib.ZSTD_isError(decoded) or decoded != size:
            raise ValueError(f"zstd decode failed for {path}: {decoded}/{size}")
        return output.raw[:decoded]


def discover_episodes(source: Path, first_n: int | None = None) -> list[EpisodeSource]:
    """Discover every selected episode by its master clock, regardless of name."""

    if not source.is_dir():
        raise FileNotFoundError(source)
    discovered: list[tuple[int, str, Path, int]] = []
    for clock_path in source.glob("*/timeline/clock.npz"):
        episode = clock_path.parent.parent
        missing = [
            name for name in REQUIRED_RAW_FILES if not (episode / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{episode} is missing required raw files: {missing}"
            )
        with np.load(clock_path, allow_pickle=False) as clock:
            time_unix_ns = np.asarray(clock["time_unix_ns"], dtype=np.int64)
            elapsed_ns = np.asarray(clock["elapsed_ns"], dtype=np.int64)
            sample_rate_hz = float(np.asarray(clock["sample_rate_hz"]).item())
        if time_unix_ns.ndim != 1 or not len(time_unix_ns):
            raise ValueError(f"{clock_path}: empty or invalid time_unix_ns")
        if elapsed_ns.shape != time_unix_ns.shape:
            raise ValueError(f"{clock_path}: clock fields have different lengths")
        if np.any(np.diff(time_unix_ns) <= 0) or np.any(np.diff(elapsed_ns) <= 0):
            raise ValueError(
                f"{clock_path}: master timeline must be strictly increasing"
            )
        if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            raise ValueError(f"{clock_path}: invalid sample_rate_hz={sample_rate_hz}")
        discovered.append(
            (int(time_unix_ns[0]), episode.name, episode, int(len(time_unix_ns)))
        )
    discovered.sort(key=lambda item: (item[0], item[1]))
    if first_n is not None:
        if first_n <= 0:
            raise ValueError("--first-n must be positive")
        discovered = discovered[:first_n]
    if not discovered:
        raise ValueError(f"no timeline-based episodes found in {source}")
    return [
        EpisodeSource(
            path=path,
            name=name,
            start_time_unix_ns=start,
            timeline_rows=rows,
            output_episode_index=index,
        )
        for index, (start, name, path, rows) in enumerate(discovered)
    ]


def load_camera_index(episode: Path, name: str) -> CameraIndex:
    if name not in CAMERA_NAMES:
        raise ValueError(f"unknown camera {name!r}")
    path = episode / "cameras" / name / "index.npz"
    with np.load(path, allow_pickle=False) as payload:
        required = (
            "frame_index",
            "timeline_row",
            "source_timestamp_ns",
            "time_unix_ns",
            "valid",
        )
        missing = [key for key in required if key not in payload.files]
        if missing:
            raise KeyError(f"{path}: missing {missing}")
        frame_index = np.asarray(payload["frame_index"], dtype=np.int64)
        timeline_row = np.asarray(payload["timeline_row"], dtype=np.int64)
        source_timestamp_ns = np.asarray(payload["source_timestamp_ns"], dtype=np.int64)
        time_unix_ns = np.asarray(payload["time_unix_ns"], dtype=np.int64)
        valid = np.asarray(payload["valid"], dtype=bool)
    length = len(frame_index)
    for label, value in (
        ("timeline_row", timeline_row),
        ("source_timestamp_ns", source_timestamp_ns),
        ("time_unix_ns", time_unix_ns),
        ("valid", valid),
    ):
        if value.shape != (length,):
            raise ValueError(
                f"{path}: {label} must have shape {(length,)}, got {value.shape}"
            )
    if not length:
        raise ValueError(f"{path}: camera index is empty")
    if np.any(np.diff(timeline_row) < 0) or np.any(np.diff(frame_index) < 0):
        raise ValueError(f"{path}: camera index is not monotonic")
    if np.any(frame_index < 0) or np.any(timeline_row < 0):
        raise ValueError(f"{path}: camera indices must be nonnegative")
    return CameraIndex(
        name=name,
        video_path=episode / "cameras" / name / "rgb.mp4",
        frame_index=frame_index,
        timeline_row=timeline_row,
        source_timestamp_ns=source_timestamp_ns,
        time_unix_ns=time_unix_ns,
        valid=valid,
    )


def _reuse_mask(indices: np.ndarray) -> np.ndarray:
    result = np.zeros(len(indices), dtype=bool)
    if len(indices) > 1:
        result[1:] = indices[1:] == indices[:-1]
    return result


def _selected_gap(rows: np.ndarray) -> np.ndarray:
    result = np.zeros(len(rows), dtype=np.int32)
    if len(rows) > 1:
        result[1:] = np.diff(rows).astype(np.int32)
    return result


def align_camera(
    index: CameraIndex,
    master_rows: np.ndarray,
    clock_time_unix_ns: np.ndarray,
) -> CameraAlignment:
    """Select the newest valid recorder-arrived frame without looking ahead."""

    rows = np.asarray(master_rows, dtype=np.int64)
    clock = np.asarray(clock_time_unix_ns, dtype=np.int64)
    if rows.shape != clock.shape:
        raise ValueError("master row and clock shapes differ")
    valid_index = np.flatnonzero(index.valid)
    if not len(valid_index):
        raise ValueError(f"{index.name}: camera index has no valid frame")
    valid_timeline_row = index.timeline_row[valid_index]
    positions = np.searchsorted(valid_timeline_row, rows, side="right") - 1
    available = positions >= 0
    source_video_index = np.full(len(rows), -1, dtype=np.int64)
    source_timestamp_ns = np.full(len(rows), -1, dtype=np.int64)
    source_timeline_row = np.full(len(rows), -1, dtype=np.int64)
    selected_valid = np.zeros(len(rows), dtype=bool)
    if np.any(available):
        selected = valid_index[positions[available]]
        source_video_index[available] = index.frame_index[selected]
        source_timestamp_ns[available] = index.source_timestamp_ns[selected]
        source_timeline_row[available] = index.timeline_row[selected]
        selected_valid[available] = True
    causal = available & (source_timeline_row <= rows)
    if not np.array_equal(causal, available):
        raise AssertionError(f"{index.name}: alignment selected a future frame")
    recorder_age_rows = np.full(len(rows), -1, dtype=np.int32)
    alignment_latency_ns = np.full(len(rows), np.iinfo(np.int64).min, dtype=np.int64)
    recorder_age_rows[available] = (
        rows[available] - source_timeline_row[available]
    ).astype(np.int32)
    alignment_latency_ns[available] = clock[available] - source_timestamp_ns[available]
    return CameraAlignment(
        name=index.name,
        source_video_index=source_video_index.astype(np.int32),
        source_timestamp_ns=source_timestamp_ns,
        source_timeline_row=source_timeline_row.astype(np.int32),
        valid=available & selected_valid,
        frame_reused=_reuse_mask(source_video_index),
        timeline_row_gap=_selected_gap(source_timeline_row),
        recorder_age_rows=recorder_age_rows,
        alignment_latency_ns=alignment_latency_ns,
        source_gap_rows=np.diff(valid_timeline_row).astype(np.int32),
    )


def common_camera_prefix(alignments: dict[str, CameraAlignment]) -> int:
    if set(alignments) != set(CAMERA_NAMES):
        raise ValueError(
            f"expected camera alignments {CAMERA_NAMES}, got {tuple(alignments)}"
        )
    common = np.logical_and.reduce([alignments[name].valid for name in CAMERA_NAMES])
    usable = np.flatnonzero(common)
    if not len(usable):
        raise ValueError("three cameras never have a common causal frame")
    return int(usable[0])


def _require_timeline_length(
    path: Path, expected: int, arrays: dict[str, np.ndarray]
) -> None:
    bad = {
        name: value.shape for name, value in arrays.items() if len(value) != expected
    }
    if bad:
        raise ValueError(f"{path}: raw arrays do not match T={expected}: {bad}")


def first_observed_row(observed: np.ndarray, *, label: str) -> int:
    """Return the first row that can seed causal latest-valid cleaning."""

    mask = np.asarray(observed, dtype=bool)
    if mask.ndim != 1:
        raise ValueError(f"{label}: observed mask must be one-dimensional")
    rows = np.flatnonzero(mask)
    if not len(rows):
        raise ValueError(f"{label}: source has no valid row")
    return int(rows[0])


def forward_fill_latest(
    values: np.ndarray,
    observed: np.ndarray,
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Causally copy the latest observed row without inventing validity."""

    source = np.asarray(values)
    mask = np.asarray(observed, dtype=bool)
    if source.ndim < 2 or mask.shape != (len(source),):
        raise ValueError(
            f"{label}: incompatible values/mask shapes {source.shape}/{mask.shape}"
        )
    if not len(source):
        raise ValueError(f"{label}: cannot fill an empty source")
    seed = first_observed_row(mask, label=label)
    latest = np.maximum.accumulate(
        np.where(mask, np.arange(len(source), dtype=np.int64), -1)
    )
    filled = source.copy()
    usable = latest >= 0
    filled[usable] = source[latest[usable]]
    if not np.isfinite(filled[seed:]).all():
        raise ValueError(
            f"{label}: latest-valid fill left non-finite values after row {seed}"
        )
    return filled, latest


def forward_fill_hand_pair(
    q: np.ndarray,
    q_cmd: np.ndarray,
    observed: np.ndarray,
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hold row-paired hand state and command from one shared raw source row."""

    q_values = np.asarray(q)
    q_commands = np.asarray(q_cmd)
    if q_values.shape != q_commands.shape or q_values.ndim != 2:
        raise ValueError(
            f"{label}: paired q/q_cmd shapes differ: {q_values.shape}/{q_commands.shape}"
        )
    paired = np.concatenate((q_values, q_commands), axis=1)
    filled, source_rows = forward_fill_latest(paired, observed, label=label)
    width = q_values.shape[1]
    return filled[:, :width], filled[:, width:], source_rows


def _hold_diagnostics(
    observed: np.ndarray,
    latest_source_row: np.ndarray,
    output_rows: np.ndarray,
) -> dict[str, Any]:
    mask = np.asarray(observed, dtype=bool)[output_rows]
    source_rows = np.asarray(latest_source_row, dtype=np.int64)[output_rows]
    held = ~mask
    age = output_rows - source_rows
    if np.any(source_rows < 0) or np.any(age < 0):
        raise ValueError("cleaned control row has no causal source")
    padded = np.concatenate((np.zeros(1, dtype=bool), held, np.zeros(1, dtype=bool)))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    stops = np.flatnonzero(padded[:-1] & ~padded[1:])
    runs = (stops - starts).astype(np.int32)
    return {
        "observed_rows": int(mask.sum()),
        "held_rows": int(held.sum()),
        "held_ratio": float(held.mean()),
        "hold_run_count": int(len(runs)),
        "hold_gap_rows": quantiles(runs),
        "hold_gap_histogram": (
            {
                str(int(gap)): int(count)
                for gap, count in zip(*np.unique(runs, return_counts=True), strict=True)
            }
            if len(runs)
            else {}
        ),
        "max_source_age_rows": int(age.max()) if len(age) else 0,
    }


def build_control_rows(
    canonical_wrist_pose9: np.ndarray,
    q_clean: np.ndarray,
    q_cmd_clean: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Build model-independent, fully cleaned 30 Hz state/action rows."""

    canonical = np.asarray(canonical_wrist_pose9, dtype=np.float32)
    count = len(canonical)
    if canonical.shape != (count, 2, 9):
        raise ValueError(
            f"canonical_wrist_pose9 must be [{count},2,9], got {canonical.shape}"
        )
    q_values = np.asarray(q_clean, dtype=np.float32)
    q_commands = np.asarray(q_cmd_clean, dtype=np.float32)
    if q_values.shape != (count, 44) or q_commands.shape != (count, 44):
        raise ValueError("cleaned q and q_cmd must both have shape [T,44]")
    if not np.isfinite(canonical).all() or not np.isfinite(q_values).all():
        raise ValueError("cleaned wrist/q still contains NaN or Inf")
    if not np.isfinite(q_commands).all():
        raise ValueError("cleaned q_cmd still contains NaN or Inf")

    q_source = q_values[:, RAW_TO_POSTTRAIN_JOINT_ORDER]
    q_cmd_source = q_commands[:, RAW_TO_POSTTRAIN_JOINT_ORDER]
    wrist = canonical.reshape(count, 18).astype(np.float32)
    q_actual = q_source.astype(np.float32)
    q_teleop = q_cmd_source.astype(np.float32)
    state = np.concatenate((wrist, q_actual), axis=1).astype(np.float32)
    wrist_exe_next = np.concatenate((wrist[1:], wrist[-1:]), axis=0)
    q_exe_next = np.concatenate((q_actual[1:], q_actual[-1:]), axis=0)
    delta_q = (q_teleop - q_exe_next).astype(np.float32)
    action = np.concatenate(
        (wrist_exe_next, q_exe_next, q_teleop, delta_q), axis=1
    ).astype(np.float32)
    if state.shape != (count, 62) or action.shape != (count, ACTION_DIM):
        raise AssertionError(f"bad state/action shapes: {state.shape}/{action.shape}")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise AssertionError("state/action sanitization failed")
    identity_error = float(
        np.max(np.abs(action[:, 18:62] + action[:, 106:150] - action[:, 62:106]))
    )
    if identity_error > 1e-6:
        raise ValueError(f"action identity error {identity_error} > 1e-6")

    anchor_valid = np.ones(count, dtype=bool)
    return {
        "state": state,
        "action": action,
        "anchor_valid": anchor_valid,
        "action_identity_max_error": identity_error,
    }


def decode_deformation(
    episode: Path,
    decoder: ZstdDecoder,
    master_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    index_path = episode / "sharpa/tactile/deform_index.npz"
    with np.load(index_path, allow_pickle=False) as payload:
        timeline_row = np.asarray(payload["timeline_row"], dtype=np.int64)
        raw_frame_index = np.asarray(payload["raw_frame_index"], dtype=np.int64)
        source_timestamp_ns = np.asarray(payload["source_timestamp_ns"], dtype=np.int64)
        sensor_time = np.asarray(payload["sensor_time"], dtype=np.float64)
        source_valid = np.asarray(payload["valid"], dtype=bool)
    count = len(timeline_row)
    expected_shapes = {
        "raw_frame_index": (count,),
        "source_timestamp_ns": (count,),
        "sensor_time": (count, 10),
        "valid": (count, 10),
    }
    for label, shape in expected_shapes.items():
        value = {
            "raw_frame_index": raw_frame_index,
            "source_timestamp_ns": source_timestamp_ns,
            "sensor_time": sensor_time,
            "valid": source_valid,
        }[label]
        if value.shape != shape:
            raise ValueError(
                f"{index_path}: {label} must have shape {shape}, got {value.shape}"
            )
    if not count or np.any(np.diff(timeline_row) < 0):
        raise ValueError(
            f"{index_path}: deformation timeline must be nonempty and monotonic"
        )
    if np.any(raw_frame_index < 0):
        raise ValueError(f"{index_path}: negative raw_frame_index")

    data_path = episode / "sharpa/tactile/deform_u8.zst"
    raw_bytes = decoder.decode(data_path)
    bytes_per_frame = 10 * DEFORMATION_SIZE * DEFORMATION_SIZE
    if len(raw_bytes) % bytes_per_frame:
        raise ValueError(
            f"{data_path}: decoded byte count is not a whole deformation frame"
        )
    raw_count = len(raw_bytes) // bytes_per_frame
    if int(raw_frame_index.max()) >= raw_count:
        raise ValueError(
            f"{index_path}: raw_frame_index max {raw_frame_index.max()} >= decoded frames {raw_count}"
        )
    raw = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(
        raw_count, 10, DEFORMATION_SIZE, DEFORMATION_SIZE
    )
    target_rows = np.asarray(master_rows, dtype=np.int64)
    positions = np.searchsorted(timeline_row, target_rows, side="right") - 1
    available = positions >= 0
    deformation = np.zeros(
        (len(target_rows), 10, DEFORMATION_SIZE, DEFORMATION_SIZE), dtype=np.uint8
    )
    valid = np.zeros((len(target_rows), 10), dtype=bool)
    if np.any(available):
        selected = positions[available]
        raw_selected = raw_frame_index[selected]
        deformation[available] = raw[raw_selected]
        selected_valid = source_valid[selected].copy()
        selected_valid &= np.isfinite(sensor_time[selected])
        valid[available] = selected_valid
    deformation[~valid] = 0
    diagnostics = {
        "source_frames": int(count),
        "decoded_frames": int(raw_count),
        "aligned_reuse_ratio": (
            float(_reuse_mask(positions)[1:].mean()) if len(positions) > 1 else 0.0
        ),
        "valid_ratio": float(valid.mean()),
        "source_timestamp_min_ns": int(source_timestamp_ns.min()),
        "source_timestamp_max_ns": int(source_timestamp_ns.max()),
    }
    return deformation, valid, diagnostics


def _pose9_rotation_error(canonical: np.ndarray) -> float:
    axes = np.asarray(canonical[..., 3:9], dtype=np.float64).reshape(-1, 2, 3)
    first = axes[:, 0]
    second = axes[:, 1]
    gram = np.stack(
        (
            np.sum(first * first, axis=1) - 1.0,
            np.sum(second * second, axis=1) - 1.0,
            np.sum(first * second, axis=1),
        ),
        axis=1,
    )
    return float(np.max(np.abs(gram)))


def wrist_audit(raw_tcp_pose6: np.ndarray, canonical: np.ndarray) -> dict[str, Any]:
    """Audit the stored canonical wrist without transforming it again.

    The recorder-side ``ur_common`` package is not part of this repository.  We
    therefore report both a direct Pose9 geometry audit and an independent
    held-out rigid-chain reconstruction from raw TCP poses.  The reconstruction
    fits the two fixed physical transforms on alternating frames and evaluates
    only the held-out frames; it is evidence, never an output transform.
    """

    from scipy.spatial.transform import Rotation

    raw = np.asarray(raw_tcp_pose6, dtype=np.float64)
    target = np.asarray(canonical, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[1:] != (2, 6):
        raise ValueError(f"raw_tcp_pose6 must be [T,2,6], got {raw.shape}")
    if target.shape != (len(raw), 2, 9):
        raise ValueError(
            f"canonical_wrist_pose9 must be [{len(raw)},2,9], got {target.shape}"
        )

    sample = np.unique(np.linspace(0, len(raw) - 1, min(64, len(raw))).astype(np.int64))
    train = sample[::2]
    held_out = sample[1::2]
    if not len(held_out):
        held_out = sample

    def raw_matrix(values: np.ndarray) -> np.ndarray:
        result = np.tile(np.eye(4, dtype=np.float64), (len(values), 1, 1))
        result[:, :3, 3] = values[:, :3]
        result[:, :3, :3] = Rotation.from_rotvec(values[:, 3:]).as_matrix()
        return result

    def canonical_matrix(values: np.ndarray) -> np.ndarray:
        first = values[:, 3:6]
        first = first / np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-12)
        second = values[:, 6:9] - first * np.sum(
            first * values[:, 6:9], axis=1, keepdims=True
        )
        second = second / np.maximum(
            np.linalg.norm(second, axis=1, keepdims=True), 1e-12
        )
        result = np.tile(np.eye(4, dtype=np.float64), (len(values), 1, 1))
        result[:, :3, 3] = values[:, :3]
        result[:, :3, :3] = np.stack((first, second, np.cross(first, second)), axis=-1)
        return result

    side_results: dict[str, Any] = {}
    for side_index, side in enumerate(("left", "right")):
        source = raw_matrix(raw[:, side_index])
        expected = canonical_matrix(target[:, side_index])
        # Solve expected ~= root_from_base @ base_from_tcp @ tcp_from_hand.
        # Both fixed transforms are estimated from calibration rows only.
        from scipy.optimize import least_squares

        def transform(parameters: np.ndarray) -> np.ndarray:
            value = np.eye(4, dtype=np.float64)
            value[:3, :3] = Rotation.from_rotvec(parameters[:3]).as_matrix()
            value[:3, 3] = parameters[3:6]
            return value

        def residual(parameters: np.ndarray) -> np.ndarray:
            left = transform(parameters[:6])
            right = transform(parameters[6:])
            reconstructed = left[None] @ source[train] @ right[None]
            return np.concatenate(
                (
                    (reconstructed[:, :3, 3] - expected[train, :3, 3]).reshape(-1),
                    (reconstructed[:, :3, :3] - expected[train, :3, :3]).reshape(-1),
                )
            )

        solved = least_squares(
            residual,
            np.zeros(12, dtype=np.float64),
            max_nfev=1000,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        root_from_base = transform(solved.x[:6])
        tcp_from_hand = transform(solved.x[6:])
        reconstructed = root_from_base[None] @ source[held_out] @ tcp_from_hand[None]
        position_error = np.abs(reconstructed[:, :3, 3] - expected[held_out, :3, 3])
        rotation_error = np.abs(reconstructed[:, :3, :3] - expected[held_out, :3, :3])
        side_results[side] = {
            "calibration_rows": train.tolist(),
            "held_out_rows": held_out.tolist(),
            "held_out_position_max_abs_m": float(position_error.max()),
            "held_out_rotation_matrix_max_abs": float(rotation_error.max()),
            "root_from_base": root_from_base.tolist(),
            "tcp_from_hand": tcp_from_hand.tolist(),
        }
    return {
        "method": "held_out_fixed_rigid_chain_reconstruction",
        "producer_geometry_dependency": "ur_common.geometry (not vendored in this repository)",
        "output_uses_stored_canonical_wrist_pose9_directly": True,
        "conversion_time_root_or_mount_transform_count": 0,
        "audit_chain_root_transform_count": 1,
        "audit_chain_tcp_to_hand_transform_count": 1,
        "canonical_pose9_orthonormality_max_error": _pose9_rotation_error(target),
        "sides": side_results,
    }


def load_canonical_episode(
    source: EpisodeSource,
    *,
    decoder: ZstdDecoder | None = None,
) -> CanonicalEpisode:
    episode = source.path
    decoder = ZstdDecoder() if decoder is None else decoder
    with np.load(episode / "timeline/clock.npz", allow_pickle=False) as payload:
        clock_time_unix_ns = np.asarray(payload["time_unix_ns"], dtype=np.int64)
        elapsed_ns = np.asarray(payload["elapsed_ns"], dtype=np.int64)
        sample_rate_hz = float(np.asarray(payload["sample_rate_hz"]).item())
    total = len(clock_time_unix_ns)
    if total != source.timeline_rows or elapsed_ns.shape != (total,):
        raise ValueError(f"{episode}: clock changed after discovery")
    fps = int(round(sample_rate_hz))
    if fps != FPS or not np.isclose(sample_rate_hz, FPS, atol=0.1):
        raise ValueError(
            f"{episode}: expected approximately {FPS} Hz, got {sample_rate_hz}"
        )

    camera_indices = {name: load_camera_index(episode, name) for name in CAMERA_NAMES}
    full_rows = np.arange(total, dtype=np.int64)
    full_alignments = {
        name: align_camera(camera_indices[name], full_rows, clock_time_unix_ns)
        for name in CAMERA_NAMES
    }
    camera_crop_start = common_camera_prefix(full_alignments)

    with np.load(episode / "ur/physical_12.npz", allow_pickle=False) as payload:
        raw_tcp_pose6 = np.asarray(payload["raw_tcp_pose6"], dtype=np.float32)
        canonical_wrist_pose9 = np.asarray(
            payload["canonical_wrist_pose9"], dtype=np.float32
        )
        ur_valid_raw = np.asarray(payload["valid"], dtype=bool)
        ur_q = np.asarray(payload["q"], dtype=np.float32)
    _require_timeline_length(
        episode / "ur/physical_12.npz",
        total,
        {
            "raw_tcp_pose6": raw_tcp_pose6,
            "canonical_wrist_pose9": canonical_wrist_pose9,
            "valid": ur_valid_raw,
            "q": ur_q,
        },
    )
    if raw_tcp_pose6.shape != (total, 2, 6) or canonical_wrist_pose9.shape != (
        total,
        2,
        9,
    ):
        raise ValueError(f"{episode}: invalid UR pose shapes")

    with np.load(episode / "sharpa/joints_44.npz", allow_pickle=False) as payload:
        q_raw = np.asarray(payload["q"], dtype=np.float32)
        q_cmd_raw = np.asarray(payload["q_cmd"], dtype=np.float32)
        q_cmd_valid_raw = np.asarray(payload["q_cmd_valid"], dtype=bool)
        q_cmd_timestamp_ns = np.asarray(payload["q_cmd_timestamp_ns"], dtype=np.int64)
        q_cmd_source_raw = np.asarray(payload["q_cmd_source"], dtype=np.uint8)
        tau_raw = np.asarray(payload["tau"], dtype=np.float32)
        q_timestamp_ns = np.asarray(payload["q_timestamp_ns"], dtype=np.int64)
        q_valid_raw = np.asarray(payload["q_valid"], dtype=bool)
        tau_valid_raw = np.asarray(payload["tau_valid"], dtype=bool)
    _require_timeline_length(
        episode / "sharpa/joints_44.npz",
        total,
        {
            "q": q_raw,
            "q_cmd": q_cmd_raw,
            "q_cmd_valid": q_cmd_valid_raw,
            "q_cmd_timestamp_ns": q_cmd_timestamp_ns,
            "q_cmd_source": q_cmd_source_raw,
            "tau": tau_raw,
            "q_timestamp_ns": q_timestamp_ns,
            "q_valid": q_valid_raw,
            "tau_valid": tau_valid_raw,
        },
    )
    for label, value in (("q", q_raw), ("q_cmd", q_cmd_raw), ("tau", tau_raw)):
        if value.shape != (total, 44):
            raise ValueError(
                f"{episode}: {label} must be [{total},44], got {value.shape}"
            )
    if not np.array_equal(q_valid_raw, q_cmd_valid_raw):
        raise ValueError(
            f"{episode}: paired hand q_valid and q_cmd_valid masks must be identical"
        )

    wrist_observed = ur_valid_raw & np.isfinite(canonical_wrist_pose9).all(axis=(1, 2))
    hand_pair_observed = (
        q_valid_raw
        & q_cmd_valid_raw
        & (q_cmd_source_raw == 2)
        & np.isfinite(q_raw).all(axis=1)
        & np.isfinite(q_cmd_raw).all(axis=1)
    )
    crop_start = max(
        camera_crop_start,
        first_observed_row(wrist_observed, label=f"{episode}: wrist"),
        first_observed_row(hand_pair_observed, label=f"{episode}: hand_pair"),
    )
    rows = full_rows[crop_start:]
    if not len(rows):
        raise ValueError(f"{episode}: no rows remain after causal seed prefix")
    alignments = {
        name: value.sliced(crop_start) for name, value in full_alignments.items()
    }
    if not all(np.all(value.valid) for value in alignments.values()):
        raise AssertionError(
            f"{episode}: latest-valid camera alignment is not fully usable"
        )
    count = len(rows)

    wrist_clean, wrist_source_row = forward_fill_latest(
        canonical_wrist_pose9, wrist_observed, label=f"{episode}: wrist"
    )
    q_clean, q_cmd_clean, hand_pair_source_row = forward_fill_hand_pair(
        q_raw,
        q_cmd_raw,
        hand_pair_observed,
        label=f"{episode}: hand_pair",
    )
    control = build_control_rows(
        wrist_clean[crop_start:],
        q_clean[crop_start:],
        q_cmd_clean[crop_start:],
    )
    cleaning = {
        "method": "raw_row_paired_hand_hold",
        "all_output_rows_usable": True,
        "camera_seed_crop_start_row": camera_crop_start,
        "common_seed_crop_start_row": crop_start,
        "cropped_prefix_rows": crop_start,
        "hand_pairing": {
            "basis": "same raw timeline row",
            "source_timestamps_used_for_alignment": False,
            "q_and_q_cmd_held_together": True,
        },
        "wrist": _hold_diagnostics(wrist_observed, wrist_source_row, rows),
        "hand_pair": _hold_diagnostics(hand_pair_observed, hand_pair_source_row, rows),
    }

    tau_source = tau_raw[crop_start:, RAW_TO_POSTTRAIN_JOINT_ORDER]
    tau_valid = tau_valid_raw[crop_start:, None] & np.isfinite(tau_source)
    tau = np.where(tau_valid, tau_source, 0.0).astype(np.float32)

    with np.load(episode / "sharpa/tactile/force6d.npz", allow_pickle=False) as payload:
        wrench_raw = np.asarray(payload["wrench"], dtype=np.float32)
        wrench_valid_raw = np.asarray(payload["valid"], dtype=bool)
    _require_timeline_length(
        episode / "sharpa/tactile/force6d.npz",
        total,
        {"wrench": wrench_raw, "valid": wrench_valid_raw},
    )
    if wrench_raw.shape != (total, 2, 5, 6) or wrench_valid_raw.shape != (total, 2, 5):
        raise ValueError(f"{episode}: invalid force6d shapes")
    wrench_flat = wrench_raw[crop_start:].reshape(count, 10, 6)[
        :, RAW_TO_POSTTRAIN_TACTILE_ORDER
    ]
    wrench_valid = wrench_valid_raw[crop_start:].reshape(count, 10)[
        :, RAW_TO_POSTTRAIN_TACTILE_ORDER
    ]
    wrench_valid &= np.isfinite(wrench_flat).all(axis=2)
    wrench = np.where(wrench_valid[..., None], wrench_flat, 0.0).astype(np.float32)

    deformation, deformation_valid, _ = decode_deformation(episode, decoder, rows)
    camera_dimensions: dict[str, tuple[int, int]] = {}
    for name, index in camera_indices.items():
        capture = cv2.VideoCapture(str(index.video_path))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if width <= 0 or height <= 0 or frames <= 0:
            raise ValueError(f"cannot probe source video: {index.video_path}")
        if int(index.frame_index.max()) >= frames:
            raise ValueError(
                f"{index.video_path}: source index max {index.frame_index.max()} >= frames {frames}"
            )
        camera_dimensions[name] = (height, width)

    return CanonicalEpisode(
        source=source,
        fps=fps,
        crop_start_row=crop_start,
        state=np.asarray(control["state"]),
        action=np.asarray(control["action"]),
        anchor_valid=np.asarray(control["anchor_valid"]),
        tau=tau,
        tau_valid=tau_valid,
        tactile_wrench=wrench,
        tactile_wrench_valid=wrench_valid,
        tactile_deformation=deformation,
        tactile_deformation_valid=deformation_valid,
        source_timeline_row=rows.astype(np.int32),
        camera_alignments=alignments,
        camera_dimensions=camera_dimensions,
        cleaning=cleaning,
        wrist_audit=wrist_audit(raw_tcp_pose6, canonical_wrist_pose9),
        action_identity_max_error=float(control["action_identity_max_error"]),
    )


def quantiles(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def summarize_episode(episode: CanonicalEpisode) -> dict[str, Any]:
    cameras: dict[str, Any] = {}
    for name, alignment in episode.camera_alignments.items():
        latency = (
            alignment.alignment_latency_ns[alignment.valid].astype(np.float64) / 1e6
        )
        cameras[name] = {
            "video_key": CAMERA_VIDEO_KEYS[name],
            "height": episode.camera_dimensions[name][0],
            "width": episode.camera_dimensions[name][1],
            "valid_ratio": float(alignment.valid.mean()),
            "reuse_ratio": (
                float(alignment.frame_reused[1:].mean()) if episode.length > 1 else 0.0
            ),
            "source_frame_count_used": int(
                np.unique(alignment.source_video_index).size
            ),
            "timeline_row_gap": quantiles(alignment.source_gap_rows),
            "timeline_row_gap_histogram": {
                str(int(gap)): int(count)
                for gap, count in zip(
                    *np.unique(alignment.source_gap_rows, return_counts=True),
                    strict=True,
                )
            },
            "recorder_age_rows": quantiles(
                alignment.recorder_age_rows[alignment.valid]
            ),
            "alignment_latency_ms": quantiles(latency),
        }
    wrist_position = max(
        float(value["held_out_position_max_abs_m"])
        for value in episode.wrist_audit["sides"].values()
    )
    wrist_rotation = max(
        float(value["held_out_rotation_matrix_max_abs"])
        for value in episode.wrist_audit["sides"].values()
    )
    return {
        "episode_index": episode.source.output_episode_index,
        "source_episode": episode.source.name,
        "source_start_time_unix_ns": episode.source.start_time_unix_ns,
        "source_timeline_rows": episode.source.timeline_rows,
        "crop_start_timeline_row": episode.crop_start_row,
        "length": episode.length,
        "valid_anchor_count": int(episode.anchor_valid.sum()),
        "all_anchor_valid": bool(episode.anchor_valid.all()),
        "cleaning": episode.cleaning,
        "tau_valid_ratio": float(episode.tau_valid.mean()),
        "tactile_wrench_valid_ratio": float(episode.tactile_wrench_valid.mean()),
        "tactile_deformation_valid_ratio": float(
            episode.tactile_deformation_valid.mean()
        ),
        "action_identity_max_error": episode.action_identity_max_error,
        "wrist_reconstruction_position_max_abs_m": wrist_position,
        "wrist_reconstruction_rotation_max_abs": wrist_rotation,
        "wrist_audit": episode.wrist_audit,
        "cameras": cameras,
    }


__all__ = [
    "ACTION_DIM",
    "CAMERA_NAMES",
    "CAMERA_VIDEO_KEYS",
    "CanonicalEpisode",
    "CameraAlignment",
    "CameraIndex",
    "DEFORMATION_SIZE",
    "EpisodeSource",
    "FPS",
    "RAW_TO_POSTTRAIN_JOINT_ORDER",
    "RAW_TO_POSTTRAIN_TACTILE_ORDER",
    "REQUIRED_RAW_FILES",
    "ZstdDecoder",
    "align_camera",
    "build_control_rows",
    "common_camera_prefix",
    "decode_deformation",
    "discover_episodes",
    "first_observed_row",
    "forward_fill_hand_pair",
    "forward_fill_latest",
    "load_camera_index",
    "load_canonical_episode",
    "quantiles",
    "summarize_episode",
    "wrist_audit",
]
