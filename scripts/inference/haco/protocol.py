from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from dexterity.data.posttrain import (
    ACTION_HORIZON,
    HISTORY_PAST_FRAMES,
    load_anchor_valid,
    strict_anchor_indices,
)


FORMAL_EVAL_SCHEMA = "haco.open_loop_eval.v1"
ANCHOR_FRACTIONS = (0.25, 0.50, 0.75)


@dataclass(frozen=True)
class FormalEvalProtocol:
    action_contract: str
    action_target: str
    checkpoint_step: int = 30_000
    episode_count: int = 115
    action_horizon: int = ACTION_HORIZON
    rtc_prefix_steps: int = 10
    eval_seed: int = 42

    def __post_init__(self) -> None:
        contracts = {
            "joint_compliance_delta": "q_compliance",
            "compliance_only": "q_compliance",
            "nominal_only": "q_nominal",
        }
        expected_target = contracts.get(self.action_contract)
        if expected_target is None:
            raise ValueError(f"unsupported HACO contract {self.action_contract!r}")
        if self.action_target != expected_target:
            raise ValueError(
                f"{self.action_contract} requires target={expected_target}, "
                f"got {self.action_target}"
            )
        if self.checkpoint_step != 30_000:
            raise ValueError("formal HACO evaluation is fixed to checkpoint-30000")
        if self.episode_count != 115:
            raise ValueError("formal unscrew_cap evaluation requires all 115 episodes")
        if not 0 < self.rtc_prefix_steps < self.action_horizon:
            raise ValueError("RTC prefix must be positive and shorter than the horizon")

    @property
    def chunk_stride(self) -> int:
        return self.action_horizon - self.rtc_prefix_steps

    @property
    def active_semantic_dim(self) -> int:
        return 106 if self.action_contract == "joint_compliance_delta" else 62

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": FORMAL_EVAL_SCHEMA,
            "action_contract": self.action_contract,
            "action_target": self.action_target,
            "active_semantic_dim": self.active_semantic_dim,
            "carrier_dim": 132,
            "checkpoint_step": self.checkpoint_step,
            "episode_count": self.episode_count,
            "native_horizon": self.action_horizon,
            "rtc_enabled": True,
            "rtc_prefix_steps": self.rtc_prefix_steps,
            "chunk_stride": self.chunk_stride,
            "eval_seed": self.eval_seed,
            "validation_split": None,
            "evaluation_kind": "train-distribution-open-loop",
            "control_metric_space": "normalized",
            "delta_q_metric_space": (
                "physical_rad"
                if self.action_contract == "joint_compliance_delta"
                else None
            ),
            "execution_semantics": "direct_main_q",
            "delta_execution": False,
        }


def _episode_rows(dataset_root: Path) -> list[dict[str, Any]]:
    path = dataset_root / "meta/episodes.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows.sort(key=lambda item: int(item["episode_index"]))
    return rows


def _parquet_path(dataset_root: Path, episode_index: int, chunks_size: int) -> Path:
    return (
        dataset_root
        / "data"
        / f"chunk-{episode_index // chunks_size:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )


def adjacent_rtc_anchors(
    *,
    episode_length: int,
    anchor_valid: np.ndarray,
    protocol: FormalEvalProtocol,
) -> np.ndarray:
    """Legal first anchors whose second, stride-shifted chunk is also strict."""
    valid = np.asarray(anchor_valid, dtype=bool)
    if valid.shape != (episode_length,):
        raise ValueError(
            f"anchor_valid must have shape {(episode_length,)}, got {valid.shape}"
        )
    strict = strict_anchor_indices(episode_length)
    strict = strict[valid[strict]]
    strict_set = set(int(value) for value in strict)
    starts = [
        int(anchor)
        for anchor in strict
        if int(anchor) + protocol.chunk_stride in strict_set
        and valid[int(anchor) + protocol.chunk_stride]
    ]
    return np.asarray(starts, dtype=np.int64)


def episode_balanced_anchors(
    eligible: np.ndarray,
    fractions: tuple[float, ...] = ANCHOR_FRACTIONS,
) -> np.ndarray:
    values = np.asarray(eligible, dtype=np.int64)
    if values.ndim != 1 or values.size < len(fractions):
        raise ValueError("episode has too few adjacent RTC anchors")
    if np.any(values[1:] <= values[:-1]):
        raise ValueError("eligible anchors must be strictly increasing")
    positions = np.asarray(
        [int(np.floor(fraction * (len(values) - 1))) for fraction in fractions],
        dtype=np.int64,
    )
    selected = values[positions]
    if len(np.unique(selected)) != len(selected):
        raise ValueError("episode-balanced anchors are not unique")
    return selected


def build_manifest(
    dataset_root: str | Path,
    protocol: FormalEvalProtocol,
) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    if info.get("split_counts") != {"train": 115, "val": 0, "test": 0}:
        raise ValueError(
            "formal HACO evaluation requires train=115, val=0, test=0"
        )
    rows = _episode_rows(root)
    if len(rows) != protocol.episode_count:
        raise ValueError(
            f"expected {protocol.episode_count} episodes, found {len(rows)}"
        )
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("every formal-eval episode must belong to train")
    chunks_size = int(info["chunks_size"])
    samples = []
    for row in rows:
        episode_index = int(row["episode_index"])
        episode_length = int(row["length"])
        valid = load_anchor_valid(
            _parquet_path(root, episode_index, chunks_size)
        )
        eligible = adjacent_rtc_anchors(
            episode_length=episode_length,
            anchor_valid=valid,
            protocol=protocol,
        )
        selected = episode_balanced_anchors(eligible)
        for quantile, anchor in zip(ANCHOR_FRACTIONS, selected, strict=True):
            samples.append(
                {
                    "sample_index": len(samples),
                    "episode_index": episode_index,
                    "anchor_fraction": quantile,
                    "first_anchor": int(anchor),
                    "second_anchor": int(anchor) + protocol.chunk_stride,
                    "first_metric_range": [0, protocol.action_horizon],
                    "second_metric_range": [
                        protocol.rtc_prefix_steps,
                        protocol.action_horizon,
                    ],
                }
            )
    expected_samples = protocol.episode_count * len(ANCHOR_FRACTIONS)
    if len(samples) != expected_samples:
        raise AssertionError(
            f"expected {expected_samples} episode-balanced samples, got {len(samples)}"
        )
    return {
        **protocol.metadata(),
        "dataset": str(root),
        "anchors_per_episode": len(ANCHOR_FRACTIONS),
        "sample_count": len(samples),
        "history_past_frames": HISTORY_PAST_FRAMES,
        "samples": samples,
    }


def metric_step_mask(protocol: FormalEvalProtocol) -> np.ndarray:
    mask = np.ones((2, protocol.action_horizon), dtype=bool)
    mask[1, : protocol.rtc_prefix_steps] = False
    return mask


def action_metrics(
    prediction_main: np.ndarray,
    target_main: np.ndarray,
    *,
    protocol: FormalEvalProtocol,
    prediction_delta_q_rad: np.ndarray | None = None,
    target_delta_q_rad: np.ndarray | None = None,
) -> dict[str, float]:
    prediction = np.asarray(prediction_main, dtype=np.float64)
    target = np.asarray(target_main, dtype=np.float64)
    expected = (2, protocol.action_horizon, 62)
    if prediction.shape != expected or target.shape != expected:
        raise ValueError(f"main action arrays must have shape {expected}")
    include = metric_step_mask(protocol)
    difference = prediction[include] - target[include]
    metrics = {
        "open-loop-eval/action_mse_wrist_18d": float(
            np.mean(difference[:, :18] ** 2)
        ),
        "open-loop-eval/action_mse_hand_joint_44d": float(
            np.mean(difference[:, 18:62] ** 2)
        ),
        "open-loop-eval/included_steps": int(include.sum()),
        "open-loop-eval/excluded_rtc_prefix_steps": int((~include).sum()),
    }
    delta_values = (prediction_delta_q_rad, target_delta_q_rad)
    if protocol.action_contract == "joint_compliance_delta":
        if any(value is None for value in delta_values):
            raise ValueError("joint contract requires physical-rad delta_q arrays")
        prediction_delta = np.asarray(prediction_delta_q_rad, dtype=np.float64)
        target_delta = np.asarray(target_delta_q_rad, dtype=np.float64)
        expected_delta = (2, protocol.action_horizon, 44)
        if prediction_delta.shape != expected_delta or target_delta.shape != expected_delta:
            raise ValueError(f"delta_q arrays must have shape {expected_delta}")
        metrics["open-loop-eval/delta_q_mae_rad"] = float(
            np.mean(np.abs(prediction_delta[include] - target_delta[include]))
        )
    elif any(value is not None for value in delta_values):
        raise ValueError(
            "delta-free HACO evaluation must not contain delta_q"
        )
    return metrics


def summarize_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    if not rows:
        raise ValueError("cannot summarize an empty HACO evaluation")
    metric_names = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key.startswith("open-loop-eval/")
            and key not in {
                "open-loop-eval/included_steps",
                "open-loop-eval/excluded_rtc_prefix_steps",
            }
            and isinstance(value, (int, float))
        }
    )
    summary = {}
    for name in metric_names:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        summary[f"{name}/mean"] = float(np.mean(values))
        summary[f"{name}/median"] = float(np.median(values))
    return {"sample_count": len(rows), "metrics": summary}


__all__ = [
    "ANCHOR_FRACTIONS",
    "FORMAL_EVAL_SCHEMA",
    "FormalEvalProtocol",
    "action_metrics",
    "adjacent_rtc_anchors",
    "build_manifest",
    "episode_balanced_anchors",
    "metric_step_mask",
    "summarize_records",
]
