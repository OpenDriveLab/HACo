from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


DROID_EMBODIMENT_TAG_NAME = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
DROID_EMBODIMENT_TAG_VALUE = "oxe_droid_relative_eef_relative_joint"


@dataclass(frozen=True)
class N17DroidContract:
    """The DROID contract used by official GR00T N1.7 checkpoints."""

    embodiment_tag_name: str = DROID_EMBODIMENT_TAG_NAME
    embodiment_tag_value: str = DROID_EMBODIMENT_TAG_VALUE
    model_video_keys: tuple[str, ...] = ("exterior_image_1_left", "wrist_image_left")
    lerobot_video_keys: tuple[str, ...] = ("exterior_1_left", "wrist_left")
    original_video_keys: tuple[str, ...] = (
        "observation.images.exterior_1_left",
        "observation.images.wrist_left",
    )
    state_keys: tuple[str, ...] = ("eef_9d", "gripper_position", "joint_position")
    action_keys: tuple[str, ...] = ("eef_9d", "gripper_position", "joint_position")
    language_key: str = "annotation.language.language_instruction"
    state_dims: tuple[int, ...] = (9, 1, 7)
    action_dims: tuple[int, ...] = (9, 1, 7)
    state_delta_indices: tuple[int, ...] = (0,)
    droid_video_delta_indices: tuple[int, ...] = (0,)
    base_video_delta_indices: tuple[int, ...] = (-15, 0)
    action_delta_indices: tuple[int, ...] = tuple(range(40))

    @property
    def state_dim(self) -> int:
        return sum(self.state_dims)

    @property
    def action_dim(self) -> int:
        return sum(self.action_dims)

    def modality_json(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Return the LeRobot `meta/modality.json` shape expected by N1.7 DROID."""

        def ranges(keys: tuple[str, ...], dims: tuple[int, ...]) -> dict[str, dict[str, int]]:
            start = 0
            out: dict[str, dict[str, int]] = {}
            for key, dim in zip(keys, dims, strict=True):
                out[key] = {"start": start, "end": start + dim}
                start += dim
            return out

        return {
            "state": ranges(self.state_keys, self.state_dims),
            "action": ranges(self.action_keys, self.action_dims),
            "video": {
                key: {"original_key": original_key}
                for key, original_key in zip(
                    self.lerobot_video_keys, self.original_video_keys, strict=True
                )
            },
            "annotation": {
                self.language_key.removeprefix("annotation."): {"original_key": "task_index"}
            },
        }


DROID_CONTRACT = N17DroidContract()


def _processor_config_path(checkpoint_path: str | Path) -> Path:
    root = Path(checkpoint_path)
    candidates = [root / "processor_config.json", root / "processor" / "processor_config.json"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No processor_config.json found under {root}")


def load_checkpoint_modality_config(
    checkpoint_path: str | Path,
    tag_value: str = DROID_EMBODIMENT_TAG_VALUE,
) -> dict[str, Any]:
    """Load a tag-specific modality config from an N1.7 checkpoint."""

    path = _processor_config_path(checkpoint_path)
    processor_config = json.loads(path.read_text())
    modality_configs = processor_config.get("processor_kwargs", {}).get("modality_configs", {})
    if tag_value not in modality_configs:
        available = ", ".join(sorted(modality_configs))
        raise KeyError(f"Checkpoint {path} does not contain tag {tag_value!r}; available: {available}")
    return modality_configs[tag_value]


def summarize_checkpoint_contract(checkpoint_path: str | Path) -> dict[str, Any]:
    """Return a dependency-free summary of a checkpoint's N1.7 DROID contract."""

    config = load_checkpoint_modality_config(checkpoint_path)
    return {
        "checkpoint": str(checkpoint_path),
        "embodiment_tag": DROID_CONTRACT.embodiment_tag_name,
        "video": {
            "delta_indices": config["video"]["delta_indices"],
            "model_keys": config["video"]["modality_keys"],
            "lerobot_keys": list(DROID_CONTRACT.lerobot_video_keys),
        },
        "state": {
            "delta_indices": config["state"]["delta_indices"],
            "keys": config["state"]["modality_keys"],
            "dim": DROID_CONTRACT.state_dim,
        },
        "action": {
            "delta_indices": config["action"]["delta_indices"],
            "keys": config["action"]["modality_keys"],
            "dim": DROID_CONTRACT.action_dim,
            "configs": config["action"].get("action_configs", []),
        },
        "language": {
            "delta_indices": config["language"]["delta_indices"],
            "keys": config["language"]["modality_keys"],
        },
    }


def validate_droid_checkpoint_contract(checkpoint_path: str | Path) -> None:
    """Validate that a checkpoint exposes the expected N1.7 DROID keys."""

    config = load_checkpoint_modality_config(checkpoint_path)
    expected = DROID_CONTRACT

    checks = {
        "video.modality_keys": (
            tuple(config["video"]["modality_keys"]),
            expected.model_video_keys,
        ),
        "state.modality_keys": (
            tuple(config["state"]["modality_keys"]),
            expected.state_keys,
        ),
        "state.delta_indices": (
            tuple(config["state"]["delta_indices"]),
            expected.state_delta_indices,
        ),
        "action.modality_keys": (
            tuple(config["action"]["modality_keys"]),
            expected.action_keys,
        ),
        "action.delta_indices": (
            tuple(config["action"]["delta_indices"]),
            expected.action_delta_indices,
        ),
        "language.modality_keys": (
            tuple(config["language"]["modality_keys"]),
            (expected.language_key,),
        ),
    }

    failures = [
        f"{name}: got {actual!r}, expected {want!r}"
        for name, (actual, want) in checks.items()
        if actual != want
    ]
    video_delta_indices = tuple(config["video"]["delta_indices"])
    expected_video_delta_indices = {
        expected.base_video_delta_indices,
        expected.droid_video_delta_indices,
    }
    if video_delta_indices not in expected_video_delta_indices:
        failures.append(
            "video.delta_indices: got "
            f"{video_delta_indices!r}, expected one of {sorted(expected_video_delta_indices)!r}"
        )
    if failures:
        joined = "\n".join(f"  - {failure}" for failure in failures)
        raise ValueError(f"Checkpoint {checkpoint_path} does not match N1.7 DROID:\n{joined}")
