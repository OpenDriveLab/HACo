"""Checkpoint and observation helpers for HACO deployment."""

from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from PIL import Image

from dexterity.runtime.sharpa62 import (
    DIRECT_EEF_INTERFACE_SCHEMA,
    model_action_to_wire_layout,
    wire_state_to_model_layout,
)


ACTION_DIM = 62
LANGUAGE_KEY = "annotation.language.task_description"
STATE_KEYS = (
    "left_wrist_eef",
    "right_wrist_eef",
    "left_hand_joints",
    "right_hand_joints",
)
STATE_SLICES = {
    "left_wrist_eef": slice(0, 9),
    "right_wrist_eef": slice(9, 18),
    "left_hand_joints": slice(18, 40),
    "right_hand_joints": slice(40, 62),
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def decode_jpeg_rgb(value: Any, label: str = "observation image") -> np.ndarray:
    """Decode one JPEG to RGB while preserving its width and height."""

    if isinstance(value, np.ndarray):
        data = value.astype(np.uint8, copy=False).tobytes()
    elif isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
    else:
        raise ValueError(f"{label} must be bytes")
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def finite_state(value: Any) -> np.ndarray:
    state = np.asarray(value, dtype=np.float32)
    if state.shape != (ACTION_DIM,):
        raise ValueError(f"expected hand pose shape (62,), got {state.shape}")
    if not np.all(np.isfinite(state)):
        raise ValueError("hand pose contains NaN or Inf")
    return state


class PosttrainContract:
    """Direct absolute-EEF boundary with the required joint-order mapping."""

    def __init__(self) -> None:
        self.report_path = DIRECT_EEF_INTERFACE_SCHEMA

    def state_to_model(self, state_deploy: np.ndarray) -> np.ndarray:
        return wire_state_to_model_layout(finite_state(state_deploy))

    def action_to_deploy(self, action_model: np.ndarray) -> np.ndarray:
        return model_action_to_wire_layout(action_model)


def trim_or_pad(action: np.ndarray, horizon: int) -> np.ndarray:
    if action.shape[0] < horizon:
        if action.shape[0] == 0:
            raise ValueError("HACO returned an empty action horizon")
        return np.concatenate(
            [action, np.repeat(action[-1:], horizon - action.shape[0], axis=0)]
        )
    return action[:horizon].astype(np.float32)


def checkpoint_view_with_local_backbone(
    checkpoint: Path, backbone_model: Path
) -> tuple[Path, tempfile.TemporaryDirectory[str]]:
    """Create a temporary checkpoint view with a relocated backbone path."""

    if not (backbone_model / "config.json").is_file():
        raise FileNotFoundError(f"backbone not found: {backbone_model}")
    owner = tempfile.TemporaryDirectory(prefix="haco-deploy-checkpoint-")
    view = Path(owner.name)
    patched_files = {"config.json", "processor_config.json"}
    for item in checkpoint.iterdir():
        if item.name not in patched_files:
            (view / item.name).symlink_to(item, target_is_directory=item.is_dir())

    # The external runtime selects the implementation from this canonical
    # model name, so expose the supplied local snapshot through that layout.
    backbone_alias = view / "backbones" / "nvidia" / "Cosmos-Reason2-2B"
    backbone_alias.parent.mkdir(parents=True)
    backbone_alias.symlink_to(backbone_model.resolve(), target_is_directory=True)
    resolved_backbone = str(backbone_alias)

    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    config["model_name"] = resolved_backbone
    write_json(view / "config.json", config)

    processor = json.loads(
        (checkpoint / "processor_config.json").read_text(encoding="utf-8")
    )
    processor.setdefault("processor_kwargs", {})["model_name"] = resolved_backbone
    write_json(view / "processor_config.json", processor)
    return view, owner


__all__ = [
    "LANGUAGE_KEY",
    "STATE_KEYS",
    "STATE_SLICES",
    "PosttrainContract",
    "checkpoint_view_with_local_backbone",
    "decode_jpeg_rgb",
    "trim_or_pad",
    "write_json",
]
