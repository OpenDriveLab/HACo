#!/usr/bin/env python3
"""Serve the posttrain GR00T N1.7 SharpA62 policy.

The robot supplies canonical absolute EEF state and owns all XYZ-direction
transforms. The selected GR00T processor may model relative or absolute wrist
actions; both decode to absolute EEF at this boundary. The server passes EEF
through unchanged and only maps the 44 hand joints between public and model
order.
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import logging
from pathlib import Path
import socket
import sys
import tempfile
import time
from typing import Any

import numpy as np
from PIL import Image
import torch

try:
    from dexterity.deploy.template.action_contract import (
        ACTION_HORIZON,
        build_policy_action,
        common_action_metadata,
    )
    from dexterity.runtime.sharpa62 import (
        DIRECT_EEF_INTERFACE_SCHEMA,
        direct_eef_interface_metadata,
        model_action_to_wire_layout,
        wire_state_to_model_layout,
    )
    from dexterity.deploy.template.model_adapter import SharpAModelAdapter
    from dexterity.deploy.template.server import SharpAPolicyServer
except ImportError:
    from dexterity.deploy.template.action_contract import (
        ACTION_HORIZON,
        build_policy_action,
        common_action_metadata,
    )
    from dexterity.runtime.sharpa62 import (
        DIRECT_EEF_INTERFACE_SCHEMA,
        direct_eef_interface_metadata,
        model_action_to_wire_layout,
        wire_state_to_model_layout,
    )
    from dexterity.deploy.template.model_adapter import SharpAModelAdapter
    from dexterity.deploy.template.server import SharpAPolicyServer


LOGGER = logging.getLogger("groot_n17_sharpa62_server")

ACTION_DIM = 62
WRIST_DIM = 9
HAND_DIM = 22
DEFAULT_ACTION_HORIZON = ACTION_HORIZON
VIDEO_KEYS = ("ego_view", "left_wrist_view", "right_wrist_view")
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


@dataclasses.dataclass
class ServerMetadata:
    schema: str = "sharpa62_policy_server.v1"
    request_schema: str = "sharpa62_model_observation.v1"
    response_schema: str = "sharpa62_policy_action.v1"
    policy_family: str = "groot"
    execute_joint_source: str = "model_q"
    model_name: str = "groot_n17"
    embodiment: str = "adam_pro_sharpa_relative_eef"
    robot: str = "sharpa62"
    image_keys: tuple[str, ...] = (
        "observation/ego_view_jpeg",
        "observation/left_wrist_view_jpeg",
        "observation/right_wrist_view_jpeg",
    )
    accepted_image_keys: tuple[str, ...] = (
        "observation/ego_view_jpeg",
        "observation/ego_view",
        "observation/left_wrist_view_jpeg",
        "observation/left_wrist_view",
        "observation/right_wrist_view_jpeg",
        "observation/right_wrist_view",
    )
    state_key: str = "observation/hand_pose_62d"
    timestamp_key: str = "observation/timestamp_unix_s"
    prompt_key: str = "prompt"
    session_key: str = "session_id"
    image_height: int | None = None
    image_width: int | None = None
    image_size_policy: str = "preserve_source_resolution"
    image_resize_owner: str = "groot_checkpoint_processor"
    state_dim: int = ACTION_DIM
    action_dim: int = ACTION_DIM
    action_horizon: int = DEFAULT_ACTION_HORIZON
    action_hz: float = 30.0
    layout: str = "left_wrist9,right_wrist9,sharpa_q44"
    model_layout: str = "left_wrist9,right_wrist9,left_hand22,right_hand22"
    action_space: str = "sharpa_dexretarget_position_62d"
    output_wrist_frame: str = "absolute"
    server_xyz_direction_transform: str = "none"
    transport: str = "websocket+msgpack_numpy"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def ensure_uint8_rgb(value: Any) -> np.ndarray:
    """Validate a decoded RGB frame without changing its spatial geometry."""

    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected RGB image shape (H,W,3), got {image.shape}")
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            minimum = float(np.nanmin(image)) if image.size else 0.0
            maximum = float(np.nanmax(image)) if image.size else 0.0
            if 0.0 <= minimum and maximum <= 1.0:
                image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    # GR00T's checkpoint processor owns letterboxing, cropping, and resizing.
    # Resizing here would distort camera geometry and apply preprocessing twice.
    return np.ascontiguousarray(image)


def decode_jpeg_rgb(value: Any, label: str = "observation image") -> np.ndarray:
    """Decode one JPEG to RGB while preserving the encoded width and height."""

    if isinstance(value, np.ndarray):
        data = value.astype(np.uint8, copy=False).tobytes()
    elif isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
    else:
        raise ValueError(f"{label} must be bytes")
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def decode_video_request(
    obs: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, str], int]:
    """Decode the synchronized ego/left-wrist/right-wrist model inputs."""

    images: dict[str, np.ndarray] = {}
    transports: dict[str, str] = {}
    ego_received_frames = 1
    for video_key in VIDEO_KEYS:
        jpeg_key = f"observation/{video_key}_jpeg"
        raw_key = f"observation/{video_key}"
        if jpeg_key in obs:
            image = decode_jpeg_rgb(obs[jpeg_key], jpeg_key)
            transports[video_key] = "jpeg"
        elif raw_key in obs:
            raw_image = np.asarray(obs[raw_key])
            if raw_image.ndim == 4:
                if len(raw_image) == 0:
                    raise ValueError(f"{raw_key} contains no frames")
                if video_key == "ego_view":
                    ego_received_frames = int(len(raw_image))
                raw_image = raw_image[-1]
            image = ensure_uint8_rgb(raw_image)
            transports[video_key] = "raw_numpy"
        else:
            raise KeyError(
                f"GR00T multiview request requires {jpeg_key!r} or {raw_key!r}"
            )
        images[video_key] = image[None, None, ...]
    return images, transports, ego_received_frames


def finite_state(value: Any) -> np.ndarray:
    state = np.asarray(value, dtype=np.float32)
    if state.shape != (ACTION_DIM,):
        raise ValueError(f"expected hand pose shape (62,), got {state.shape}")
    if not np.all(np.isfinite(state)):
        raise ValueError("hand pose contains NaN or Inf")
    return state


class PosttrainContract:
    """Direct absolute-EEF boundary plus the required joint-order mapping."""

    def __init__(self) -> None:
        self.report_path = DIRECT_EEF_INTERFACE_SCHEMA

    def state_to_model(self, state_deploy: np.ndarray) -> np.ndarray:
        return wire_state_to_model_layout(finite_state(state_deploy))

    def action_to_deploy(self, action_model: np.ndarray) -> np.ndarray:
        return model_action_to_wire_layout(action_model)


def concat_action_dict(action: dict[str, Any]) -> np.ndarray:
    pieces = []
    for key in STATE_KEYS:
        if key not in action:
            raise KeyError(f"GR00T action missing {key!r}; got {sorted(action)}")
        value = np.asarray(action[key], dtype=np.float32)
        if value.ndim == 3:
            value = value[0]
        if value.ndim != 2:
            raise ValueError(f"action {key!r} must be (T,D), got {value.shape}")
        pieces.append(value)
    result = np.concatenate(pieces, axis=-1).astype(np.float32)
    if result.shape[1] != ACTION_DIM:
        raise ValueError(f"concatenated GR00T action must be 62D, got {result.shape}")
    return result


def trim_or_pad(action: np.ndarray, horizon: int) -> np.ndarray:
    if action.shape[0] < horizon:
        if action.shape[0] == 0:
            raise ValueError("GR00T returned an empty action horizon")
        return np.concatenate(
            [action, np.repeat(action[-1:], horizon - action.shape[0], axis=0)]
        )
    return action[:horizon].astype(np.float32)


def checkpoint_view_with_local_backbone(
    checkpoint: Path, backbone_model: Path
) -> tuple[Path, tempfile.TemporaryDirectory[str]]:
    """Create a read-only checkpoint view with relocated backbone paths.

    The training checkpoint records the old machine-local Cosmos directory in
    both config files.  Symlink all immutable checkpoint assets into a temporary
    directory and rewrite only those two JSON files there.
    """

    if not (backbone_model / "config.json").is_file():
        raise FileNotFoundError(f"Cosmos backbone not found: {backbone_model}")
    owner = tempfile.TemporaryDirectory(prefix="groot-deploy-checkpoint-")
    view = Path(owner.name)
    patched_files = {"config.json", "processor_config.json"}
    for item in checkpoint.iterdir():
        if item.name not in patched_files:
            (view / item.name).symlink_to(item, target_is_directory=item.is_dir())

    # Isaac-GR00T selects the backbone implementation from the model-name
    # string, so retain the canonical Cosmos basename while pointing it at the
    # relocated local snapshot.
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


def load_policy(
    checkpoint: Path,
    reference_repo: Path,
    backbone_model: Path,
    embodiment_tag: str,
    device: str,
):
    if str(reference_repo) not in sys.path:
        sys.path.insert(0, str(reference_repo))
    if embodiment_tag == "real_r1_pro_sharpa_absolute_eef":
        from scripts.train.groot_n17.embodiment import (
            register_sharpa_absolute_eef_embodiment,
        )

        register_sharpa_absolute_eef_embodiment()
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    checkpoint_view, checkpoint_view_owner = checkpoint_view_with_local_backbone(
        checkpoint, backbone_model
    )
    policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=str(checkpoint_view),
        device=device,
        strict=True,
    )
    modality = policy.get_modality_config()
    failures = []
    if tuple(modality["video"].modality_keys) != VIDEO_KEYS:
        failures.append(f"video keys={modality['video'].modality_keys}")
    if tuple(modality["state"].modality_keys) != STATE_KEYS:
        failures.append(f"state keys={modality['state'].modality_keys}")
    if tuple(modality["action"].modality_keys) != STATE_KEYS:
        failures.append(f"action keys={modality['action'].modality_keys}")
    if tuple(modality["language"].modality_keys) != (LANGUAGE_KEY,):
        failures.append(f"language keys={modality['language'].modality_keys}")
    if len(modality["action"].delta_indices) < DEFAULT_ACTION_HORIZON:
        failures.append(f"action horizon={len(modality['action'].delta_indices)}")
    if failures:
        raise ValueError(
            "checkpoint does not match posttrain N1.7 contract: " + "; ".join(failures)
        )
    return policy, len(modality["action"].delta_indices), checkpoint_view_owner


class GrootPolicy:
    def __init__(
        self,
        policy: Any,
        contract: PosttrainContract,
        action_horizon: int,
        checkpoint_view_owner: tempfile.TemporaryDirectory[str],
    ) -> None:
        self.policy = policy
        self.contract = contract
        self.action_horizon = int(action_horizon)
        # Retain the temporary symlink view for the whole server lifetime.
        self.checkpoint_view_owner = checkpoint_view_owner
        self.current_session_id: str | None = None
        self.current_prompt = ""
        self.request_index = 0

    def reset(self, info: dict[str, Any]) -> str:
        self.current_session_id = str(
            info.get("session_id", self.current_session_id or "default")
        )
        self.current_prompt = ""
        self.request_index = 0
        self.policy.reset(info)
        return "reset successful"

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        if obs.get("schema") != "sharpa62_model_observation.v1":
            raise ValueError(
                "schema must be 'sharpa62_model_observation.v1', "
                f"got {obs.get('schema')!r}"
            )
        session_id = str(obs.get("session_id", "default"))
        if self.current_session_id is None:
            self.current_session_id = session_id
        elif session_id != self.current_session_id:
            self.reset({"session_id": session_id})

        if obs.get("prompt"):
            self.current_prompt = str(obs["prompt"])
        video, image_transports, received_frames = decode_video_request(obs)
        raw_states = np.asarray(obs["observation/hand_pose_62d"], dtype=np.float32)
        state_deploy = finite_state(
            raw_states[-1] if raw_states.ndim == 2 else raw_states
        )
        state_model = self.contract.state_to_model(state_deploy)
        n17_observation = {
            "video": video,
            "state": {
                key: state_model[value][None, None, :].astype(np.float32)
                for key, value in STATE_SLICES.items()
            },
            "language": {LANGUAGE_KEY: [[self.current_prompt]]},
        }
        with torch.inference_mode():
            action_dict, info = self.policy.get_action(n17_observation)
        action_model = trim_or_pad(concat_action_dict(action_dict), self.action_horizon)
        action_deploy = self.contract.action_to_deploy(action_model)
        request_index = self.request_index
        self.request_index += 1
        return build_policy_action(
            action_deploy,
            session_id=session_id,
            request_index=request_index,
            policy_family="groot",
            execute_joint_source="model_q",
            diagnostics={
                "model": "groot_n17_posttrain",
                "input_frames_received": np.asarray(received_frames, dtype=np.int64),
                "input_frames_used": np.asarray(1, dtype=np.int64),
                "input_views_used": np.asarray(len(VIDEO_KEYS), dtype=np.int64),
                "image_transport": ",".join(
                    f"{key}:{image_transports[key]}" for key in VIDEO_KEYS
                ),
                "anchor_state_deploy_62d": state_deploy,
                "eef_interface": str(self.contract.report_path),
                "elapsed_s": np.asarray(
                    time.perf_counter() - started, dtype=np.float32
                ),
                "policy_info": _pack_info(info),
            },
        )


def _pack_info(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        key: item
        for key, item in value.items()
        if isinstance(item, (str, int, float, bool, np.ndarray))
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve posttrain GR00T N1.7 SharpA62.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5500)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-repo", required=True)
    parser.add_argument("--backbone-model", required=True)
    parser.add_argument("--embodiment-tag", default="real_r1_pro_sharpa_relative_eef")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--action-horizon", type=int, default=DEFAULT_ACTION_HORIZON)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    checkpoint = Path(args.checkpoint).expanduser().absolute()
    reference_repo = Path(args.reference_repo).expanduser().absolute()
    backbone_model = Path(args.backbone_model).expanduser().absolute()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if args.action_horizon != DEFAULT_ACTION_HORIZON:
        raise ValueError(
            f"unified SharpA62 contract requires --action-horizon={DEFAULT_ACTION_HORIZON}"
        )

    policy, raw_horizon, checkpoint_view_owner = load_policy(
        checkpoint,
        reference_repo,
        backbone_model,
        args.embodiment_tag,
        args.device,
    )
    if args.action_horizon > raw_horizon:
        raise ValueError(
            f"requested horizon {args.action_horizon} exceeds checkpoint horizon {raw_horizon}"
        )
    contract = PosttrainContract()
    output_dir = (
        Path(args.output_dir) if args.output_dir else checkpoint / "deploy_groot_n17"
    )
    output_dir = output_dir.expanduser().absolute()
    metadata = dataclasses.asdict(ServerMetadata(action_horizon=args.action_horizon))
    metadata.update(
        common_action_metadata(
            policy_family="groot",
            execute_joint_source="model_q",
        )
    )
    metadata.update(
        {
            "embodiment": args.embodiment_tag,
            "checkpoint": str(checkpoint),
            "reference_repo": str(reference_repo),
            "backbone_model": str(backbone_model),
            "eef_interface": direct_eef_interface_metadata(),
            "embodiment_tag": args.embodiment_tag,
            "raw_action_horizon": raw_horizon,
            "output_dir": str(output_dir),
            "host": socket.gethostname(),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "device": args.device,
        }
    )
    write_json(output_dir / "server_metadata.json", metadata)
    adapter = SharpAModelAdapter(
        GrootPolicy(
            policy,
            contract,
            args.action_horizon,
            checkpoint_view_owner,
        ),
        model_kind="groot",
        policy_family="groot",
    )
    SharpAPolicyServer(
        adapter,
        policy_family="groot",
        model_name="groot_n17",
        checkpoint_path=checkpoint,
        host=args.host,
        port=args.port,
    ).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
