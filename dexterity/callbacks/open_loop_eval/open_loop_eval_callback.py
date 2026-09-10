from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image
import torch
from transformers import TrainerCallback


CONTROL_GROUP_SLICES = {
    "left_wrist_eef": slice(0, 9),
    "right_wrist_eef": slice(9, 18),
    "left_hand_joints": slice(18, 40),
    "right_hand_joints": slice(40, 62),
}
CONTROL_GROUPS = tuple(CONTROL_GROUP_SLICES)

# Component-major action groups, in concatenation order. Models with a torque
# head use the full tuple (150-D); models with only a compliance residual use
# COMPLIANCE_COMPONENT_MAJOR_ACTION_GROUPS (106-D). Either way delta_q is last,
# which is all the renderer needs.
COMPONENT_MAJOR_ACTION_GROUPS = (
    "left_wrist_eef",
    "right_wrist_eef",
    "left_hand_joints",
    "right_hand_joints",
    "left_hand_tau",
    "right_hand_tau",
    "left_hand_delta_q",
    "right_hand_delta_q",
)
COMPLIANCE_COMPONENT_MAJOR_ACTION_GROUPS = (
    "left_wrist_eef",
    "right_wrist_eef",
    "left_hand_joints",
    "right_hand_joints",
    "left_hand_delta_q",
    "right_hand_delta_q",
)


@dataclass(frozen=True)
class OpenLoopEvalConfig:
    dreamzero_root: str
    embodiment_tag: str
    action_horizon: int = 40
    every_n_steps: int = 500
    first_step: bool = False
    log_name: str = "open_loop_eval"
    sample_seed: int = 42

    def __post_init__(self) -> None:
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if self.every_n_steps <= 0:
            raise ValueError("every_n_steps must be positive")


@dataclass
class PreparedRollout:
    policy_inputs: dict[str, Any]
    state_abs: np.ndarray
    target_control_abs: np.ndarray
    target_control_norm: np.ndarray
    hip_pose: Any
    input_image: Any
    chunk_count: int
    horizon: int
    # Optional component-major action; present only for models that predict
    # a compliance residual. render_delta_q_slice locates delta_q within it.
    target_action_for_render: np.ndarray | None = None
    tactile_wrench_for_render: np.ndarray | None = None
    tactile_wrench_valid_for_render: np.ndarray | None = None
    tactile_wrench_pred_for_render: np.ndarray | None = None
    tactile_wrench_pred_valid_for_render: np.ndarray | None = None
    model_visual_inputs: Mapping[str, Any] | None = None


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _rollout_artifact_dir(state) -> Path | None:
    root = os.environ.get("OPEN_LOOP_EVAL_ARTIFACT_DIR")
    if not root:
        return None
    output = Path(root) / f"step-{int(state.global_step):06d}"
    output.mkdir(parents=True, exist_ok=True)
    return output


def _save_rollout_input_image(value: Any, output: Path) -> None:
    image = _to_numpy(value)
    if image is None:
        return
    while image.ndim > 3:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"rollout input image must be three-dimensional, got {image.shape}")
    if image.shape[0] in {1, 3} and image.shape[-1] not in {1, 3}:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] == 1:
        image = image[..., 0]
    if np.issubdtype(image.dtype, np.floating) and float(np.nanmax(image)) <= 1.0:
        image = image * 255.0
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(output)


def _persist_rollout_artifacts(
    state,
    prepared: "PreparedRollout",
    metrics: Mapping[str, float],
    temporary_dir: str | Path,
    filenames: Mapping[str, str],
) -> None:
    output = _rollout_artifact_dir(state)
    if output is None:
        return
    temporary = Path(temporary_dir)
    for destination, source in filenames.items():
        path = temporary / source
        if path.is_file():
            shutil.copy2(path, output / destination)
    if prepared.input_image is not None:
        _save_rollout_input_image(prepared.input_image, output / "input_image.png")
    (output / "metrics.json").write_text(
        json.dumps({key: float(value) for key, value in metrics.items()}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _split_control(values: np.ndarray) -> dict[str, np.ndarray]:
    array = np.asarray(values, dtype=np.float32)
    if array.shape[-1] < 62:
        raise ValueError(f"control tensor must end in at least 62, got {array.shape}")
    return {
        key: array[..., group_slice]
        for key, group_slice in CONTROL_GROUP_SLICES.items()
    }


def _concat_control(parts: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([parts[key] for key in CONTROL_GROUPS], axis=-1).astype(
        np.float32
    )


def _state_dict_from_abs(state_abs: np.ndarray) -> dict[str, np.ndarray]:
    return {
        f"state.{key}": value
        for key, value in _split_control(state_abs).items()
    }


def _action_dict_control(action_dict: Mapping[str, np.ndarray]) -> np.ndarray:
    return _concat_control(
        {key: action_dict[f"action.{key}"] for key in CONTROL_GROUPS}
    )


def _first_image_patch_count(inputs: Mapping[str, Any]) -> int | None:
    grid = _to_numpy(inputs.get("image_grid_thw"))
    if grid is None:
        return None
    while grid.ndim > 2:
        grid = grid[0]
    if grid.ndim == 1:
        grid = grid.reshape(1, -1)
    if grid.shape[0] == 0 or grid.shape[-1] < 3:
        return None
    # ``inputs`` has already been reduced to one training sample here.  A
    # multiview sample therefore has V grid rows, and its pixel carrier spans
    # the patches from every row rather than only the first camera.
    count = int(np.prod(grid[:, :3].astype(np.int64), axis=-1).sum())
    return count if count > 0 else None


def _batch_size(inputs: Mapping[str, Any]) -> int:
    for key in (
        "action",
        "norm_actions",
        "noisy_actions",
        "target",
        "state_raw",
        "viz_rollout_state_abs_62d",
        "state",
        "viz_input_image",
        "image_grid_thw",
    ):
        value = inputs.get(key)
        if torch.is_tensor(value) or isinstance(value, np.ndarray):
            if value.ndim > 0:
                return int(value.shape[0])
        if isinstance(value, (list, tuple)):
            return len(value)
    raise ValueError("open-loop evaluation could not infer the training batch size")


def _select_training_sample(
    inputs: Mapping[str, Any], sample_index: int
) -> dict[str, Any]:
    """Select one current-batch sample while preserving a B=1 dimension."""

    batch_size = _batch_size(inputs)
    if sample_index < 0 or sample_index >= batch_size:
        raise IndexError(f"sample_index={sample_index} outside batch_size={batch_size}")

    grid = inputs.get("image_grid_thw")
    grid_start = grid_end = patch_start = patch_end = None
    if torch.is_tensor(grid) or isinstance(grid, np.ndarray):
        grid_array = _to_numpy(grid)
        if grid_array is not None and grid_array.ndim >= 2:
            grid_array = grid_array.reshape(-1, grid_array.shape[-1])
            if grid_array.shape[0] % batch_size != 0:
                raise ValueError(
                    "image_grid_thw rows must be divisible by the training "
                    f"batch size, got rows={grid_array.shape[0]} batch={batch_size}"
                )
            views_per_sample = grid_array.shape[0] // batch_size
            grid_start = sample_index * views_per_sample
            grid_end = grid_start + views_per_sample
            patch_counts = np.prod(grid_array[:, :3], axis=-1).astype(int)
            patch_start = int(patch_counts[:grid_start].sum())
            patch_end = int(patch_counts[:grid_end].sum())

    selected: dict[str, Any] = {}
    for key, value in inputs.items():
        if key == "pixel_values" and patch_start is not None and patch_end is not None:
            selected[key] = value[patch_start:patch_end]
        elif key == "image_grid_thw" and grid_start is not None and grid_end is not None:
            selected[key] = value[grid_start:grid_end]
        elif torch.is_tensor(value) or isinstance(value, np.ndarray):
            selected[key] = (
                value[sample_index : sample_index + 1]
                if value.ndim > 0 and value.shape[0] == batch_size
                else value
            )
        elif isinstance(value, list) and len(value) == batch_size:
            selected[key] = [value[sample_index]]
        elif isinstance(value, tuple) and len(value) == batch_size:
            selected[key] = (value[sample_index],)
        else:
            selected[key] = value
    return selected


def _repeat_first_batch(
    value: Any,
    repeats: int,
    key: str,
    inputs: Mapping[str, Any],
) -> Any:
    if value is None:
        return None
    if key == "pixel_values":
        patch_count = _first_image_patch_count(inputs)
        if patch_count is not None:
            if torch.is_tensor(value):
                first = value[:patch_count]
                return first.repeat(repeats, *([1] * (first.ndim - 1)))
            if isinstance(value, np.ndarray) and value.ndim > 0:
                first = value[:patch_count]
                return np.concatenate([first.copy() for _ in range(repeats)], axis=0)
    if key == "image_grid_thw":
        # The input has already been reduced to B=1, so retain all V view rows
        # and repeat the complete view group for each rollout chunk.
        if torch.is_tensor(value):
            return value.repeat(repeats, *([1] * (value.ndim - 1)))
        if isinstance(value, np.ndarray) and value.ndim > 0:
            return np.concatenate([value.copy() for _ in range(repeats)], axis=0)
    if torch.is_tensor(value):
        first = value[:1] if value.ndim > 0 else value
        return first.repeat(repeats, *([1] * (first.ndim - 1)))
    if isinstance(value, np.ndarray) and value.ndim > 0:
        return np.repeat(value[:1], repeats, axis=0)
    if isinstance(value, list) and value:
        return [value[0] for _ in range(repeats)]
    if isinstance(value, tuple) and value:
        return tuple(value[0] for _ in range(repeats))
    return value


def _repeat_policy_inputs(
    inputs: Mapping[str, Any],
    repeats: int,
    *,
    skip_state: bool,
) -> dict[str, Any]:
    output = {}
    for key, value in inputs.items():
        if skip_state and key == "state":
            continue
        output[key] = _repeat_first_batch(value, repeats, str(key), inputs)
    return output


def _state_tensor_like(reference: Any, values: np.ndarray, device: torch.device):
    tensor = torch.from_numpy(values)
    if reference is None:
        return tensor.to(device=device)
    if reference.ndim == 3 and tensor.ndim == 2:
        tensor = tensor[:, None, :]
    elif reference.ndim == 2 and tensor.ndim == 3 and tensor.shape[1] == 1:
        tensor = tensor[:, 0, :]
    return tensor.to(device=reference.device, dtype=reference.dtype)


def _identity_hip_pose() -> np.ndarray:
    hip = np.zeros(9, dtype=np.float32)
    hip[3] = 1.0
    hip[7] = 1.0
    return hip


def _first_sequence(value: Any, last_dim: int) -> np.ndarray:
    array = _to_numpy(value)
    if array is None:
        return np.zeros((0, last_dim), dtype=np.float32)
    while array.ndim > 2:
        array = array[0]
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return array.astype(np.float32, copy=False)


def _mse(prediction: np.ndarray, target: np.ndarray) -> float:
    steps = min(prediction.shape[-2], target.shape[-2])
    dims = min(prediction.shape[-1], target.shape[-1])
    if steps <= 0 or dims <= 0:
        return float("nan")
    difference = prediction[..., :steps, :dims] - target[..., :steps, :dims]
    return float(np.mean(difference * difference))


def decode_component_major(
    processor,
    action_pred: "torch.Tensor",
    embodiment,
    prepared: "PreparedRollout",
    groups: tuple[str, ...],
) -> np.ndarray:
    """Decode a normalized prediction into a component-major action array.

    Shared by every model whose action carries more than the 62-D control
    vector, so the per-chunk unapply loop exists once.
    """
    chunks = []
    for chunk_index in range(min(prepared.chunk_count, action_pred.shape[0])):
        normalized = (
            action_pred[chunk_index : chunk_index + 1].detach().float().cpu().numpy()
        )
        action_dict = processor.unapply(
            normalized,
            embodiment,
            state=_state_dict_from_abs(
                prepared.state_abs[chunk_index : chunk_index + 1]
            ),
        )
        component_major = np.concatenate(
            [
                np.asarray(action_dict[f"action.{key}"], dtype=np.float32)
                for key in groups
            ],
            axis=-1,
        )
        if component_major.ndim == 3:
            component_major = component_major[0]
        chunks.append(component_major)
    if not chunks:
        raise ValueError("component-major decode produced no chunks")
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


class OpenLoopEvalCallback(TrainerCallback):
    def __init__(self, trainer, config: OpenLoopEvalConfig) -> None:
        self.trainer = trainer
        self.config = config
        self._inputs = None
        self._model = None
        # Trainer callbacks can be dispatched more than once for the same
        # state during wrapper composition. Never repeat an expensive rollout
        # or upload duplicate media for an already-processed optimizer step.
        self._last_rollout_step = -1

    def is_due(self, global_step: int) -> bool:
        return global_step % self.config.every_n_steps == 0 or (
            self.config.first_step and global_step == 1
        )

    def capture(self, model, inputs) -> None:
        self._model = model
        batch = inputs.get("inputs", inputs) if isinstance(inputs, Mapping) else inputs
        if not isinstance(batch, Mapping):
            raise TypeError("open-loop evaluation requires a mapping training batch")
        next_step = int(self.trainer.state.global_step) + 1
        generator = np.random.default_rng(self.config.sample_seed + next_step)
        sample_index = int(generator.integers(0, _batch_size(batch)))
        self._inputs = _select_training_sample(batch, sample_index)

    @staticmethod
    def _prediction(outputs: Any) -> torch.Tensor:
        if isinstance(outputs, Mapping):
            prediction = outputs.get("action_pred")
        else:
            prediction = getattr(outputs, "action_pred", None)
        if not torch.is_tensor(prediction):
            raise TypeError("model.get_action() must return tensor action_pred")
        return prediction

    def prediction_for_unapply(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape[-1] < 62:
            raise ValueError(f"N1.7 action must end in at least 62, got {action.shape}")
        return action[..., :62]

    def control_norm(self, action: torch.Tensor) -> torch.Tensor:
        return self.prediction_for_unapply(action)

    def extra_rollout_metrics(
        self,
        action_pred: torch.Tensor,
        batch: Mapping[str, Any],
    ) -> Mapping[str, torch.Tensor | float]:
        del action_pred, batch
        return {}

    def extra_visual_media(self, prepared: PreparedRollout) -> Mapping[str, Any]:
        del prepared
        return {}

    def capture_rollout_outputs(
        self,
        outputs: Any,
        model,
        batch: Mapping[str, Any],
        prepared: PreparedRollout | None,
    ) -> None:
        """Optional variant hook for auxiliary predictions and media."""
        del outputs, model, batch, prepared

    def rollout_outputs(self, model, prepared: PreparedRollout) -> Any:
        """Run policy inference for a prepared rollout.

        The default keeps the historical batched, independent-chunk behavior.
        Stateful policies (for example PACE V4 RTC) override this narrow hook
        to generate chunks sequentially without duplicating callback logging,
        decoding, rendering, or artifact handling.
        """
        return model.get_action(prepared.policy_inputs)

    def rollout_metric_step_mask(
        self,
        *,
        chunk_count: int,
        horizon: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return which generated steps participate in scalar metrics."""
        return torch.ones(
            (chunk_count, horizon), dtype=torch.bool, device=device
        )

    def extra_decoded_rollout_metrics(
        self,
        prediction_action_for_render: np.ndarray | None,
        prepared: PreparedRollout,
        metric_step_mask: np.ndarray,
    ) -> Mapping[str, float]:
        """Optional metrics that require actions decoded to physical units."""
        del prediction_action_for_render, prepared, metric_step_mask
        return {}

    def target_control_norm(
        self,
        batch: Mapping[str, Any],
        processor,
        target_control_abs: np.ndarray,
        state_abs: np.ndarray,
        tag_value: str,
        chunk_count: int,
        horizon: int,
    ) -> np.ndarray:
        del batch
        chunks = []
        for chunk_index in range(chunk_count):
            start = chunk_index * horizon
            end = start + horizon
            chunks.append(
                self._normalize_abs_control(
                    processor,
                    target_control_abs[start:end],
                    state_abs[chunk_index : chunk_index + 1],
                    tag_value,
                )
            )
        return np.concatenate(chunks, axis=0)

    def _policy_inputs(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        output = dict(batch)
        for key in ("action", "action_mask", "action_physical_scale"):
            output.pop(key, None)
        for key in list(output):
            if str(key).startswith("viz_"):
                output.pop(key, None)
        return output

    @staticmethod
    def _processor(trainer):
        processor = getattr(trainer, "processor", None)
        if processor is None:
            processor = getattr(
                getattr(trainer, "train_dataset", None), "processor", None
            )
        if processor is None or not hasattr(processor, "decode_action"):
            raise AttributeError(
                "train_dataset.processor.decode_action is required"
            )
        return processor

    def _normalize_state(
        self,
        processor,
        state_abs: np.ndarray,
        tag_value: str,
    ) -> np.ndarray:
        normalized = processor.state_action_processor.apply_state(
            _split_control(state_abs), tag_value
        )
        values = _concat_control(normalized)
        max_state_dim = int(getattr(processor, "max_state_dim", values.shape[-1]))
        if values.shape[-1] < max_state_dim:
            padding = np.zeros(
                (*values.shape[:-1], max_state_dim - values.shape[-1]),
                dtype=np.float32,
            )
            values = np.concatenate((values, padding), axis=-1)
        return values

    def _normalize_abs_control(
        self,
        processor,
        action_abs: np.ndarray,
        state_abs: np.ndarray,
        tag_value: str,
    ) -> np.ndarray:
        state = state_abs[0] if state_abs.ndim == 3 and state_abs.shape[0] == 1 else state_abs
        normalized = processor.state_action_processor.apply_action(
            _split_control(action_abs),
            tag_value,
            state=_split_control(state),
        )
        return _concat_control(normalized)

    def _unapply_control(
        self,
        processor,
        normalized_action: torch.Tensor,
        embodiment,
        state_abs: np.ndarray,
    ) -> np.ndarray:
        # Use the exact public decode path used by Gr00tPolicy at deployment.
        # This keeps online W&B prediction videos and rollout inference on one
        # action-decoding contract.
        action = normalized_action.detach().float().cpu().numpy()
        action_dict = processor.decode_action(
            action,
            embodiment,
            state=_split_control(state_abs),
        )
        return _concat_control(action_dict)

    def _prediction_action_for_render(
        self,
        processor,
        action_pred: torch.Tensor,
        embodiment,
        prepared: PreparedRollout,
        prediction_control_abs: np.ndarray,
    ) -> np.ndarray | None:
        """The component-major prediction, or None when there is no such layout.

        Models that predict nothing beyond the 62-D control (groot_n17, t_rex)
        have no component-major array at all, and the renderer must then read
        the control itself; the subclasses that do predict a compliance
        residual or a torque override this and return their own layout.
        """
        del processor, action_pred, embodiment, prepared, prediction_control_abs
        return None

    def _prepare_sidechannel_rollout(
        self,
        batch: Mapping[str, Any],
        processor,
        tag_value: str,
        model_device: torch.device,
    ) -> PreparedRollout | None:
        state_value = batch.get("viz_rollout_state_abs_62d")
        target_value = batch.get("viz_rollout_gt_abs_action_62d")
        if state_value is None or target_value is None:
            return None
        state_abs = _to_numpy(state_value)[0]
        target_abs = _to_numpy(target_value)[0]
        chunk_count = int(state_abs.shape[0])
        valid_value = _to_numpy(batch.get("viz_rollout_valid_chunks"))
        if valid_value is not None:
            chunk_count = min(
                chunk_count,
                max(1, int(np.asarray(valid_value).reshape(-1)[0])),
            )
        state_abs = state_abs[:chunk_count]
        if chunk_count <= 0:
            return None
        horizon = min(
            self.config.action_horizon,
            max(1, int(target_abs.shape[0] // chunk_count)),
        )
        target_abs = target_abs[: chunk_count * horizon, :62]
        state_norm = self._normalize_state(processor, state_abs, tag_value)
        base_policy_inputs = self._policy_inputs(batch)
        policy_inputs = _repeat_policy_inputs(
            base_policy_inputs, chunk_count, skip_state=True
        )
        policy_inputs["state"] = _state_tensor_like(
            base_policy_inputs.get("state"), state_norm, model_device
        )
        target_norm = np.asarray(
            self.target_control_norm(
                batch,
                processor,
                target_abs,
                state_abs,
                tag_value,
                chunk_count,
                horizon,
            ),
            dtype=np.float32,
        )
        # Render side channels use one model-agnostic naming scheme. The
        # component-major array is 150-D for models with a torque head and
        # 106-D for models with only a compliance residual; render_delta_q_slice
        # tells the renderer where delta_q sits in either case.
        side_channels = {
            name: _to_numpy(batch.get(key))
            for name, key in (
                (
                    "target_action_for_render",
                    "viz_rollout_gt_component_major_action",
                ),
                ("tactile_wrench_for_render", "viz_rollout_tactile_wrench"),
                (
                    "tactile_wrench_valid_for_render",
                    "viz_rollout_tactile_wrench_valid",
                ),
                (
                    "tactile_wrench_pred_for_render",
                    "viz_rollout_tactile_wrench_pred",
                ),
                (
                    "tactile_wrench_pred_valid_for_render",
                    "viz_rollout_tactile_wrench_pred_valid",
                ),
            )
        }
        for name, value in side_channels.items():
            if value is None:
                continue
            if value.ndim < 2 or value.shape[0] == 0:
                raise ValueError(f"{name} must contain a batch dimension")
            side_channels[name] = value[0]
        if (
            self.render_delta_q_slice is not None
            and side_channels["target_action_for_render"] is None
        ):
            raise KeyError(
                f"{type(self).__name__} renders the compliance layer and "
                "therefore requires the "
                "viz_rollout_gt_component_major_action side channel"
            )
        if (
            side_channels["tactile_wrench_for_render"] is not None
            and side_channels["tactile_wrench_valid_for_render"] is None
        ):
            raise KeyError(
                "viz_rollout_tactile_wrench requires "
                "viz_rollout_tactile_wrench_valid"
            )
        if (
            side_channels["tactile_wrench_pred_for_render"] is not None
            and side_channels["tactile_wrench_for_render"] is None
        ):
            raise KeyError(
                "viz_rollout_tactile_wrench_pred requires the observed "
                "viz_rollout_tactile_wrench to compare against"
            )
        expected_target_shape = (chunk_count * horizon, 62)
        if target_norm.shape != expected_target_shape:
            raise ValueError(
                "normalized rollout target must have shape "
                f"{expected_target_shape}, got {target_norm.shape}"
            )
        return PreparedRollout(
            policy_inputs=policy_inputs,
            state_abs=state_abs,
            target_control_abs=target_abs,
            target_control_norm=target_norm,
            hip_pose=batch.get("viz_rollout_hip_pose_9d"),
            input_image=batch.get("viz_input_image"),
            chunk_count=chunk_count,
            horizon=horizon,
            **side_channels,
        )

    def _prepare_rollout(
        self,
        batch: Mapping[str, Any],
        processor,
        tag_value: str,
        model_device: torch.device,
    ) -> PreparedRollout:
        prepared = self._prepare_sidechannel_rollout(
            batch, processor, tag_value, model_device
        )
        if prepared is None:
            raise KeyError(
                "missing viz_rollout_state_abs_62d or "
                "viz_rollout_gt_abs_action_62d"
            )
        return prepared

    def _base_metrics(
        self,
        prediction_norm: np.ndarray,
        target_norm: np.ndarray,
        prediction_abs: np.ndarray,
        target_abs: np.ndarray,
    ) -> dict[str, float]:
        steps = min(
            prediction_norm.shape[0],
            target_norm.shape[0],
            prediction_abs.shape[0],
            target_abs.shape[0],
        )
        if steps <= 0:
            return {}
        prediction_norm = prediction_norm[:steps, :62]
        target_norm = target_norm[:steps, :62]
        # Wrist and hand joints live on very different scales, so they are
        # reported separately and never pooled into a single scalar.
        return {
            "open-loop-eval/action_mse_wrist_18d": _mse(
                prediction_norm[None, :, :18], target_norm[None, :, :18]
            ),
            "open-loop-eval/action_mse_hand_joint_44d": _mse(
                prediction_norm[None, :, 18:], target_norm[None, :, 18:]
            ),
        }

    # Where delta_q sits inside the component-major render arrays, for models
    # that predict a compliance residual. None means the model has no residual,
    # so the compliance layer stays off.
    render_delta_q_slice: slice | None = None
    component_major_groups: tuple[str, ...] | None = None

    def _render_payload(
        self,
        prepared: PreparedRollout,
        prediction_abs: np.ndarray,
        prediction_action_for_render: np.ndarray | None,
        render_steps: int,
    ) -> dict[str, np.ndarray]:
        """Build the layer arrays for the paired-motion renderer.

        Which layers appear is decided here by what the model actually produced,
        never by which model it is: delta_q turns on the compliance overlay and
        sliders, wrench turns on the force/torque gauges, and a predicted wrench
        turns those gauges into a GT-versus-prediction comparison.
        """
        gt_component_major = prepared.target_action_for_render
        pred_component_major = prediction_action_for_render
        if (gt_component_major is None) != (pred_component_major is None):
            raise ValueError(
                "component-major GT and prediction actions must both be "
                "present or both absent"
            )
        if gt_component_major is None:
            gt_action = prepared.target_control_abs[:render_steps, :62]
            pred_action = prediction_abs[:render_steps, :62]
            delta_q: tuple[np.ndarray, np.ndarray] | None = None
        else:
            gt_action = gt_component_major[:render_steps, :62]
            pred_action = pred_component_major[:render_steps, :62]
            if self.render_delta_q_slice is None:
                raise ValueError(
                    f"{type(self).__name__} renders component-major actions but "
                    "does not declare render_delta_q_slice"
                )
            delta_q = (
                gt_component_major[:render_steps, self.render_delta_q_slice],
                pred_component_major[:render_steps, self.render_delta_q_slice],
            )

        payload: dict[str, np.ndarray] = {
            "action_62d_gt": np.asarray(gt_action, dtype=np.float32),
            "action_62d_pred": np.asarray(pred_action, dtype=np.float32),
            "chunk_size": np.asarray([prepared.horizon], dtype=np.int32),
        }
        hip_pose = _first_sequence(prepared.hip_pose, 9)
        if hip_pose.shape[0] == 0:
            hip_pose = _identity_hip_pose()[None]
        payload["hip_pose_9d"] = hip_pose[: prepared.chunk_count]
        if delta_q is not None:
            payload["delta_q_44d_gt"] = np.asarray(delta_q[0], dtype=np.float32)
            payload["delta_q_44d_pred"] = np.asarray(delta_q[1], dtype=np.float32)
        if prepared.tactile_wrench_for_render is not None:
            if prepared.tactile_wrench_valid_for_render is None:
                raise ValueError("wrench rendering requires its validity mask")
            payload["wrench_gt"] = np.asarray(
                prepared.tactile_wrench_for_render[:render_steps], dtype=np.float32
            )
            payload["wrench_valid_gt"] = np.asarray(
                prepared.tactile_wrench_valid_for_render[:render_steps], dtype=bool
            )
            if prepared.tactile_wrench_pred_for_render is not None:
                if prepared.tactile_wrench_pred_valid_for_render is None:
                    raise ValueError(
                        "predicted wrench rendering requires its validity mask"
                    )
                payload["wrench_pred"] = np.asarray(
                    prepared.tactile_wrench_pred_for_render[:render_steps],
                    dtype=np.float32,
                )
                payload["wrench_valid_pred"] = np.asarray(
                    prepared.tactile_wrench_pred_valid_for_render[:render_steps],
                    dtype=bool,
                )
        return payload

    def _render_and_log(
        self,
        state,
        prepared: PreparedRollout,
        prediction_abs: np.ndarray,
        metrics: dict[str, float],
        prediction_action_for_render: np.ndarray | None = None,
    ) -> None:
        import wandb

        if wandb.run is None:
            return
        candidate_lengths = [
            prepared.chunk_count * prepared.horizon,
            prediction_abs.shape[0],
            prepared.target_control_abs.shape[0],
        ]
        if prepared.target_action_for_render is not None:
            candidate_lengths.append(prepared.target_action_for_render.shape[0])
        if prediction_action_for_render is not None:
            candidate_lengths.append(prediction_action_for_render.shape[0])
        render_steps = min(candidate_lengths)

        media: dict[str, Any] = dict(metrics)
        if prepared.input_image is not None:
            image = _to_numpy(prepared.input_image)
            while image.ndim > 3:
                image = image[0]
            if image.ndim == 3:
                if image.shape[0] in {1, 3} and image.shape[-1] not in {1, 3}:
                    image = np.transpose(image, (1, 2, 0))
                media["open-loop-vis/input_image"] = wandb.Image(image)

        if render_steps <= 0:
            print(
                f"[{self.config.log_name}] no renderable steps; "
                "logging input image and scalars only",
                flush=True,
            )
            media.update(self.extra_visual_media(prepared))
            self._wandb_log(state, media)
            return

        with tempfile.TemporaryDirectory(
            prefix=f"open_loop_eval_step{state.global_step:06d}_"
        ) as tmpdir:
            render_input = Path(tmpdir) / "motion_pair.npz"
            np.savez_compressed(
                render_input,
                **self._render_payload(
                    prepared,
                    prediction_abs,
                    prediction_action_for_render,
                    render_steps,
                ),
            )
            renderer_python = os.environ.get(
                "DREAMZERO_RENDER_PYTHON",
                sys.executable,
            )
            render_env = os.environ.copy()
            render_env["PYTHONPATH"] = (
                f"{self.config.dreamzero_root}:" + render_env.get("PYTHONPATH", "")
            )
            completed = subprocess.run(
                [
                    renderer_python,
                    "-m",
                    "dexterity.rendering.cli.paired_motion_video",
                    "--input-npz",
                    str(render_input),
                    "--out-dir",
                    tmpdir,
                    "--fps",
                    "15",
                ],
                cwd=self.config.dreamzero_root,
                env=render_env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if completed.stdout:
                print(completed.stdout.rstrip())
            if completed.returncode != 0:
                print(
                    f"[{self.config.log_name}] renderer failed with code "
                    f"{completed.returncode}; logging scalars only",
                    flush=True,
                )
            _persist_rollout_artifacts(
                state,
                prepared,
                metrics,
                tmpdir,
                {
                    "rollout_input.npz": "motion_pair.npz",
                    "gt_hand_motion.mp4": "viz__gt_hand_motion.mp4",
                    "pred_hand_motion.mp4": "viz__pred_hand_motion.mp4",
                    "render_report.json": "render_report.json",
                },
            )
            for key, filename in (
                ("open-loop-vis/gt_hand_motion", "viz__gt_hand_motion.mp4"),
                ("open-loop-vis/pred_hand_motion", "viz__pred_hand_motion.mp4"),
            ):
                path = Path(tmpdir) / filename
                if path.exists():
                    media[key] = wandb.Video(str(path), format="mp4")
            media.update(self.extra_visual_media(prepared))
            self._wandb_log(state, media)

    def _wandb_log(self, state, payload: Mapping[str, Any]) -> None:
        import wandb

        if wandb.run is None or not payload:
            return
        run_step = int(getattr(wandb.run, "step", 0) or 0)
        logged = {"train/global_step": int(state.global_step), **dict(payload)}
        wandb.log(
            logged,
            step=max(int(state.global_step), run_step),
            commit=False,
        )
        print(
            f"[{self.config.log_name}] step {state.global_step} logged "
            f"{len(logged)} media/scalars to wandb",
            flush=True,
        )

    def on_step_end(self, args, state, control, **kwargs):
        del args, kwargs
        global_step = int(state.global_step)
        if (
            not self.trainer.is_world_process_zero()
            or not self.is_due(global_step)
            or self._last_rollout_step == global_step
        ):
            return control
        self._last_rollout_step = global_step
        captured_inputs, model = self._inputs, self._model
        self._inputs = None
        self._model = None
        if captured_inputs is None or model is None or not hasattr(model, "get_action"):
            return control
        batch = captured_inputs.get("inputs", captured_inputs)
        if not isinstance(batch, Mapping) or "action" not in batch:
            return control
        was_training = bool(model.training)
        cpu_rng_state = torch.random.get_rng_state()
        model_device = next(iter(model.parameters())).device
        cuda_rng_state = None
        if model_device.type == "cuda":
            cuda_rng_state = torch.cuda.get_rng_state(model_device)
        try:
            from gr00t.data.embodiment_tags import EmbodimentTag

            processor = self._processor(self.trainer)
            embodiment = EmbodimentTag.resolve(self.config.embodiment_tag)
            # A missing rollout side channel is a dataset wiring bug, not a
            # reason to quietly downgrade to scalars-only: let it raise.
            prepared = self._prepare_rollout(
                batch,
                processor,
                embodiment.value,
                model_device,
            )
            model.eval()
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if model_device.type == "cuda"
                else nullcontext()
            )
            with torch.no_grad(), autocast_context:
                outputs = self.rollout_outputs(model, prepared)
            self.capture_rollout_outputs(outputs, model, batch, prepared)
            action_pred = self._prediction(outputs)
            batch_size = int(action_pred.shape[0])
            horizon = min(self.config.action_horizon, int(action_pred.shape[1]))
            if batch_size <= 0 or horizon <= 0:
                return control
            action_pred = action_pred[:, :horizon]
            metrics: dict[str, float] = {}
            for name, value in self.extra_rollout_metrics(action_pred, batch).items():
                tensor = torch.as_tensor(value).detach().float()
                if tensor.numel() != 1 or not torch.isfinite(tensor).all():
                    raise ValueError(f"rollout metric {name} must be one finite scalar")
                metrics[name] = float(tensor.cpu().item())
            chunk_count = min(prepared.chunk_count, batch_size)
            horizon = min(prepared.horizon, horizon)
            action_pred = action_pred[:chunk_count, :horizon]
            metric_step_mask = self.rollout_metric_step_mask(
                chunk_count=chunk_count,
                horizon=horizon,
                device=action_pred.device,
            )
            if metric_step_mask.shape != (chunk_count, horizon):
                raise ValueError(
                    "rollout metric mask must have shape "
                    f"{(chunk_count, horizon)}, got "
                    f"{tuple(metric_step_mask.shape)}"
                )
            metric_step_mask_flat = (
                metric_step_mask.detach().cpu().numpy().reshape(-1)
            )
            prediction_norm = (
                self.control_norm(action_pred)
                .detach()
                .float()
                .cpu()
                .numpy()
                .reshape(-1, 62)
            )[metric_step_mask_flat]
            prediction_abs_chunks = []
            for chunk_index in range(chunk_count):
                prediction_abs = self._unapply_control(
                    processor,
                    action_pred[chunk_index : chunk_index + 1],
                    embodiment,
                    prepared.state_abs[chunk_index : chunk_index + 1],
                )
                prediction_abs_chunks.append(prediction_abs[0])
            prediction_abs = np.concatenate(prediction_abs_chunks, axis=0)
            prediction_abs_for_render = prediction_abs
            target_norm_aligned = prepared.target_control_norm[
                : chunk_count * prepared.horizon
            ].reshape(chunk_count, prepared.horizon, 62)[:, :horizon].reshape(-1, 62)
            target_abs_aligned = prepared.target_control_abs[
                : chunk_count * prepared.horizon
            ].reshape(chunk_count, prepared.horizon, 62)[:, :horizon].reshape(-1, 62)
            target_norm_for_metrics = target_norm_aligned[metric_step_mask_flat]
            target_abs_for_metrics = target_abs_aligned[metric_step_mask_flat]
            prediction_abs_for_metrics = prediction_abs[
                : chunk_count * horizon
            ][metric_step_mask_flat]
            metrics.update(
                self._base_metrics(
                    prediction_norm,
                    target_norm_for_metrics,
                    prediction_abs_for_metrics,
                    target_abs_for_metrics,
                )
            )
            prediction_action_for_render = self._prediction_action_for_render(
                processor,
                action_pred,
                embodiment,
                prepared,
                prediction_abs_for_render,
            )
            decoded_metrics = self.extra_decoded_rollout_metrics(
                prediction_action_for_render,
                prepared,
                metric_step_mask_flat,
            )
            for name, value in decoded_metrics.items():
                scalar = np.asarray(value, dtype=np.float64)
                if scalar.size != 1 or not np.isfinite(scalar).all():
                    raise ValueError(
                        f"decoded rollout metric {name} must be one finite scalar"
                    )
                metrics[name] = float(scalar.reshape(-1)[0])
            self._render_and_log(
                state,
                prepared,
                prediction_abs_for_render,
                metrics,
                prediction_action_for_render=prediction_action_for_render,
            )
        except Exception as exc:
            print(
                f"[{self.config.log_name}] failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
        finally:
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, model_device)
            model.train(was_training)
        return control


def build_open_loop_eval_trainer(
    trainer_class,
    callback_factory: Callable[[Any], OpenLoopEvalCallback],
    *,
    callback_attribute: str = "_open_loop_eval_callback",
):
    """Return an explicit Trainer subclass that receives the current batch.

    Hugging Face callback events do not expose the training inputs. The local
    subclass is therefore the narrow integration point: it captures the
    already-consumed batch and leaves scheduling, inference, metrics, media,
    and W&B logging in the callback. No global Trainer method is monkey-patched.
    """

    class TrainerWithOpenLoopEval(trainer_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            callback = callback_factory(self)
            setattr(self, callback_attribute, callback)
            self.add_callback(callback)
            print(
                f"[{callback.config.log_name}] callback registered "
                f"every={callback.config.every_n_steps} "
                f"first_step={callback.config.first_step}",
                flush=True,
            )

        def training_step(self, model, inputs, *args, **kwargs):
            callback = getattr(self, callback_attribute)
            next_global_step = int(self.state.global_step) + 1
            if self.is_world_process_zero() and callback.is_due(next_global_step):
                try:
                    callback.capture(self.accelerator.unwrap_model(model), inputs)
                except Exception as exc:
                    print(
                        f"[{callback.config.log_name}] capture failed: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
            return super().training_step(model, inputs, *args, **kwargs)

    TrainerWithOpenLoopEval.__name__ = f"{trainer_class.__name__}WithOpenLoopEval"
    TrainerWithOpenLoopEval.__qualname__ = TrainerWithOpenLoopEval.__name__
    return TrainerWithOpenLoopEval
