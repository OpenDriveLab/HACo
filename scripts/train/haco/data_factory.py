"""Strict train-only dataset factory for HACO.

The low-level sensor storage and streaming primitives are shared utilities,
while every action and sensor semantic is resolved by the HACO contract.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from dexterity.data.posttrain import action_indices
from scripts.train.pace.data_factory import (
    PaceDatasetFactory,
    PaceShardedSingleStepDataset,
    PaceStreamingMixtureDataset,
    _EmbodimentTag,
    _MessageType,
    _extract_step_data,
)
from scripts.train.haco.config import HacoExperiment, get_experiment


STATE_GROUPS = (
    "left_wrist_eef",
    "right_wrist_eef",
    "left_hand_joints",
    "right_hand_joints",
)
WRIST_GROUPS = ("left_wrist_eef", "right_wrist_eef")
Q_COMPLIANCE_GROUPS = ("left_hand_q_teleop", "right_hand_q_teleop")
Q_NOMINAL_GROUPS = ("left_hand_joints", "right_hand_joints")
DELTA_Q_GROUPS = ("left_hand_delta_q", "right_hand_delta_q")


def action_groups(experiment: HacoExperiment) -> tuple[str, ...]:
    if experiment.action_contract == "joint_compliance_delta":
        return (*WRIST_GROUPS, *Q_COMPLIANCE_GROUPS, *DELTA_Q_GROUPS)
    if experiment.action_contract == "compliance_only":
        return (*WRIST_GROUPS, *Q_COMPLIANCE_GROUPS)
    if experiment.action_contract == "nominal_only":
        return (*WRIST_GROUPS, *Q_NOMINAL_GROUPS)
    raise AssertionError(experiment.action_contract)


def control_groups(experiment: HacoExperiment) -> tuple[str, ...]:
    if experiment.action_target == "q_compliance":
        return (*WRIST_GROUPS, *Q_COMPLIANCE_GROUPS)
    return (*WRIST_GROUPS, *Q_NOMINAL_GROUPS)


class HacoShardedSingleStepDataset(PaceShardedSingleStepDataset):
    """HACO streaming sample with explicit action and eval sidechannels."""

    def __init__(
        self,
        dataset_path: str | Path,
        *args: Any,
        experiment_id: str,
        sensor_stats_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        self.experiment = get_experiment(experiment_id)
        self._loader_index_by_episode_index: dict[int, int] | None = None
        super().__init__(dataset_path, *args, **kwargs)
        if sensor_stats_path is not None:
            stats = json.loads(
                Path(sensor_stats_path).read_text(encoding="utf-8")
            )
            if stats.get("split") != "train":
                raise ValueError("HACO sensor normalization must be train-only")
            self.pace_store.stats = stats
        self.haco_store = self.pace_store

    def _episode_index_mapping(self) -> dict[int, int]:
        if self._loader_index_by_episode_index is None:
            mapping: dict[int, int] = {}
            for loader_index, item in enumerate(
                self.episode_loader.episodes_metadata
            ):
                episode_index = int(item["episode_index"])
                if episode_index in mapping:
                    raise ValueError(
                        f"duplicate canonical episode_index={episode_index}"
                    )
                mapping[episode_index] = loader_index
            self._loader_index_by_episode_index = mapping
        return self._loader_index_by_episode_index

    def get_datapoint_by_episode_anchor(
        self, episode_index: int, anchor: int
    ) -> dict[str, Any]:
        """Build one deterministic sample for a manifest (episode, anchor).

        This is the stable entry point used by the formal open-loop runner.  It
        bypasses shard order but preserves the exact same processor path as
        training.
        """

        mapping = self._episode_index_mapping()
        if int(episode_index) not in mapping:
            raise KeyError(f"unknown episode_index={episode_index}")
        loader_index = mapping[int(episode_index)]
        if not self._exact_anchors_by_loader_index:
            self._load_and_validate_anchor_contract()
        valid = self._exact_anchors_by_loader_index[loader_index]
        if int(anchor) not in set(valid.tolist()):
            raise ValueError(
                f"anchor {anchor} is not a strict training anchor for episode "
                f"{episode_index}"
            )
        episode_data = self.episode_loader[loader_index]
        return self.get_datapoint(
            episode_data,
            int(anchor),
            episode_index=int(episode_index),
        )

    def get_datapoint(self, episode_data, step_index: int, *, episode_index: int):
        if self.processor is None:
            raise AssertionError("Processor must be set before HACO datapoints")
        vla_step_data = _extract_step_data(
            episode_data,
            int(step_index),
            self.modality_configs,
            self.embodiment_tag,
            allow_padding=False,
        )
        if self.experiment.sensor_encoder_mode == "none":
            sensor_metadata = {}
        else:
            sensor_metadata = self.haco_store.metadata(
                int(episode_index), int(step_index)
            )
        # Rebuild the validity mask from the selected HACO semantic contract;
        # delta-q never enters a delta-free sample.
        sensor_metadata["action_component_valid"] = np.ones(
            (40, self.experiment.semantic_action_dim), dtype=bool
        )
        vla_step_data.metadata["haco"] = sensor_metadata
        raw_state = np.concatenate(
            tuple(vla_step_data.states[key] for key in STATE_GROUPS), axis=-1
        ).astype(np.float32)
        active_groups = action_groups(self.experiment)
        execution_groups = control_groups(self.experiment)
        raw_model_action = np.concatenate(
            tuple(vla_step_data.actions[key] for key in active_groups), axis=-1
        ).astype(np.float32)
        raw_control_action = np.concatenate(
            tuple(vla_step_data.actions[key] for key in execution_groups), axis=-1
        ).astype(np.float32)
        if raw_state.shape != (1, 62):
            raise ValueError(
                f"HACO rollout state must have shape (1, 62), got {raw_state.shape}"
            )
        if raw_control_action.shape != (40, 62):
            raise ValueError(
                "HACO executed target must have shape (40, 62), got "
                f"{raw_control_action.shape}"
            )
        expected_model_shape = (40, self.experiment.semantic_action_dim)
        if raw_model_action.shape != expected_model_shape:
            raise ValueError(
                f"HACO semantic action must have shape {expected_model_shape}, "
                f"got {raw_model_action.shape}"
            )

        messages = [{"type": _MessageType.EPISODE_STEP.value, "content": vla_step_data}]
        transformed = self.processor(messages)
        transformed["viz_rollout_state_abs_62d"] = raw_state
        transformed["viz_rollout_gt_abs_action_62d"] = raw_control_action
        transformed["viz_rollout_gt_haco_action"] = raw_model_action
        transformed["viz_rollout_episode_index"] = np.asarray(
            int(episode_index), dtype=np.int64
        )
        transformed["viz_rollout_anchor_index"] = np.asarray(
            int(step_index), dtype=np.int64
        )
        if self.experiment.sensor_encoder_mode != "none":
            sidecar = self.haco_store.episode(int(episode_index))
            future = action_indices(int(step_index), sidecar.length)
            transformed["viz_rollout_tactile_wrench"] = sidecar.tactile_wrench[
                future
            ].copy()
            transformed["viz_rollout_tactile_wrench_valid"] = (
                sidecar.tactile_wrench_valid[future].copy()
            )
        transformed["viz_rollout_hip_pose_9d"] = np.asarray(
            [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        transformed["viz_rollout_valid_chunks"] = np.asarray(1, dtype=np.int32)
        vlm_content = transformed.get("vlm_content")
        if isinstance(vlm_content, dict) and vlm_content.get("images"):
            transformed["viz_input_image"] = np.asarray(
                vlm_content["images"][0], dtype=np.uint8
            )
        return transformed


class HacoDatasetFactory(PaceDatasetFactory):
    """Build a streaming train-only mixture for one HACO experiment."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.experiment = get_experiment(config.model.experiment_id)

    def build(self, processor):
        if self.config.training.eval_strategy != "no":
            raise ValueError("HACO requires eval_strategy='no'")
        all_datasets = []
        all_weights = []
        normalization_value = os.environ.get("HACO_MULTITASK_NORMALIZATION_DIR")
        normalization_dir = (
            Path(normalization_value).expanduser().resolve()
            if normalization_value
            else None
        )
        sensor_stats_path = (
            normalization_dir / "sensor_stats.json"
            if normalization_dir is not None
            else None
        )
        for dataset_spec in self.config.data.datasets:
            datasets = []
            for dataset_path in dataset_spec.dataset_paths:
                embodiment_tag = dataset_spec.embodiment_tag
                if embodiment_tag is None:
                    raise ValueError("HACO embodiment tag is required")
                root = Path(dataset_path)
                required = (
                    root / "meta/stats.json",
                    root / "meta/sensor_stats.json",
                    root / "meta/stats_provenance.json",
                )
                missing = [str(path) for path in required if not path.is_file()]
                if missing:
                    raise FileNotFoundError(
                        "HACO requires precomputed train-only statistics: "
                        + ", ".join(missing)
                    )
                provenance = json.loads(required[2].read_text(encoding="utf-8"))
                sensor_stats = json.loads(required[1].read_text(encoding="utf-8"))
                if (
                    provenance.get("split") != "train"
                    or sensor_stats.get("split") != "train"
                ):
                    raise ValueError("HACO statistics must be train-only")
                datasets.append(
                    HacoShardedSingleStepDataset(
                        dataset_path=dataset_path,
                        embodiment_tag=_EmbodimentTag(embodiment_tag),
                        modality_configs=self.config.data.modality_configs[
                            embodiment_tag
                        ],
                        video_backend=self.config.data.video_backend,
                        shard_size=self.config.data.shard_size,
                        episode_sampling_rate=self.config.data.episode_sampling_rate,
                        seed=self.config.data.seed,
                        allow_padding=False,
                        split="train",
                        experiment_id=self.experiment.experiment_id,
                        sensor_stats_path=sensor_stats_path,
                    )
                )
            lengths = np.asarray(
                [dataset.selected_anchor_count for dataset in datasets],
                dtype=np.float64,
            )
            for dataset, relative_length in zip(datasets, lengths / lengths.sum()):
                all_datasets.append(dataset)
                all_weights.append(float(relative_length * dataset_spec.mix_ratio))
        # HACO uses absolute wrist actions. Some older task datasets retain a
        # legacy single-step relative-action statistic while newer tasks store
        # horizon-aware [40, 9] values. GR00T tries to merge that unused field
        # before we can install the dedicated MT5 statistics, so remove it from
        # the process-local loader metadata only.
        for dataset in all_datasets:
            dataset.episode_loader.stats.pop("relative_action", None)
        train_dataset = PaceStreamingMixtureDataset(
            datasets=all_datasets,
            weights=all_weights,
            processor=processor,
            seed=self.config.data.seed,
            training=True,
            num_shards_per_epoch=self.config.data.num_shards_per_epoch,
            override_pretraining_statistics=(
                self.config.data.override_pretraining_statistics
            ),
        )
        if normalization_dir is not None:
            raw_stats = json.loads(
                (normalization_dir / "stats.json").read_text(encoding="utf-8")
            )
            reference = all_datasets[0]
            loader = reference.episode_loader
            original_stats = loader.stats
            try:
                loader.stats = raw_stats
                processor_stats = reference.get_dataset_statistics()
            finally:
                loader.stats = original_stats
            embodiment = self.config.data.datasets[0].embodiment_tag
            global_stats = {embodiment: processor_stats}
            processor.set_statistics(global_stats, override=True)
            for dataset in all_datasets:
                dataset.set_processor(processor)
            train_dataset.global_stats = global_stats
        return train_dataset, None


__all__ = [
    "HacoDatasetFactory",
    "HacoShardedSingleStepDataset",
    "action_groups",
    "control_groups",
]
