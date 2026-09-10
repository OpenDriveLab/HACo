from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from dexterity.data.posttrain import (
    action_indices,
    exact_anchor_indices,
    strict_anchor_indices,
)

from .dataset import PaceMetadataStore

ROLLOUT_CONTROL_GROUPS = (
    "left_wrist_eef",
    "right_wrist_eef",
    "left_hand_joints",
    "right_hand_joints",
)
ROLLOUT_PACE_ACTION_GROUPS = (
    *ROLLOUT_CONTROL_GROUPS,
    "left_hand_delta_q",
    "right_hand_delta_q",
)
ROLLOUT_COMMAND_GROUPS = (
    "left_wrist_eef",
    "right_wrist_eef",
    "left_hand_q_teleop",
    "right_hand_q_teleop",
)


def _reference_types():
    from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
    from gr00t.data.dataset.sharded_single_step_dataset import (
        ShardedSingleStepDataset,
        extract_step_data,
    )
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import MessageType

    return (
        ShardedMixtureDataset,
        ShardedSingleStepDataset,
        extract_step_data,
        EmbodimentTag,
        MessageType,
    )


try:
    (
        _ShardedMixtureDataset,
        _BaseDataset,
        _extract_step_data,
        _EmbodimentTag,
        _MessageType,
    ) = _reference_types()
except ImportError:
    _ShardedMixtureDataset = None
    _BaseDataset = object
    _extract_step_data = None
    _EmbodimentTag = None
    _MessageType = None


def _anchor_count(item: dict[str, Any]) -> int:
    if "valid_anchor_count" in item:
        return int(item["valid_anchor_count"])
    return int(item["quality"]["valid_anchor_count"])


class PaceShardedSingleStepDataset(_BaseDataset):
    def __init__(
        self,
        dataset_path: str | Path,
        *args: Any,
        split: str = "train",
        **kwargs: Any,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"unsupported PACE split {split!r}")
        self.pace_store = PaceMetadataStore(dataset_path)
        self.split = split
        self._exact_anchors_by_loader_index: dict[int, np.ndarray] = {}
        super().__init__(dataset_path=dataset_path, *args, **kwargs)

    def _load_and_validate_anchor_contract(self) -> None:
        split_counts = {name: 0 for name in ("train", "val", "test")}
        for loader_index, item in enumerate(self.episode_loader.episodes_metadata):
            episode_index = int(item["episode_index"])
            anchor_valid = self.pace_store.anchor_valid(episode_index)
            episode_length = self.episode_loader.get_episode_length(loader_index)
            if len(anchor_valid) != episode_length:
                raise ValueError(
                    f"episode {episode_index} sidecar/metadata length mismatch"
                )
            declared_anchors = exact_anchor_indices(anchor_valid)
            if declared_anchors.size != _anchor_count(item):
                raise ValueError(f"episode {episode_index} exact anchor count mismatch")
            # ``anchor_valid`` records validity of the converted source row.  Some
            # datasets (including UR/SharpA) deliberately mark the full timeline
            # valid, while PACE additionally needs nine history samples and a
            # strict 40-step future action chunk.  Intersect the two contracts so
            # extract_step_data(..., allow_padding=False) can never index beyond
            # an episode boundary.
            anchors = np.intersect1d(
                declared_anchors,
                strict_anchor_indices(episode_length),
                assume_unique=True,
            )
            self._exact_anchors_by_loader_index[loader_index] = anchors
            episode_split = str(item.get("split", ""))
            if episode_split not in split_counts:
                raise ValueError(
                    f"episode {episode_index} has bad split {episode_split!r}"
                )
            split_counts[episode_split] += int(anchors.size)
        if split_counts["train"] <= 0:
            raise ValueError(f"PACE requires non-empty train anchors: {split_counts}")
        self.full_split_anchor_counts = split_counts
        self.full_anchor_count = int(sum(split_counts.values()))

    def get_effective_episode_length(self, episode_index: int) -> int:
        if not self._exact_anchors_by_loader_index:
            self._load_and_validate_anchor_contract()
        return int(self._exact_anchors_by_loader_index[int(episode_index)].size)

    def shard_dataset(self) -> None:
        self._load_and_validate_anchor_contract()
        eligible = np.asarray(
            [
                loader_index
                for loader_index, item in enumerate(
                    self.episode_loader.episodes_metadata
                )
                if item.get("split") == self.split
            ],
            dtype=np.int64,
        )
        if eligible.size == 0:
            raise ValueError(f"PACE dataset has no episodes with split={self.split!r}")
        if not 0 < self.episode_sampling_rate <= 1:
            raise ValueError("episode_sampling_rate must be in (0,1]")
        shuffled = self.rng.permutation(eligible)
        selected = {}
        for loader_index in shuffled:
            anchors = self._exact_anchors_by_loader_index[int(loader_index)].copy()
            self.rng.shuffle(anchors)
            selected[int(loader_index)] = anchors
        total_steps = int(sum(len(value) for value in selected.values()))
        if total_steps == 0:
            raise ValueError("PACE dataset has no exact anchors")
        shard_count = max(1, int(np.ceil(total_steps / self.shard_size)))
        sharded_episodes = [[] for _ in range(shard_count)]
        shard_lengths = np.zeros(shard_count, dtype=np.int64)
        split_count = max(1, int(1.0 / self.episode_sampling_rate))
        for loader_index in shuffled:
            anchors = selected[int(loader_index)]
            # A shard is step-sized, while the old implementation contributed
            # only one chunk per episode at sampling_rate=1.  Whenever the
            # average episode contains more than ``shard_size`` anchors that
            # creates more shards than chunks and leaves the tail empty (the
            # 115-episode UR dataset hits this on every 4-GPU launch).  Split
            # each episode finely enough to populate the requested step shards
            # while retaining the sampling-rate lower bound.
            episode_piece_count = max(
                split_count,
                int(np.ceil(len(anchors) / self.shard_size)),
            )
            for split_index in range(episode_piece_count):
                chunk = anchors[split_index::episode_piece_count]
                if chunk.size == 0:
                    continue
                shard_index = int(np.argmin(shard_lengths))
                sharded_episodes[shard_index].append((int(loader_index), chunk))
                shard_lengths[shard_index] += len(chunk)
        if np.any(shard_lengths == 0):
            raise AssertionError("PACE sharding produced an empty shard")
        self.sharded_episodes = sharded_episodes
        self.shard_lengths = shard_lengths
        self.selected_anchor_count = total_steps
        print(
            f"Generated {shard_count} exact PACE shards for {self.dataset_path}; "
            f"split={self.split}, exact_anchors={total_steps}"
        )

    def get_datapoint(self, episode_data, step_index: int, *, episode_index: int):
        if self.processor is None:
            raise AssertionError("Processor must be set before PACE datapoints")
        vla_step_data = _extract_step_data(
            episode_data,
            int(step_index),
            self.modality_configs,
            self.embodiment_tag,
            allow_padding=False,
        )
        vla_step_data.metadata["pace"] = self.pace_store.metadata(
            int(episode_index), int(step_index)
        )
        raw_state = np.concatenate(
            tuple(vla_step_data.states[key] for key in ROLLOUT_CONTROL_GROUPS),
            axis=-1,
        ).astype(np.float32)
        is_command_variant = all(
            key in vla_step_data.actions for key in ROLLOUT_COMMAND_GROUPS
        )
        rollout_control_groups = (
            ROLLOUT_COMMAND_GROUPS if is_command_variant else ROLLOUT_CONTROL_GROUPS
        )
        rollout_model_groups = (
            ROLLOUT_COMMAND_GROUPS if is_command_variant else ROLLOUT_PACE_ACTION_GROUPS
        )
        raw_control_action = np.concatenate(
            tuple(vla_step_data.actions[key] for key in rollout_control_groups),
            axis=-1,
        ).astype(np.float32)
        raw_pace_action = np.concatenate(
            tuple(vla_step_data.actions[key] for key in rollout_model_groups),
            axis=-1,
        ).astype(np.float32)
        if raw_state.shape != (1, 62):
            raise ValueError(
                f"PACE rollout state must have shape (1, 62), got {raw_state.shape}"
            )
        if raw_control_action.shape != (40, 62):
            raise ValueError(
                "PACE rollout control target must have shape (40, 62), "
                f"got {raw_control_action.shape}"
            )
        expected_model_dim = 62 if is_command_variant else 106
        if raw_pace_action.shape != (40, expected_model_dim):
            raise ValueError(
                f"PACE rollout action must have shape (40, {expected_model_dim}), "
                f"got {raw_pace_action.shape}"
            )
        sidecar = self.pace_store.episode(int(episode_index))
        future = action_indices(int(step_index), sidecar.length)
        messages = [{"type": _MessageType.EPISODE_STEP.value, "content": vla_step_data}]
        transformed = self.processor(messages)
        transformed["viz_rollout_state_abs_62d"] = raw_state
        transformed["viz_rollout_gt_abs_action_62d"] = raw_control_action
        if not is_command_variant:
            transformed["viz_rollout_gt_component_major_action"] = raw_pace_action
        transformed["viz_rollout_tactile_wrench"] = sidecar.tactile_wrench[
            future
        ].copy()
        transformed["viz_rollout_tactile_wrench_valid"] = sidecar.tactile_wrench_valid[
            future
        ].copy()
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

    def iter_shard(self, idx: int, *, rng: np.random.Generator):
        """Yield a shard without materializing all processed samples in RAM.

        The upstream sharded mixture eagerly builds a list for every prefetched
        shard.  A PACE sample contains ten 240x240 tactile frames in addition to
        the three RGB views, so eager loading amplifies both startup latency and
        memory use.  DataLoader workers already prefetch complete batches; keep
        episode decoding local to a worker and stream the processed samples into
        that bounded queue instead.
        """
        episodes = list(self.sharded_episodes[idx])
        rng.shuffle(episodes)
        for loader_index, shard_step_indices in episodes:
            episode_data = self.episode_loader[loader_index]
            episode_index = int(
                self.episode_loader.episodes_metadata[loader_index]["episode_index"]
            )
            step_indices = np.asarray(shard_step_indices, dtype=np.int64).copy()
            rng.shuffle(step_indices)
            for step_index in step_indices:
                yield self.get_datapoint(
                    episode_data, int(step_index), episode_index=episode_index
                )

    def get_shard(self, idx: int) -> list:
        """Compatibility path for callers outside PACE's streaming mixture."""
        return list(self.iter_shard(idx, rng=self.rng))


class PaceStreamingMixtureDataset(
    _ShardedMixtureDataset if _ShardedMixtureDataset is not None else object
):
    """PACE mixture that lets DataLoader prefetch batches, not whole shards."""

    def __iter__(self):
        self.worker_shard_sampling_schedule = self.filter_shard_sample_schedule()
        if not self.worker_shard_sampling_schedule:
            raise RuntimeError("PACE worker received an empty shard schedule")

        while True:
            # Include rank and worker id so independently forked workers do not
            # produce the same within-shard order.
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [self.seed, self.epoch, self.rank, int(self.worker_id or 0)]
                )
            )
            for dataset_index, shard_index in self.worker_shard_sampling_schedule:
                dataset = self.datasets[dataset_index]
                if not hasattr(dataset, "iter_shard"):
                    raise TypeError(
                        "PACE streaming mixture requires datasets with iter_shard()"
                    )
                yield from dataset.iter_shard(shard_index, rng=rng)

            self.epoch += 1
            self.shard_sampling_schedule = self.generate_shard_sampling_schedule()
            self.worker_shard_sampling_schedule = self.filter_shard_sample_schedule()


class PaceDatasetFactory:
    def __init__(self, config) -> None:
        if _ShardedMixtureDataset is None or _EmbodimentTag is None:
            raise ImportError("Isaac-GR00T must be importable for PACE datasets")
        self.config = config

    def build(self, processor):
        if self.config.training.eval_strategy != "no":
            raise ValueError(
                "PACE sharded training currently requires eval_strategy='no'"
            )
        all_datasets = []
        all_weights = []
        for dataset_spec in self.config.data.datasets:
            datasets = []
            for dataset_path in dataset_spec.dataset_paths:
                embodiment_tag = dataset_spec.embodiment_tag
                if embodiment_tag is None:
                    raise ValueError("PACE embodiment tag is required")
                root = Path(dataset_path)
                stats_path = root / "meta/stats.json"
                relative_stats_path = root / "meta/relative_stats.json"
                sensor_stats_path = root / "meta/sensor_stats.json"
                provenance_path = root / "meta/stats_provenance.json"
                required = [
                    stats_path,
                    sensor_stats_path,
                    provenance_path,
                ]
                if self.config.model.use_relative_action:
                    required.append(relative_stats_path)
                missing = [str(path) for path in required if not path.is_file()]
                if missing:
                    raise FileNotFoundError(
                        "PACE requires precomputed train-only statistics; missing "
                        + ", ".join(missing)
                    )
                provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                sensor_stats = json.loads(sensor_stats_path.read_text(encoding="utf-8"))
                if (
                    provenance.get("split") != "train"
                    or sensor_stats.get("split") != "train"
                ):
                    raise ValueError("PACE statistics must be train-only")
                datasets.append(
                    PaceShardedSingleStepDataset(
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
                    )
                )
            lengths = np.asarray(
                [dataset.selected_anchor_count for dataset in datasets],
                dtype=np.float64,
            )
            for dataset, relative_length in zip(datasets, lengths / lengths.sum()):
                all_datasets.append(dataset)
                all_weights.append(float(relative_length * dataset_spec.mix_ratio))
        train_dataset = PaceStreamingMixtureDataset(
            datasets=all_datasets,
            weights=all_weights,
            processor=processor,
            seed=self.config.data.seed,
            training=True,
            num_shards_per_epoch=self.config.data.num_shards_per_epoch,
            override_pretraining_statistics=self.config.data.override_pretraining_statistics,
        )
        return train_dataset, None
