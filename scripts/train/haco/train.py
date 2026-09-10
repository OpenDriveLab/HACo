#!/usr/bin/env python3
"""Train one frozen HACO experiment from official GR00T N1.7 weights."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_ROOT = Path(
    os.environ.get("ISAAC_GROOT_DIR", PROJECT_ROOT / "third_party/Isaac-GR00T")
)
if str(REFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFERENCE_ROOT))

import accelerate.utils.other as accelerate_other  # noqa: E402
import tyro  # noqa: E402

from gr00t.configs.base_config import get_default_config  # noqa: E402
from gr00t.configs.finetune_config import FinetuneConfig  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.experiment.experiment import run  # noqa: E402
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7  # noqa: E402

from dexterity.callbacks.open_loop_eval.open_loop_eval_callback_haco import (  # noqa: E402
    install_haco_open_loop_eval_callback,
)
from dexterity.models.haco import HacoConfig  # noqa: E402
from scripts.train.haco.config import (  # noqa: E402
    CONTINUATION_GLOBAL_BATCH_SIZE,
    CONTINUATION_GPUS_PER_RUN,
    DELTA_Q_LOSS_WEIGHT,
    GLOBAL_BATCH_SIZE,
    LEARNING_RATE,
    MULTITASK_GLOBAL_BATCH_SIZE,
    MULTITASK_CONTINUATION_300K_PROFILE,
    MULTITASK_CONTINUATION_400K_PROFILE,
    MULTITASK_CONTINUATION_500K_PROFILE,
    MULTITASK_CONTINUATION_PROFILE,
    MULTITASK_PROFILE,
    RTC_INFERENCE_PREFIX_STEPS,
    RTC_PREFIX_WEIGHTS,
    RTC_TRAINING_MAX_PREFIX_STEPS,
    SEED,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    get_experiment,
    validate_continuation_training_values,
    validate_fixed_training_values,
    validate_multitask_normalization,
    validate_multitask_training_values,
    validate_official_checkpoint,
    validate_training_dataset,
)
from scripts.train.haco.pipeline import HacoPipeline  # noqa: E402,F401


def _load_modality_config(path_value: str | None) -> None:
    if path_value is None:
        raise ValueError("HACO requires an explicit modality config path")
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load modality config {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


def _training_dataset_paths(args: FinetuneConfig) -> list[Path]:
    raw = os.environ.get("HACO_DATASET_PATHS_JSON")
    if not raw:
        return [Path(args.dataset_path).expanduser().resolve()]
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("HACO_DATASET_PATHS_JSON must be valid JSON") from error
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, str) or not value.strip() for value in values)
    ):
        raise ValueError("HACO_DATASET_PATHS_JSON must be a nonempty string list")
    paths = [Path(value).expanduser().resolve() for value in values]
    if len(set(paths)) != len(paths):
        raise ValueError("HACO_DATASET_PATHS_JSON contains duplicate datasets")
    return paths


def build_model_config(args: FinetuneConfig) -> HacoConfig:
    experiment = get_experiment(os.environ.get("HACO_EXPERIMENT_ID", ""))
    source = Gr00tN1d7.config_class.from_pretrained(args.base_model_path)
    values = source.to_dict()
    values.update(
        {
            "model_name": os.environ.get(
                "GR00T_N1D7_MODEL_NAME", values.get("model_name")
            ),
            "tune_llm": args.tune_llm,
            "tune_visual": args.tune_visual,
            "tune_projector": args.tune_projector,
            "tune_diffusion_model": args.tune_diffusion_model,
            "state_dropout_prob": args.state_dropout_prob,
            "random_rotation_angle": args.random_rotation_angle,
            "color_jitter_params": args.color_jitter_params,
            "extra_augmentation_config": (
                json.loads(args.extra_augmentation_config)
                if args.extra_augmentation_config
                else None
            ),
            "load_bf16": False,
            "reproject_vision": False,
            "use_relative_action": False,
            "backbone_trainable_params_fp32": True,
            "shortest_image_edge": None,
            "crop_fraction": None,
            "experiment_id": experiment.experiment_id,
            "sensor_encoder_mode": experiment.sensor_encoder_mode,
            "physical_integration": experiment.physical_integration,
            "action_contract": experiment.action_contract,
            "camera_mode": experiment.camera_mode,
            "rtc_training_max_prefix_steps": RTC_TRAINING_MAX_PREFIX_STEPS,
            "rtc_inference_prefix_steps": RTC_INFERENCE_PREFIX_STEPS,
            "rtc_training_prefix_weights": list(RTC_PREFIX_WEIGHTS),
            "delta_q_flow_weight": DELTA_Q_LOSS_WEIGHT,
        }
    )
    return HacoConfig(**values)


def build_training_config(args: FinetuneConfig):
    nnodes = int(os.environ.get("HACO_NNODES", "1"))
    gpus_per_node = int(os.environ.get("HACO_GPUS_PER_NODE", "4"))
    per_device = int(os.environ.get("HACO_PER_DEVICE_BATCH_SIZE", "12"))
    seed = int(os.environ.get("HACO_SEED", str(SEED)))
    smoke = _env_bool("HACO_SMOKE", False)
    continuation = _env_bool("HACO_CONTINUATION", False)
    dataset_paths = _training_dataset_paths(args)
    multitask_profile = os.environ.get("HACO_MULTITASK_PROFILE", "")
    if multitask_profile and multitask_profile not in {
        MULTITASK_PROFILE,
        MULTITASK_CONTINUATION_PROFILE,
        MULTITASK_CONTINUATION_300K_PROFILE,
        MULTITASK_CONTINUATION_400K_PROFILE,
        MULTITASK_CONTINUATION_500K_PROFILE,
    }:
        raise ValueError(f"unsupported HACO multi-task profile {multitask_profile!r}")
    multitask = bool(multitask_profile)
    if continuation and (smoke or multitask):
        raise ValueError(
            "HACO_CONTINUATION is mutually exclusive with smoke and multi-task"
        )
    if multitask and not smoke:
        validate_multitask_training_values(
            profile=multitask_profile,
            dataset_count=len(dataset_paths),
            nnodes=nnodes,
            gpus_per_node=gpus_per_node,
            per_device_batch_size=per_device,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_steps=args.max_steps,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            seed=seed,
        )
    elif continuation:
        validate_continuation_training_values(
            nnodes=nnodes,
            gpus_per_node=gpus_per_node,
            per_device_batch_size=per_device,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_steps=args.max_steps,
            save_steps=args.save_steps,
            seed=seed,
        )
    elif not smoke:
        validate_fixed_training_values(
            nnodes=nnodes,
            gpus_per_node=gpus_per_node,
            per_device_batch_size=per_device,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_steps=args.max_steps,
            save_steps=args.save_steps,
            seed=seed,
        )
    elif min(args.num_gpus, args.global_batch_size, args.max_steps, args.save_steps) <= 0:
        raise ValueError("HACO smoke values must be positive")
    if multitask and not smoke and (
        args.num_gpus != 16
        or args.global_batch_size != MULTITASK_GLOBAL_BATCH_SIZE
    ):
        raise ValueError(
            "HACO multi-task training requires num_gpus=16 and global_batch_size=192"
        )
    if continuation and (
        args.num_gpus != CONTINUATION_GPUS_PER_RUN
        or args.global_batch_size != CONTINUATION_GLOBAL_BATCH_SIZE
    ):
        raise ValueError(
            "HACO continuation requires num_gpus=8 and global_batch_size=96"
        )
    if not smoke and not continuation and not multitask and (
        args.num_gpus != 4 or args.global_batch_size != GLOBAL_BATCH_SIZE
    ):
        raise ValueError("HACO requires num_gpus=4 and global_batch_size=48")
    if args.learning_rate != LEARNING_RATE:
        raise ValueError(f"HACO learning_rate must be {LEARNING_RATE}")
    if args.weight_decay != WEIGHT_DECAY:
        raise ValueError(f"HACO weight_decay must be {WEIGHT_DECAY}")
    if args.warmup_ratio != WARMUP_RATIO:
        raise ValueError(f"HACO warmup_ratio must be {WARMUP_RATIO}")
    if args.episode_sampling_rate != 1.0:
        raise ValueError("HACO must sample every episode in the task dataset")
    if args.skip_weight_loading and not _env_bool("HACO_ALLOW_SMOKE_SKIP_LOAD", False):
        raise ValueError("formal HACO runs may not skip official checkpoint weights")

    validate_official_checkpoint(args.base_model_path)
    for dataset_path in dataset_paths:
        validate_training_dataset(dataset_path)
    if multitask:
        normalization = os.environ.get("HACO_MULTITASK_NORMALIZATION_DIR")
        if normalization is None:
            raise ValueError("set HACO_MULTITASK_NORMALIZATION_DIR")
        validate_multitask_normalization(
            normalization, expected_dataset_paths=dataset_paths
        )
    _load_modality_config(args.modality_config_path)
    args.embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [str(dataset_path)],
                        "mix_ratio": 1.0,
                        "embodiment_tag": args.embodiment_tag.value,
                    }
                    for dataset_path in dataset_paths
                ],
            }
        }
    )
    config.load_config_path = None
    config.data.seed = seed
    config.data.allow_padding = False
    config.data.override_pretraining_statistics = True
    config.model = build_model_config(args)
    action_modality = config.data.modality_configs[args.embodiment_tag.value][
        "action"
    ]
    if len(action_modality.delta_indices) != 40:
        raise ValueError("HACO action horizon must be 40")
    if len(action_modality.modality_keys) not in {4, 6}:
        raise ValueError("HACO action modality must contain four or six groups")

    training = config.training
    training.experiment_name = args.experiment_name
    training.start_from_checkpoint = str(
        validate_official_checkpoint(args.base_model_path)
    )
    training.transformers_local_files_only = True
    training.transformers_cache_dir = os.environ.get(
        "HF_HUB_CACHE", str(PROJECT_ROOT / ".cache/huggingface/hub")
    )
    training.optim = "adamw_torch"
    training.use_ddp = True
    training.deepspeed_stage = 0
    training.global_batch_size = (
        args.global_batch_size
        if smoke or continuation or multitask
        else GLOBAL_BATCH_SIZE
    )
    training.dataloader_num_workers = args.dataloader_num_workers
    training.learning_rate = LEARNING_RATE
    training.lr_scheduler_type = "constant_with_warmup"
    training.gradient_accumulation_steps = args.gradient_accumulation_steps
    training.output_dir = args.output_dir
    training.logging_steps = int(os.environ.get("HACO_LOGGING_STEPS", "10"))
    training.save_steps = args.save_steps
    training.save_total_limit = args.save_total_limit
    training.num_gpus = args.num_gpus if smoke or continuation or multitask else 4
    training.use_wandb = args.use_wandb
    training.max_steps = args.max_steps
    training.weight_decay = WEIGHT_DECAY
    training.warmup_ratio = WARMUP_RATIO
    training.wandb_project = args.wandb_project
    training.save_only_model = args.save_only_model
    training.skip_weight_loading = args.skip_weight_loading
    training.eval_strategy = "no"
    training.eval_set_split_ratio = 0.0
    training.save_best_eval_metric_name = ""
    config.data.shard_size = args.shard_size
    config.data.episode_sampling_rate = 1.0
    config.data.num_shards_per_epoch = args.num_shards_per_epoch
    return config


def main() -> int:
    args = tyro.cli(FinetuneConfig, description=__doc__)
    config = build_training_config(args)
    if config.training.deepspeed_stage == 0:
        accelerate_other.is_deepspeed_available = lambda: False
    install_haco_open_loop_eval_callback()
    run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
