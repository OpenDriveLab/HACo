#!/usr/bin/env python3
"""Validate immutable HACO inputs without importing the model runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.train.haco.config import (
    CONTINUATION_GLOBAL_BATCH_SIZE,
    EXPERIMENTS,
    GLOBAL_BATCH_SIZE,
    MULTITASK_GLOBAL_BATCH_SIZE,
    MULTITASK_PROFILE,
    RTC_INFERENCE_PREFIX_STEPS,
    RTC_PREFIX_WEIGHTS,
    RTC_TRAINING_MAX_PREFIX_STEPS,
    get_experiment,
    validate_continuation_training_values,
    validate_fixed_training_values,
    validate_multitask_normalization,
    validate_multitask_training_values,
    validate_official_checkpoint,
    validate_training_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True, choices=tuple(EXPERIMENTS))
    parser.add_argument("--base-model-path", required=True, type=Path)
    parser.add_argument(
        "--dataset-path", required=True, type=Path, action="append"
    )
    parser.add_argument("--multitask-profile", default="")
    parser.add_argument("--normalization-dir", type=Path)
    parser.add_argument("--nnodes", required=True, type=int)
    parser.add_argument("--gpus-per-node", required=True, type=int)
    parser.add_argument("--per-device-batch-size", required=True, type=int)
    parser.add_argument("--gradient-accumulation-steps", required=True, type=int)
    parser.add_argument("--max-steps", required=True, type=int)
    parser.add_argument("--save-steps", required=True, type=int)
    parser.add_argument("--save-total-limit", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--continuation", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    experiment = get_experiment(args.experiment_id)
    base = validate_official_checkpoint(args.base_model_path)
    datasets = [validate_training_dataset(path) for path in args.dataset_path]
    dataset_infos = [
        json.loads((path / "meta" / "info.json").read_text(encoding="utf-8"))
        for path in datasets
    ]
    if args.smoke and args.continuation:
        raise ValueError("--smoke and --continuation are mutually exclusive")
    if args.multitask_profile and args.continuation:
        raise ValueError("multi-task and continuation are mutually exclusive")
    if args.multitask_profile and not args.smoke:
        validate_multitask_training_values(
            profile=args.multitask_profile,
            dataset_count=len(datasets),
            nnodes=args.nnodes,
            gpus_per_node=args.gpus_per_node,
            per_device_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_steps=args.max_steps,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            seed=args.seed,
        )
    elif args.multitask_profile:
        if (
            args.multitask_profile != MULTITASK_PROFILE
            or len(datasets) != 5
            or args.nnodes != 2
            or args.gpus_per_node != 8
            or args.per_device_batch_size != 12
            or args.gradient_accumulation_steps != 1
            or args.seed != 42
        ):
            raise ValueError("HACO multi-task smoke must preserve data and topology")
    elif args.continuation:
        validate_continuation_training_values(
            nnodes=args.nnodes,
            gpus_per_node=args.gpus_per_node,
            per_device_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_steps=args.max_steps,
            save_steps=args.save_steps,
            seed=args.seed,
        )
    elif not args.smoke:
        validate_fixed_training_values(
            nnodes=args.nnodes,
            gpus_per_node=args.gpus_per_node,
            per_device_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_steps=args.max_steps,
            save_steps=args.save_steps,
            seed=args.seed,
        )
    elif min(
        args.nnodes,
        args.gpus_per_node,
        args.per_device_batch_size,
        args.gradient_accumulation_steps,
        args.max_steps,
        args.save_steps,
    ) <= 0:
        raise ValueError("smoke launch values must be positive")
    normalization = None
    if args.multitask_profile:
        if args.normalization_dir is None:
            raise ValueError("multi-task HACO requires --normalization-dir")
        normalization = validate_multitask_normalization(
            args.normalization_dir, expected_dataset_paths=datasets
        )
    result = {
        "status": "ready",
        "experiment": experiment.as_dict(),
        "official_base_model_path": str(base),
        "dataset_paths": [str(path) for path in datasets],
        "train_episodes": sum(int(info["total_episodes"]) for info in dataset_infos),
        "validation_episodes": sum(
            int(info["split_counts"]["val"]) for info in dataset_infos
        ),
        "multitask_profile": args.multitask_profile or None,
        "normalization_dir": str(normalization) if normalization else None,
        "global_batch_size": (
            args.nnodes
            * args.gpus_per_node
            * args.per_device_batch_size
            * args.gradient_accumulation_steps
        ),
        "formal_global_batch_size": GLOBAL_BATCH_SIZE,
        "continuation_global_batch_size": CONTINUATION_GLOBAL_BATCH_SIZE,
        "multitask_global_batch_size": MULTITASK_GLOBAL_BATCH_SIZE,
        "max_steps": args.max_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "seed": args.seed,
        "rtc": {
            "training_max_prefix_steps": RTC_TRAINING_MAX_PREFIX_STEPS,
            "inference_prefix_steps": RTC_INFERENCE_PREFIX_STEPS,
            "prefix_weights": list(RTC_PREFIX_WEIGHTS),
        },
        "smoke": args.smoke,
        "continuation": args.continuation,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
