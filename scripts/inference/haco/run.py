#!/usr/bin/env python3
"""Run the fixed 115-episode HACO trained-RTC open-loop diagnostic."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
REFERENCE_ROOT = Path(
    os.environ.get("ISAAC_GROOT_DIR", ROOT / "third_party/Isaac-GR00T")
)
if str(REFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFERENCE_ROOT))

from dexterity.callbacks.open_loop_eval.open_loop_eval_callback_haco import (  # noqa: E402
    _action_prediction,
    _clean_active_prefix,
    decode_haco_action,
)
from scripts.inference.haco.protocol import (  # noqa: E402
    FORMAL_EVAL_SCHEMA,
    FormalEvalProtocol,
    action_metrics,
    summarize_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument(
        "--embodiment-tag", default="real_r1_pro_sharpa_absolute_eef"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Smoke-only prefix of the manifest; 0 is required for formal output.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _checkpoint_step(checkpoint: Path) -> int:
    name = checkpoint.resolve().name
    prefix = "checkpoint-"
    if not name.startswith(prefix):
        raise ValueError("HACO formal checkpoint must be named checkpoint-30000")
    return int(name[len(prefix) :])


def _formal_protocol(manifest: Mapping[str, Any]) -> FormalEvalProtocol:
    if manifest.get("schema") != FORMAL_EVAL_SCHEMA:
        raise ValueError("unsupported HACO open-loop manifest")
    return FormalEvalProtocol(
        action_contract=str(manifest["action_contract"]),
        action_target=str(manifest["action_target"]),
        checkpoint_step=int(manifest["checkpoint_step"]),
        episode_count=int(manifest["episode_count"]),
        action_horizon=int(manifest["native_horizon"]),
        rtc_prefix_steps=int(manifest["rtc_prefix_steps"]),
        eval_seed=int(manifest["eval_seed"]),
    )


def _load_runtime(checkpoint: Path, device: torch.device):
    from dexterity.models.haco import Haco, HacoProcessor

    kwargs = {"trust_remote_code": True, "local_files_only": True}
    model = Haco.from_pretrained(
        str(checkpoint),
        torch_dtype=torch.bfloat16,
        transformers_loading_kwargs=kwargs,
        **kwargs,
    )
    processor = HacoProcessor.from_pretrained(
        str(checkpoint),
        transformers_loading_kwargs=kwargs,
        **kwargs,
    )
    model.to(device=device, dtype=torch.bfloat16).eval()
    return model, processor


def _build_dataset(
    *,
    dataset_root: Path,
    processor,
    model_config,
    embodiment_tag: str,
    video_backend: str,
    seed: int,
):
    from gr00t.data.embodiment_tags import EmbodimentTag
    from scripts.train.haco.data_factory import (
        HacoShardedSingleStepDataset,
    )

    embodiment = EmbodimentTag.resolve(embodiment_tag)
    dataset = HacoShardedSingleStepDataset(
        dataset_path=str(dataset_root),
        embodiment_tag=embodiment,
        modality_configs=processor.modality_configs[embodiment.value],
        video_backend=video_backend,
        shard_size=1024,
        episode_sampling_rate=1.0,
        seed=seed,
        allow_padding=False,
        split="train",
        experiment_id=str(model_config.experiment_id),
    )
    dataset.processor = processor
    return dataset, embodiment


def _collated_sample(
    processor, sample: Mapping[str, Any]
) -> tuple[dict[str, Any], torch.Tensor]:
    batch = dict(processor.collator([dict(sample)])["inputs"])
    target = batch.get("action")
    if not torch.is_tensor(target) or target.shape != (1, 40, 132):
        raise ValueError(
            "HACO formal sample must collate action to [1,40,132]"
        )
    for key in ("action", "action_mask", "action_physical_scale"):
        batch.pop(key, None)
    for key in list(batch):
        if str(key).startswith("viz_"):
            batch.pop(key, None)
    return batch, target.float()


def _raw_sidechannel(sample: Mapping[str, Any], key: str, shape) -> np.ndarray:
    value = np.asarray(sample[key], dtype=np.float32)
    if value.shape != shape:
        raise ValueError(f"{key} must have shape {shape}, got {value.shape}")
    return value


def _reset_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate_sample(
    *,
    model,
    processor,
    dataset,
    embodiment,
    sample_record: Mapping[str, Any],
    protocol: FormalEvalProtocol,
) -> dict[str, Any]:
    first = dataset.get_datapoint_by_episode_anchor(
        int(sample_record["episode_index"]), int(sample_record["first_anchor"])
    )
    second = dataset.get_datapoint_by_episode_anchor(
        int(sample_record["episode_index"]), int(sample_record["second_anchor"])
    )
    first_inputs, first_target_norm = _collated_sample(processor, first)
    second_inputs, second_target_norm = _collated_sample(processor, second)
    with torch.no_grad(), torch.autocast(
        device_type=model.device.type,
        dtype=torch.bfloat16,
        enabled=model.device.type == "cuda",
    ):
        first_output = model.get_action(first_inputs)
        first_prediction = _action_prediction(first_output)
        prefix = _clean_active_prefix(
            first_prediction,
            prefix_steps=protocol.rtc_prefix_steps,
            active_mask=model.action_contract.active_mask(),
        )
        second_output = model.get_action(
            second_inputs,
            options={
                "rtc_prefix_steps": protocol.rtc_prefix_steps,
                "rtc_prefix_action": prefix,
            },
        )
        second_prediction = _action_prediction(second_output)
    prediction_norm = torch.cat(
        (first_prediction, second_prediction), dim=0
    ).detach().float().cpu().numpy()
    target_norm = torch.cat(
        (first_target_norm, second_target_norm), dim=0
    ).numpy()

    prediction_main_abs = []
    prediction_delta_abs = []
    for prediction, source in (
        (first_prediction, first),
        (second_prediction, second),
    ):
        state = _raw_sidechannel(
            source, "viz_rollout_state_abs_62d", (1, 62)
        )
        main, delta = decode_haco_action(
            processor,
            prediction,
            embodiment,
            state,
            action_contract=protocol.action_contract,
        )
        prediction_main_abs.append(main[0])
        if delta is not None:
            prediction_delta_abs.append(delta[0])
    target_main_abs = np.stack(
        (
            _raw_sidechannel(first, "viz_rollout_gt_abs_action_62d", (40, 62)),
            _raw_sidechannel(second, "viz_rollout_gt_abs_action_62d", (40, 62)),
        )
    )
    prediction_main_abs_array = np.stack(prediction_main_abs)
    if prediction_main_abs_array.shape != target_main_abs.shape:
        raise ValueError("decoded HACO main action has the wrong shape")

    prediction_delta = target_delta = None
    if protocol.action_contract == "joint_compliance_delta":
        prediction_delta = np.stack(prediction_delta_abs)
        target_delta = np.stack(
            (
                _raw_sidechannel(
                    first, "viz_rollout_gt_haco_action", (40, 106)
                )[:, 62:106],
                _raw_sidechannel(
                    second, "viz_rollout_gt_haco_action", (40, 106)
                )[:, 62:106],
            )
        )
    metrics = action_metrics(
        prediction_norm[..., :62],
        target_norm[..., :62],
        protocol=protocol,
        prediction_delta_q_rad=prediction_delta,
        target_delta_q_rad=target_delta,
    )
    return {
        **dict(sample_record),
        **metrics,
        "action_target": protocol.action_target,
        "execution_semantics": "direct_main_q",
        "prediction_main_abs_mean": float(np.mean(prediction_main_abs_array)),
        "target_main_abs_mean": float(np.mean(target_main_abs)),
    }


def _episode_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["episode_index"])].append(record)
    summaries = []
    for episode_index in sorted(grouped):
        summary = summarize_records(grouped[episode_index])
        summaries.append(
            {"episode_index": episode_index, **summary["metrics"]}
        )
    return summaries


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if _checkpoint_step(checkpoint) != 30_000:
        raise ValueError("formal HACO evaluation uses checkpoint-30000 only")
    manifest = _read_json(args.manifest)
    protocol = _formal_protocol(manifest)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("HACO formal evaluation requires CUDA")
    model, processor = _load_runtime(checkpoint, device)
    if model.config.action_contract != protocol.action_contract:
        raise ValueError("checkpoint and eval manifest action contracts differ")
    if model.config.action_target != protocol.action_target:
        raise ValueError("checkpoint and eval manifest target semantics differ")
    if int(model.config.rtc_inference_prefix_steps) != protocol.rtc_prefix_steps:
        raise ValueError("checkpoint and eval manifest RTC prefix lengths differ")
    dataset, embodiment = _build_dataset(
        dataset_root=Path(manifest["dataset"]),
        processor=processor,
        model_config=model.config,
        embodiment_tag=args.embodiment_tag,
        video_backend=args.video_backend,
        seed=protocol.eval_seed,
    )
    samples = list(manifest["samples"])
    is_formal = args.max_samples <= 0
    if not is_formal:
        samples = samples[: args.max_samples]
    if is_formal and len(samples) != 345:
        raise ValueError("formal HACO evaluation requires all 345 samples")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    started = time.monotonic()
    for index, sample_record in enumerate(samples):
        _reset_seed(protocol.eval_seed + int(sample_record["sample_index"]))
        record = _evaluate_sample(
            model=model,
            processor=processor,
            dataset=dataset,
            embodiment=embodiment,
            sample_record=sample_record,
            protocol=protocol,
        )
        records.append(record)
        print(
            f"[haco-eval] {index + 1}/{len(samples)} "
            f"episode={record['episode_index']} anchor={record['first_anchor']}",
            flush=True,
        )
    summary = summarize_records(records)
    payload = {
        **protocol.metadata(),
        "checkpoint": str(checkpoint),
        "formal_protocol": is_formal,
        "elapsed_seconds": time.monotonic() - started,
        "sample_count": len(records),
        "summary": summary["metrics"],
        "episode_summaries": _episode_summaries(records),
    }
    (args.output_dir / "records.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print((args.output_dir / "summary.json").resolve(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
