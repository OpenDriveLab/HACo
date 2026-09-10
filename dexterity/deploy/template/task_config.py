#!/usr/bin/env python3
"""Resolve one task/model/run into deployment environment variables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

TASK_CONFIG_SCHEMA = "sharpa.deploy.task_config.v2"

ASSET_ENV = {
    "backbone": ("SHARPA_DEPLOY_BACKBONE_MODEL", "dir"),
    "backbone_model": ("SHARPA_DEPLOY_BACKBONE_MODEL", "dir"),
    "conversion_report": ("SHARPA_DEPLOY_CONVERSION_REPORT", "file"),
    "dataset_info": ("SHARPA_DEPLOY_DATASET_INFO", "file"),
    "dataset_meta": ("SHARPA_DEPLOY_DATASET_META", "dir"),
    "stats": ("SHARPA_DEPLOY_STATS", "file"),
    "stats_provenance": ("SHARPA_DEPLOY_STATS_PROVENANCE", "file"),
    "tokenizer": ("SHARPA_DEPLOY_TOKENIZER", "dir"),
    "tokenizer_model": ("SHARPA_DEPLOY_TOKENIZER_MODEL", "file"),
}

CODE_ENV = {
    "modality_config": ("SHARPA_DEPLOY_MODALITY_CONFIG", "file"),
}


class TaskConfigError(RuntimeError):
    """A task configuration cannot be resolved safely."""


def parse_args() -> argparse.Namespace:
    root = Path(__file__).absolute().parents[3] / "scripts" / "deploy" / "task_config"
    parser = argparse.ArgumentParser(
        description="Resolve <task> <model> <run_id> from scripts/deploy/task_config."
    )
    parser.add_argument("--config-root", type=Path, default=root)
    parser.add_argument(
        "--repository-root",
        type=Path,
        help="local DreamZero checkout containing deployment code",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        help="root for relative checkpoint, dataset, and model asset paths",
    )
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--model",
        required=True,
        help="Canonical model id; must equal scripts/deploy/<model>.",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--format", choices=("env0", "json", "summary"), default="env0")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--skip-path-check", action="store_true")
    return parser.parse_args()


def canonical_model(value: str) -> str:
    """Validate, but never rewrite, the server-owned model id."""
    model = str(value).strip()
    if not model or Path(model).name != model or model in {".", "..", "task_config"}:
        raise TaskConfigError(f"invalid model id: {value!r}")
    return model


def load_task_config(config_root: Path, task_id: str) -> dict[str, Any]:
    if not task_id or Path(task_id).name != task_id:
        raise TaskConfigError(f"invalid task id: {task_id!r}")
    path = config_root.expanduser().absolute() / task_id / "task.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise TaskConfigError(f"task config not found: {path}") from error
    except json.JSONDecodeError as error:
        raise TaskConfigError(f"invalid task config {path}: {error}") from error
    if data.get("schema") != TASK_CONFIG_SCHEMA:
        raise TaskConfigError(f"unsupported task config schema: {path}")
    task = data.get("task")
    if not isinstance(task, dict) or task.get("id") != task_id:
        raise TaskConfigError(f"task.id must be {task_id!r}: {path}")
    runs = data.get("runs")
    if not isinstance(runs, list):
        raise TaskConfigError(f"task config runs must be a list: {path}")
    seen_run_ids: set[str] = set()
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise TaskConfigError(f"runs[{index}] must be an object: {path}")
        model = canonical_model(str(run.get("model") or ""))
        run_id = str(run.get("run_id") or "").strip()
        if not run_id:
            raise TaskConfigError(f"runs[{index}].run_id must be nonempty: {path}")
        if run_id in seen_run_ids:
            raise TaskConfigError(f"duplicate run_id {run_id!r}: {path}")
        seen_run_ids.add(run_id)
        run["model"] = model
    data["_path"] = str(path)
    return data


def repository_root(data: dict[str, Any], override: Path | None = None) -> Path:
    if override is not None:
        return override.expanduser().absolute()
    # snapshot.repository_root is provenance from the training machine, not a
    # deployment location. Runtime paths are always selected by the server
    # profile or fall back to the current checkout.
    return Path(__file__).absolute().parents[3]


def absolute_path(value: str, root: Path) -> str:
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else root / path).absolute())


def choose_run(
    data: dict[str, Any], model: str, requested_run_id: str | None
) -> dict[str, Any]:
    model = canonical_model(model)
    if not requested_run_id:
        raise TaskConfigError(
            f"run_id is required for task={data['task']['id']} model={model}"
        )
    matches = [
        run
        for run in data["runs"]
        if run.get("model") == model and run.get("run_id") == requested_run_id
    ]
    if not matches:
        raise TaskConfigError(
            f"run_id {requested_run_id!r} not found for "
            f"task={data['task']['id']} model={model}"
        )
    if len(matches) != 1:
        raise TaskConfigError(
            f"run_id {requested_run_id!r} is ambiguous for model={model}"
        )
    return matches[0]


def _dataset_prompt(dataset: Path) -> str | None:
    path = dataset / "meta" / "tasks.jsonl"
    if not path.is_file():
        return None
    prompts: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise TaskConfigError(
                f"invalid task prompt at {path}:{line_number}"
            ) from error
        prompt = value.get("task")
        if isinstance(prompt, str) and prompt.strip():
            prompts.add(prompt.strip())
    if len(prompts) > 1:
        raise TaskConfigError(f"dataset contains multiple task prompts: {path}")
    return next(iter(prompts), None)


def resolve_prompt(task: dict[str, Any], dataset: Path) -> str:
    recorded = task.get("prompt")
    recorded = recorded.strip() if isinstance(recorded, str) else None
    dataset_value = _dataset_prompt(dataset)
    if dataset_value and recorded and dataset_value != recorded:
        raise TaskConfigError(
            "task config prompt disagrees with dataset meta/tasks.jsonl: "
            f"{recorded!r} != {dataset_value!r}"
        )
    prompt = dataset_value or recorded
    if not prompt:
        raise TaskConfigError(
            f"task={task.get('id')} has no prompt in dataset or task config"
        )
    return prompt


def resolve_environment(
    data: dict[str, Any],
    model: str,
    run: dict[str, Any],
    root: Path,
    *,
    code_root: Path | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    model = canonical_model(model)
    if run.get("model") != model:
        raise TaskConfigError(
            f"run model {run.get('model')!r} does not match launcher {model!r}"
        )
    task = data["task"]
    run_id = str(run.get("run_id") or "")
    run_alias = str(run.get("id") or run_id)
    checkpoint = run.get("checkpoint")
    dataset_value = task.get("dataset")
    if not run_id or not checkpoint or not dataset_value:
        raise TaskConfigError("run_id, checkpoint, and task.dataset are required")
    dataset = Path(absolute_path(str(dataset_value), root))
    code_root = (code_root or root).expanduser().absolute()
    root = root.expanduser().absolute()
    env = {
        "SHARPA_DEPLOY_REPOSITORY_ROOT": str(code_root),
        "SHARPA_DEPLOY_ASSET_ROOT": str(root),
        "SHARPA_DEPLOY_TASK": str(task["id"]),
        "SHARPA_DEPLOY_ID": run_alias,
        "SHARPA_DEPLOY_RUN_ID": run_id,
        "SHARPA_DEPLOY_MODEL": model,
        "SHARPA_DEPLOY_DATASET": str(dataset),
        "SHARPA_DEPLOY_PROMPT": resolve_prompt(task, dataset),
        "SHARPA_DEPLOY_CHECKPOINT": absolute_path(str(checkpoint), root),
    }
    model_implementation = run.get("model_implementation")
    if model_implementation:
        env["SHARPA_DEPLOY_MODEL_IMPLEMENTATION"] = str(model_implementation)
    path_kinds = {
        "SHARPA_DEPLOY_REPOSITORY_ROOT": "dir",
        "SHARPA_DEPLOY_ASSET_ROOT": "dir",
        "SHARPA_DEPLOY_DATASET": "dir",
        "SHARPA_DEPLOY_CHECKPOINT": "dir",
    }

    assets = dict(task.get("meta", {}))
    assets.update(run.get("assets", {}))
    for name, value in assets.items():
        if name in {"tokenizer", "tokenizer_model"} and model != "pi05":
            continue
        setting = ASSET_ENV.get(name) or CODE_ENV.get(name)
        if setting is None or value in (None, ""):
            continue
        key, kind = setting
        path_root = code_root if name in CODE_ENV else root
        env[key] = absolute_path(str(value), path_root)
        path_kinds[key] = kind
    task_normalization = task.get("meta", {}).get("normalization") or task.get(
        "meta", {}
    ).get("sensor_stats")
    if model == "t_rex":
        task_normalization = (
            task.get("meta", {}).get("trex_stats") or task_normalization
        )
    elif model == "vitacformer":
        task_normalization = (
            task.get("meta", {}).get("vitac_stats") or task_normalization
        )
    if task_normalization:
        env["SHARPA_DEPLOY_NORMALIZATION"] = absolute_path(
            str(task_normalization), root
        )
        path_kinds["SHARPA_DEPLOY_NORMALIZATION"] = "file"
    normalization = run.get("normalization")
    if normalization:
        env["SHARPA_DEPLOY_NORMALIZATION"] = absolute_path(str(normalization), root)
        path_kinds["SHARPA_DEPLOY_NORMALIZATION"] = "file"
        if model == "vitacformer":
            env["SHARPA_DEPLOY_STATS"] = env["SHARPA_DEPLOY_NORMALIZATION"]
            path_kinds["SHARPA_DEPLOY_STATS"] = "file"
    embodiment_tag = run.get("deployment", {}).get("embodiment_tag")
    if embodiment_tag:
        env["SHARPA_DEPLOY_EMBODIMENT_TAG"] = str(embodiment_tag)
    if model == "vitacformer" and "SHARPA_DEPLOY_STATS" not in env:
        normalization = env.get("SHARPA_DEPLOY_NORMALIZATION")
        if normalization:
            env["SHARPA_DEPLOY_STATS"] = normalization
            path_kinds["SHARPA_DEPLOY_STATS"] = "file"
    return env, path_kinds


def validate_run(run: dict[str, Any], allow_incomplete: bool) -> None:
    status = run.get("status")
    deployment = run.get("deployment", {})
    eligible = deployment.get(
        "eligible",
        deployment.get(
            "eligible_for_initial_bash",
            deployment.get("eligible_for_launcher_assembly"),
        ),
    )
    if eligible is False:
        raise TaskConfigError(
            f"run {run.get('run_id')} is not deployable: "
            f"{deployment.get('reason', 'marked ineligible')}"
        )
    # A checkpoint produced by a still-running formal job is a valid deployment
    # target when the task config opts in explicitly.  Keep failed, interrupted,
    # or merely unspecified runs behind the manual development override.
    deployable_intermediate = status == "in_progress" and eligible is True
    if status != "complete" and not deployable_intermediate and not allow_incomplete:
        raise TaskConfigError(
            f"run {run.get('run_id')} has status={status}; use --allow-incomplete only intentionally"
        )


def validate_paths(env: dict[str, str], path_kinds: dict[str, str]) -> None:
    missing = []
    for key, kind in path_kinds.items():
        path = Path(env[key])
        valid = path.is_dir() if kind == "dir" else path.is_file()
        if not valid:
            missing.append(f"{key}={path} (expected {kind})")
    if missing:
        raise TaskConfigError(
            "required deployment paths are missing:\n  " + "\n  ".join(missing)
        )


def emit(env: dict[str, str], output_format: str) -> None:
    if output_format == "json":
        json.dump(env, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    elif output_format == "summary":
        for key in sorted(env):
            print(f"{key}={env[key]}")
    else:
        for key, value in env.items():
            sys.stdout.buffer.write(key.encode() + b"\0" + value.encode() + b"\0")


def list_runs(data: dict[str, Any], model: str) -> None:
    model = canonical_model(model)
    for run in data["runs"]:
        if run.get("model") != model:
            continue
        print(
            "\t".join(
                (
                    model,
                    str(run.get("id", "")),
                    str(run.get("run_id", "")),
                    str(run.get("status", "")),
                )
            )
        )


def main() -> int:
    args = parse_args()
    try:
        data = load_task_config(args.config_root, args.task)
        model = canonical_model(args.model)
        if args.list:
            list_runs(data, model)
            return 0
        run = choose_run(data, model, args.run_id)
        validate_run(run, args.allow_incomplete)
        code_root = repository_root(data, args.repository_root)
        asset_root = (
            args.asset_root.expanduser().absolute()
            if args.asset_root is not None
            else code_root
        )
        env, path_kinds = resolve_environment(
            data, model, run, asset_root, code_root=code_root
        )
        if not args.skip_path_check:
            validate_paths(env, path_kinds)
        emit(env, args.format)
        return 0
    except TaskConfigError as error:
        print(f"task config error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
