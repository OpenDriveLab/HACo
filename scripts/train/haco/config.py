"""Frozen HACO experiment contracts and launch-time validation.

This module deliberately has no Isaac-GR00T imports, so launchers can validate
the experiment matrix before importing the training runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


BASE_MODEL_TYPE = "Gr00tN1d7"
ACTION_HORIZON = 40
ACTION_CARRIER_DIM = 132
GPUS_PER_RUN = 4
PER_DEVICE_BATCH_SIZE = 12
GLOBAL_BATCH_SIZE = 48
CONTINUATION_GPUS_PER_RUN = 8
CONTINUATION_GLOBAL_BATCH_SIZE = 96
CONTINUATION_TARGET_STEPS = 50_000
CONTINUATION_SAVE_STEPS = 10_000
MULTITASK_PROFILE = "mt5_equal_2n16g_100k"
MULTITASK_CONTINUATION_PROFILE = "mt5_equal_2n16g_200k"
MULTITASK_CONTINUATION_300K_PROFILE = "mt5_equal_2n16g_300k"
MULTITASK_CONTINUATION_400K_PROFILE = "mt5_equal_2n16g_400k"
MULTITASK_CONTINUATION_500K_PROFILE = "mt5_equal_2n16g_500k"
MULTITASK_TASK_COUNT = 5
MULTITASK_NNODES = 2
MULTITASK_GPUS_PER_NODE = 8
MULTITASK_GLOBAL_BATCH_SIZE = 192
MULTITASK_TARGET_STEPS = 100_000
MULTITASK_CONTINUATION_TARGET_STEPS = 200_000
MULTITASK_CONTINUATION_300K_TARGET_STEPS = 300_000
MULTITASK_CONTINUATION_400K_TARGET_STEPS = 400_000
MULTITASK_CONTINUATION_500K_TARGET_STEPS = 500_000
MULTITASK_TARGET_STEPS_BY_PROFILE = {
    MULTITASK_PROFILE: MULTITASK_TARGET_STEPS,
    MULTITASK_CONTINUATION_PROFILE: MULTITASK_CONTINUATION_TARGET_STEPS,
    MULTITASK_CONTINUATION_300K_PROFILE: MULTITASK_CONTINUATION_300K_TARGET_STEPS,
    MULTITASK_CONTINUATION_400K_PROFILE: MULTITASK_CONTINUATION_400K_TARGET_STEPS,
    MULTITASK_CONTINUATION_500K_PROFILE: MULTITASK_CONTINUATION_500K_TARGET_STEPS,
}
MULTITASK_SAVE_STEPS = 10_000
SEED = 42
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-5
WARMUP_RATIO = 0.05
DELTA_Q_LOSS_WEIGHT = 0.5
RTC_TRAINING_MAX_PREFIX_STEPS = 12
RTC_INFERENCE_PREFIX_STEPS = 10
RTC_PREFIX_WEIGHTS = (1.0,) * 10 + (2.0, 1.0, 1.0)

SENSOR_MODES = frozenset(
    {"none", "tactile_only", "torque_only", "separate", "fused"}
)
PHYSICAL_INTEGRATIONS = frozenset(
    {"suffix", "imgmem", "physcross_ungated", "physcross_gated"}
)
ACTION_CONTRACTS = frozenset(
    {"joint_compliance_delta", "compliance_only", "nominal_only"}
)
CAMERA_MODES = frozenset({"three", "ego"})


@dataclass(frozen=True)
class HacoExperiment:
    """One independently trained HACO model in the 15-model matrix."""

    experiment_id: str
    paper_id: str
    group: str
    sensor_encoder_mode: str
    physical_integration: str
    action_contract: str
    camera_mode: str = "three"
    launch_priority: int = 0

    @property
    def action_target(self) -> str:
        return (
            "q_nominal"
            if self.action_contract == "nominal_only"
            else "q_compliance"
        )

    @property
    def semantic_action_dim(self) -> int:
        return 106 if self.action_contract == "joint_compliance_delta" else 62

    @property
    def has_delta_q(self) -> bool:
        return self.action_contract == "joint_compliance_delta"

    def validate(self) -> None:
        if self.sensor_encoder_mode not in SENSOR_MODES:
            raise ValueError(
                f"{self.experiment_id}: invalid sensor mode "
                f"{self.sensor_encoder_mode!r}"
            )
        if self.physical_integration not in PHYSICAL_INTEGRATIONS:
            raise ValueError(
                f"{self.experiment_id}: invalid physical integration "
                f"{self.physical_integration!r}"
            )
        if self.action_contract not in ACTION_CONTRACTS:
            raise ValueError(
                f"{self.experiment_id}: invalid action contract "
                f"{self.action_contract!r}"
            )
        if self.camera_mode not in CAMERA_MODES:
            raise ValueError(
                f"{self.experiment_id}: invalid camera mode {self.camera_mode!r}"
            )
        if self.launch_priority <= 0:
            raise ValueError(
                f"{self.experiment_id}: launch_priority must be positive"
            )

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "action_target": self.action_target,
                "semantic_action_dim": self.semantic_action_dim,
                "action_carrier_dim": ACTION_CARRIER_DIM,
                "rtc_enabled": True,
            }
        )
        return result


def _experiment(
    experiment_id: str,
    paper_id: str,
    group: str,
    sensor_encoder_mode: str,
    physical_integration: str,
    action_contract: str,
    *,
    camera_mode: str = "three",
    priority: int,
) -> HacoExperiment:
    value = HacoExperiment(
        experiment_id=experiment_id,
        paper_id=paper_id,
        group=group,
        sensor_encoder_mode=sensor_encoder_mode,
        physical_integration=physical_integration,
        action_contract=action_contract,
        camera_mode=camera_mode,
        launch_priority=priority,
    )
    value.validate()
    return value


# The complete HACO model is trained once. These are the eleven independent
# HACO models in the final fifteen-model matrix, in launch-priority order.
EXPERIMENTS = {
    item.experiment_id: item
    for item in (
        _experiment(
            "haco",
            "haco",
            "main",
            "fused",
            "physcross_gated",
            "joint_compliance_delta",
            priority=1,
        ),
        _experiment(
            "hp_wo_haptic",
            "hp_wo_haptic",
            "perception",
            "none",
            "physcross_gated",
            "joint_compliance_delta",
            priority=2,
        ),
        _experiment(
            "hp_wo_torque",
            "hp_wo_torque",
            "perception",
            "tactile_only",
            "physcross_gated",
            "joint_compliance_delta",
            priority=3,
        ),
        _experiment(
            "hp_wo_tactile",
            "hp_wo_tactile",
            "perception",
            "torque_only",
            "physcross_gated",
            "joint_compliance_delta",
            priority=4,
        ),
        _experiment(
            "hp_wo_coupled_en",
            "hp_wo_coupled_en",
            "perception",
            "separate",
            "physcross_gated",
            "joint_compliance_delta",
            priority=5,
        ),
        _experiment(
            "ac_wo_intent_sup",
            "ac_wo_intent_sup",
            "active_compliance",
            "fused",
            "physcross_gated",
            "compliance_only",
            priority=6,
        ),
        _experiment(
            "ac_wo_active_comp",
            "ac_wo_active_comp",
            "active_compliance",
            "fused",
            "physcross_gated",
            "nominal_only",
            priority=7,
        ),
        _experiment(
            "cg_action_suf",
            "cg_action_suf",
            "pfi",
            "fused",
            "suffix",
            "joint_compliance_delta",
            priority=8,
        ),
        _experiment(
            "cg_visuo_haptic",
            "cg_visuo_haptic",
            "pfi",
            "fused",
            "imgmem",
            "joint_compliance_delta",
            priority=9,
        ),
        _experiment(
            "cg_ungated_comp_attn",
            "cg_ungated_comp_attn",
            "pfi",
            "fused",
            "physcross_ungated",
            "joint_compliance_delta",
            priority=10,
        ),
        _experiment(
            "wc_wo_wrist",
            "wc_wo_wrist",
            "wrist_camera",
            "fused",
            "physcross_gated",
            "joint_compliance_delta",
            camera_mode="ego",
            priority=11,
        ),
    )
}


def get_experiment(experiment_id: str) -> HacoExperiment:
    try:
        return EXPERIMENTS[experiment_id]
    except KeyError as error:
        raise ValueError(
            f"unknown HACO experiment {experiment_id!r}; expected one of "
            f"{tuple(EXPERIMENTS)}"
        ) from error


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def validate_official_checkpoint(path_value: str | Path) -> Path:
    """Accept only an untouched pretrained source checkpoint."""

    path = Path(path_value).expanduser().resolve()
    lowered_parts = {part.lower() for part in path.parts}
    forbidden = {"posttrain", "postrain", "logs"}
    intersection = sorted(lowered_parts & forbidden)
    if intersection:
        raise ValueError(
            "HACO must initialize from a pretrained base, not a task checkpoint; "
            f"checkpoint; forbidden path components={intersection}: {path}"
        )
    config_path = path / "config.json"
    processor_path = path / "processor_config.json"
    if not config_path.is_file() or not processor_path.is_file():
        raise FileNotFoundError(
            f"official checkpoint requires config.json and processor_config.json: {path}"
        )
    payload = _read_json(config_path)
    if payload.get("model_type") != BASE_MODEL_TYPE:
        raise ValueError(
            "HACO source model_type must be "
            f"{BASE_MODEL_TYPE!r}, got {payload.get('model_type')!r}"
        )
    if int(payload.get("action_horizon", -1)) != ACTION_HORIZON:
        raise ValueError(
            f"official checkpoint action_horizon must be {ACTION_HORIZON}"
        )
    return path


def validate_training_dataset(path_value: str | Path) -> Path:
    """Enforce the task-independent UR-SharpA action and split contract."""

    path = Path(path_value).expanduser().resolve()
    info_path = path / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    info = _read_json(info_path)
    expected_scalars = {"state_dim": 62, "action_dim": 150, "total_tasks": 1}
    for key, expected in expected_scalars.items():
        if int(info.get(key, -1)) != expected:
            raise ValueError(
                f"dataset {key} must be {expected}, got {info.get(key)!r}"
            )
    total_episodes = int(info.get("total_episodes", -1))
    total_frames = int(info.get("total_frames", -1))
    if total_episodes <= 0 or total_frames <= 0:
        raise ValueError("dataset must contain positive episode and frame counts")
    expected_splits = {"train": total_episodes, "val": 0, "test": 0}
    if info.get("split_counts") != expected_splits:
        raise ValueError(
            f"dataset split_counts must be {expected_splits}, "
            f"got {info.get('split_counts')!r}"
        )
    expected_range = {"train": f"0:{total_episodes}"}
    if info.get("splits") != expected_range:
        raise ValueError(
            f"dataset must expose only the complete train split {expected_range}"
        )
    if info.get("action_order") != [
        "wrist_exe_next18",
        "q_exe_next44",
        "q_teleop44",
        "delta_q44",
    ]:
        raise ValueError("dataset has the wrong 150-D action order")
    if info.get("delta_q_definition") != "q_teleop - q_exe_next":
        raise ValueError("dataset has the wrong delta_q definition")
    required = (
        path / "meta" / "stats.json",
        path / "meta" / "sensor_stats.json",
        path / "meta" / "stats_provenance.json",
        path / "meta" / "tasks.jsonl",
    )
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(
            "HACO requires train-only statistics: " + ", ".join(missing)
        )
    provenance = _read_json(path / "meta" / "stats_provenance.json")
    sensor_stats = _read_json(path / "meta" / "sensor_stats.json")
    if provenance.get("split") != "train" or sensor_stats.get("split") != "train":
        raise ValueError("HACO statistics must be computed from train only")
    prompts: set[str] = set()
    for line_number, line in enumerate(
        (path / "meta" / "tasks.jsonl").read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        try:
            task = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid task prompt at meta/tasks.jsonl:{line_number}"
            ) from error
        prompt = task.get("task") if isinstance(task, dict) else None
        if isinstance(prompt, str) and prompt.strip():
            prompts.add(prompt.strip())
    if len(prompts) != 1:
        raise ValueError("HACO dataset must contain exactly one nonempty task prompt")
    return path


def validate_multitask_normalization(
    path_value: str | Path,
    *,
    expected_dataset_paths: list[str | Path] | None = None,
) -> Path:
    """Validate the immutable equal-task normalization artifact."""

    path = Path(path_value).expanduser().resolve()
    required = (
        path / "stats.json",
        path / "sensor_stats.json",
        path / "stats_provenance.json",
        path / "mixture_manifest.json",
    )
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(
            "HACO multi-task normalization is incomplete: " + ", ".join(missing)
        )
    manifest = _read_json(path / "mixture_manifest.json")
    if manifest.get("schema") != "sharpa.haco_multitask_normalization.v1":
        raise ValueError("unsupported HACO multi-task normalization schema")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != MULTITASK_TASK_COUNT:
        raise ValueError(
            f"HACO multi-task normalization requires {MULTITASK_TASK_COUNT} datasets"
        )
    expected_weight = 1.0 / MULTITASK_TASK_COUNT
    if any(abs(float(item.get("weight", -1.0)) - expected_weight) > 1e-12 for item in datasets):
        raise ValueError("HACO multi-task normalization must use equal task weights")
    if expected_dataset_paths is not None:
        expected_paths = [
            str(Path(value).expanduser().resolve()) for value in expected_dataset_paths
        ]
        actual_paths = [
            str(Path(item.get("path", "")).expanduser().resolve())
            for item in datasets
        ]
        if actual_paths != expected_paths:
            raise ValueError(
                "HACO multi-task normalization datasets do not match training data"
            )
    provenance = _read_json(path / "stats_provenance.json")
    sensor = _read_json(path / "sensor_stats.json")
    if (
        provenance.get("schema")
        != "sharpa.haco_multitask_equal_stats_provenance.v1"
        or provenance.get("split") != "train"
        or sensor.get("schema") != "sharpa.sensor_normalization.multitask.v1"
        or sensor.get("split") != "train"
    ):
        raise ValueError("HACO multi-task statistics provenance is invalid")
    return path


def validate_fixed_training_values(
    *,
    nnodes: int,
    gpus_per_node: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    max_steps: int,
    save_steps: int,
    seed: int,
) -> None:
    values = {
        "nnodes": (nnodes, 1),
        "gpus_per_node": (gpus_per_node, GPUS_PER_RUN),
        "per_device_batch_size": (
            per_device_batch_size,
            PER_DEVICE_BATCH_SIZE,
        ),
        "gradient_accumulation_steps": (gradient_accumulation_steps, 1),
        "seed": (seed, SEED),
    }
    problems = [
        f"{name}={actual} (expected {expected})"
        for name, (actual, expected) in values.items()
        if actual != expected
    ]
    global_batch = (
        nnodes
        * gpus_per_node
        * per_device_batch_size
        * gradient_accumulation_steps
    )
    if global_batch != GLOBAL_BATCH_SIZE:
        problems.append(
            f"global_batch_size={global_batch} (expected {GLOBAL_BATCH_SIZE})"
        )
    if max_steps <= 0 or save_steps <= 0 or max_steps % save_steps != 0:
        problems.append(
            "max_steps and save_steps must be positive with "
            "max_steps divisible by save_steps"
        )
    if problems:
        raise ValueError("formal HACO training contract violated: " + "; ".join(problems))


def validate_continuation_training_values(
    *,
    nnodes: int,
    gpus_per_node: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    max_steps: int,
    save_steps: int,
    seed: int,
) -> None:
    """Validate the explicit 30k-to-50k, eight-GPU continuation contract."""

    values = {
        "nnodes": (nnodes, 1),
        "gpus_per_node": (gpus_per_node, CONTINUATION_GPUS_PER_RUN),
        "per_device_batch_size": (per_device_batch_size, PER_DEVICE_BATCH_SIZE),
        "gradient_accumulation_steps": (gradient_accumulation_steps, 1),
        "max_steps": (max_steps, CONTINUATION_TARGET_STEPS),
        "save_steps": (save_steps, CONTINUATION_SAVE_STEPS),
        "seed": (seed, SEED),
    }
    problems = [
        f"{name}={actual} (expected {expected})"
        for name, (actual, expected) in values.items()
        if actual != expected
    ]
    global_batch = (
        nnodes
        * gpus_per_node
        * per_device_batch_size
        * gradient_accumulation_steps
    )
    if global_batch != CONTINUATION_GLOBAL_BATCH_SIZE:
        problems.append(
            "global_batch_size="
            f"{global_batch} (expected {CONTINUATION_GLOBAL_BATCH_SIZE})"
        )
    if problems:
        raise ValueError(
            "formal HACO continuation contract violated: " + "; ".join(problems)
        )


def validate_multitask_training_values(
    *,
    profile: str,
    dataset_count: int,
    nnodes: int,
    gpus_per_node: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    max_steps: int,
    save_steps: int,
    save_total_limit: int,
    seed: int,
) -> None:
    """Validate the formal five-task, two-node HACO training contract."""

    expected_steps = MULTITASK_TARGET_STEPS_BY_PROFILE.get(profile)
    values = {
        "dataset_count": (dataset_count, MULTITASK_TASK_COUNT),
        "nnodes": (nnodes, MULTITASK_NNODES),
        "gpus_per_node": (gpus_per_node, MULTITASK_GPUS_PER_NODE),
        "per_device_batch_size": (per_device_batch_size, PER_DEVICE_BATCH_SIZE),
        "gradient_accumulation_steps": (gradient_accumulation_steps, 1),
        "save_steps": (save_steps, MULTITASK_SAVE_STEPS),
        "save_total_limit": (save_total_limit, 4),
        "seed": (seed, SEED),
    }
    problems = [
        f"{name}={actual!r} (expected {expected!r})"
        for name, (actual, expected) in values.items()
        if actual != expected
    ]
    if expected_steps is None:
        problems.append(f"unsupported multi-task profile {profile!r}")
    elif max_steps != expected_steps:
        problems.append(f"max_steps={max_steps!r} (expected {expected_steps!r})")
    global_batch = (
        nnodes
        * gpus_per_node
        * per_device_batch_size
        * gradient_accumulation_steps
    )
    if global_batch != MULTITASK_GLOBAL_BATCH_SIZE:
        problems.append(
            "global_batch_size="
            f"{global_batch} (expected {MULTITASK_GLOBAL_BATCH_SIZE})"
        )
    if problems:
        raise ValueError(
            "formal HACO multi-task training contract violated: "
            + "; ".join(problems)
        )
