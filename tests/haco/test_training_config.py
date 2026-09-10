from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.train.haco.config import (
    EXPERIMENTS,
    RTC_PREFIX_WEIGHTS,
    get_experiment,
    validate_continuation_training_values,
    validate_fixed_training_values,
    validate_multitask_normalization,
    validate_multitask_training_values,
    validate_official_checkpoint,
    validate_training_dataset,
)


ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "scripts/launch/haco/matrix_ur_unscrew_cap.json"


def test_eleven_independent_haco_models_have_frozen_priority() -> None:
    assert len(EXPERIMENTS) == 11
    assert [item.launch_priority for item in EXPERIMENTS.values()] == list(
        range(1, 12)
    )
    assert get_experiment("haco").paper_id == "haco"
    assert get_experiment("wc_wo_wrist").camera_mode == "ego"


@pytest.mark.parametrize(
    "legacy_id",
    (
        "m5_full",
        "p1_vision_state",
        "p2_tactile_only",
        "p3_torque_only",
        "p4_separate",
        "ac2_compliance_only",
        "ac3_nominal_only",
        "f1_suffix",
        "f2_imgmem",
        "f3_physcross_ungated",
        "w2_ego_only",
    ),
)
def test_haco_rejects_every_legacy_experiment_id(legacy_id: str) -> None:
    with pytest.raises(ValueError, match="unknown HACO experiment"):
        get_experiment(legacy_id)


def test_active_compliance_action_contracts_are_genuinely_delta_free() -> None:
    full = get_experiment("haco")
    ac2 = get_experiment("ac_wo_intent_sup")
    ac3 = get_experiment("ac_wo_active_comp")
    assert (full.action_target, full.semantic_action_dim, full.has_delta_q) == (
        "q_compliance",
        106,
        True,
    )
    assert (ac2.action_target, ac2.semantic_action_dim, ac2.has_delta_q) == (
        "q_compliance",
        62,
        False,
    )
    assert (ac3.action_target, ac3.semantic_action_dim, ac3.has_delta_q) == (
        "q_nominal",
        62,
        False,
    )


def test_rtc_prefix_weights_lock_the_agreed_distribution() -> None:
    assert len(RTC_PREFIX_WEIGHTS) == 13
    assert RTC_PREFIX_WEIGHTS[10] == 2.0
    assert sum(value == 2.0 for value in RTC_PREFIX_WEIGHTS) == 1


def test_formal_training_values_accept_task_specific_30k_or_50k() -> None:
    validate_fixed_training_values(
        nnodes=1,
        gpus_per_node=4,
        per_device_batch_size=12,
        gradient_accumulation_steps=1,
        max_steps=30000,
        save_steps=10000,
        seed=42,
    )
    validate_fixed_training_values(
        nnodes=1,
        gpus_per_node=4,
        per_device_batch_size=12,
        gradient_accumulation_steps=1,
        max_steps=50000,
        save_steps=10000,
        seed=42,
    )
    with pytest.raises(ValueError, match="global_batch_size"):
        validate_fixed_training_values(
            nnodes=1,
            gpus_per_node=4,
            per_device_batch_size=11,
            gradient_accumulation_steps=1,
            max_steps=30000,
            save_steps=10000,
            seed=42,
        )


def test_formal_continuation_values_are_explicitly_eight_gpu_bs12() -> None:
    validate_continuation_training_values(
        nnodes=1,
        gpus_per_node=8,
        per_device_batch_size=12,
        gradient_accumulation_steps=1,
        max_steps=50000,
        save_steps=10000,
        seed=42,
    )
    with pytest.raises(ValueError, match="gpus_per_node=4"):
        validate_continuation_training_values(
            nnodes=1,
            gpus_per_node=4,
            per_device_batch_size=12,
            gradient_accumulation_steps=1,
            max_steps=50000,
            save_steps=10000,
            seed=42,
        )


def test_formal_multitask_values_lock_two_nodes_and_four_checkpoints() -> None:
    validate_multitask_training_values(
        profile="mt5_equal_2n16g_100k",
        dataset_count=5,
        nnodes=2,
        gpus_per_node=8,
        per_device_batch_size=12,
        gradient_accumulation_steps=1,
        max_steps=100000,
        save_steps=10000,
        save_total_limit=4,
        seed=42,
    )
    validate_multitask_training_values(
        profile="mt5_equal_2n16g_200k",
        dataset_count=5,
        nnodes=2,
        gpus_per_node=8,
        per_device_batch_size=12,
        gradient_accumulation_steps=1,
        max_steps=200000,
        save_steps=10000,
        save_total_limit=4,
        seed=42,
    )
    validate_multitask_training_values(
        profile="mt5_equal_2n16g_300k",
        dataset_count=5,
        nnodes=2,
        gpus_per_node=8,
        per_device_batch_size=12,
        gradient_accumulation_steps=1,
        max_steps=300000,
        save_steps=10000,
        save_total_limit=4,
        seed=42,
    )
    validate_multitask_training_values(
        profile="mt5_equal_2n16g_400k",
        dataset_count=5,
        nnodes=2,
        gpus_per_node=8,
        per_device_batch_size=12,
        gradient_accumulation_steps=1,
        max_steps=400000,
        save_steps=10000,
        save_total_limit=4,
        seed=42,
    )
    validate_multitask_training_values(
        profile="mt5_equal_2n16g_500k",
        dataset_count=5,
        nnodes=2,
        gpus_per_node=8,
        per_device_batch_size=12,
        gradient_accumulation_steps=1,
        max_steps=500000,
        save_steps=10000,
        save_total_limit=4,
        seed=42,
    )
    with pytest.raises(ValueError, match="save_total_limit=5"):
        validate_multitask_training_values(
            profile="mt5_equal_2n16g_100k",
            dataset_count=5,
            nnodes=2,
            gpus_per_node=8,
            per_device_batch_size=12,
            gradient_accumulation_steps=1,
            max_steps=100000,
            save_steps=10000,
            save_total_limit=5,
            seed=42,
        )


def test_multitask_normalization_requires_equal_task_provenance(
    tmp_path: Path,
) -> None:
    datasets = [
        {"task_id": f"task_{index}", "weight": 0.2} for index in range(5)
    ]
    (tmp_path / "stats.json").write_text("{}")
    (tmp_path / "sensor_stats.json").write_text(
        json.dumps(
            {
                "schema": "sharpa.sensor_normalization.multitask.v1",
                "split": "train",
            }
        )
    )
    (tmp_path / "stats_provenance.json").write_text(
        json.dumps(
            {
                "schema": "sharpa.haco_multitask_equal_stats_provenance.v1",
                "split": "train",
            }
        )
    )
    (tmp_path / "mixture_manifest.json").write_text(
        json.dumps(
            {
                "schema": "sharpa.haco_multitask_normalization.v1",
                "datasets": datasets,
            }
        )
    )
    assert validate_multitask_normalization(tmp_path) == tmp_path.resolve()


def test_official_checkpoint_validator_rejects_rtc_or_posttrain(tmp_path: Path) -> None:
    checkpoint = tmp_path / "official"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"model_type": "Gr00tN1d7", "action_horizon": 40})
    )
    (checkpoint / "processor_config.json").write_text("{}")
    assert validate_official_checkpoint(checkpoint) == checkpoint.resolve()

    forbidden = tmp_path / "logs" / "checkpoint"
    forbidden.mkdir(parents=True)
    (forbidden / "config.json").write_text(
        json.dumps({"model_type": "Gr00tN1d7", "action_horizon": 40})
    )
    (forbidden / "processor_config.json").write_text("{}")
    with pytest.raises(ValueError, match="task/RTC"):
        validate_official_checkpoint(forbidden)


def test_dataset_validator_enforces_full_train_split_and_no_validation(
    tmp_path: Path,
) -> None:
    meta = tmp_path / "meta"
    meta.mkdir()
    info = {
        "total_episodes": 115,
        "total_frames": 146323,
        "total_tasks": 1,
        "state_dim": 62,
        "action_dim": 150,
        "split_counts": {"train": 115, "val": 0, "test": 0},
        "splits": {"train": "0:115"},
        "action_order": [
            "wrist_exe_next18",
            "q_exe_next44",
            "q_teleop44",
            "delta_q44",
        ],
        "delta_q_definition": "q_teleop - q_exe_next",
    }
    (meta / "info.json").write_text(json.dumps(info))
    (meta / "stats.json").write_text("{}")
    (meta / "sensor_stats.json").write_text(json.dumps({"split": "train"}))
    (meta / "stats_provenance.json").write_text(json.dumps({"split": "train"}))
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "Demo task"}) + "\n"
    )
    assert validate_training_dataset(tmp_path) == tmp_path.resolve()
    info["split_counts"]["val"] = 1
    (meta / "info.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="split_counts"):
        validate_training_dataset(tmp_path)


def test_machine_matrix_matches_training_contracts_and_priority() -> None:
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    runs = matrix["experiments"]
    assert len(runs) == 11
    haco_runs = [item for item in runs if item["model_family"] == "haco"]
    assert len(haco_runs) == 11
    assert [item["experiment_id"] for item in haco_runs] == list(EXPERIMENTS)
    for item in haco_runs:
        expected = EXPERIMENTS[item["experiment_id"]]
        assert item["launch_priority"] == expected.launch_priority
        assert item["sensor_encoder_mode"] == expected.sensor_encoder_mode
        assert item["physical_integration"] == expected.physical_integration
        assert item["action_contract"] == expected.action_contract
        assert item["action_target"] == expected.action_target
        assert item["semantic_action_dim"] == expected.semantic_action_dim
        assert item["camera_mode"] == expected.camera_mode
        assert item["rtc_enabled"] is True
        assert (ROOT / item["launcher"]).is_file()
