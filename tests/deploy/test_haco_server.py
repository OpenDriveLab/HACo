from __future__ import annotations

from copy import deepcopy
import io
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image
import pytest
import torch

from dexterity.deploy.models.haco.server import (
    HacoAdapter,
    HacoCheckpointSpec,
    HacoPolicy,
    HacoRequestBuilder,
    metadata_format,
    parse_args,
)
from dexterity.deploy.template.protocol import SharpAObservation
from dexterity.models.haco.processor import HacoProcessor
from dexterity.runtime.sharpa62 import model_joints_to_wire


HACO_VARIANTS = (
    (
        "haco", "fused", "physcross_gated",
        "joint_compliance_delta", "q_compliance", "three",
    ),
    (
        "hp_wo_haptic", "none", "physcross_gated",
        "joint_compliance_delta", "q_compliance", "three",
    ),
    (
        "hp_wo_torque", "tactile_only", "physcross_gated",
        "joint_compliance_delta", "q_compliance", "three",
    ),
    (
        "hp_wo_tactile", "torque_only", "physcross_gated",
        "joint_compliance_delta", "q_compliance", "three",
    ),
    (
        "hp_wo_coupled_en", "separate", "physcross_gated",
        "joint_compliance_delta", "q_compliance", "three",
    ),
    (
        "ac_wo_intent_sup", "fused", "physcross_gated",
        "compliance_only", "q_compliance", "three",
    ),
    (
        "ac_wo_active_comp", "fused", "physcross_gated",
        "nominal_only", "q_nominal", "three",
    ),
    (
        "cg_action_suf", "fused", "suffix",
        "joint_compliance_delta", "q_compliance", "three",
    ),
    (
        "cg_visuo_haptic", "fused", "imgmem",
        "joint_compliance_delta", "q_compliance", "three",
    ),
    (
        "cg_ungated_comp_attn", "fused", "physcross_ungated",
        "joint_compliance_delta", "q_compliance", "three",
    ),
    (
        "wc_wo_wrist", "fused", "physcross_gated",
        "joint_compliance_delta", "q_compliance", "ego",
    ),
)


def _spec(**overrides: Any) -> HacoCheckpointSpec:
    values = {
        "experiment_id": "haco",
        "sensor_encoder_mode": "fused",
        "physical_integration": "physcross_gated",
        "action_contract": "joint_compliance_delta",
        "action_target": "q_compliance",
        "camera_mode": "three",
        "rtc_steps": 10,
    }
    values.update(overrides)
    return HacoCheckpointSpec(**values)


def _camera() -> dict[str, Any]:
    return {
        "encoding": "jpeg",
        "data": b"jpeg",
        "timestamp_ns": 1,
        "valid": True,
    }


def _sensor_current(value: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    return {
        "timestamp_ns": 1,
        "left": value.copy(),
        "right": (value + 100).copy(),
        "valid": {"left": valid.copy(), "right": valid.copy()},
    }


def _sensor_history(value: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    return {
        "timestamp_ns": np.arange(8, dtype=np.int64),
        "left": value.copy(),
        "right": (value + 100).copy(),
        "valid": {"left": valid.copy(), "right": valid.copy()},
    }


def _observation(
    spec: HacoCheckpointSpec,
    *,
    request_id: int = 0,
    last_action_id: str | None = None,
    executed_steps: int = 0,
) -> SharpAObservation:
    tau_history = np.arange(8 * 22, dtype=np.float32).reshape(8, 22)
    tau_current = np.arange(22, dtype=np.float32)
    wrench_history = np.arange(8 * 5 * 6, dtype=np.float32).reshape(8, 5, 6)
    wrench_current = np.arange(5 * 6, dtype=np.float32).reshape(5, 6)
    deformation = np.zeros((5, 240, 240), dtype=np.uint8)
    return cast(
        SharpAObservation,
        {
            "schema": "sharpa_policy_observation.v3",
            "metadata_format_id": metadata_format(spec)["format_id"],
            "session_id": "episode",
            "request_id": request_id,
            "timestamp_ns": request_id + 1,
            "prompt": "unscrew the cap",
            "image": {
                name: {"history": [], "current": _camera()}
                for name in ("ego_cam", "left_wrist_cam", "right_wrist_cam")
            },
            "state": {
                "history": None,
                "current": {
                    "timestamp_ns": request_id + 1,
                    "left_wrist": {
                        "joint": None,
                        "eef": np.zeros(9, dtype=np.float32),
                        "eef_def": "absolute",
                    },
                    "right_wrist": {
                        "joint": None,
                        "eef": np.zeros(9, dtype=np.float32),
                        "eef_def": "absolute",
                    },
                    "hand_joint": {
                        "left": np.zeros(22, dtype=np.float32),
                        "right": np.zeros(22, dtype=np.float32),
                    },
                    "valid": True,
                },
            },
            "sensor": {
                "tau": {
                    "history": _sensor_history(
                        tau_history, np.ones((8, 22), dtype=bool)
                    ),
                    "current": _sensor_current(
                        tau_current, np.ones(22, dtype=bool)
                    ),
                },
                "wrench": {
                    "history": _sensor_history(
                        wrench_history, np.ones((8, 5), dtype=bool)
                    ),
                    "current": _sensor_current(
                        wrench_current, np.ones(5, dtype=bool)
                    ),
                },
                "deformation": {
                    "history": None,
                    "current": _sensor_current(
                        deformation, np.ones(5, dtype=bool)
                    ),
                },
            },
            "execution_feedback": {
                "last_action_id": last_action_id,
                "executed_steps": executed_steps,
                "success": True,
            },
        },
    )


def _write_checkpoint(
    path: Path,
    *,
    experiment_id: str = "haco",
    sensor_encoder_mode: str = "fused",
    physical_integration: str = "physcross_gated",
    action_contract: str = "joint_compliance_delta",
    action_target: str = "q_compliance",
    camera_mode: str = "three",
) -> None:
    path.mkdir()
    config = {
        "model_type": "Haco",
        "haco_schema": "haco.checkpoint.v1",
        "experiment_id": experiment_id,
        "sensor_encoder_mode": sensor_encoder_mode,
        "physical_integration": physical_integration,
        "action_contract": action_contract,
        "action_target": action_target,
        "camera_mode": camera_mode,
        "rtc_inference_prefix_steps": 10,
        "action_horizon": 40,
        "max_action_dim": 132,
    }
    processor = {
        "processor_kwargs": {
            key: config[key]
            for key in (
                "experiment_id",
                "sensor_encoder_mode",
                "physical_integration",
                "action_contract",
                "camera_mode",
            )
        }
    }
    processor["processor_kwargs"].update(
        {"max_action_horizon": 40, "max_action_dim": 132}
    )
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "processor_config.json").write_text(
        json.dumps(processor), encoding="utf-8"
    )
    (path / "statistics.json").write_text("{}", encoding="utf-8")


def test_checkpoint_spec_is_checkpoint_owned_and_strict(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)
    spec = HacoCheckpointSpec.from_checkpoint(checkpoint)
    assert spec == _spec()

    processor_path = checkpoint / "processor_config.json"
    processor = json.loads(processor_path.read_text(encoding="utf-8"))
    processor["processor_kwargs"]["action_contract"] = "nominal_only"
    processor_path.write_text(json.dumps(processor), encoding="utf-8")
    with pytest.raises(ValueError, match="model/processor action_contract mismatch"):
        HacoCheckpointSpec.from_checkpoint(checkpoint)


@pytest.mark.parametrize(
    (
        "experiment_id",
        "sensor_encoder_mode",
        "physical_integration",
        "action_contract",
        "action_target",
        "camera_mode",
    ),
    HACO_VARIANTS,
)
def test_checkpoint_spec_accepts_every_frozen_haco_variant(
    tmp_path: Path,
    experiment_id: str,
    sensor_encoder_mode: str,
    physical_integration: str,
    action_contract: str,
    action_target: str,
    camera_mode: str,
) -> None:
    checkpoint = tmp_path / experiment_id
    _write_checkpoint(
        checkpoint,
        experiment_id=experiment_id,
        sensor_encoder_mode=sensor_encoder_mode,
        physical_integration=physical_integration,
        action_contract=action_contract,
        action_target=action_target,
        camera_mode=camera_mode,
    )
    spec = HacoCheckpointSpec.from_checkpoint(checkpoint)
    assert spec.experiment_id == experiment_id
    assert spec.sensor_encoder_mode == sensor_encoder_mode
    assert spec.physical_integration == physical_integration
    assert spec.action_contract == action_contract
    assert spec.action_target == action_target
    assert spec.camera_mode == camera_mode


def test_checkpoint_spec_rejects_an_experiment_id_contract_mismatch(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "bad-p1"
    _write_checkpoint(checkpoint, experiment_id="hp_wo_haptic")
    with pytest.raises(ValueError, match="frozen hp_wo_haptic contract"):
        HacoCheckpointSpec.from_checkpoint(checkpoint)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("model_type", "PaceV4", "model_type"),
        ("haco_schema", "dreamzero.haco_checkpoint.v1", "haco_schema"),
        ("experiment_id", "m5_full", "unknown HACO experiment"),
    ),
)
def test_checkpoint_spec_rejects_legacy_pace_v4_identity(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    checkpoint = tmp_path / field
    _write_checkpoint(checkpoint)
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config[field] = value
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        HacoCheckpointSpec.from_checkpoint(checkpoint)


def test_processor_accepts_only_the_auto_processor_dispatch_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "processor_config.json").write_text(
        json.dumps({"processor_kwargs": {}}), encoding="utf-8"
    )
    (tmp_path / "statistics.json").write_text("{}", encoding="utf-8")
    received: dict[str, Any] = {}

    def fake_init(self, **kwargs: Any) -> None:
        received.update(kwargs)

    monkeypatch.setattr(HacoProcessor, "__init__", fake_init)
    HacoProcessor.from_pretrained(tmp_path, _from_auto=True)
    assert "_from_auto" not in received

    with pytest.raises(TypeError, match="unsupported HACO processor overrides"):
        HacoProcessor.from_pretrained(tmp_path, unsupported=True)


def test_metadata_is_minimal_for_each_checkpoint_variant() -> None:
    full = metadata_format(_spec())
    assert full["image"]["left_wrist_cam"]["current"] is True
    assert full["sensor"]["tau"] == {"history_len": 8, "current": True}
    assert full["sensor"]["wrench"] == {"history_len": 8, "current": True}
    assert full["sensor"]["deformation"] == {"history_len": 0, "current": True}

    vision = metadata_format(
        _spec(experiment_id="hp_wo_haptic", sensor_encoder_mode="none")
    )
    assert all(
        value == {"history_len": 0, "current": False}
        for value in vision["sensor"].values()
    )
    ego = metadata_format(_spec(experiment_id="w2_ego", camera_mode="ego"))
    assert ego["image"]["left_wrist_cam"]["current"] is False
    assert ego["image"]["right_wrist_cam"]["current"] is False


def test_adapter_advertises_the_first_rtc_execution_window() -> None:
    spec = _spec(sensor_encoder_mode="none")
    adapter = HacoAdapter(cast(Any, object()), spec)
    assert adapter.interface_metadata()["execution"] == {
        "frequency_hz": 30.0,
        "action_length": 40,
        "execute_start": 0,
        "execute_length": 40,
        "rtc": 10,
    }


def test_validate_only_is_an_explicit_server_mode() -> None:
    args = parse_args(
        [
            "--checkpoint",
            "/checkpoint",
            "--reference-repo",
            "/reference",
            "--backbone-model",
            "/backbone",
            "--validate-only",
        ]
    )
    assert args.validate_only is True


def test_request_builder_builds_physical_inputs_and_enforces_rtc_timing() -> None:
    spec = _spec()
    builder = HacoRequestBuilder(spec)
    first = builder.build(_observation(spec))
    assert first["force_history"].shape == (44, 9)
    assert first["tactile_wrench_history"].shape == (10, 9, 6)
    assert first["tactile_deformation"].shape == (10, 240, 240)
    assert int(first["rtc/prefix_steps"]) == 0
    np.testing.assert_array_equal(
        first["tactile_wrench_history"][0, 0],
        np.arange(6, dtype=np.float32) + 100,
    )

    action = np.zeros((40, 62), dtype=np.float32)
    builder.record_result(
        {"execute_start": 0, "execute_stop": 40, "rtc": 10},
        action,
        "episode:chunk:0",
    )
    steady = builder.build(
        _observation(
            spec,
            request_id=1,
            last_action_id="episode:chunk:0",
            executed_steps=30,
        )
    )
    assert int(steady["rtc/prefix_steps"]) == 10

    builder.record_result(
        {"execute_start": 10, "execute_stop": 40, "rtc": 10},
        action,
        "episode:chunk:1",
    )
    with pytest.raises(ValueError, match="expected executed_steps=20"):
        builder.build(
            _observation(
                spec,
                request_id=2,
                last_action_id="episode:chunk:1",
                executed_steps=30,
            )
        )


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (1, 2, 3)).save(buffer, format="JPEG")
    return buffer.getvalue()


class _FakeModelPolicy:
    def __init__(self) -> None:
        self.options: list[dict[str, Any]] = []

    def reset(self, info: dict[str, Any]) -> str:
        self.options.clear()
        return "ok"

    def get_action(self, observation: dict[str, Any], options: dict[str, Any]):
        self.options.append(deepcopy(options))
        prefix_steps = int(options["rtc_prefix_steps"])
        normalized = torch.zeros(1, 40, 132)
        if prefix_steps:
            normalized[:, :prefix_steps] = options["rtc_prefix_action"]
        q = np.broadcast_to(np.arange(44, dtype=np.float32), (1, 40, 44)).copy()
        nominal = q + 200.0
        delta = np.full((1, 40, 44), 1000.0, dtype=np.float32)
        wrist = np.zeros((1, 40, 9), dtype=np.float32)
        return (
            {
                "left_wrist_eef": wrist,
                "right_wrist_eef": wrist,
                "left_hand_q_teleop": q[..., :22],
                "right_hand_q_teleop": q[..., 22:],
                "left_hand_joints": nominal[..., :22],
                "right_hand_joints": nominal[..., 22:],
                "left_hand_delta_q": delta[..., :22],
                "right_hand_delta_q": delta[..., 22:],
            },
            {"normalized_action": normalized},
        )


def _model_request(prefix_steps: int) -> dict[str, Any]:
    image = _jpeg()
    return {
        "session_id": "episode",
        "prompt": "unscrew",
        "observation/hand_pose_62d": np.zeros(62, dtype=np.float32),
        "observation/ego_view_jpeg": image,
        "observation/left_wrist_view_jpeg": image,
        "observation/right_wrist_view_jpeg": image,
        "rtc/prefix_steps": np.asarray(prefix_steps, dtype=np.int64),
    }


def test_policy_executes_main_q_directly_and_keeps_full_normalized_rtc() -> None:
    low_level = _FakeModelPolicy()
    policy = HacoPolicy(low_level, _spec(sensor_encoder_mode="none"))
    first = policy.infer(_model_request(0))

    expected_q = np.broadcast_to(
        model_joints_to_wire(np.arange(44, dtype=np.float32)), (40, 44)
    )
    np.testing.assert_array_equal(first["action_chunk_62d"][:, 18:], expected_q)
    assert np.max(first["action_chunk_62d"][:, 18:]) < 1000
    assert first["diagnostics"]["delta_q_execution"] == "never_added"
    assert first["diagnostics"]["delta_q_diagnostic_rad_40x44"].shape == (40, 44)
    assert tuple(policy.last_normalized.shape) == (1, 40, 132)

    second = policy.infer(_model_request(10))
    assert int(second["execute_start"]) == 10
    prefix = low_level.options[1]["rtc_prefix_action"]
    assert tuple(prefix.shape) == (1, 10, 132)
    torch.testing.assert_close(prefix, policy.last_normalized[:, -10:])


@pytest.mark.parametrize(
    ("action_contract", "action_target", "q_offset"),
    (
        ("compliance_only", "q_compliance", 0.0),
        ("nominal_only", "q_nominal", 200.0),
    ),
)
def test_delta_free_contracts_execute_their_declared_q_directly(
    action_contract: str,
    action_target: str,
    q_offset: float,
) -> None:
    low_level = _FakeModelPolicy()
    policy = HacoPolicy(
        low_level,
        _spec(
            experiment_id=(
                "ac_wo_intent_sup"
                if action_contract == "compliance_only"
                else "ac_wo_active_comp"
            ),
            sensor_encoder_mode="fused",
            action_contract=action_contract,
            action_target=action_target,
        ),
    )
    result = policy.infer(_model_request(0))
    expected = np.broadcast_to(
        model_joints_to_wire(np.arange(44, dtype=np.float32) + q_offset),
        (40, 44),
    )
    np.testing.assert_array_equal(result["action_chunk_62d"][:, 18:], expected)
    assert "delta_q_diagnostic_rad_40x44" not in result["diagnostics"]
