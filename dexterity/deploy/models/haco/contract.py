"""Catalog contract for the trained-RTC HACO server."""

from __future__ import annotations


DEPLOY_STATUS = "ready"
DEFAULT_LAUNCHER = "scripts/deploy/haco/launch.sh"
EXPERIMENT_CONTRACTS = {
    "haco": {
        "sensor_encoder_mode": "fused",
        "physical_integration": "physcross_gated",
        "action_contract": "joint_compliance_delta",
        "action_target": "q_compliance",
        "camera_mode": "three",
    },
    "hp_wo_haptic": {
        "sensor_encoder_mode": "none",
        "physical_integration": "physcross_gated",
        "action_contract": "joint_compliance_delta",
        "action_target": "q_compliance",
        "camera_mode": "three",
    },
    "hp_wo_torque": {
        "sensor_encoder_mode": "tactile_only",
        "physical_integration": "physcross_gated",
        "action_contract": "joint_compliance_delta",
        "action_target": "q_compliance",
        "camera_mode": "three",
    },
    "hp_wo_tactile": {
        "sensor_encoder_mode": "torque_only",
        "physical_integration": "physcross_gated",
        "action_contract": "joint_compliance_delta",
        "action_target": "q_compliance",
        "camera_mode": "three",
    },
    "hp_wo_coupled_en": {
        "sensor_encoder_mode": "separate",
        "physical_integration": "physcross_gated",
        "action_contract": "joint_compliance_delta",
        "action_target": "q_compliance",
        "camera_mode": "three",
    },
    "ac_wo_intent_sup": {
        "sensor_encoder_mode": "fused",
        "physical_integration": "physcross_gated",
        "action_contract": "compliance_only",
        "action_target": "q_compliance",
        "camera_mode": "three",
    },
    "ac_wo_active_comp": {
        "sensor_encoder_mode": "fused",
        "physical_integration": "physcross_gated",
        "action_contract": "nominal_only",
        "action_target": "q_nominal",
        "camera_mode": "three",
    },
    "cg_action_suf": {
        "sensor_encoder_mode": "fused",
        "physical_integration": "suffix",
        "action_contract": "joint_compliance_delta",
        "action_target": "q_compliance",
        "camera_mode": "three",
    },
    "cg_visuo_haptic": {
        "sensor_encoder_mode": "fused",
        "physical_integration": "imgmem",
        "action_contract": "joint_compliance_delta",
        "action_target": "q_compliance",
        "camera_mode": "three",
    },
    "cg_ungated_comp_attn": {
        "sensor_encoder_mode": "fused",
        "physical_integration": "physcross_ungated",
        "action_contract": "joint_compliance_delta",
        "action_target": "q_compliance",
        "camera_mode": "three",
    },
    "wc_wo_wrist": {
        "sensor_encoder_mode": "fused",
        "physical_integration": "physcross_gated",
        "action_contract": "joint_compliance_delta",
        "action_target": "q_compliance",
        "camera_mode": "ego",
    },
}
POLICY_CONTRACT = {
    "policy": "haco",
    "observation_schema": "sharpa_policy_observation.v3",
    "action_schema": "sharpa_policy_action.v5",
    "action": "float32[40,62] at 30 Hz",
    "execution_semantics": "direct_main_q",
    "delta_q_semantics": "diagnostic_only_never_added",
    "rtc": {
        "prefix_steps": 10,
        "first_execute_start": 0,
        "steady_state_execute_start": 10,
    },
    "server_xyz_direction_transform": "none",
}


def main() -> int:
    from dexterity.deploy.models.haco.server import main as serve

    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
