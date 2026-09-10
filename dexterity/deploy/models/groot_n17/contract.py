"""GR00T SharpA62 interactive policy: required runtime input and output.

Input (WebSocket binary msgpack, one inference request):
  schema: ``sharpa62_model_observation.v1``
  observation/{ego,left_wrist,right_wrist}_view_jpeg: JPEG bytes
  observation/hand_pose_62d: float32[62] in
      left_wrist9 | right_wrist9 | SharpA q44 order
  prompt: str; session_id: str

Output (WebSocket binary msgpack):
  schema: ``sharpa62_policy_action.v1``
  action_hand_pose_62d: float32[40,62], 30 Hz, absolute wrist EEF
  hand action: model-predicted q; no torque/tactile input and no residual controller
  XYZ direction conversion: none in the model server; owned by the robot

``reset`` requests contain endpoint="reset" and optional session_id and return a
text acknowledgement.  This module is the canonical catalog entry; the model
implementation is exposed by ``dexterity.deploy.models.groot_n17.server``.
"""

from __future__ import annotations

DEPLOY_STATUS = "ready"
DEFAULT_LAUNCHER = "scripts/deploy/groot_n17/launch.sh"
POLICY_CONTRACT = {
    "policy": "groot",
    "request_schema": "sharpa62_model_observation.v1",
    "required_inputs": {
        "ego_rgb": "JPEG bytes or uint8[H,W,3]",
        "left_wrist_rgb": "JPEG bytes or uint8[H,W,3]",
        "right_wrist_rgb": "JPEG bytes or uint8[H,W,3]",
        "state": "float32[62]",
        "prompt": "str",
        "session_id": "str",
    },
    "response_schema": "sharpa62_policy_action.v1",
    "action": "float32[40,62] at 30 Hz",
    "action_eef_def": "absolute",
    "server_xyz_direction_transform": "none",
    "execute_joint_source": "model_q",
}


def main() -> int:
    from dexterity.deploy.models.groot_n17.server import main as serve

    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
