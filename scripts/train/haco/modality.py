"""Dynamic UR-SharpA modality contract for the frozen HACO matrix.

The selected experiment is resolved once at process startup. In particular,
the two active-compliance ablations register genuine 62-D action modalities:
delta-q is absent from their action keys, statistics, processor input, head
target, and loss mask. ``wc_wo_wrist`` removes wrist cameras entirely.
"""

from __future__ import annotations

import os

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

from scripts.train.groot_n17.embodiment import (
    register_sharpa_absolute_eef_embodiment,
)
from scripts.train.haco.config import get_experiment


EMBODIMENT_TAG = os.environ.get(
    "HACO_EMBODIMENT_TAG", "real_r1_pro_sharpa_absolute_eef"
)
EXPERIMENT_ID = os.environ.get("HACO_EXPERIMENT_ID", "haco")
EXPERIMENT = get_experiment(EXPERIMENT_ID)

if EMBODIMENT_TAG == "real_r1_pro_sharpa_absolute_eef":
    register_sharpa_absolute_eef_embodiment()


def _action_groups() -> tuple[list[str], list[str]]:
    wrist = ["left_wrist_eef", "right_wrist_eef"]
    if EXPERIMENT.action_contract == "joint_compliance_delta":
        return (
            [
                *wrist,
                "left_hand_q_teleop",
                "right_hand_q_teleop",
                "left_hand_delta_q",
                "right_hand_delta_q",
            ],
            ["left_hand_delta_q", "right_hand_delta_q"],
        )
    if EXPERIMENT.action_contract == "compliance_only":
        return (
            [*wrist, "left_hand_q_teleop", "right_hand_q_teleop"],
            [],
        )
    if EXPERIMENT.action_contract == "nominal_only":
        return (
            [*wrist, "left_hand_joints", "right_hand_joints"],
            [],
        )
    raise AssertionError(EXPERIMENT.action_contract)


ACTION_GROUPS, MEAN_STD_GROUPS = _action_groups()
VIDEO_GROUPS = (
    ["ego_view", "left_wrist_view", "right_wrist_view"]
    if EXPERIMENT.camera_mode == "three"
    else ["ego_view"]
)


haco_ur_sharpa_config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=VIDEO_GROUPS),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_wrist_eef",
            "right_wrist_eef",
            "left_hand_joints",
            "right_hand_joints",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(40)),
        modality_keys=ACTION_GROUPS,
        mean_std_embedding_keys=MEAN_STD_GROUPS,
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
            ),
            *[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                )
                for _ in ACTION_GROUPS[2:]
            ],
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.language.task_description"],
    ),
}

register_modality_config(
    haco_ur_sharpa_config,
    embodiment_tag=EmbodimentTag.resolve(EMBODIMENT_TAG),
)


__all__ = [
    "ACTION_GROUPS",
    "EMBODIMENT_TAG",
    "EXPERIMENT",
    "VIDEO_GROUPS",
    "haco_ur_sharpa_config",
]
