from __future__ import annotations

SHARPA_SINGLE_VIEW_EMBODIMENT = "adam_pro_sharpa_relative_eef"
SHARPA_SINGLE_VIEW_ENUM_NAME = "ADAM_PRO_SHARPA_RELATIVE_EEF"
SHARPA_SINGLE_VIEW_PROJECTOR_ID = 26
SHARPA_ABSOLUTE_EEF_EMBODIMENT = "real_r1_pro_sharpa_absolute_eef"
SHARPA_ABSOLUTE_EEF_ENUM_NAME = "REAL_R1_PRO_SHARPA_ABSOLUTE_EEF"
SHARPA_ABSOLUTE_EEF_PROJECTOR_ID = 26


def _register_pretrain_embodiment(*, value: str, enum_name: str, projector_id: int):
    from gr00t.data import embodiment_tags as embodiment_tags_module

    embodiment_tag = embodiment_tags_module.EmbodimentTag
    member = embodiment_tag._value2member_map_.get(value)
    if member is None:
        member = object.__new__(embodiment_tag)
        member._name_ = enum_name
        member._value_ = value
        embodiment_tag._member_names_.append(enum_name)
        embodiment_tag._member_map_[enum_name] = member
        embodiment_tag._value2member_map_[value] = member

    embodiment_tags_module.PRETRAIN_TAGS = frozenset(
        {*embodiment_tags_module.PRETRAIN_TAGS, member}
    )

    from gr00t.model.gr00t_n1d7 import processing_gr00t_n1d7

    processing_gr00t_n1d7.EMBODIMENT_TAG_TO_PROJECTOR_INDEX[value] = projector_id
    return member


def register_sharpa_single_view_embodiment():
    """Register the SharpA single-view tag used by our N1.7 checkpoints."""
    return _register_pretrain_embodiment(
        value=SHARPA_SINGLE_VIEW_EMBODIMENT,
        enum_name=SHARPA_SINGLE_VIEW_ENUM_NAME,
        projector_id=SHARPA_SINGLE_VIEW_PROJECTOR_ID,
    )


def register_sharpa_absolute_eef_embodiment():
    """Register the official checkpoint's absolute-EEF SharpA tag."""
    return _register_pretrain_embodiment(
        value=SHARPA_ABSOLUTE_EEF_EMBODIMENT,
        enum_name=SHARPA_ABSOLUTE_EEF_ENUM_NAME,
        projector_id=SHARPA_ABSOLUTE_EEF_PROJECTOR_ID,
    )
