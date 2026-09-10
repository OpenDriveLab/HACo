"""External model-base integration used by HACO."""

import json
from pathlib import Path

from gr00t.model.gr00t_n1d7.gr00t_n1d7 import (
    Gr00tN1d7,
    Gr00tN1d7ActionHead,
    get_backbone_cls as _official_get_backbone_cls,
)


def get_backbone_cls(config):
    """Resolve the backbone from either a Hub id or a local snapshot.

    The external implementation recognizes its canonical Hub id. A local
    Qwen3-VL snapshot is equivalent and is detected from ``config.json``.
    """

    try:
        return _official_get_backbone_cls(config)
    except ValueError:
        model_path = Path(str(config.model_name)).expanduser()
        config_path = model_path / "config.json"
        if config_path.is_file():
            model_config = json.loads(config_path.read_text(encoding="utf-8"))
            if model_config.get("model_type") == "qwen3_vl":
                from gr00t.model.modules.qwen3_backbone import Qwen3Backbone

                return Qwen3Backbone
        raise

__all__ = [
    "Gr00tN1d7",
    "Gr00tN1d7ActionHead",
    "get_backbone_cls",
]
