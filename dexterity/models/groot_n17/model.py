"""Canonical GR00T N1.7 model base.

Derived policies import the base from this module instead of reaching into the
upstream checkout directly. This keeps the inheritance boundary visible in
our code while the implementation remains the official GR00T N1.7 model.
"""

import json
from pathlib import Path

from gr00t.model.gr00t_n1d7.gr00t_n1d7 import (
    Gr00tN1d7,
    Gr00tN1d7ActionHead,
    get_backbone_cls as _official_get_backbone_cls,
)


GrootN17 = Gr00tN1d7
GrootN17ActionHead = Gr00tN1d7ActionHead


def get_backbone_cls(config):
    """Resolve the official backbone from either a Hub id or a local mirror.

    Upstream only recognizes model names containing the original Hub id.  Our
    checkpoint layout intentionally stores the Cosmos mirror at
    ``checkpoints/cosmos_reason2_2b``.  Treat a local Qwen3-VL config as the
    same official backbone without changing the path recorded in checkpoints.
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
    "GrootN17",
    "GrootN17ActionHead",
    "Gr00tN1d7",
    "Gr00tN1d7ActionHead",
    "get_backbone_cls",
]
