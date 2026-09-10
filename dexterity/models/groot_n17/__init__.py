"""GR00T N1.7 peer integration helpers.

This package is intentionally lightweight: importing it must not import the
official Isaac-GR00T stack. Runtime bridges live under ``scripts/inference``.
"""

from .contract import (
    DROID_CONTRACT,
    DROID_EMBODIMENT_TAG_NAME,
    DROID_EMBODIMENT_TAG_VALUE,
    load_checkpoint_modality_config,
    summarize_checkpoint_contract,
    validate_droid_checkpoint_contract,
)
__all__ = [
    "DROID_CONTRACT",
    "DROID_EMBODIMENT_TAG_NAME",
    "DROID_EMBODIMENT_TAG_VALUE",
    "load_checkpoint_modality_config",
    "summarize_checkpoint_contract",
    "validate_droid_checkpoint_contract",
]
