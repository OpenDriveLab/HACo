"""Initialize an independent HACO model from official GR00T N1.7 weights."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import torch

from dexterity.models.haco import Haco
from scripts.train.haco.config import validate_official_checkpoint


def _loading_keys(loading_info: dict[str, Any], name: str) -> set[str]:
    values = loading_info.get(name, [])
    return {
        str(value[0]) if isinstance(value, (list, tuple)) else str(value)
        for value in values
    }


def load_official_as_haco(
    source_model_path: str | Path,
    *,
    config,
    transformers_loading_kwargs: dict[str, Any],
) -> tuple[Haco, dict[str, Any]]:
    """Load shared official weights and retain only intentional HACO extensions."""

    source = validate_official_checkpoint(source_model_path)
    # The model-owned loader validates the complete shared state dict and only
    # permits named HACO extension prefixes to be absent.  The official 132-D
    # The official action boundary is preserved in full; HACO semantic masking
    # selects the active 106-D or 62-D vocabulary.
    model, loading_info = Haco.from_official_checkpoint(
        source,
        config=config,
        output_loading_info=True,
        transformers_loading_kwargs=transformers_loading_kwargs,
        **transformers_loading_kwargs,
    )
    missing = _loading_keys(loading_info, "missing_keys")
    reinitialized = _loading_keys(
        loading_info, "reinitialized_haco_extension_keys"
    )
    unexpected = _loading_keys(loading_info, "unexpected_keys")
    mismatched = _loading_keys(loading_info, "mismatched_keys")

    if unexpected or mismatched:
        raise RuntimeError(
            "invalid official-GR00T-to-HACO load: "
            f"unexpected={sorted(unexpected)}, mismatched={sorted(mismatched)}"
        )
    if reinitialized != missing:
        raise RuntimeError(
            "not every HACO extension was explicitly initialized: "
            f"missing={sorted(missing)}, reinitialized={sorted(reinitialized)}"
        )
    model_tensors = {
        **dict(model.named_parameters()),
        **dict(model.named_buffers()),
    }
    extension_ranges: dict[str, float] = {}
    missing_tensors = sorted(key for key in missing if key not in model_tensors)
    nonfinite_tensors = []
    out_of_range_tensors = []
    for key in sorted(missing):
        tensor = model_tensors.get(key)
        if tensor is None or tensor.numel() == 0:
            continue
        finite = torch.isfinite(tensor)
        if not finite.all().item():
            nonfinite_tensors.append(key)
            continue
        max_abs = float(tensor.detach().abs().max().float().cpu())
        extension_ranges[key] = max_abs
        # Fresh Linear/Embedding/Norm/gate parameters in this architecture are
        # O(1). Values above 100 indicate uninitialized allocator contents,
        # which can still be finite while poisoning the first BF16 step.
        if max_abs > 100.0:
            out_of_range_tensors.append((key, max_abs))
    if missing_tensors or nonfinite_tensors or out_of_range_tensors:
        raise RuntimeError(
            "HACO extension initialization preflight failed: "
            f"missing={missing_tensors}, nonfinite={nonfinite_tensors}, "
            f"out_of_range={out_of_range_tensors}"
        )
    manifest = {
        "schema": "haco.initialization.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "source_model_type": "Gr00tN1d7",
        "source_is_task_posttrain": False,
        "target_model_type": config.model_type,
        "experiment_id": config.experiment_id,
        "action_contract": config.action_contract,
        "action_target": config.action_target,
        "missing_haco_extension_keys": sorted(missing),
        "reinitialized_haco_extension_keys": sorted(reinitialized),
        "haco_extension_init_seed": int(
            loading_info["haco_extension_init_seed"]
        ),
        "haco_extension_parameters_finite": bool(
            loading_info["haco_extension_parameters_finite"]
        ),
        "extension_parameter_max_abs": extension_ranges,
        "extension_preflight_max_abs_limit": 100.0,
        "shared_pretrained_keys_preserved": True,
        "official_action_boundary_132d_preserved": True,
    }
    return model, manifest


def write_initialization_manifest(
    path: str | Path, manifest: dict[str, Any]
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


__all__ = ["load_official_as_haco", "write_initialization_manifest"]
