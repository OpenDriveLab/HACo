from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


def resolve_local_model_path(
    model_name: str | Path, transformers_loading_kwargs: dict[str, Any]
) -> str:
    model_path = Path(model_name).expanduser()
    if model_path.exists():
        return str(model_path.resolve())
    if not transformers_loading_kwargs.get("local_files_only", False):
        return str(model_name)
    cache_dir = transformers_loading_kwargs.get("cache_dir")
    if cache_dir is None:
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        cache_dir = hf_home / "hub"
    try:
        return snapshot_download(
            repo_id=str(model_name),
            cache_dir=str(cache_dir),
            revision=transformers_loading_kwargs.get("revision"),
            local_files_only=True,
        )
    except Exception as error:
        raise FileNotFoundError(
            f"local Hugging Face snapshot not found for {model_name!s} in {cache_dir}"
        ) from error
