from __future__ import annotations

from pathlib import Path


def model_stats_path(
    model_name: str,
    dataset_root: str | Path,
    suffix: str,
) -> Path:
    """Return ``meta/<model_name>_stats<suffix>`` for a LeRobot dataset."""

    if not model_name or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_" for char in model_name):
        raise ValueError(f"invalid model_name: {model_name!r}")
    if not suffix.startswith("."):
        raise ValueError(f"stats suffix must start with '.', got {suffix!r}")
    return Path(dataset_root).resolve() / "meta" / f"{model_name}_stats{suffix}"


__all__ = ["model_stats_path"]
