#!/usr/bin/env python3
"""Shared helpers for deterministic pick/place baseline evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from dexterity.data.posttrain import Anchor, PickPlaceStore


def anchors_for_split(store: PickPlaceStore, split: str) -> list[Anchor]:
    """Return all valid anchors for an explicit dataset split."""
    anchors: list[Anchor] = []
    episode_count = 0
    for episode in sorted(store.episodes.values(), key=lambda item: item.episode_index):
        if episode.split != split:
            continue
        episode_count += 1
        with np.load(store.sidecar_path(episode.episode_index)) as sidecar:
            frame_indices = np.flatnonzero(sidecar["anchor_valid"])
        anchors.extend(
            Anchor(episode.episode_index, int(frame_index))
            for frame_index in frame_indices
        )
    if not anchors:
        raise ValueError(f"split {split!r} has no valid anchors")
    print(
        f"[eval] split={split} episodes={episode_count} anchors={len(anchors)}",
        flush=True,
    )
    return anchors


def evenly_spaced(items: list[Any], limit: int) -> list[Any]:
    """Choose a stable subset spanning the full ordered collection."""
    if limit <= 0 or limit >= len(items):
        return list(items)
    indices = np.linspace(0, len(items) - 1, num=limit, dtype=np.int64)
    return [items[int(index)] for index in indices]


def write_results(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = payload["results"]
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (output_dir / "eval_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_step_paths(values: Iterable[str]) -> list[tuple[int, Path]]:
    parsed: list[tuple[int, Path]] = []
    for value in values:
        step_text, separator, path_text = value.partition("=")
        if not separator:
            raise ValueError(f"checkpoint must use STEP=PATH syntax: {value}")
        step = int(step_text)
        path = Path(path_text).resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        parsed.append((step, path))
    return sorted(parsed)
