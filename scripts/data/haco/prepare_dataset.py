"""Compute train-split statistics and validate a HACO LeRobot dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from dexterity.data.normalization import write_train_only_stats
from scripts.data.haco.validate_dataset import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    root = args.dataset.expanduser().resolve()
    write_train_only_stats(root)
    result = validate_dataset(root)
    print(
        f"prepared HACO dataset: {result['episodes']} episode(s), "
        f"{result['frames']} frames, {result['anchors']} training anchors"
    )


if __name__ == "__main__":
    main()
