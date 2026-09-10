#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time

from .sensor_dataset import ensure_tactile_cache, sensor_cache_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build node-local mmap caches for compressed HACO tactile frames."
    )
    parser.add_argument("--dataset-path", action="append", required=True)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    jobs: list[tuple[Path, Path]] = []
    for root_value in args.dataset_path:
        root = Path(root_value).resolve()
        sources = sorted((root / "sensors/episodes").glob("episode_*.npz"))
        if not sources:
            raise FileNotFoundError(f"no HACO sensor episodes under {root}")
        jobs.extend((root, source) for source in sources)

    start = time.monotonic()
    completed_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(ensure_tactile_cache, root, source): (root, source)
            for root, source in jobs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            target = future.result()
            completed_bytes += target.stat().st_size
            if completed == len(jobs) or completed % 10 == 0:
                print(
                    f"HACO sensor cache: {completed}/{len(jobs)} episodes, "
                    f"{completed_bytes / 2**30:.1f} GiB ready",
                    flush=True,
                )

    roots = sorted({str(sensor_cache_root(root)) for root, _ in jobs})
    print(
        f"HACO sensor cache ready in {time.monotonic() - start:.1f}s: "
        + ", ".join(roots),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
