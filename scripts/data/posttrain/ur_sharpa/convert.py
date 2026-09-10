#!/usr/bin/env python3
"""Convert timeline-based UR-SharpA captures to canonical LeRobot episodes."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.posttrain.ur_sharpa.contract import (  # noqa: E402
    CAMERA_NAMES,
    EpisodeSource,
    discover_episodes,
    load_canonical_episode,
    summarize_episode,
)
from scripts.data.posttrain.ur_sharpa.storage import (  # noqa: E402
    DATASET_SCHEMA,
    aggregate_report,
    episode_output_paths,
    publish_episode,
    verify_raw_checksums,
    write_json,
    write_metadata,
    write_rowwise_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_names",
        nargs="+",
        help="Dataset directory names under --raw-root.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="Directory containing one raw capture directory per dataset.",
    )
    parser.add_argument(
        "--task-text",
        required=True,
        help="Language instruction written to every converted episode.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination parent; each dataset keeps its input directory name.",
    )
    parser.add_argument("--first-n", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunks-size", type=int, default=1000)
    parser.add_argument("--video-crf", type=int, default=18)
    return parser.parse_args()


def _validate_name(value: str, *, label: str) -> str:
    name = value.strip()
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"{label} must be one directory name, got {value!r}")
    return name


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prepare_output(
    output: Path,
    *,
    source: Path,
    selected: list[EpisodeSource],
    task_text: str,
    chunks_size: int,
    video_crf: int,
) -> dict[str, Any]:
    state_path = output / "meta/conversion_state.json"
    expected = {
        "schema": DATASET_SCHEMA,
        "source": str(source.resolve()),
        "task_text": task_text,
        "chunks_size": chunks_size,
        "video_crf": video_crf,
        "selected_episodes": [item.name for item in selected],
    }
    if output.exists():
        if not state_path.is_file():
            raise FileExistsError(
                f"{output} exists but has no resumable conversion state; "
                "use a new output root"
            )
        existing = _load_json(state_path)
        for key, value in expected.items():
            if existing.get(key) != value:
                raise ValueError(f"cannot resume {output}: conversion state differs at {key}")
        return existing
    output.mkdir(parents=True)
    state = {
        **expected,
        "complete": False,
        "successful_episode_indices": [],
        "failures": [],
    }
    write_json(state_path, state)
    return state


def _convert_one(
    source: EpisodeSource,
    *,
    output: Path,
    chunks_size: int,
    video_crf: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checksum = verify_raw_checksums(source.path)
    episode = load_canonical_episode(source)
    summary = summarize_episode(episode)
    summary["checksum"] = checksum
    publish_episode(
        output,
        episode,
        summary,
        task_index=0,
        chunks_size=chunks_size,
        video_crf=video_crf,
    )
    return summary, checksum


def _committed_summary(
    output: Path, source: EpisodeSource, chunks_size: int
) -> dict[str, Any] | None:
    paths = episode_output_paths(output, source.output_episode_index, chunks_size)
    if not paths["commit"].is_file() or not paths["report"].is_file():
        return None
    commit = _load_json(paths["commit"])
    report = _load_json(paths["report"])
    if (
        commit.get("schema") != DATASET_SCHEMA
        or commit.get("source_episode") != source.name
        or int(commit.get("episode_index", -1)) != source.output_episode_index
        or report.get("source_episode") != source.name
    ):
        raise ValueError(f"{paths['commit']}: committed episode does not match source")
    for key in ("parquet", "sensors", *(f"video_{name}" for name in CAMERA_NAMES)):
        if not paths[key].is_file():
            raise FileNotFoundError(f"committed episode is missing {paths[key]}")
    return report


def convert_dataset(
    dataset_name: str,
    *,
    task_text: str,
    raw_root: Path,
    output_root: Path,
    first_n: int | None,
    workers: int,
    chunks_size: int,
    video_crf: int,
) -> Path:
    if workers < 1:
        raise ValueError("--workers must be positive")
    if chunks_size < 1:
        raise ValueError("--chunks-size must be positive")
    if not task_text.strip():
        raise ValueError("--task-text must not be empty")
    source = raw_root.expanduser().resolve() / dataset_name
    output = output_root.expanduser().resolve() / dataset_name
    selected = discover_episodes(source, first_n=first_n)
    state = _prepare_output(
        output,
        source=source,
        selected=selected,
        task_text=task_text.strip(),
        chunks_size=chunks_size,
        video_crf=video_crf,
    )

    summaries: dict[int, dict[str, Any]] = {}
    checksum_results: dict[str, dict[str, Any]] = {}
    pending: list[EpisodeSource] = []
    for item in selected:
        committed = _committed_summary(output, item, chunks_size)
        if committed is None:
            pending.append(item)
        else:
            summaries[item.output_episode_index] = committed
            checksum_results[item.name] = dict(committed.get("checksum", {}))
            print(
                f"[ur_sharpa] resume episode={item.output_episode_index:06d} "
                f"source={item.name}",
                flush=True,
            )

    failures: list[dict[str, Any]] = []
    if pending:
        with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as pool:
            futures = {
                pool.submit(
                    _convert_one,
                    item,
                    output=output,
                    chunks_size=chunks_size,
                    video_crf=video_crf,
                ): item
                for item in pending
            }
            for future in as_completed(futures):
                item = futures[future]
                try:
                    summary, checksum = future.result()
                except BaseException as error:
                    failure = {
                        "episode_index": item.output_episode_index,
                        "source_episode": item.name,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": "".join(
                            traceback.format_exception(type(error), error, error.__traceback__)
                        ),
                    }
                    failures.append(failure)
                    print(
                        f"[ur_sharpa] FAILED episode={item.output_episode_index:06d} "
                        f"source={item.name}: {error}",
                        flush=True,
                    )
                    continue
                summaries[item.output_episode_index] = summary
                checksum_results[item.name] = checksum
                print(
                    f"[ur_sharpa] episode={item.output_episode_index:06d} "
                    f"source={item.name} rows={summary['length']} "
                    f"anchors={summary['valid_anchor_count']}",
                    flush=True,
                )
                state["successful_episode_indices"] = sorted(summaries)
                state["failures"] = failures
                write_json(output / "meta/conversion_state.json", state)

    ordered = [summaries[index] for index in sorted(summaries)]
    report = aggregate_report(
        source=source,
        output=output,
        selected_count=len(selected),
        successes=ordered,
        failures=failures,
        checksum_results=checksum_results,
    )
    write_json(output / "meta/conversion_report.json", report)
    if failures or len(ordered) != len(selected):
        state["failures"] = failures
        write_json(output / "meta/conversion_state.json", state)
        raise RuntimeError(
            f"UR-SharpA conversion completed with {len(ordered)}/{len(selected)} successes; "
            f"see {output / 'meta/conversion_report.json'}"
        )

    write_metadata(
        output,
        source,
        ordered,
        task_text=task_text.strip(),
        chunks_size=chunks_size,
        fps=30,
    )
    provenance = write_rowwise_stats(output)
    from scripts.data.posttrain.ur_sharpa.verify import verify_dataset

    verification = verify_dataset(output, report_path=output / "meta/verification_report.json")
    state.update(
        {
            "complete": True,
            "successful_episode_indices": list(range(len(selected))),
            "failures": [],
            "stats_provenance": provenance,
            "verification_ok": bool(verification["ok"]),
        }
    )
    write_json(output / "meta/conversion_state.json", state)
    final_report = aggregate_report(
        source=source,
        output=output,
        selected_count=len(selected),
        successes=ordered,
        failures=[],
        checksum_results=checksum_results,
        verifier_report=str((output / "meta/verification_report.json").resolve()),
    )
    write_json(output / "meta/conversion_report.json", final_report)
    if not verification["ok"]:
        raise RuntimeError(f"verification failed: {output / 'meta/verification_report.json'}")
    print(
        json.dumps(
            {
                "dataset": dataset_name,
                "source_episodes": len(selected),
                "output_rows": final_report["total_output_rows"],
                "output": str(output.resolve()),
                "verification_report": str(
                    (output / "meta/verification_report.json").resolve()
                ),
            },
            indent=2,
        ),
        flush=True,
    )
    return output


def main() -> int:
    args = parse_args()
    names: list[str] = []
    for value in args.dataset_names:
        name = _validate_name(value, label="dataset name")
        if name in names:
            raise ValueError(f"duplicate dataset name: {name}")
        names.append(name)
    for name in names:
        convert_dataset(
            name,
            task_text=args.task_text,
            raw_root=args.raw_root,
            output_root=args.output_root,
            first_n=args.first_n,
            workers=args.workers,
            chunks_size=args.chunks_size,
            video_crf=args.video_crf,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
