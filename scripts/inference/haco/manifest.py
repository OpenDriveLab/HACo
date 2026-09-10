#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.inference.haco.protocol import FormalEvalProtocol, build_manifest


ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the fixed 115-episode HACO RTC open-loop manifest."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "datasets/postrain/ur-sharpa/lerobot_data/unscrew_cap",
    )
    parser.add_argument(
        "--action-contract",
        required=True,
        choices=(
            "joint_compliance_delta",
            "compliance_only",
            "nominal_only",
        ),
    )
    parser.add_argument(
        "--action-target", required=True, choices=("q_cmp", "q_obs")
    )
    parser.add_argument("--checkpoint-step", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = FormalEvalProtocol(
        action_contract=args.action_contract,
        action_target=args.action_target,
        checkpoint_step=args.checkpoint_step,
        eval_seed=args.seed,
    )
    manifest = build_manifest(args.dataset, protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
