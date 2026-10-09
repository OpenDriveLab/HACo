"""Launch a HACO server and run one sample-dataset inference request."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import aiohttp
import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from dexterity.deploy.template.serialization import packb, unpackb
from dexterity.runtime.sharpa62 import DEPLOY_TO_MODEL_JOINT, MODEL_TO_DEPLOY_JOINT

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = ROOT / "dataset/sample"
DEFAULT_OUTPUT = ROOT / "outputs/quick_test"
OBSERVATION_SCHEMA = "sharpa_policy_observation.v3"


def _frame_jpeg(path: Path, frame_index: int) -> bytes:
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index == frame_index:
                image = Image.fromarray(frame.to_ndarray(format="rgb24"))
                output = io.BytesIO()
                image.save(output, format="JPEG", quality=95)
                return output.getvalue()
    raise IndexError(f"video {path} has no frame {frame_index}")


def _load_sample(dataset: Path, anchor: int) -> tuple[dict, dict[str, np.ndarray]]:
    info = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))
    if info.get("action_order") != ["wrist_obs18", "q_obs44", "q_cmp44", "delta_q44"]:
        raise ValueError("sample does not use the HACO q_obs/q_cmp action contract")
    prompt = json.loads(
        (dataset / "meta/tasks.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )["task"]
    table = pq.read_table(dataset / "data/chunk-000/episode_000000.parquet")
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    if anchor < 8 or anchor + 40 >= len(table):
        raise ValueError("anchor needs 8 history frames and a strict 40-frame future")
    with np.load(
        dataset / "sensors/episodes/episode_000000.npz", allow_pickle=False
    ) as source:
        sensor = {name: source[name].copy() for name in source.files}

    def video(name: str) -> bytes:
        return _frame_jpeg(
            dataset / f"videos/chunk-000/observation.images.{name}/episode_000000.mp4",
            anchor,
        )

    now_ns = time.time_ns()
    period_ns = round(1e9 / 30)
    history_time = now_ns + np.arange(-8, 0, dtype=np.int64) * period_ns
    state_wire = state[anchor].copy()
    state_wire[18:] = state_wire[18:][MODEL_TO_DEPLOY_JOINT]
    tau = sensor["tau"][anchor - 8 : anchor + 1][:, MODEL_TO_DEPLOY_JOINT]
    tau_valid = sensor["tau_valid_mask"][anchor - 8 : anchor + 1][
        :, MODEL_TO_DEPLOY_JOINT
    ]
    wrench = sensor["tactile_wrench"][anchor - 8 : anchor + 1]
    wrench_valid = sensor["tactile_wrench_valid_mask"][anchor - 8 : anchor + 1]
    # Dataset fingertips are right-first; public sensor pairs are left-first.
    wrench = np.concatenate((wrench[:, 5:], wrench[:, :5]), axis=1)
    wrench_valid = np.concatenate((wrench_valid[:, 5:], wrench_valid[:, :5]), axis=1)
    deformation = sensor["tactile_deformation"][anchor]
    deformation_valid = sensor["tactile_deformation_valid_mask"][anchor]

    def temporal(values: np.ndarray, valid: np.ndarray, split: int) -> dict:
        return {
            "history": {
                "left": values[:-1, :split],
                "right": values[:-1, split:],
                "timestamp_ns": history_time,
                "valid": {
                    "left": valid[:-1, :split],
                    "right": valid[:-1, split:],
                },
            },
            "current": {
                "left": values[-1, :split],
                "right": values[-1, split:],
                "timestamp_ns": now_ns,
                "valid": {
                    "left": valid[-1, :split],
                    "right": valid[-1, split:],
                },
            },
        }

    observation = {
        "schema": OBSERVATION_SCHEMA,
        "metadata_format_id": "pending",
        "session_id": "haco-quick-test",
        "request_id": 0,
        "timestamp_ns": now_ns,
        "prompt": prompt,
        "image": {
            key: {
                "history": [],
                "current": {
                    "encoding": "jpeg",
                    "data": video(dataset_name),
                    "timestamp_ns": now_ns,
                    "valid": True,
                },
            }
            for key, dataset_name in (
                ("ego_cam", "ego_view"),
                ("left_wrist_cam", "left_wrist_view"),
                ("right_wrist_cam", "right_wrist_view"),
            )
        },
        "state": {
            "history": None,
            "current": {
                "timestamp_ns": now_ns,
                "left_wrist": {
                    "joint": None,
                    "eef": state_wire[:9],
                    "eef_def": "absolute",
                },
                "right_wrist": {
                    "joint": None,
                    "eef": state_wire[9:18],
                    "eef_def": "absolute",
                },
                "hand_joint": {"left": state_wire[18:40], "right": state_wire[40:]},
                "valid": True,
            },
        },
        "sensor": {
            "tau": temporal(tau, tau_valid, 22),
            "wrench": temporal(wrench, wrench_valid, 5),
            "deformation": {
                "history": None,
                "current": {
                    "left": deformation[5:],
                    "right": deformation[:5],
                    "timestamp_ns": now_ns,
                    "valid": {
                        "left": deformation_valid[5:],
                        "right": deformation_valid[:5],
                    },
                },
            },
        },
        "execution_feedback": {
            "last_action_id": None,
            "executed_steps": 0,
            "success": True,
        },
    }
    future = slice(anchor, anchor + 40)
    truth = {
        "q_obs_action": np.concatenate(
            (action[future, :18], action[future, 18:62]), axis=-1
        ),
        "q_cmp_action": np.concatenate(
            (action[future, :18], action[future, 62:106]), axis=-1
        ),
        "delta_q": action[future, 106:150],
        "wrench": sensor["tactile_wrench"][future],
        "wrench_valid": sensor["tactile_wrench_valid_mask"][future],
    }
    return observation, truth


async def _wait_for_server(base_url: str, process: subprocess.Popen | None) -> dict:
    deadline = time.monotonic() + 300
    error: Exception | None = None
    async with aiohttp.ClientSession() as session:
        while time.monotonic() < deadline:
            if process is not None and process.poll() is not None:
                raise RuntimeError(f"HACO server exited with code {process.returncode}")
            try:
                async with session.get(base_url + "/metadata") as response:
                    if response.status == 200:
                        return unpackb(await response.read())
            except (aiohttp.ClientError, OSError) as current:
                error = current
            await asyncio.sleep(1)
    raise TimeoutError(f"server did not become ready: {error}")


async def _infer(base_url: str, observation: dict) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            base_url + "/reset",
            data=packb({"session_id": observation["session_id"], "request_id": 0}),
        ) as response:
            response.raise_for_status()
            reset = unpackb(await response.read())
        observation["metadata_format_id"] = reset["metadata_format"]["format_id"]
        async with session.ws_connect(base_url + "/infer", max_msg_size=64 << 20) as ws:
            await ws.send_bytes(packb(observation))
            message = await ws.receive(timeout=300)
            if message.type != aiohttp.WSMsgType.BINARY:
                raise RuntimeError(
                    f"server returned unexpected WebSocket message: {message}"
                )
            result = unpackb(message.data)
    if result.get("schema") != "sharpa_policy_action.v5":
        raise RuntimeError(f"inference failed: {result}")
    return result


def _render(result: dict, truth: dict[str, np.ndarray], output: Path) -> None:
    action = result["action"]
    pred_wire = np.concatenate(
        (
            action["left_wrist"]["eef"],
            action["right_wrist"]["eef"],
            action["hand_joint"]["left"],
            action["hand_joint"]["right"],
        ),
        axis=-1,
    ).astype(np.float32)
    pred_q_cmp = pred_wire.copy()
    pred_q_cmp[:, 18:] = pred_q_cmp[:, 18:][:, DEPLOY_TO_MODEL_JOINT]
    pred_delta = np.asarray(
        result["diagnostics"]["delta_q_diagnostic_rad_40x44"], dtype=np.float32
    )[:, DEPLOY_TO_MODEL_JOINT]
    pred_q_obs = pred_q_cmp.copy()
    pred_q_obs[:, 18:] -= pred_delta
    output.mkdir(parents=True, exist_ok=True)
    payload = output / "quick_test_render.npz"
    np.savez_compressed(
        payload,
        action_62d_gt=truth["q_obs_action"],
        action_62d_pred=pred_q_obs,
        delta_q_44d_gt=truth["delta_q"],
        delta_q_44d_pred=pred_delta,
        wrench_gt=truth["wrench"],
        wrench_valid_gt=truth["wrench_valid"],
        chunk_size=np.asarray(40),
        hip_pose_9d=np.asarray([[0, 0, 0, 1, 0, 0, 0, 1, 0]], dtype=np.float32),
    )
    subprocess.run(
        [
            os.environ.get("PYTHON_BIN", os.sys.executable),
            "-m",
            "dexterity.rendering.cli.paired_motion_video",
            "--input-npz",
            str(payload),
            "--out-dir",
            str(output),
            "--fps",
            "30",
        ],
        cwd=ROOT,
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--reference-repo", type=Path)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--anchor", type=int, default=8)
    parser.add_argument("--port", type=int, default=15501)
    parser.add_argument("--server-url", help="use an already-running HACO server")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process: subprocess.Popen | None = None
    base_url = (
        args.server_url.rstrip("/")
        if args.server_url
        else f"http://127.0.0.1:{args.port}"
    )
    if not args.server_url:
        if not all((args.checkpoint, args.backbone, args.reference_repo)):
            raise SystemExit(
                "--checkpoint, --backbone, and --reference-repo are required "
                "when quick_test launches the server"
            )
        env = os.environ.copy()
        env.update(
            {
                "HACO_REFERENCE_REPO": str(args.reference_repo),
                "HACO_DEPLOY_HOST": "127.0.0.1",
                "HACO_DEPLOY_PORT": str(args.port),
            }
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        log = (args.output_dir / "server.log").open("w", encoding="utf-8")
        process = subprocess.Popen(
            [
                "bash",
                str(ROOT / "scripts/deploy/haco/launch.sh"),
                str(args.checkpoint),
                str(args.backbone),
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        metadata = asyncio.run(_wait_for_server(base_url, process))
        observation, truth = _load_sample(args.dataset.resolve(), args.anchor)
        result = asyncio.run(_infer(base_url, observation))
        _render(result, truth, args.output_dir.resolve())
        summary = {
            "checkpoint_id": metadata.get("checkpoint_id"),
            "prompt": observation["prompt"],
            "frames": int(result["execution"]["action_length"]),
            "gt_video": "viz__gt_hand_motion.mp4",
            "pred_video": "viz__pred_hand_motion.mp4",
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)


if __name__ == "__main__":
    main()
