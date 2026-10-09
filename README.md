<h1 align="center">HACo: Learning Haptic Active Compliance<br>for Force-Aware Dexterous Manipulation</h1>

<p align="center">
  Naisheng Ye<sup>1,2,†</sup>, Yinzhe Zhou<sup>2,3</sup>, Junkai Zhao<sup>2</sup>, Yuhang Lu<sup>1,2</sup>,<br>
  Checheng Yu<sup>1</sup>, Zhenjie Yang<sup>1</sup>, Pengwei Wang<sup>2</sup>, Hongyang Li<sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>The University of Hong Kong &nbsp;
  <sup>2</sup>Beijing Academy of Artificial Intelligence (BAAI) &nbsp;
  <sup>3</sup>Johns Hopkins University<br>
  <sup>†</sup>Work done during an internship at BAAI.
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2609.36596"><img src="https://img.shields.io/badge/arXiv-2609.36596-b31b1b" alt="arXiv"></a>
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white" alt="Python 3.10">
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="Apache 2.0 License"></a>
  <a href="https://opendrivelab.github.io/Haco-Page/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
</p>

**HACo** learns active compliance for dexterous manipulation by combining
fingertip touch with joint-torque measurements. Contact-sensitive tasks require
more than reproducing observed motion: the policy must adapt its commands to
the forces acting on the hand. We collect demonstrations using teleoperation
that regulates contact loads, then train HACo to predict the resulting
compliant commands. The difference between commanded and observed joint
positions supplies additional supervision for motion intent under contact.
Our haptic representation combines fingertip deformation and wrench signals
with torque feedback, capturing both local contact and loads transmitted
through the hand. Gated cross-attention connects this representation to action
generation. On five real-world tasks involving friction, tangential forces,
fragile surfaces, rotational torque, and deformable objects, HACo succeeds in
83% of trials on average, compared with 35% for the strongest evaluated baseline.

[Watch the demo video](https://opendrivelab.github.io/Haco-Page/)

## Get Started

### Setup

Use Linux, Python 3.10, CUDA 12.8, and [uv](https://docs.astral.sh/uv/getting-started/installation/).
FFmpeg is used to read and write videos.

```bash
sudo apt install ffmpeg
git clone https://github.com/OpenDriveLab/HACo.git
cd HACo
git clone https://github.com/NVIDIA/Isaac-GR00T.git third_party/Isaac-GR00T
git -C third_party/Isaac-GR00T checkout 4b1dca9d88d2a0b9ea5a65aa61c82ff89f5c4f0e
git -C third_party/Isaac-GR00T submodule update --init --recursive
uv sync --directory third_party/Isaac-GR00T --python 3.10
source third_party/Isaac-GR00T/.venv/bin/activate
uv pip install -e '.[deployment]'
```

HACo initializes from GR00T N1.7. Cosmos-Reason2 is its internal vision-language
backbone; the current loader also needs its files locally.

```bash
hf download nvidia/GR00T-N1.7-3B --local-dir checkpoints/base_model
hf download nvidia/Cosmos-Reason2-2B --local-dir checkpoints/cosmos_reason2_2b
```

### Dataset

Data follows the **LeRobot v2** format at **30 Hz**: state and action sequences
in Parquet, three RGB camera streams in MP4, haptic measurements in per-episode
NPZ files, and task descriptions and metadata in `meta/`.
See [dataset/sample](dataset/sample) for an example.

| Modality | Shape | Description |
| --- | --- | --- |
| State | `62` | Two wrist poses (18) and hand joint positions (44) |
| Action | `150` | Wrist poses (18), observed joints (44), compliant joint commands (44), and their difference (44) |
| Joint torque | `44` | Measured hand-joint torque |
| Tactile wrench | `10 × 6` | Force and torque at each fingertip |
| Tactile deformation | `10 × 240 × 240` | One uint8 deformation map per fingertip |
| RGB | Three streams | Ego, left wrist, and right wrist cameras |
| Language | Task description | Natural-language instruction |

Each action targets the next frame, with `delta_q = q_cmp - q_obs`.
Field layouts, sensor ordering, and normalization statistics are provided in
[dataset/sample/meta](dataset/sample/meta).

### Training

Weights & Biases is disabled by default. Optionally enable local logging before
starting training:

```bash
export WANDB_MODE=offline
```

For online logging, run `wandb login` and set `WANDB_MODE=online` instead.

```bash
export HACO_DATASET_PATH=/path/to/your/lerobot_dataset
bash scripts/launch/haco/haco.sh
```

The default run uses 4 GPUs, batch size 12 per GPU, and 30,000 steps, with
outputs saved to `logs/haco/`. Model paths default to the download locations above.

### Inference

Start the policy server with a trained HACo checkpoint:

```bash
bash scripts/deploy/haco/launch.sh \
  /path/to/haco-checkpoint checkpoints/cosmos_reason2_2b
```

The server returns 40-frame action chunks over `ws://localhost:5500/infer`.
See the [deployment guide](scripts/deploy/haco/README.md) for the client protocol.
