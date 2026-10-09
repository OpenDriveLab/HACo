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
  <a href="https://opendrivelab.github.io/Haco-Page/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white" alt="Python 3.10">
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="Apache 2.0 License"></a>
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

HACo was trained and tested with CUDA 12.8, Python 3.10, and PyTorch 2.7.1.

```bash
git clone https://github.com/OpenDriveLab/HACo.git
cd HACo
git clone https://github.com/NVIDIA/Isaac-GR00T.git third_party/Isaac-GR00T
git -C third_party/Isaac-GR00T checkout 4b1dca9d88d2a0b9ea5a65aa61c82ff89f5c4f0e
git -C third_party/Isaac-GR00T submodule update --init --recursive
uv sync --directory third_party/Isaac-GR00T --python 3.10
source third_party/Isaac-GR00T/.venv/bin/activate
uv pip install -e '.[deployment]'
```

Download GR00T N1.7 and Cosmos-Reason2, as required by the official loading pipeline.

```bash
hf download nvidia/GR00T-N1.7-3B --local-dir checkpoints/base_model
hf download nvidia/Cosmos-Reason2-2B --local-dir checkpoints/cosmos_reason2_2b
```

### Dataset

Data follows the **LeRobot v2** format at **30 Hz**.
See [dataset/sample](dataset/sample) for a two-second unscrew-cap example.

```text
dataset/sample/
├── data/chunk-000/                       # State and action sequences
│   └── episode_000000.parquet
├── sensors/episodes/                    # Haptic measurements
│   └── episode_000000.npz
├── videos/chunk-000/                     # Three synchronized RGB streams
│   ├── observation.images.ego_view/
│   │   └── episode_000000.mp4
│   ├── observation.images.left_wrist_view/
│   │   └── episode_000000.mp4
│   └── observation.images.right_wrist_view/
│       └── episode_000000.mp4
└── meta/
    ├── info.json                        # Dataset metadata
    ├── episodes.jsonl                   # Episode metadata
    ├── tasks.jsonl                      # Task descriptions
    ├── modality.json                    # Input/output field layouts
    ├── stats.json                       # State/action normalization
    ├── relative_stats.json              # Relative-action normalization
    ├── sensor_stats.json                # Sensor normalization
    ├── stats_provenance.json            # Statistics provenance
    └── source.json                      # Sample source episode and frame range
```

| Modality | Shape per frame | Format | Description |
| :--- | :---: | :---: | :--- |
| State | `62` | Parquet | Wrist poses and observed joint positions |
| Action | `150` | Parquet | Wrist poses + `q_obs` + `q_cmp` + `delta_q` |
| Joint torque | `44` | NPZ | Measured hand-joint torque |
| Tactile wrench | `10 × 6` | NPZ | 3D force and 3D torque at each fingertip |
| Tactile deformation | `10 × 240 × 240` | NPZ | One uint8 deformation map per fingertip |
| RGB | `H × W × 3` per camera | MP4 | Ego, left wrist, and right wrist views |
| Language | — | JSONL | Task instruction in `meta/tasks.jsonl` |

Each action targets the next frame, with `delta_q = q_cmp - q_obs`.
Field layouts, sensor ordering, and normalization statistics are provided in
[dataset/sample/meta](dataset/sample/meta).

### Training

Configure Weights & Biases:

```bash
# Online logging (default).
wandb login
export WANDB_PROJECT=haco
export WANDB_MODE=online

# Alternatively, uncomment this line to save logs locally without online syncing.
# export WANDB_MODE=offline
```

Start training:

```bash
# Default: 4 GPUs, batch size 12 per GPU, 30k steps.
# Outputs are saved to logs/haco/; model paths use the download locations above.
export HACO_DATASET_PATH=/path/to/your/lerobot_dataset
bash scripts/launch/haco/haco.sh
```

### Inference

Start the policy server with a trained HACo checkpoint:

```bash
bash scripts/deploy/haco/launch.sh \
  /path/to/haco-checkpoint checkpoints/cosmos_reason2_2b
```

The server returns 40-frame action chunks over `ws://localhost:5500/infer`.
See the [deployment guide](scripts/deploy/haco/README.md) for the client protocol.

Open-loop example from the sample dataset, using checkpoint-500000.
Both videos show the same 40 frames at 30 Hz. Blue shows `q_obs`; dashed
orange shows `q_cmp`. Force/torque gauges show measured signals in both videos.

**Ground truth**

<video src="https://github.com/user-attachments/assets/7ac60c0e-92f3-40c3-a5ef-bc344b55381a" controls width="100%"></video>

**Prediction**

<video src="https://github.com/user-attachments/assets/580bfd12-9820-4198-96a5-2fe465cfed02" controls width="100%"></video>
