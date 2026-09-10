# HACO

HACO is a vision-language-action policy for contact-rich bimanual dexterous
manipulation. It combines three-view RGB observations, robot state, joint
torque history, fingertip wrench/deformation sensing, active-compliance action
representations, and trained real-time chunking (RTC).

This repository is the standalone release workspace extracted from
[OpenDriveLab/Dexhand](https://github.com/OpenDriveLab/Dexhand) at source
commit `bd0c503`. Datasets, checkpoints, experiment logs, machine-specific
cluster orchestration, and credentials are intentionally not included.

## Layout

```text
dexterity/models/haco/       HACO model, processor, contracts, and RTC
dexterity/data/              SharpA/LeRobot data and normalization utilities
dexterity/deploy/            unified SharpA WebSocket protocol and HACO server
dexterity/callbacks/         training-time open-loop evaluation
scripts/train/haco/          training driver and frozen experiment matrix
scripts/launch/haco/         one launcher per HACO/ablation configuration
scripts/inference/haco/      reproducible open-loop evaluation
scripts/deploy/haco/         checkpoint deployment launcher
tests/haco/                  HACO contracts and experiment tests
```

The small `groot_n17`, `groot_rtc`, and `pace` compatibility modules are
required building blocks used by HACO. They are not separate model releases.

## Requirements

- Python 3.10 or 3.11
- CUDA-compatible PyTorch for training and deployment
- NVIDIA Isaac-GR00T checked out locally (the code imports its `gr00t`
  package)
- an official GR00T N1.7 checkpoint and the configured vision-language
  backbone
- a converted UR-SharpA LeRobot dataset for training/evaluation

Create an environment and install this repository:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
git clone https://github.com/NVIDIA/Isaac-GR00T.git third_party/Isaac-GR00T
pip install -e third_party/Isaac-GR00T
```

The exact CUDA/PyTorch build should match the target machine. HACO deployment
currently validates `transformers==4.57.3` and `tokenizers==0.22.2`.
Install `.[deployment]` for hand-retargeting support, `.[video]` for the
optional Decord/TorchCodec backends, or `.[tensorflow]` when TensorFlow tensor
conversion is required.

## Training

All launchers are configured with environment variables. A no-training
preflight for the full model is:

```bash
export ISAAC_GROOT_DIR="$PWD/third_party/Isaac-GR00T"
export HACO_DATASET_PATH=/path/to/lerobot_dataset
export HACO_BASE_MODEL_PATH=/path/to/groot_n17_checkpoint
export HACO_VLM_MODEL_PATH=/path/to/vlm_backbone
HACO_DRY_RUN=1 bash scripts/launch/haco/haco.sh
```

Remove `HACO_DRY_RUN=1` to launch distributed training. Hardware and schedule
defaults can be overridden with `HACO_GPUS_PER_NODE`,
`HACO_PER_DEVICE_BATCH_SIZE`, `HACO_MAX_STEPS`, and related variables in
`scripts/train/haco/run.sh`.

## Evaluation and deployment

Build an evaluation manifest:

```bash
python -m scripts.inference.haco.manifest \
  --dataset /path/to/lerobot_dataset \
  --action-contract joint_compliance_delta \
  --action-target q_compliance \
  --output /tmp/haco_manifest.json
```

Serve a checkpoint over the SharpA WebSocket protocol:

```bash
HACO_REFERENCE_REPO="$PWD/third_party/Isaac-GR00T" \
  bash scripts/deploy/haco/launch.sh \
  /path/to/checkpoint /path/to/vlm_backbone
```

Model weights and data are not distributed by this source repository. Their
licenses and access conditions must be handled separately.

## Verification

```bash
python -m compileall -q dexterity scripts
find scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
pytest
```

## License

Apache License 2.0. See `LICENSE`. Third-party components and external model
weights remain subject to their own licenses.
