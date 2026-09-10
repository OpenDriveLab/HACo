# HACO

HACO is a vision-language-action policy for contact-rich bimanual dexterous
manipulation. It combines RGB observations, robot state, torque and tactile
sensing, active-compliance actions, and real-time chunking (RTC).

## Setup

HACO uses Python 3.10 and CUDA 12.8 on Linux. Install `git-lfs`, FFmpeg, and
[`uv`](https://docs.astral.sh/uv/), then create the environment:

```bash
sudo apt-get update
sudo apt-get install -y git-lfs ffmpeg
git lfs install

git clone https://github.com/OpenDriveLab/HACo.git
cd HACo
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T.git \
  third_party/Isaac-GR00T
git -C third_party/Isaac-GR00T checkout \
  4b1dca9d88d2a0b9ea5a65aa61c82ff89f5c4f0e
git -C third_party/Isaac-GR00T submodule update --init --recursive

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd third_party/Isaac-GR00T
uv sync --python 3.10
cd ../..

uv pip install --python third_party/Isaac-GR00T/.venv/bin/python -e .
source third_party/Isaac-GR00T/.venv/bin/activate
```

Download the pretrained base model and VLM backbone:

```bash
mkdir -p checkpoints
hf download nvidia/GR00T-N1.7-3B --local-dir checkpoints/base_model
hf download nvidia/Cosmos-Reason2-2B \
  --local-dir checkpoints/cosmos_reason2_2b
```

Training data must use the LeRobot format and include `meta/info.json` and
`meta/modality.json`.

## Training

Log in to Weights & Biases and select the account or team that will own the
run:

```bash
wandb login
export WANDB_ENTITY="your-wandb-entity"
export WANDB_PROJECT=haco
export WANDB_MODE=online
```

Set the data and model paths:

```bash
export HACO_DATASET_PATH=/path/to/lerobot_dataset
export HACO_BASE_MODEL_PATH="$PWD/checkpoints/base_model"
export HACO_VLM_MODEL_PATH="$PWD/checkpoints/cosmos_reason2_2b"
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

Check the configuration without starting a training job:

```bash
HACO_DRY_RUN=1 bash scripts/launch/haco/haco.sh
```

Start training:

```bash
bash scripts/launch/haco/haco.sh
```

The default run uses 4 GPUs, a per-GPU batch size of 12, and 30,000 training
steps. Checkpoints and W&B run files are written to `logs/haco/`.

## Inference

Start the policy server with a trained HACO checkpoint:

```bash
source third_party/Isaac-GR00T/.venv/bin/activate
CUDA_VISIBLE_DEVICES=0 bash scripts/deploy/haco/launch.sh \
  /path/to/haco-checkpoint \
  "$PWD/checkpoints/cosmos_reason2_2b"
```

The server listens on port `5500`. Its health endpoint is
`http://localhost:5500/healthz`, and inference uses the binary MessagePack
WebSocket endpoint `ws://localhost:5500/infer`.

## License

HACO is released under the [Apache License 2.0](LICENSE).
