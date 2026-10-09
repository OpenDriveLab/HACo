#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REFERENCE_REPO="${HACO_REFERENCE_REPO:-${ROOT}/third_party/Isaac-GR00T}"
PYTHON_BIN="${PYTHON_BIN:-${REFERENCE_REPO}/.venv/bin/python}"

if [[ "$#" -lt 1 ]]; then
  echo "usage: bash scripts/deploy/haco/launch.sh CHECKPOINT [BACKBONE] [server args...]" >&2
  exit 2
fi
CHECKPOINT="$1"
shift
BACKBONE_MODEL="${HACO_VLM_MODEL_PATH:-${ROOT}/checkpoints/cosmos_reason2_2b}"
if [[ "$#" -gt 0 && "$1" != -* ]]; then
  BACKBONE_MODEL="$1"
  shift
fi

HOST="${HACO_DEPLOY_HOST:-0.0.0.0}"
PORT="${HACO_DEPLOY_PORT:-5500}"
DEVICE="${HACO_DEVICE:-cuda:0}"
EMBODIMENT_TAG="${HACO_EMBODIMENT_TAG:-real_r1_pro_sharpa_absolute_eef}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Isaac-GR00T Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -d "${REFERENCE_REPO}" ]]; then
  echo "Isaac-GR00T repository not found: ${REFERENCE_REPO}" >&2
  exit 1
fi
if [[ ! -d "${CHECKPOINT}" ]]; then
  echo "HACO checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${BACKBONE_MODEL}/config.json" ]]; then
  echo "HACO backbone config not found: ${BACKBONE_MODEL}/config.json" >&2
  exit 1
fi

export ISAAC_GROOT_DIR="${REFERENCE_REPO}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONPATH="${ROOT}:${REFERENCE_REPO}:${PYTHONPATH:-}"

if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

PYTHONPATH= "${PYTHON_BIN}" -c '
import tokenizers
import transformers
if transformers.__version__ != "4.57.3" or tokenizers.__version__ != "0.22.2":
    raise SystemExit("HACO requires transformers=4.57.3 and tokenizers=0.22.2")
'

cd "${ROOT}"
exec "${PYTHON_BIN}" -m dexterity.deploy.models.haco.server \
  --host "${HOST}" \
  --port "${PORT}" \
  --checkpoint "${CHECKPOINT}" \
  --reference-repo "${REFERENCE_REPO}" \
  --backbone-model "${BACKBONE_MODEL}" \
  --embodiment-tag "${EMBODIMENT_TAG}" \
  --device "${DEVICE}" \
  "$@"
