#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HACO_ROOT="${HACO_ROOT:-${PROJECT_ROOT}}"
ISAAC_GROOT_DIR="${ISAAC_GROOT_DIR:-${HACO_ROOT}/third_party/Isaac-GR00T}"
PYTHON_BIN="${PYTHON_BIN:-${ISAAC_GROOT_DIR}/.venv/bin/python}"

: "${HACO_EXPERIMENT_ID:?set one frozen HACO experiment id}"

HACO_DATASET_PATH="${HACO_DATASET_PATH:-${HACO_ROOT}/datasets/postrain/ur-sharpa/lerobot_data/unscrew_cap}"
HACO_DATASET_PATHS_JSON="${HACO_DATASET_PATHS_JSON:-}"
HACO_MULTITASK_PROFILE="${HACO_MULTITASK_PROFILE:-}"
HACO_MULTITASK_NORMALIZATION_DIR="${HACO_MULTITASK_NORMALIZATION_DIR:-}"
HACO_BASE_MODEL_PATH="${HACO_BASE_MODEL_PATH:-${HACO_ROOT}/checkpoints/groot_n17/pretrain}"
HACO_VLM_MODEL_PATH="${HACO_VLM_MODEL_PATH:-${HACO_ROOT}/checkpoints/cosmos_reason2_2b}"
HACO_MODALITY_CONFIG_PATH="${HACO_MODALITY_CONFIG_PATH:-${HACO_ROOT}/scripts/train/haco/modality.py}"
HACO_EMBODIMENT_TAG="${HACO_EMBODIMENT_TAG:-real_r1_pro_sharpa_absolute_eef}"
HACO_RUN_NAME="${HACO_RUN_NAME:-posttrain-ur_unscrew_cap-haco-${HACO_EXPERIMENT_ID}-official-1n4g-bs12-gbs48-30k-seed42-${RUN_DATETIME:-$(date -u +%Y%m%d_%H%M%S)}}"
HACO_RUN_ROOT="${HACO_RUN_ROOT:-${HACO_ROOT}/logs/haco/${HACO_RUN_NAME}}"
HACO_OUTPUT_DIR="${HACO_OUTPUT_DIR:-${HACO_RUN_ROOT}/checkpoints}"

HACO_NNODES="${HACO_NNODES:-1}"
HACO_NODE_RANK="${HACO_NODE_RANK:-0}"
HACO_MASTER_ADDR="${HACO_MASTER_ADDR:-127.0.0.1}"
HACO_MASTER_PORT="${HACO_MASTER_PORT:-29840}"
HACO_GPUS_PER_NODE="${HACO_GPUS_PER_NODE:-4}"
HACO_PER_DEVICE_BATCH_SIZE="${HACO_PER_DEVICE_BATCH_SIZE:-12}"
HACO_GRADIENT_ACCUMULATION_STEPS="${HACO_GRADIENT_ACCUMULATION_STEPS:-1}"
HACO_MAX_STEPS="${HACO_MAX_STEPS:-30000}"
HACO_SAVE_STEPS="${HACO_SAVE_STEPS:-10000}"
HACO_SAVE_TOTAL_LIMIT="${HACO_SAVE_TOTAL_LIMIT:-3}"
HACO_SEED="${HACO_SEED:-42}"
HACO_SMOKE="${HACO_SMOKE:-0}"
HACO_CONTINUATION="${HACO_CONTINUATION:-0}"

if [[ "${HACO_SMOKE}" == 1 ]]; then
    HACO_MAX_STEPS="${HACO_SMOKE_STEPS:-20}"
    HACO_SAVE_STEPS="${HACO_SMOKE_SAVE_STEPS:-20}"
fi
if [[ "${HACO_SMOKE}" == 1 && "${HACO_CONTINUATION}" == 1 ]]; then
    echo "HACO_SMOKE and HACO_CONTINUATION are mutually exclusive" >&2
    exit 2
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing HACO Python executable: ${PYTHON_BIN}" >&2
    exit 1
fi
HACO_DATASET_PATHS=("${HACO_DATASET_PATH}")
if [[ -n "${HACO_DATASET_PATHS_JSON}" ]]; then
    mapfile -t HACO_DATASET_PATHS < <(
        "${PYTHON_BIN}" -c \
            'import json,sys; print(*json.loads(sys.argv[1]), sep="\n")' \
            "${HACO_DATASET_PATHS_JSON}"
    )
fi
if [[ "${#HACO_DATASET_PATHS[@]}" -eq 0 ]]; then
    echo "HACO requires at least one dataset" >&2
    exit 1
fi

for required in \
    "${HACO_MODALITY_CONFIG_PATH}" \
    "${HACO_VLM_MODEL_PATH}/config.json" \
    "${HACO_BASE_MODEL_PATH}/config.json"; do
    if [[ ! -e "${required}" ]]; then
        echo "Missing HACO input: ${required}" >&2
        exit 1
    fi
done
for dataset_path in "${HACO_DATASET_PATHS[@]}"; do
    if [[ ! -f "${dataset_path}/meta/info.json" ]]; then
        echo "Missing HACO dataset metadata: ${dataset_path}/meta/info.json" >&2
        exit 1
    fi
done

PREFLIGHT_ARGS=(
    --experiment-id "${HACO_EXPERIMENT_ID}"
    --base-model-path "${HACO_BASE_MODEL_PATH}"
    --nnodes "${HACO_NNODES}"
    --gpus-per-node "${HACO_GPUS_PER_NODE}"
    --per-device-batch-size "${HACO_PER_DEVICE_BATCH_SIZE}"
    --gradient-accumulation-steps "${HACO_GRADIENT_ACCUMULATION_STEPS}"
    --max-steps "${HACO_MAX_STEPS}"
    --save-steps "${HACO_SAVE_STEPS}"
    --save-total-limit "${HACO_SAVE_TOTAL_LIMIT}"
    --seed "${HACO_SEED}"
)
for dataset_path in "${HACO_DATASET_PATHS[@]}"; do
    PREFLIGHT_ARGS+=(--dataset-path "${dataset_path}")
done
if [[ -n "${HACO_MULTITASK_PROFILE}" ]]; then
    PREFLIGHT_ARGS+=(
        --multitask-profile "${HACO_MULTITASK_PROFILE}"
        --normalization-dir "${HACO_MULTITASK_NORMALIZATION_DIR}"
    )
fi
if [[ "${HACO_SMOKE}" == 1 ]]; then
    PREFLIGHT_ARGS+=(--smoke)
elif [[ "${HACO_CONTINUATION}" == 1 ]]; then
    PREFLIGHT_ARGS+=(--continuation)
fi

cd "${HACO_ROOT}"
export PYTHONPATH="${HACO_ROOT}:${ISAAC_GROOT_DIR}:${PYTHONPATH:-}"
"${PYTHON_BIN}" -m scripts.train.haco.preflight "${PREFLIGHT_ARGS[@]}"

if [[ "${HACO_DRY_RUN:-0}" == 1 ]]; then
    echo "HACO dry-run complete; training was not started."
    exit 0
fi

export HACO_ROOT ISAAC_GROOT_DIR
export HACO_EXPERIMENT_ID HACO_DATASET_PATH HACO_BASE_MODEL_PATH
export HACO_DATASET_PATHS_JSON HACO_MULTITASK_PROFILE
export HACO_MULTITASK_NORMALIZATION_DIR
export HACO_NNODES HACO_GPUS_PER_NODE HACO_PER_DEVICE_BATCH_SIZE
export HACO_SEED HACO_SMOKE HACO_CONTINUATION HACO_EMBODIMENT_TAG
export GR00T_N1D7_MODEL_NAME="${HACO_VLM_MODEL_PATH}"
export HACO_TRAINING_MAX_PREFIX_STEPS=12
export HACO_INFERENCE_PREFIX_STEPS=10
export HACO_RTC_PREFIX_STEPS=10
export HACO_PREFIX_WEIGHTS='[1,1,1,1,1,1,1,1,1,1,2,1,1]'
export HACO_DELTA_Q_FLOW_WEIGHT=0.5
export HF_HOME="${HF_HOME:-${HACO_ROOT}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-haco}"
export WANDB_NAME="${HACO_RUN_NAME}"
export WANDB_DIR="${WANDB_DIR:-${HACO_RUN_ROOT}}"
export WANDB_RESUME=never
export WANDB_X_DISABLE_VIEWER="${WANDB_X_DISABLE_VIEWER:-true}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-1200}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-1200}"
export OPEN_LOOP_EVAL_ENABLE="${OPEN_LOOP_EVAL_ENABLE:-1}"
export OPEN_LOOP_EVAL_FIRST_STEP="${OPEN_LOOP_EVAL_FIRST_STEP:-1}"
export OPEN_LOOP_EVAL_EVERY_N_STEPS="${OPEN_LOOP_EVAL_EVERY_N_STEPS:-${HACO_SAVE_STEPS}}"
export OPEN_LOOP_EVAL_ARTIFACT_DIR="${OPEN_LOOP_EVAL_ARTIFACT_DIR:-${HACO_RUN_ROOT}/open_loop_eval}"
export HACO_SENSOR_CACHE_DIR="${HACO_SENSOR_CACHE_DIR:-/tmp/haco-${UID}/sensor_cache}"
# The shared cache primitive currently reads this internal variable.
export PACE_SENSOR_CACHE_DIR="${HACO_SENSOR_CACHE_DIR}"
unset WANDB_RUN_ID VIZ_ENABLE DZ_VIZ_MODE

HACO_CONDA_NVCC_HOME="${HACO_CONDA_NVCC_HOME:-}"
if [[ -z "${CUDA_HOME:-}" && -x /usr/local/cuda-12.9/bin/nvcc ]]; then
    export CUDA_HOME=/usr/local/cuda-12.9
elif [[ -z "${CUDA_HOME:-}" && -n "${HACO_CONDA_NVCC_HOME}" && -x "${HACO_CONDA_NVCC_HOME}/bin/nvcc" ]]; then
    export CUDA_HOME="${HACO_CONDA_NVCC_HOME}"
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

if [[ "${HACO_PREBUILD_SENSOR_CACHE:-1}" == 1 ]]; then
    for dataset_path in "${HACO_DATASET_PATHS[@]}"; do
        "${PYTHON_BIN}" -m scripts.train.pace.cache_sensors \
            --dataset-path "${dataset_path}" \
            --workers "${HACO_SENSOR_CACHE_BUILD_WORKERS:-16}"
    done
fi

mkdir -p "${HACO_RUN_ROOT}" "${HACO_OUTPUT_DIR}" "${WANDB_DIR}"

USE_WANDB=()
if [[ "${WANDB_MODE}" != disabled ]]; then
    USE_WANDB+=(--use_wandb)
fi

exec "${PYTHON_BIN}" -m torch.distributed.run \
    --nnodes "${HACO_NNODES}" \
    --node_rank "${HACO_NODE_RANK}" \
    --master_addr "${HACO_MASTER_ADDR}" \
    --master_port "${HACO_MASTER_PORT}" \
    --nproc_per_node "${HACO_GPUS_PER_NODE}" \
    -m scripts.train.haco.train \
    --base_model_path "${HACO_BASE_MODEL_PATH}" \
    --dataset_path "${HACO_DATASET_PATHS[0]}" \
    --embodiment_tag "${HACO_EMBODIMENT_TAG}" \
    --modality_config_path "${HACO_MODALITY_CONFIG_PATH}" \
    --output_dir "${HACO_OUTPUT_DIR}" \
    --experiment_name "${HACO_RUN_NAME}" \
    --num_gpus "$((HACO_NNODES * HACO_GPUS_PER_NODE))" \
    --global_batch_size "$((HACO_NNODES * HACO_GPUS_PER_NODE * HACO_PER_DEVICE_BATCH_SIZE * HACO_GRADIENT_ACCUMULATION_STEPS))" \
    --gradient_accumulation_steps "${HACO_GRADIENT_ACCUMULATION_STEPS}" \
    --dataloader_num_workers "${HACO_DATALOADER_WORKERS:-4}" \
    --shard_size "${HACO_SHARD_SIZE:-1024}" \
    --episode_sampling_rate 1.0 \
    --num_shards_per_epoch 100000 \
    --learning_rate 2e-5 \
    --weight_decay 1e-5 \
    --warmup_ratio 0.05 \
    --max_steps "${HACO_MAX_STEPS}" \
    --save_steps "${HACO_SAVE_STEPS}" \
    --save_total_limit "${HACO_SAVE_TOTAL_LIMIT}" \
    --state_dropout_prob 0.0 \
    --no-tune-llm \
    --no-tune-visual \
    --tune-projector \
    --tune-diffusion-model \
    --wandb-project "${WANDB_PROJECT}" \
    "${USE_WANDB[@]}" \
    "$@"
