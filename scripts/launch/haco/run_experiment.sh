#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
    echo "usage: $0 EXPERIMENT_ID [training args...]" >&2
    exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HACO_ROOT="${HACO_ROOT:-${PROJECT_ROOT}}"
HACO_EXPERIMENT_ID="$1"
shift

case "${HACO_EXPERIMENT_ID}" in
    haco) DEFAULT_PORT=29840 ;;
    hp_wo_haptic) DEFAULT_PORT=29841 ;;
    hp_wo_torque) DEFAULT_PORT=29842 ;;
    hp_wo_tactile) DEFAULT_PORT=29843 ;;
    hp_wo_coupled_en) DEFAULT_PORT=29844 ;;
    ac_wo_intent_sup) DEFAULT_PORT=29845 ;;
    ac_wo_active_comp) DEFAULT_PORT=29846 ;;
    cg_action_suf) DEFAULT_PORT=29847 ;;
    cg_visuo_haptic) DEFAULT_PORT=29848 ;;
    cg_ungated_comp_attn) DEFAULT_PORT=29849 ;;
    wc_wo_wrist) DEFAULT_PORT=29850 ;;
    *)
        echo "Unknown HACO experiment: ${HACO_EXPERIMENT_ID}" >&2
        exit 2
        ;;
esac

export HACO_ROOT HACO_EXPERIMENT_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HACO_MASTER_PORT="${HACO_MASTER_PORT:-${DEFAULT_PORT}}"
export HACO_BASE_MODEL_PATH="${HACO_BASE_MODEL_PATH:-${HACO_ROOT}/checkpoints/base_model}"
if [[ "${HACO_EXPERIMENT_ID}" == hp_wo_haptic ]]; then
    export HACO_PREBUILD_SENSOR_CACHE="${HACO_PREBUILD_SENSOR_CACHE:-0}"
fi

exec bash "${HACO_ROOT}/scripts/train/haco/run.sh" "$@"
