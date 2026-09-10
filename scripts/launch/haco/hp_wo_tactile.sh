#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec bash "${HACO_ROOT:-${PROJECT_ROOT}}/scripts/launch/haco/run_experiment.sh" hp_wo_tactile "$@"
