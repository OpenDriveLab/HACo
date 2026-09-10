#!/usr/bin/env bash

sharpa_load_task_config() {
  if [[ "$#" -ne 3 ]]; then
    echo "usage: sharpa_load_task_config <model> <task> <run_id>" >&2
    return 2
  fi

  local model="$1"
  local task="$2"
  local run_id="$3"
  local template_root
  template_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  local repository_root
  repository_root="$(cd "${template_root}/../../.." && pwd)"
  local output_file
  output_file="$(mktemp "${TMPDIR:-/tmp}/sharpa-task-config.XXXXXX")" || return 1

  local -a resolver_args=(
    "${template_root}/task_config.py"
    --config-root "${SHARPA_DEPLOY_TASK_CONFIG_ROOT:-${repository_root}/scripts/deploy/task_config}"
    --repository-root "${repository_root}"
    --asset-root "${SHARPA_DEPLOY_ASSET_ROOT:-${repository_root}}"
    --model "${model}"
    --task "${task}"
    --run-id "${run_id}"
    --format env0
  )
  if [[ "${SHARPA_DEPLOY_ALLOW_INCOMPLETE:-0}" == "1" ]]; then
    resolver_args+=(--allow-incomplete)
  fi
  if [[ "${SHARPA_DEPLOY_SKIP_PATH_CHECK:-0}" == "1" ]]; then
    resolver_args+=(--skip-path-check)
  fi
  if ! "${SHARPA_DEPLOY_CONFIG_PYTHON:-python3}" "${resolver_args[@]}" >"${output_file}"; then
    rm -f "${output_file}"
    return 2
  fi

  local key value
  while IFS= read -r -d '' key && IFS= read -r -d '' value; do
    if [[ ! "${key}" =~ ^SHARPA_DEPLOY_[A-Z0-9_]+$ ]]; then
      echo "task config emitted an invalid environment key: ${key}" >&2
      rm -f "${output_file}"
      return 2
    fi
    printf -v "${key}" '%s' "${value}"
    export "${key}"
  done <"${output_file}"
  rm -f "${output_file}"

  echo "[task config] task=${SHARPA_DEPLOY_TASK} model=${model} id=${SHARPA_DEPLOY_ID} run_id=${SHARPA_DEPLOY_RUN_ID}" >&2
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "load_task_config.sh must be sourced by a model launch.sh" >&2
  exit 2
fi
