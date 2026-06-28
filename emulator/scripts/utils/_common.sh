#!/bin/bash

set -euo pipefail

resolve_repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
}

activate_env_if_needed() {
  local env_name="${UPD_EMULATOR_ENV_NAME:-fuxi_ens}"
  if [[ "${CONDA_DEFAULT_ENV:-}" != "$env_name" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "$env_name"
  fi
}

resolve_python_bin() {
  if [[ -n "${UPD_EMULATOR_PYTHON_BIN:-}" && -x "${UPD_EMULATOR_PYTHON_BIN}" ]]; then
    printf '%s\n' "$UPD_EMULATOR_PYTHON_BIN"
    return 0
  fi

  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    printf '%s\n' "${CONDA_PREFIX}/bin/python"
    return 0
  fi

  local env_name="${UPD_EMULATOR_ENV_NAME:-fuxi_ens}"
  local fallback="/home/nishidh/miniconda3/envs/${env_name}/bin/python3.10"
  if [[ -x "$fallback" ]]; then
    printf '%s\n' "$fallback"
    return 0
  fi

  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi

  command -v python3
}

ensure_allocation() {
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit this launcher with sbatch on partition GPU-AI_prio, or run it inside an existing Slurm allocation."
    echo "Example:"
    echo "  sbatch scripts/run_train_stage1_forecast_2gpu.sh"
    exit 1
  fi
}
