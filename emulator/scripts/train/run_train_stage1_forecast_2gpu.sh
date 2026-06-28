#!/bin/bash
#SBATCH --job-name=upd_s1
#SBATCH --partition=gpu_prio
#SBATCH --nodes=3
#SBATCH --exclude=cn1,cn3,cn15
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

set -euo pipefail

if [[ -n "${UPD_EMULATOR_REPO_ROOT:-}" ]]; then
  REPO_ROOT="$UPD_EMULATOR_REPO_ROOT"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
export UPD_EMULATOR_REPO_ROOT="$REPO_ROOT"
source "$REPO_ROOT/scripts/utils/_common.sh"
ensure_allocation
activate_env_if_needed
PYTHON_BIN=$(resolve_python_bin)

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export UPD_EMULATOR_DATA_ROOT="${UPD_EMULATOR_DATA_ROOT:-/home/nishidh/fuxi-ens/data}"
export UPD_EMULATOR_CACHE_DIR="${UPD_EMULATOR_CACHE_DIR:-$REPO_ROOT/outputs/cache}"
export UPD_EMULATOR_STATS_PATH="${UPD_EMULATOR_STATS_PATH:-$REPO_ROOT/resources/stats/exac_input_stats.pt}"
export UPD_EMULATOR_STAGE1_OUTPUT_DIR="${UPD_EMULATOR_STAGE1_OUTPUT_DIR:-$REPO_ROOT/outputs/forecast_stage1}"
export UPD_EMULATOR_STAGE1_SAVE_EVERY="${UPD_EMULATOR_STAGE1_SAVE_EVERY:-1000}"
export UPD_EMULATOR_STAGE1_LOG_EVERY="${UPD_EMULATOR_STAGE1_LOG_EVERY:-50}"
if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
  DEFAULT_MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
else
  DEFAULT_MASTER_ADDR="127.0.0.1"
fi
DEFAULT_MASTER_PORT="${SLURM_JOB_ID:-29500}"
DEFAULT_MASTER_PORT=$((29500 + DEFAULT_MASTER_PORT % 1000))
export MASTER_ADDR="${MASTER_ADDR:-$DEFAULT_MASTER_ADDR}"
export MASTER_PORT="${MASTER_PORT:-$DEFAULT_MASTER_PORT}"

# Submit directly with:
#   sbatch scripts/train/run_train_stage1_forecast_2gpu.sh

srun --nodes=3 --ntasks=6 --ntasks-per-node=2 --cpu-bind=none \
  "$PYTHON_BIN" "$REPO_ROOT/scripts/train/train_stage1_forecast.py" \
  --data-root "$UPD_EMULATOR_DATA_ROOT" \
  --stats-path "$UPD_EMULATOR_STATS_PATH" \
  --cache-dir "$UPD_EMULATOR_CACHE_DIR" \
  --output-dir "$UPD_EMULATOR_STAGE1_OUTPUT_DIR" \
  --log-path "$UPD_EMULATOR_STAGE1_OUTPUT_DIR/training_log.csv" \
  --save-every "$UPD_EMULATOR_STAGE1_SAVE_EVERY" \
  --log-every "$UPD_EMULATOR_STAGE1_LOG_EVERY" \
  "$@"
