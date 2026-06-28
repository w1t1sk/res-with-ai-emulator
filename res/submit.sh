#!/bin/bash
# submit.sh — Single parametrized SLURM launcher for the QDMC pipeline.
#
# Usage (arguments are passed straight through to run.py):
#   sbatch submit.sh spinup
#   sbatch submit.sh dns
#   sbatch submit.sh airres --emulator new
#   sbatch submit.sh airres --emulator old --scheme ck
#
# Resource directives below are sized for the paper-scale run (N=400). To scale
# down for a quick test, request fewer CPUs/GPUs on the sbatch command line, e.g.
#   sbatch --cpus-per-task=4 --gpus=1 submit.sh airres --emulator new --walkers 4 --members 2
#
#SBATCH --job-name=qdmc_res
#SBATCH --nodes=1
#SBATCH --exclude=cn3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=100
#SBATCH --partition=gpu_prio
#SBATCH --gpus=2
#SBATCH --gres=gpu:2
#SBATCH --output=/home/nishidh/res_aiemulator_sem8/res/slurm-logs/%x-%j.log

set -euo pipefail

REPO=/home/nishidh/res_aiemulator_sem8
cd "$REPO/res"
mkdir -p slurm-logs

echo "=== QDMC pipeline: run.py $* ==="
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK:-?} | GPUs allocated: ${SLURM_GPUS:-0}"

source /home/nishidh/miniconda3/etc/profile.d/conda.sh
conda activate climt
source "$REPO/setup_env.sh"

# Pin numerical libraries to one thread each so the physics workers don't oversubscribe.
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AIRES_TARGET_REGION="${AIRES_TARGET_REGION:-nw_box}"

/home/nishidh/miniconda3/envs/climt/bin/python -u run.py "$@"
