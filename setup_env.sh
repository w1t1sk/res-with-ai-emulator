#!/usr/bin/env bash
# setup_env.sh — Environment configuration for the QDMC rare event sampling project.
#
# Usage:
#   conda activate climt
#   source /home/nishidh/res_aiemulator_sem8/setup_env.sh
#   cd res && python run.py {spinup,dns,airres ...}
#
# Required conda environment: "climt"
# (provides sympl, the compiled climt GCM, PyTorch, NumPy, xarray, scipy)
#
# Sets:
#   PYTHONPATH            — adds the climt GCM package (not pip-installed)
#   AIRES_EMULATOR_DIR    — path to the new FuXi-ENS latent-hierswin emulator
#   AIRES_OLD_EMULATOR_DIR— path to the old lightweight fuxiens emulator
#   AIRES_REPO_DIR        — path to the res/ simulation directory

export PYTHONPATH=/home/nishidh/res_aiemulator_sem8/gcm:$PYTHONPATH
export AIRES_EMULATOR_DIR=/home/nishidh/res_aiemulator_sem8/emulator
export AIRES_OLD_EMULATOR_DIR=/home/nishidh/res_aiemulator_sem8/res/old_emulator
export AIRES_REPO_DIR=/home/nishidh/res_aiemulator_sem8/res
