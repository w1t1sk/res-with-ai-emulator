# Rare Event Sampling with a Global Climate Model and AI Emulator

This repository implements **QDMC rare event sampling** for climate extremes:
an aquaplanet Global Climate Model (GCM, via [`climt`](https://github.com/CliMT/climt))
acts as the physics engine that evolves an ensemble of trajectories ("walkers"),
and an AI weather-forecast emulator scores those trajectories so that promising
ones are cloned and unpromising ones killed at fixed resampling times. The result
is a heavily biased ensemble that probes rare warm extremes far more efficiently
than a direct (unbiased) simulation of the same size.

The method follows Lancelin et al. (2025), *AI-boosted rare event sampling to
characterize extreme weather*; the emulator architecture follows FuXi-ENS
(Zhong et al., 2025). See the project report for the full description.

---

## The experiment

- **Target region:** a 3×3 box over a northwest-India analog, 26.5–32.1°N,
  70.3–75.9°E (grid indices `[20:23, 25:28]` on the 64×128 aquaplanet grid).
- **Observable:** final 7-day average regional-mean surface temperature (`L = 7`).
- **Population:** `N = 400` walkers; the AI emulator draws `M = 100` ensemble
  members per walker when scoring.
- **Schedule:** resample every 5 days with `C_k = [0, 0, 0, 1.8, 2.2, 2.6]`
  (the first three intervals evolve unbiased; the bias ramps up later).
- **Baseline:** DNS — the same 400 initial walkers integrated to the end with no
  resampling — is the reference distribution.

---

## Repository layout

```
res_aiemulator_sem8/
├── README.md            ← you are here
├── requirements.txt     ← Python dependencies
├── setup_env.sh         ← sets PYTHONPATH / emulator paths (source before running)
├── emulator/            ← new emulator (FuXiENSModelV3LatentHierSwin) + checkpoints
└── res/
    ├── config.json      ← single experiment config (paper-faithful defaults)
    ├── core.py          ← GCM physics, QDMC algorithms, DNS + spinup (emulator-free)
    ├── driver_new.py    ← AI+RES driver using the NEW emulator
    ├── driver_old.py    ← AI+RES driver using the OLD lightweight emulator (report)
    ├── run.py           ← single CLI entry point (spinup / dns / airres)
    ├── submit.sh        ← single parametrized SLURM launcher
    ├── analysis.ipynb   ← reproduces report Figures 5, 6a, 6b
    ├── old_emulator/    ← old fuxiens checkpoint + model code + stats
    └── outputs/         ← spinup state and run outputs (git-ignored)
```

---

## Setup

```bash
conda activate climt                       # provides climt (compiled GCM), sympl, torch, numpy
source /home/nishidh/res_aiemulator_sem8/setup_env.sh
pip install -r requirements.txt            # only the pure-Python deps; climt/sympl come from conda
```

`climt` and `sympl` are **not** on PyPI — they are pre-built in the `climt` conda
environment. `setup_env.sh` adds the compiled `climt` package to `PYTHONPATH`.

---

## Running the pipeline

Everything goes through `run.py` (or `submit.sh` on SLURM, which forwards its
arguments to `run.py`).

```bash
cd res/

# 1. Generate the aquaplanet spinup (run once; ~hours of GCM integration).
python run.py spinup

# 2. DNS baseline (no resampling).
python run.py dns

# 3. AI+RES experiment. Pick the emulator with --emulator:
python run.py airres --emulator new           # new latent-hierswin emulator
python run.py airres --emulator old --scheme ck   # old emulator (reproduces the report)

# 4. Analysis: open res/analysis.ipynb and run all cells (set SCHEME at the top).
```

On the cluster:

```bash
sbatch submit.sh spinup
sbatch submit.sh dns
sbatch submit.sh airres --emulator new
```

### Quick smoke test (no GPU, tiny ensemble)

`--walkers` / `--members` override the config so you don't need a second config file:

```bash
python run.py dns    --walkers 4
python run.py airres --emulator new --walkers 4 --members 2
```

Outputs land in `res/outputs/global_dns_output/` and
`res/outputs/ai_res_nw_box_<scheme>_output/`.

---

## Choosing the emulator

| | New (`--emulator new`) | Old (`--emulator old`) |
|---|---|---|
| Class | `FuXiENSModelV3LatentHierSwin` | `fuxiens` (lightweight) |
| Perturbation | latent-space hierarchical Swin | physical-space input noise |
| Checkpoint | `emulator/checkpoints/stage2/…` | `res/old_emulator/fuxiens_stage2_best_ctd.pth` |
| Used for | new runs | the figures in the report |

Both are scored identically (M-member rollout, final-window observable); only the
model load and the single inference call differ. The drivers are deliberately kept
separate (`driver_new.py`, `driver_old.py`) so each is self-contained.

---

## Reproducing the report figures

`res/analysis.ipynb` recreates:

- **Figure 5** — AI+RES trajectories + final-day distribution.
- **Figure 6a** — observable (7-day temperature) distribution with Johnson-SU fits.
- **Figure 6b** — return-period curves (corrected DMC importance-sampling estimator).

Set `SCHEME` (default `"ck"`) and, if analysing an archived run, repoint `DNS_DIR`
/ `AI_DIR` at the top of the notebook. The original report runs (old emulator,
400×100, ~6 GB each) are in `res/report_data/` (git-ignored). Uncomment the two
lines in the notebook's setup cell to point at them directly.

---

## Notes & known discrepancies

- **Ck schedule:** `config.json` uses `C_schedule_ck = [0, 0, 0, 1.8, 2.2, 2.6]`,
  matching the report text and the Figure 5 legend.
- **Perturbation amplitude:** the report (§3.2) states √2×10⁻⁴, but the runs used
  `perturbation_magnitude = 0.003` (the value kept here). The perturbation is iid
  noise added to the surface-pressure spherical-harmonic coefficients. If you need
  to match the report's stated amplitude exactly, change this value — but note it
  alters the scientific result.
- **GCM fragility:** the `climt` physics in `core.py` (`get_dycore`, the physics
  worker loop, `extract_14_channels_fixed`, the spinup integration) is compiled
  C/Fortran and sensitive to change. Treat that code as fixed.
