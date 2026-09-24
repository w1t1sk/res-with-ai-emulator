# Rare Event Sampling with a Global Climate Model and AI Emulator

This repository tried to implement the **QDMC rare event sampling** for climate extremes:
an aquaplanet Global Climate Model (GCM, via [`climt`](https://github.com/CliMT/climt))
acts as the physics engine that evolves an ensemble of trajectories ("walkers"),
and an AI weather-forecast emulator scores those trajectories so that promising
ones are cloned and unpromising ones killed at fixed resampling times. The result
is a heavily biased ensemble that probes rare warm extremes far more efficiently
than a direct (unbiased) simulation of the same size.

The method here closely follows Lancelin et al. (2025), *AI-boosted rare event sampling to
characterize extreme weather*; the emulator architecture follows FuXi-ENS
(Zhong et al., 2025). See the project report for the full description.

---

## Experiment setup

The default experiment uses:

| Setting | Value |
|---|---|
| Target region | 3 x 3 grid box over a northwest India analog |
| Latitude range | 26.5 to 32.1 N |
| Longitude range | 70.3 to 75.9 E |
| Grid indices | `[20:23, 25:28]` |
| Observable | Final 7-day mean regional surface temperature |
| Walkers | 400 |
| Emulator ensemble members | 100 |
| Resampling interval | Every 5 days |
| Bias schedule | `[0, 0, 0, 1.8, 2.2, 2.6]` |
| Baseline | Direct simulation with the same walkers and no resampling |

The full experiment configuration is stored in:

```text
res/config.json
```

## Repository structure

```text
res_aiemulator_sem8/
├── README.md
├── requirements.txt
├── setup_env.sh
├── emulator/
│   ├── checkpoints/
│   └── ...
└── res/
    ├── config.json
    ├── core.py
    ├── driver_new.py
    ├── driver_old.py
    ├── run.py
    ├── submit.sh
    ├── analysis.ipynb
    ├── old_emulator/
    └── outputs/
```

The main files are:

```text
core.py
GCM setup, spinup, direct simulation, and rare event sampling logic

driver_new.py
Rare event sampling with the newer emulator

driver_old.py
Rare event sampling with the older lightweight emulator

run.py
Main command-line interface

submit.sh
Slurm launcher

analysis.ipynb
Basic analysis and plotting
```

## Environment setup

Activate the existing `climt` Conda environment:

```bash
conda activate climt
```

Then load the local environment setup:

```bash
source /home/nishidh/res_aiemulator_sem8/setup_env.sh
```

Install the remaining Python dependencies:

```bash
pip install -r requirements.txt
```

`climt` and `sympl` are provided through the Conda environment rather than through `pip`.

The setup script also adds the local compiled climate-model packages to `PYTHONPATH`.

## Running the pipeline

All main experiments are launched through:

```text
res/run.py
```

Move into the `res` directory first:

```bash
cd res
```

### 1. Generate the spinup

Run the aquaplanet model long enough to create the initial climate state:

```bash
python run.py spinup
```

This only needs to be done once unless the spinup state is changed.

### 2. Run the direct-simulation baseline

Run the same walker population without any rare-event resampling:

```bash
python run.py dns
```

This gives the reference distribution used to compare against the biased experiment.

### 3. Run rare event sampling

Using the newer emulator:

```bash
python run.py airres --emulator new
```

Using the older emulator:

```bash
python run.py airres --emulator old --scheme ck
```

## Running on Slurm

The same jobs can be submitted through `submit.sh`.

```bash
sbatch submit.sh spinup
sbatch submit.sh dns
sbatch submit.sh airres --emulator new
```

Additional arguments are forwarded to `run.py`.

## Small smoke test

For a quick test, reduce the number of walkers and emulator ensemble members.

```bash
python run.py dns \
    --walkers 4
```

```bash
python run.py airres \
    --emulator new \
    --walkers 4 \
    --members 2
```

This is useful for checking that the pipeline starts correctly before launching a full run.

## Emulator options

Two emulator implementations are available.

| Emulator | Option | Main use |
|---|---|---|
| New emulator | `--emulator new` | Current experiments |
| Old emulator | `--emulator old` | Earlier experiments and comparison |

The newer model uses:

```text
FuXiENSModelV3LatentHierSwin
```

The older model uses the lightweight `fuxiens` implementation.

Their checkpoints are stored separately:

```text
New emulator:
emulator/checkpoints/

Old emulator:
res/old_emulator/
```

Both are used in the same rare-event sampling workflow. The main difference is how the emulator is loaded and how one forecast call is performed.

## Outputs

Direct-simulation outputs are written to:

```text
res/outputs/global_dns_output/
```

Rare-event sampling outputs are written to:

```text
res/outputs/ai_res_nw_box_<scheme>_output/
```

The exact output directory depends on the selected sampling scheme.

## Analysis

The repository includes:

```text
res/analysis.ipynb
```

It can be used to inspect:

- sampled trajectories
- final temperature distributions
- direct-simulation and rare-event comparisons
- return-period behaviour

Run the notebook after both the direct baseline and rare-event experiment have completed.

## Useful configuration

Most experiment settings can be changed in:

```text
res/config.json
```

The default bias schedule is:

```text
[0, 0, 0, 1.8, 2.2, 2.6]
```

The first three resampling intervals are effectively unbiased. The rare-event bias is introduced gradually in the later intervals.

The default perturbation magnitude used by the experiment is:

```text
0.003
```

Because this parameter affects the sampled trajectories, changing it can change the resulting extreme-event distribution.

## Typical workflow

A complete run looks like this:

```bash
conda activate climt
source /home/nishidh/res_aiemulator_sem8/setup_env.sh

cd res

python run.py spinup

python run.py dns

python run.py airres --emulator new
```

For a cluster run:

```bash
sbatch submit.sh spinup
sbatch submit.sh dns
sbatch submit.sh airres --emulator new
```
