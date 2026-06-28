"""
core.py — Emulator-free foundation for the QDMC rare event sampling framework.

This module holds everything that does NOT depend on which AI emulator is used:
the climt GCM physics, the QDMC algorithms (splitting function + pivotal
resampling), walker lifecycle management, checkpoint I/O, and the two
emulator-free run entry points — the DNS baseline and the aquaplanet spinup.

The two emulator-specific drivers (driver_new.py, driver_old.py) import from
this module and add only the AI scoring + AI+RES loop.

Contents (in logical order):
  - Constants            (timestep, daily steps)
  - GCM physics          (climt aquaplanet dycore, state extraction)
  - Configuration        (region resolution, multi-region tracking, validation)
  - QDMC algorithms      (splitting function, pivotal resampling, ancestry)
  - Scoring helpers      (normalization, region means, fallback scoring)
  - Run helpers          (paths, spinup I/O, walker initialization)
  - Runtime tuning       (physics pool sizing, AI batch/microbatch sizing)
  - Checkpoint I/O       (atomic save/load for fault tolerance)
  - Walker lifecycle     (explosion recovery, trajectory filtering, seeds)
  - Physics workers      (subprocess GCM integration entry points)
  - Emulator-free runs   (run_dns_baseline, run_spinup)
"""

import copy
import gzip
import json
import multiprocessing
import os
import pickle
import time
import warnings
from datetime import timedelta

import climt
import numpy as np
import torch
import xarray as xr
from sympl import TimeDifferencingWrapper

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DAILY_STEPS = 72                        # number of 20-minute timesteps per day
MODEL_TIMESTEP = timedelta(minutes=20)


# ---------------------------------------------------------------------------
# GCM physics (climt aquaplanet)
# ---------------------------------------------------------------------------

def get_dycore():
    """Assemble and return the climt aquaplanet dynamical core.

    The configuration matches the FuXi training-climate setup:
    Emanuel convection, simple boundary layer (scaling_land=0.7),
    grey longwave radiation, slab ocean surface, 5 damped levels.
    """
    convection = climt.EmanuelConvection()
    boundary = TimeDifferencingWrapper(climt.SimpleBoundaryLayer(scaling_land=0.7))
    radiation = climt.GrayLongwaveRadiation()
    slab_surface = climt.SlabSurface()
    return climt.GFSDynamicalCore(
        [boundary, radiation, convection, slab_surface],
        number_of_damped_levels=5,
    )


def format_data_to_xarray_fixed(state):
    """Convert a climt state dict to an xarray Dataset with sorted pressure levels.

    Pressure levels are sorted by mean pressure and tiny duplicates are nudged
    apart to keep the coordinate monotone for interpolation.
    """
    variables_needed = [
        "air_temperature",
        "specific_humidity",
        "northward_wind",
        "eastward_wind",
        "surface_air_pressure",
        "surface_temperature",
    ]
    arrays = [state[v].rename(v).astype("float32") for v in variables_needed if v in state]
    data = xr.merge(arrays)
    data = data.assign_coords(
        {
            "lat": state["latitude"].values[:, 0],
            "lon": state["longitude"].values[0, :],
        }
    )

    if "mid_levels" in data.dims:
        mean_pressure = state["air_pressure"].mean(dim=("lat", "lon")).values.astype(np.float64)
        order = np.argsort(mean_pressure)
        sorted_pressure = mean_pressure[order].copy()

        # Nudge exact duplicates so the pressure coordinate stays strictly monotone.
        for i in range(1, len(sorted_pressure)):
            if sorted_pressure[i] <= sorted_pressure[i - 1]:
                sorted_pressure[i] = sorted_pressure[i - 1] + 1e-3

        data = data.isel(mid_levels=order)
        data = data.rename_dims({"mid_levels": "lev"}).assign_coords(lev=("lev", sorted_pressure))

    return data


def extract_14_channels_fixed(state) -> np.ndarray:
    """Extract the 14-channel (C, 64, 128) array used by the FuXi emulator.

    Channels (0-indexed):
      0    surface temperature
      1    surface pressure
      2-4  air temperature at 860/500/260 hPa
      5-7  specific humidity at 860/500/260 hPa
      8-10 eastward wind at 860/500/260 hPa
      11-13 northward wind at 860/500/260 hPa

    Returns zeros on interpolation failure so the caller can detect and handle
    exploded walkers gracefully.
    """
    ds_current = format_data_to_xarray_fixed(state)
    try:
        ds_interp = ds_current.interp(
            lev=[86000, 50000, 26000],
            method="linear",
            kwargs={"fill_value": "extrapolate"},
        )
    except Exception:
        return np.zeros((14, 64, 128), dtype=np.float32)

    tensor = np.zeros((14, 64, 128), dtype=np.float32)
    tensor[0] = ds_current["surface_temperature"].transpose("lat", "lon").values
    tensor[1] = ds_current["surface_air_pressure"].transpose("lat", "lon").values
    tensor[2:5] = ds_interp["air_temperature"].transpose("lev", "lat", "lon").values
    tensor[5:8] = ds_interp["specific_humidity"].transpose("lev", "lat", "lon").values
    tensor[8:11] = ds_interp["eastward_wind"].transpose("lev", "lat", "lon").values
    tensor[11:14] = ds_interp["northward_wind"].transpose("lev", "lat", "lon").values
    return tensor


# ---------------------------------------------------------------------------
# Configuration utilities
# ---------------------------------------------------------------------------

def get_base_dir() -> str:
    """Absolute directory containing this module (the res/ directory)."""
    return os.path.dirname(os.path.abspath(__file__))


def load_config() -> dict:
    """Load the single experiment config (config.json)."""
    with open(os.path.join(get_base_dir(), "config.json"), "r") as f:
        return json.load(f)


def get_tracked_regions(config: dict) -> dict:
    """Return the dict of tracked regions from config, or a singleton fallback."""
    tracked_regions = config.get("TRACKED_REGIONS")
    if tracked_regions:
        return tracked_regions
    return {
        config["REGION_NAME"]: {
            "label": config["REGION_NAME"].replace("_", " ").title(),
            "lat_start": config["LAT_S"],
            "lat_end": config["LAT_E"],
            "lon_start": config["LON_S"],
            "lon_end": config["LON_E"],
            "display_bounds": "Configured via top-level bounds",
        }
    }


def resolve_target_region(config: dict, target_region_name: str | None = None) -> dict:
    """Return a copy of config with REGION_NAME and grid bounds resolved.

    Resolution order: explicit argument → AIRES_TARGET_REGION env var →
    DEFAULT_REGION_NAME in config → REGION_NAME in config.
    """
    tracked_regions = get_tracked_regions(config)
    region_name = (
        target_region_name
        or os.environ.get("AIRES_TARGET_REGION")
        or config.get("DEFAULT_REGION_NAME")
        or config["REGION_NAME"]
    )
    if region_name not in tracked_regions:
        available = ", ".join(sorted(tracked_regions))
        raise ValueError(f"Unknown target region '{region_name}'. Available: {available}")

    region_spec = tracked_regions[region_name]
    resolved = copy.deepcopy(config)
    resolved["REGION_NAME"] = region_name
    resolved["TARGET_REGION_NAME"] = region_name
    resolved["REGION_LABEL"] = region_spec.get("label", region_name.replace("_", " ").title())
    resolved["LAT_S"] = region_spec["lat_start"]
    resolved["LAT_E"] = region_spec["lat_end"]
    resolved["LON_S"] = region_spec["lon_start"]
    resolved["LON_E"] = region_spec["lon_end"]
    resolved["ACTIVE_REGION_SPEC"] = region_spec
    return resolved


def validate_config(config: dict, schedule_key: str | None = None) -> None:
    """Raise ValueError if the config is internally inconsistent."""
    total_days = config["total_days"]
    resample_interval = config["resample_interval"]
    if total_days % resample_interval != 0:
        raise ValueError(
            f"total_days={total_days} must be divisible by resample_interval={resample_interval}."
        )

    if schedule_key is not None:
        expected_steps = total_days // resample_interval
        actual_steps = len(config[schedule_key])
        if actual_steps != expected_steps:
            raise ValueError(
                f"{schedule_key} has {actual_steps} entries but needs {expected_steps} "
                f"(total_days={total_days}, resample_interval={resample_interval})."
            )

    tracked_regions = get_tracked_regions(config)
    default_region = config.get("DEFAULT_REGION_NAME", config["REGION_NAME"])
    if default_region not in tracked_regions:
        available = ", ".join(sorted(tracked_regions))
        raise ValueError(
            f"DEFAULT_REGION_NAME '{default_region}' is not in tracked regions. Available: {available}"
        )

    for region_name, region_spec in tracked_regions.items():
        for key in ("lat_start", "lat_end", "lon_start", "lon_end"):
            if key not in region_spec:
                raise ValueError(f"Region '{region_name}' is missing required key '{key}'.")
        if not (0 <= region_spec["lat_start"] < region_spec["lat_end"] <= 64):
            raise ValueError(f"Region '{region_name}' has invalid latitude bounds: {region_spec}")
        if not (0 <= region_spec["lon_start"] < region_spec["lon_end"] <= 128):
            raise ValueError(f"Region '{region_name}' has invalid longitude bounds: {region_spec}")


# ---------------------------------------------------------------------------
# QDMC algorithms
# ---------------------------------------------------------------------------

def compute_splitting_function(raw_scores: np.ndarray, c_k: float) -> np.ndarray:
    """Compute the splitting potential V_k = c_k * (score - mean) / std.

    When c_k == 0 the potential is zero everywhere, so no resampling occurs.
    """
    if c_k == 0.0:
        return np.zeros_like(raw_scores)
    mu = np.mean(raw_scores)
    sigma = np.std(raw_scores) + 1e-8
    return c_k * ((raw_scores - mu) / sigma)


def pivotal_resampling(normalized_weights: np.ndarray) -> np.ndarray:
    """Pivotal (deterministic) resampling of N walkers with given relative weights.

    Guarantees exactly N clones in output and minimises variance relative to
    multinomial resampling. See Algorithm 1 in the report (after Lancelin et al.).

    Parameters
    ----------
    normalized_weights:
        Array of floats summing to N (the population size).

    Returns
    -------
    clones : ndarray of int
        Number of offspring for each walker; sums to N.
    """
    n = len(normalized_weights)
    base_clones = np.floor(normalized_weights)
    delta = normalized_weights - base_clones
    active_indices = [i for i in range(n) if 0.0 < delta[i] < 1.0]

    while len(active_indices) >= 2:
        i, j = active_indices[0], active_indices[1]
        total = delta[i] + delta[j]
        if total <= 1.0:
            if np.random.rand() < (delta[i] / total):
                delta[i], delta[j] = total, 0.0
            else:
                delta[j], delta[i] = total, 0.0
        else:
            if np.random.rand() < ((1.0 - delta[j]) / (2.0 - total)):
                delta[i], delta[j] = 1.0, total - 1.0
            else:
                delta[i], delta[j] = total - 1.0, 1.0
        active_indices = [idx for idx in range(n) if 0.0 < delta[idx] < 1.0]

    for idx in active_indices:
        delta[idx] = np.round(delta[idx])

    clones = (base_clones + delta).astype(int)
    if clones.sum() != n:
        raise RuntimeError(f"Pivotal resampling produced {clones.sum()} walkers instead of {n}.")
    return clones


def resample_walkers(walkers: list, v_k: np.ndarray, config: dict):
    """Apply importance weights and pivotal resampling to the walker population.

    Returns
    -------
    new_walkers : list
        Resampled population (same size as input).
    clones : ndarray of int
        Offspring count per original walker.
    ancestry : list of int
        Parent walker id for each entry in new_walkers.
    incremental_weights : ndarray of float
        exp(v_k[i] - V_{k-1}[i]) for each original walker.
    w_bar_k : float
        Mean incremental weight (normalisation factor for the IS estimator).
    """
    n_walkers = len(walkers)
    log_incremental_weights = np.array(
        [v_k[i] - walkers[i]["V_k_minus_1"] for i in range(n_walkers)],
        dtype=np.float64,
    )

    with np.errstate(over="ignore", under="ignore"):
        incremental_weights = np.exp(log_incremental_weights)

    # Numerically stable computation of the mean weight.
    max_log = float(np.max(log_incremental_weights))
    shifted = np.exp(log_incremental_weights - max_log)
    shifted_mean = float(np.mean(shifted))
    normalized = shifted / (shifted_mean + 1e-8)
    w_bar_log = np.log(shifted_mean + 1e-300) + max_log
    w_bar_k = float(np.exp(w_bar_log)) if w_bar_log < 700 else float("inf")

    clones = pivotal_resampling(normalized)
    new_walkers: list = []
    ancestry: list[int] = []
    new_id_tracker = 0

    for i in range(n_walkers):
        for clone_idx in range(clones[i]):
            new_walker = copy.deepcopy(walkers[i])
            delta_v = v_k[i] - walkers[i]["V_k_minus_1"]
            new_walker["V_k_minus_1"] = float(v_k[i])
            new_walker["id"] = new_id_tracker
            new_walker["parent_id"] = walkers[i]["id"]
            new_walker["log_weight"] = walkers[i].get("log_weight", 0.0) + float(delta_v)
            new_walker["pert_flag"] = 1 if clone_idx > 0 else 0
            new_walkers.append(new_walker)
            ancestry.append(walkers[i]["id"])
            new_id_tracker += 1

    return new_walkers, clones, ancestry, incremental_weights, w_bar_k


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def normalize_history_pair(
    history_pair: torch.Tensor,
    data_mean: torch.Tensor,
    data_std: torch.Tensor,
) -> torch.Tensor:
    """Z-score normalise a concatenated (t-1, t) daily-mean state pair.

    Parameters
    ----------
    history_pair:
        Shape (B, 28, H, W); first 14 channels are day t-1, last 14 are day t.

    Returns
    -------
    Normalised tensor of same shape.
    """
    history_tm1 = (history_pair[:, :14] - data_mean) / (data_std + 1e-6)
    history_t = (history_pair[:, 14:] - data_mean) / (data_std + 1e-6)
    return torch.cat([history_tm1, history_t], dim=1)


def region_mean_from_day(day_tensor: np.ndarray, config: dict) -> float:
    """Mean surface temperature (channel 0) over the configured target region."""
    return float(
        np.mean(
            day_tensor[
                0,
                config["LAT_S"]: config["LAT_E"],
                config["LON_S"]: config["LON_E"],
            ]
        )
    )


def region_mean_from_spec(day_tensor: np.ndarray, region_spec: dict) -> float:
    """Mean surface temperature (channel 0) over an arbitrary region spec dict."""
    return float(
        np.mean(
            day_tensor[
                0,
                region_spec["lat_start"]: region_spec["lat_end"],
                region_spec["lon_start"]: region_spec["lon_end"],
            ]
        )
    )


def final_observable_from_trajectory(trajectory: list, region_spec: dict, l_days: int) -> float:
    """Compute A_{L,t_f}: mean surface temperature over the last L days of a trajectory."""
    final_l_days = trajectory[-l_days:]
    region_temps = [region_mean_from_spec(day, region_spec) for day in final_l_days]
    return float(np.mean(region_temps))


def fallback_recent_score(walker: dict, config: dict) -> float:
    """Estimate the walker score from its trajectory tail when emulator data is unavailable."""
    if walker.get("trajectory"):
        tail = walker["trajectory"][-min(config["L"], len(walker["trajectory"])):]
        return float(np.mean([region_mean_from_day(day, config) for day in tail]))
    daily_history = walker.get("daily_history_tensors", [])
    if daily_history:
        return region_mean_from_day(daily_history[-1], config)
    return 0.0


def reset_previous_potential(walkers: list, value: float = 0.0) -> None:
    """Set V_{k-1} to a fixed value for all walkers (used when c_k == 0)."""
    for walker in walkers:
        walker["V_k_minus_1"] = float(value)


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------

def expected_traj_days(config: dict) -> int:
    """Total trajectory length including the L-day observable window."""
    return int(config["total_days"] + config["L"])


def spinup_paths(config: dict) -> tuple[str, str]:
    """Return (spinup_pkl_path, spinup_metadata_json_path) from config relative paths."""
    base_dir = get_base_dir()
    return (
        os.path.join(base_dir, config["spinup_relpath"]),
        os.path.join(base_dir, config["spinup_metadata_relpath"]),
    )


def load_spinup_payload(config: dict) -> tuple:
    """Load and return (state, spec, metadata) from the aquaplanet spinup file."""
    spinup_path, spinup_metadata_path = spinup_paths(config)
    if not os.path.exists(spinup_path):
        raise FileNotFoundError(
            f"Spinup file not found at '{spinup_path}'. Run `python run.py spinup` first."
        )

    opener = gzip.open if spinup_path.endswith(".gz") else open
    with opener(spinup_path, "rb") as f:
        state, spec = pickle.load(f)

    metadata: dict = {}
    if os.path.exists(spinup_metadata_path):
        with open(spinup_metadata_path, "r") as f:
            metadata = json.load(f)
    return state, spec, metadata


def build_initial_walkers(config: dict) -> list:
    """Create N_walkers from the shared spinup state with deterministic perturbation seeds.

    All schemes (DNS and AI+RES) use the same perturbation bank so that walker i
    starts from an identical state across experiments, enabling fair comparison
    of the C_k schedule effect.
    """
    base_state, base_spec, spinup_metadata = load_spinup_payload(config)
    seed_base = int(config.get("perturbation_seed_base", 1))
    perturbation_mode = str(config.get("perturbation_mode", "iid_uniform"))
    perturbation_magnitude = float(config["perturbation_magnitude"])

    walkers = []
    for i in range(config["N_walkers"]):
        walkers.append(
            {
                "id": i,
                "parent_id": i,
                "state": copy.deepcopy(base_state),
                "spec": copy.deepcopy(base_spec),
                "trajectory": [],
                "daily_history_tensors": [],
                "V_k_minus_1": 0.0,
                "log_weight": 0.0,
                "pert_flag": 1,
                "perturbation_magnitude": perturbation_magnitude,
                "perturbation_seed": seed_base + i,
                "perturbation_mode": perturbation_mode,
                "initial_walker_index": i,
                "spinup_time": str(base_state.get("time", "")),
                "spinup_metadata": {
                    "spinup_years": spinup_metadata.get("spinup_years", config.get("spinup_years")),
                    "soil_conf": spinup_metadata.get("soil_conf", config.get("spinup_soil_conf")),
                    "rad_conf": spinup_metadata.get("rad_conf", config.get("spinup_rad_conf")),
                },
            }
        )
    return walkers


# ---------------------------------------------------------------------------
# Runtime tuning helpers (SLURM env-var overrides)
# ---------------------------------------------------------------------------

def _runtime_int(config: dict, env_name: str, config_key: str, default: int) -> int:
    """Read an integer setting from an env var, falling back to config, then default."""
    raw = os.environ.get(env_name) or config.get(config_key, default)
    value = int(raw)
    if value < 0:
        raise ValueError(f"Setting {env_name}/{config_key} must be non-negative, got {value}.")
    return value


def resolve_physics_worker_count(config: dict) -> int:
    """Number of parallel physics worker processes to launch."""
    available = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
    configured = _runtime_int(
        config, "AIRES_MAX_PHYSICS_WORKERS", "physics_max_workers", available
    )
    return max(1, min(available, configured, int(config["N_walkers"])))


def resolve_physics_pool_chunksize(config: dict) -> int:
    """Pool.map chunksize for physics jobs."""
    return max(1, _runtime_int(config, "AIRES_PHYSICS_POOL_CHUNKSIZE", "physics_pool_chunksize", 1))


def resolve_physics_maxtasksperchild(config: dict) -> int | None:
    """Max tasks per worker child process (None = unbounded)."""
    configured = _runtime_int(
        config, "AIRES_PHYSICS_MAXTASKSPERCHILD", "physics_maxtasksperchild", 8
    )
    return None if configured == 0 else max(1, configured)


def resolve_ai_scoring_walker_batch_size(config: dict, n_walkers: int) -> int:
    """Number of walkers scored per GPU batch."""
    configured = _runtime_int(
        config, "AIRES_AI_WALKER_BATCH_SIZE", "ai_scoring_walker_batch_size", 4
    )
    return max(1, min(configured, n_walkers))


def resolve_ai_member_microbatch_size(config: dict) -> int:
    """Number of ensemble members processed per emulator forward pass (memory budget)."""
    return max(
        1, _runtime_int(config, "AIRES_AI_MEMBER_MICROBATCH_SIZE", "ai_member_microbatch_size", 16)
    )


# ---------------------------------------------------------------------------
# Multiprocessing pool helpers
# ---------------------------------------------------------------------------

def build_worker_args(walkers: list, days_to_run: int, tmp_dir: str) -> list:
    """Serialise walkers to tmp_dir and build the args list for pool.map."""
    args_list = []
    for walker in walkers:
        in_file = os.path.join(tmp_dir, f"in_walker_{walker['id']}.pkl")
        with open(in_file, "wb") as f:
            pickle.dump(walker, f)
        args_list.append((walker["id"], days_to_run, tmp_dir))
    return args_list


def make_spawn_pool(max_workers: int, maxtasksperchild: int | None):
    """Create a spawn-context multiprocessing pool."""
    ctx = multiprocessing.get_context("spawn")
    return ctx.Pool(processes=max_workers, maxtasksperchild=maxtasksperchild)


def run_pool_map(pool, worker_fn, args_list: list, chunksize: int) -> list:
    """Run pool.map, or fall back to sequential execution if pool is None."""
    if pool is None:
        return [worker_fn(args) for args in args_list]
    return pool.map(worker_fn, args_list, chunksize=chunksize)


# ---------------------------------------------------------------------------
# Checkpoint management (atomic writes via tmp → rename)
# ---------------------------------------------------------------------------

def ai_checkpoint_paths(out_dir: str) -> tuple[str, str]:
    """Return (checkpoint_pkl_path, checkpoint_meta_json_path)."""
    return (
        os.path.join(out_dir, "run_checkpoint.pkl"),
        os.path.join(out_dir, "run_checkpoint.json"),
    )


def load_ai_checkpoint(checkpoint_path: str) -> dict | None:
    """Load a run checkpoint, or return None if it does not exist."""
    if not os.path.exists(checkpoint_path):
        return None
    with open(checkpoint_path, "rb") as f:
        return pickle.load(f)


def save_ai_checkpoint(checkpoint_path: str, checkpoint_meta_path: str, payload: dict) -> None:
    """Atomically save a run checkpoint (write to tmp then rename)."""
    tmp_path = f"{checkpoint_path}.tmp"
    tmp_meta_path = f"{checkpoint_meta_path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f)
    with open(tmp_meta_path, "w") as f:
        json.dump(
            {
                "scheme_name": payload.get("scheme_name"),
                "region_name": payload.get("region_name"),
                "phase": payload.get("phase"),
                "next_step_idx": payload.get("next_step_idx"),
                "num_walkers": len(payload.get("walkers", [])),
                "diagnostic_steps_saved": len(
                    payload.get("diagnostics", {}).get("global_weights", [])
                ),
            },
            f,
            indent=2,
        )
    os.replace(tmp_path, checkpoint_path)
    os.replace(tmp_meta_path, checkpoint_meta_path)


def clear_ai_checkpoint(checkpoint_path: str, checkpoint_meta_path: str) -> None:
    """Remove checkpoint files after a successful run completes."""
    for path in (checkpoint_path, checkpoint_meta_path):
        if os.path.exists(path):
            os.remove(path)


# ---------------------------------------------------------------------------
# Walker lifecycle helpers
# ---------------------------------------------------------------------------

def collect_worker_outputs(tmp_dir: str, completed_ids: list[int], walkers: list) -> None:
    """Load subprocess outputs back into the walkers list in-place."""
    for idx, w_id in enumerate(completed_ids):
        out_file = os.path.join(tmp_dir, f"out_walker_{w_id}.pkl")
        with open(out_file, "rb") as f:
            walkers[idx] = pickle.load(f)
        os.remove(out_file)


def replace_exploded_walkers(walkers: list, step_label: str, diagnostics: dict) -> list:
    """Replace any exploded walkers (pert_flag==2) by cloning healthy branches.

    Records replacement details in ``diagnostics["explosion_replacements_per_step"]``.
    Raises RuntimeError if all walkers have exploded (unrecoverable state).
    """
    exploded_indices = [idx for idx, w in enumerate(walkers) if w.get("pert_flag") == 2]
    if not exploded_indices:
        diagnostics["explosion_replacements_per_step"].append([])
        return walkers

    healthy_templates = [copy.deepcopy(w) for w in walkers if w.get("pert_flag") != 2]
    if not healthy_templates:
        raise RuntimeError(
            f"All walkers exploded before recovery could be applied at '{step_label}'."
        )

    replacement_details = []
    for idx in exploded_indices:
        dead_walker = walkers[idx]
        donor = copy.deepcopy(healthy_templates[np.random.randint(len(healthy_templates))])
        donor_source_id = donor.get("id")
        donor["id"] = dead_walker["id"]
        donor["parent_id"] = donor_source_id
        donor["replacement_parent_id"] = donor_source_id
        donor["explosion_recovery_step"] = step_label
        donor["pert_flag"] = 0
        walkers[idx] = donor
        replacement_details.append(
            {
                "walker_id": dead_walker.get("id"),
                "donor_id": donor_source_id,
                "trajectory_days_before_replacement": len(dead_walker.get("trajectory", [])),
            }
        )

    diagnostics["explosion_replacements_per_step"].append(replacement_details)
    print(
        f"Recovered {len(replacement_details)} exploded walker(s) at '{step_label}' "
        f"by cloning healthy branches."
    )
    return walkers


def filter_complete_walkers(walkers: list, config: dict) -> tuple[list, list]:
    """Separate walkers into complete and incomplete/exploded sets.

    Returns
    -------
    keep : list
        Walkers whose trajectory covers the full simulation period and are not exploded.
    removed : list
        Metadata dicts describing walkers that were excluded.
    """
    keep = []
    removed = []
    target_days = expected_traj_days(config)
    for walker in walkers:
        traj_len = len(walker.get("trajectory", []))
        if traj_len >= target_days and walker.get("pert_flag") != 2:
            keep.append(walker)
        else:
            removed.append(
                {
                    "id": walker.get("id"),
                    "trajectory_days": traj_len,
                    "pert_flag": walker.get("pert_flag"),
                }
            )
    return keep, removed


def refresh_branch_perturbation_seeds(walkers: list, config: dict, branch_generation: int) -> None:
    """Assign fresh deterministic perturbation seeds to newly cloned walkers.

    Cloned walkers (pert_flag==1) receive seeds unique per generation and per
    position in the walker list, ensuring reproducibility.
    """
    seed_base = int(config.get("perturbation_seed_base", 1))
    for local_idx, walker in enumerate(walkers):
        if walker.get("pert_flag") == 1:
            walker["perturbation_seed"] = seed_base + branch_generation * 1000 + local_idx


# ---------------------------------------------------------------------------
# Physics workers (run in spawned subprocesses — no GPU)
# ---------------------------------------------------------------------------

def _physics_worker_common(args: tuple, keep_state: bool) -> int:
    """Integrate one walker for a fixed number of days using the climt dycore.

    Designed to be called by multiprocessing.Pool in a freshly spawned
    subprocess. CUDA is disabled in workers to avoid conflicts with the
    GPU-based emulator in the parent process. Walker state is read from and
    written to ``tmp_dir`` as pickle files so only picklable objects cross the
    process boundary.

    Parameters
    ----------
    args:
        (walker_id, days_to_run, tmp_dir)
    keep_state:
        If True, save the GCM state after the run (needed between resampling
        steps). If False, discard it (DNS baseline — saves memory).

    Returns
    -------
    walker_id : int
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    w_id, days_to_run, tmp_dir = args
    steps_to_run = days_to_run * DAILY_STEPS

    in_file = os.path.join(tmp_dir, f"in_walker_{w_id}.pkl")
    with open(in_file, "rb") as f:
        walker = pickle.load(f)

    # Initialise the dycore inside the subprocess so each worker gets its own
    # C/Fortran state without sharing memory with other processes.
    dycore = get_dycore()
    grid = climt.get_grid(nx=128, ny=64)
    my_state = climt.get_default_state([dycore], grid_state=grid)
    dycore(my_state, MODEL_TIMESTEP)

    state = walker["state"]
    spec = copy.deepcopy(walker["spec"])
    my_state.update(state)
    dycore._gfs_cython.reinit_spectral_arrays(spec)

    perturbation_seed = int(walker.get("perturbation_seed", walker["id"]))
    rng = np.random.default_rng(perturbation_seed)

    if walker.get("pert_flag", 0) == 1:
        dycore.set_flag(False)
        pert_mag = float(walker["perturbation_magnitude"])
        n_pert = rng.uniform(-1.0, 1.0, np.shape(spec[4])) * pert_mag
        spec[4][:] = (spec[4] + n_pert)[:]
        dycore._gfs_cython.reinit_spectral_arrays(spec)
    else:
        dycore.set_flag(True)

    daily_accumulator = np.zeros((14, 64, 128), dtype=np.float32)
    daily_history = list(walker.get("daily_history_tensors", []))

    for step in range(steps_to_run):
        if walker.get("pert_flag", 0) == 1 and step == 1:
            dycore.set_flag(True)

        diag, my_state = dycore(my_state, MODEL_TIMESTEP)
        my_state.update(diag)
        my_state["time"] += MODEL_TIMESTEP

        if np.isnan(my_state["air_temperature"].values).any():
            print(f"Walker {walker['id']} exploded. Terminating branch safely.")
            walker["pert_flag"] = 2
            break

        daily_accumulator += extract_14_channels_fixed(my_state)

        current_step_count = step + 1
        if current_step_count % DAILY_STEPS == 0:
            daily_avg = daily_accumulator / DAILY_STEPS
            walker["trajectory"].append(daily_avg.copy())
            daily_history.append(daily_avg.copy())
            if len(daily_history) > 2:
                daily_history = daily_history[-2:]
            daily_accumulator.fill(0.0)

    walker["daily_history_tensors"] = daily_history
    if keep_state:
        walker["state"] = my_state
        walker["spec"] = dycore._gfs_cython.get_spectral_arrays()
        if walker.get("pert_flag", 0) != 2:
            walker["pert_flag"] = 0
    else:
        for key in ("state", "spec"):
            walker.pop(key, None)

    out_file = os.path.join(tmp_dir, f"out_walker_{w_id}.pkl")
    with open(out_file, "wb") as f:
        pickle.dump(walker, f)

    return w_id


def step_physics_chunk_ai(args: tuple) -> int:
    """Physics worker for AI+RES steps: preserves GCM state for resampling."""
    return _physics_worker_common(args, keep_state=True)


def step_single_walker_dns(args: tuple) -> int:
    """Physics worker for DNS runs: discards GCM state after integration."""
    return _physics_worker_common(args, keep_state=False)


# ---------------------------------------------------------------------------
# Emulator-free run: DNS baseline
# ---------------------------------------------------------------------------

def build_diagnostics(config: dict, scheme: str, c_schedule: list, emulator: str) -> dict:
    """Build the initial diagnostics record for a fresh AI+RES run.

    Emulator-agnostic bookkeeping shared by both drivers; the per-step lists are
    appended to as the QDMC loop progresses.
    """
    return {
        "scheme": scheme,
        "emulator": emulator,
        "raw_scores_per_step": [],
        "V_k_per_step": [],
        "clone_counts_per_step": [],
        "ancestry_per_step": [],
        "per_walker_weights_per_step": [],
        "score_meta_per_step": [],
        "global_weights": [],
        "explosion_replacements_per_step": [],
        "final_excluded_walkers": [],
        "C_schedule": c_schedule,
        "N_walkers": config["N_walkers"],
        "M_members": config["M_members"],
        "total_days": config["total_days"],
        "L": config["L"],
        "resample_interval": config["resample_interval"],
        "region": {
            "name": config["REGION_NAME"],
            "label": config["REGION_LABEL"],
            "LAT_S": config["LAT_S"],
            "LAT_E": config["LAT_E"],
            "LON_S": config["LON_S"],
            "LON_E": config["LON_E"],
            "display_bounds": config["ACTIVE_REGION_SPEC"].get("display_bounds"),
        },
        "tracked_regions": get_tracked_regions(config),
        "score_settings": {
            "history_type": "daily_mean",
            "input_normalization": "per-day z-score",
            "score_rule": str(config.get("score_rule", "member_mean_final_L_day_temperature")),
            "perturbation_magnitude": float(config["perturbation_magnitude"]),
            "perturbation_mode": str(config.get("perturbation_mode", "iid_uniform")),
            "experiment_seed": int(config.get("experiment_seed", 314159)),
            "config_file": "config.json",
        },
    }


def finalize_and_save_airres(
    walkers: list,
    config: dict,
    diagnostics: dict,
    out_dir: str,
    checkpoint_path: str,
    checkpoint_meta_path: str,
) -> None:
    """Compute final observables, strip heavy fields, and write AI+RES outputs.

    Emulator-agnostic: called identically by driver_new and driver_old after
    the final post-t_f integration chunk.
    """
    walkers = sorted(walkers, key=lambda w: w["id"])
    walkers, removed_final = filter_complete_walkers(walkers, config)
    diagnostics["final_excluded_walkers"] = removed_final
    if removed_final:
        print(f"Excluded {len(removed_final)} incomplete/exploded walker(s) before saving.")

    tracked_regions = get_tracked_regions(config)
    for walker in walkers:
        walker["A_L_tf_by_region"] = {
            rname: final_observable_from_trajectory(walker["trajectory"], rspec, config["L"])
            for rname, rspec in tracked_regions.items()
        }
        walker["A_L_tf"] = walker["A_L_tf_by_region"][config["REGION_NAME"]]

    print("\nSimulation complete. Saving outputs...")
    for walker in walkers:
        for key in ("state", "spec", "daily_history_tensors"):
            walker.pop(key, None)

    with open(os.path.join(out_dir, "ai_res_final_walkers.pkl"), "wb") as f:
        pickle.dump(walkers, f)
    with open(os.path.join(out_dir, "global_weights.pkl"), "wb") as f:
        pickle.dump(diagnostics["global_weights"], f)
    with open(os.path.join(out_dir, "diagnostics.pkl"), "wb") as f:
        pickle.dump(diagnostics, f)
    clear_ai_checkpoint(checkpoint_path, checkpoint_meta_path)
    print(f"Outputs saved to: {out_dir}")


def run_dns_baseline(config: dict, target_region_name: str | None = None) -> None:
    """Run the DNS baseline: N unbiased trajectories with no resampling.

    Every walker is integrated for (total_days + L) days. The resulting
    empirical distribution is the reference against which AI+RES is compared.
    Outputs are written to outputs/global_dns_output/.
    """
    config = resolve_target_region(config, target_region_name=target_region_name)
    validate_config(config)
    tracked_regions = get_tracked_regions(config)
    default_region = config.get("DEFAULT_REGION_NAME", config["REGION_NAME"])

    base_dir = get_base_dir()
    out_dir = os.path.join(base_dir, "outputs", "global_dns_output")
    tmp_dir = os.path.join(base_dir, "tmp", "global_dns_tmp")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    max_workers = resolve_physics_worker_count(config)
    pool_chunksize = resolve_physics_pool_chunksize(config)
    maxtasksperchild = resolve_physics_maxtasksperchild(config)
    np.random.seed(int(config.get("experiment_seed", 314159)))
    walkers = build_initial_walkers(config)
    total_sim_days = config["total_days"] + config["L"]
    args_list = build_worker_args(walkers, total_sim_days, tmp_dir)

    print(
        f"\n=== DNS Baseline | region={config['REGION_LABEL']} | "
        f"N={config['N_walkers']} walkers | {total_sim_days} days | workers={max_workers} ==="
    )

    with make_spawn_pool(max_workers, maxtasksperchild) as pool:
        completed_ids = run_pool_map(pool, step_single_walker_dns, args_list, pool_chunksize)

    final_walkers = []
    for w_id in completed_ids:
        out_file = os.path.join(tmp_dir, f"out_walker_{w_id}.pkl")
        with open(out_file, "rb") as f:
            walker = pickle.load(f)
        os.remove(out_file)
        final_walkers.append(walker)

    final_walkers = sorted(final_walkers, key=lambda w: w["id"])
    final_walkers, removed_final = filter_complete_walkers(final_walkers, config)
    if removed_final:
        print(f"Excluded {len(removed_final)} incomplete/exploded DNS walker(s) before saving.")

    for walker in final_walkers:
        walker["A_L_tf_by_region"] = {
            rname: final_observable_from_trajectory(walker["trajectory"], rspec, config["L"])
            for rname, rspec in tracked_regions.items()
        }
        walker["A_L_tf"] = walker["A_L_tf_by_region"][default_region]
        walker.pop("daily_history_tensors", None)

    with open(os.path.join(out_dir, "dns_baseline_global.pkl"), "wb") as f:
        pickle.dump(final_walkers, f)

    print(f"DNS baseline complete. Outputs saved to: {out_dir}")


# ---------------------------------------------------------------------------
# Emulator-free run: aquaplanet spinup
# ---------------------------------------------------------------------------

def _spinup_checkpoint_paths(spinup_path: str) -> tuple[str, str]:
    """Return (checkpoint_pkl_gz_path, checkpoint_json_path) for the spinup file."""
    if spinup_path.endswith(".pkl.gz"):
        stem = spinup_path[:-7]
    elif spinup_path.endswith(".gz"):
        stem = spinup_path[:-3]
    else:
        stem = spinup_path
    return f"{stem}.checkpoint.pkl.gz", f"{stem}.checkpoint.json"


def run_spinup(config: dict) -> None:
    """Generate the aquaplanet spinup state used to initialise all walkers.

    Integrates the climt aquaplanet GCM for the configured number of years to
    reach statistical equilibrium, then saves the final state. Checkpointing is
    built in: re-running after an interruption resumes from the last checkpoint.

    NOTE: The GCM integration below is fragile compiled physics — the sequence
    of climt calls is preserved verbatim from the original spinup script.
    """
    spinup_path, spinup_metadata_path = spinup_paths(config)
    checkpoint_path, checkpoint_meta_path = _spinup_checkpoint_paths(spinup_path)
    os.makedirs(os.path.dirname(spinup_path), exist_ok=True)
    os.makedirs(os.path.dirname(spinup_metadata_path), exist_ok=True)

    soil_conf = float(config.get("spinup_soil_conf", 0.7))
    rad_conf = float(config.get("spinup_rad_conf", 6.0))
    sw_max = float(config.get("spinup_sw_max", 150.0))
    spinup_years = int(config.get("spinup_years", 3))
    progress_every_days = int(config.get("spinup_progress_every_days", 10))
    checkpoint_every_days = int(config.get("spinup_checkpoint_every_days", 30))

    print("1. Initializing aquaplanet training-match spinup...", flush=True)
    model_time_step = timedelta(minutes=20)

    convection = climt.EmanuelConvection()
    boundary = TimeDifferencingWrapper(climt.SimpleBoundaryLayer(scaling_land=soil_conf))
    radiation = climt.GrayLongwaveRadiation()
    slab_surface = climt.SlabSurface()
    optical_depth = climt.Frierson06LongwaveOpticalDepth(
        linear_optical_depth_parameter=1,
        longwave_optical_depth_at_equator=rad_conf,
    )
    dycore = climt.GFSDynamicalCore(
        [boundary, radiation, convection, slab_surface],
        number_of_damped_levels=5,
    )

    grid = climt.get_grid(nx=128, ny=64)
    my_state = climt.get_default_state([dycore], grid_state=grid)

    latitudes = my_state["latitude"].values
    sw_flux_profile = sw_max * (1 + 1.4 * (0.25 * (1 - 3 * np.sin(np.radians(latitudes - 10)) ** 2)))
    sw_flux_profile[np.where(latitudes < -80)] = sw_max * (1 + 1.4 / 4 * (1 - 3))

    my_state["downwelling_shortwave_flux_in_air"].values[:] = sw_flux_profile[np.newaxis, :]
    my_state.update(optical_depth(my_state))
    my_state["surface_temperature"].values[:] = float(config.get("spinup_surface_temperature", 290.0))
    my_state["ocean_mixed_layer_thickness"].values[:] = float(
        config.get("spinup_ocean_mixed_layer_thickness", 2.0)
    )
    my_state["soil_layer_thickness"].values[:] = float(config.get("spinup_soil_layer_thickness", 1.0))
    my_state["eastward_wind"].values[:] = np.random.randn(*my_state["eastward_wind"].shape)

    start_time = time.time()
    steps_per_year = 26280
    total_steps = steps_per_year * spinup_years
    progress_every_steps = max(1, progress_every_days * 72)
    checkpoint_every_steps = max(1, checkpoint_every_days * 72)
    start_step = 0

    checkpoint_payload = None
    if os.path.exists(checkpoint_path):
        with gzip.open(checkpoint_path, "rb") as f:
            checkpoint_payload = pickle.load(f)
    if checkpoint_payload is not None:
        print("2. Found existing checkpoint. Resuming spinup...", flush=True)
        my_state = checkpoint_payload["state"]
        dycore._gfs_cython.reinit_spectral_arrays(checkpoint_payload["spec"])
        start_step = int(checkpoint_payload.get("step_index", 0))
        print(f"   Resuming from step {start_step}/{total_steps} (day {start_step / 72:.1f}).", flush=True)

    print(
        f"3. Starting aquaplanet spinup integration for {spinup_years} years ({total_steps} steps)...",
        flush=True,
    )

    for i in range(start_step, total_steps):
        diag, my_state = dycore(my_state, model_time_step)
        my_state.update(diag)
        my_state["time"] += model_time_step

        if (i + 1) % progress_every_steps == 0 or i + 1 == total_steps:
            elapsed_minutes = (time.time() - start_time) / 60.0
            print(
                f"   Progress: {i + 1}/{total_steps} steps (day {(i + 1) / 72:.1f}). "
                f"Elapsed: {elapsed_minutes:.1f} min.",
                flush=True,
            )

        if (i + 1) % checkpoint_every_steps == 0 and i + 1 < total_steps:
            spec = dycore._gfs_cython.get_spectral_arrays()
            payload = {
                "state": my_state,
                "spec": spec,
                "step_index": int(i + 1),
                "elapsed_hours": float((time.time() - start_time) / 3600.0),
            }
            tmp_path = f"{checkpoint_path}.tmp"
            tmp_meta_path = f"{checkpoint_meta_path}.tmp"
            with gzip.open(tmp_path, "wb") as f:
                pickle.dump(payload, f)
            with open(tmp_meta_path, "w") as f:
                json.dump(
                    {
                        "step_index": int(i + 1),
                        "total_steps": int(total_steps),
                        "elapsed_hours": float((time.time() - start_time) / 3600.0),
                        "time": str(my_state.get("time", "")),
                    },
                    f,
                    indent=2,
                )
            os.replace(tmp_path, checkpoint_path)
            os.replace(tmp_meta_path, checkpoint_meta_path)
            print(f"   Checkpoint saved at step {i + 1}/{total_steps} (day {(i + 1) / 72:.1f}).", flush=True)

    print("4. Spinup complete. Saving state...", flush=True)
    spec = dycore._gfs_cython.get_spectral_arrays()
    with gzip.open(spinup_path, "wb") as f:
        pickle.dump([my_state, spec], f)

    for path in (checkpoint_path, checkpoint_meta_path):
        if os.path.exists(path):
            os.remove(path)

    metadata = {
        "spinup_years": spinup_years,
        "soil_conf": soil_conf,
        "rad_conf": rad_conf,
        "sw_max": sw_max,
        "surface_temperature": float(config.get("spinup_surface_temperature", 290.0)),
        "ocean_mixed_layer_thickness": float(config.get("spinup_ocean_mixed_layer_thickness", 2.0)),
        "soil_layer_thickness": float(config.get("spinup_soil_layer_thickness", 1.0)),
        "spinup_path": spinup_path,
        "progress_every_days": progress_every_days,
        "checkpoint_every_days": checkpoint_every_days,
        "notes": "Aquaplanet spinup matched to the FuXi training-climate generation. No land-band modification.",
        "elapsed_hours": (time.time() - start_time) / 3600.0,
    }
    with open(spinup_metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved aquaplanet spinup to {spinup_path}", flush=True)
    print(f"Saved spinup metadata to {spinup_metadata_path}", flush=True)
    print(
        f"Successfully generated aquaplanet spinup in {(time.time() - start_time) / 60.0:.1f} minutes!",
        flush=True,
    )
