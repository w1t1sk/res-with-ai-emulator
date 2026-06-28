"""
driver_old.py — AI+RES driver for the OLD emulator (lightweight fuxiens).

Self-contained driver for the QDMC rare event sampling loop using the original
lightweight fuxiens emulator (input-perturbation in physical space). This is the
emulator that produced the figures in the project report. It provides:

    load_model(device)          → (model, data_mean, data_std)
    score_walkers_with_ai(...)  → (raw_scores, score_meta)
    run_ai_res_scheme(...)      → runs the full AI+RES experiment

All emulator-agnostic machinery is imported from core.py. The only
emulator-specific pieces live here: the model construction in load_model() and
the single inference call `model(x, y=None)` inside score_walkers_with_ai().

This mirrors driver_new.py; the two differ only in load_model() and the one
inference-call line, so the QDMC loop reads identically in both files.
"""

import os
import sys

import numpy as np
import torch

import core

# Make the old emulator's modules importable (fuxi_model, swin_layers).
_old_emulator_dir = os.environ.get(
    "AIRES_OLD_EMULATOR_DIR", "/home/nishidh/res_aiemulator_sem8/res/old_emulator"
)
sys.path.insert(0, _old_emulator_dir)
from fuxi_model import PerturbationModel, fuxibase, fuxiens


EMULATOR_NAME = "old (fuxiens lightweight)"


# ---------------------------------------------------------------------------
# Emulator loading
# ---------------------------------------------------------------------------

def load_model(device: torch.device):
    """Load the old fuxiens stage-2 checkpoint.

    Reads checkpoint + stats from AIRES_OLD_EMULATOR_DIR (default:
    res_aiemulator_sem8/res/old_emulator/).

    Returns
    -------
    model : fuxiens  (eval mode, on `device`; wrapped in DataParallel if >1 GPU)
    data_mean, data_std : torch.Tensor  (1, 14, 1, 1) physical-space stats
    """
    old_emulator_dir = os.environ.get(
        "AIRES_OLD_EMULATOR_DIR", "/home/nishidh/res_aiemulator_sem8/res/old_emulator"
    )
    forecast_model = fuxibase(in_channels=28, out_channels=14, embed_dim=64, img_size=(64, 128))
    model = fuxiens(forecast_model, PerturbationModel(28, 28), PerturbationModel(42, 28))
    model.load_state_dict(
        torch.load(os.path.join(old_emulator_dir, "fuxiens_stage2_best_ctd.pth"), map_location="cpu")
    )
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.eval()

    stats = torch.load(os.path.join(old_emulator_dir, "stats_full.pt"), map_location="cpu")
    data_mean = stats["mean"].view(1, 14, 1, 1).to(device)
    data_std = stats["std"].view(1, 14, 1, 1).to(device)
    return model, data_mean, data_std


def _emulator_step(model, current_tensor: torch.Tensor) -> torch.Tensor:
    """One emulator rollout step. THE old-emulator-specific inference call."""
    return model(current_tensor, y=None)


# ---------------------------------------------------------------------------
# AI scoring (member-microbatched rollout)
# ---------------------------------------------------------------------------

def score_walkers_with_ai(
    walkers: list,
    config: dict,
    days_remaining: int,
    model,
    data_mean: torch.Tensor,
    data_std: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    """Score each walker by rolling out the AI emulator ensemble forward in time.

    Identical in structure to driver_new.score_walkers_with_ai; the only
    difference is the per-step inference call (`_emulator_step`). See that
    function's docstring for the full algorithm description.

    Returns
    -------
    raw_scores : ndarray (N_walkers,)
    score_meta : dict
    """
    forecast_horizon = days_remaining + config["L"]
    target_start = days_remaining
    target_end = target_start + config["L"]
    target_window_len = max(1, target_end - target_start)
    n_walkers = len(walkers)

    raw_scores = np.zeros(n_walkers, dtype=np.float64)
    score_meta = {
        "fallback_count": 0,
        "nonfinite_member_score_count": 0,
        "flagged_walker_count": 0,
        "forecast_horizon_days_used": int(forecast_horizon),
        "target_window_start_day": int(target_start),
        "target_window_end_day": int(target_end),
        "target_mode": "final_window",
        "walker_batch_size_used": int(core.resolve_ai_scoring_walker_batch_size(config, n_walkers)),
        "member_microbatch_size_used": int(core.resolve_ai_member_microbatch_size(config)),
    }

    region_slice = (
        slice(None),
        0,
        slice(config["LAT_S"], config["LAT_E"]),
        slice(config["LON_S"], config["LON_E"]),
    )
    member_microbatch_size = core.resolve_ai_member_microbatch_size(config)
    walker_batch_size = core.resolve_ai_scoring_walker_batch_size(config, n_walkers)

    with torch.inference_mode():
        for batch_start in range(0, n_walkers, walker_batch_size):
            batch_end = min(batch_start + walker_batch_size, n_walkers)
            batch_walkers = walkers[batch_start:batch_end]

            valid_items = []
            valid_positions = []
            for local_idx, walker in enumerate(batch_walkers):
                if walker.get("pert_flag") == 2:
                    score_meta["flagged_walker_count"] += 1
                    score_meta["fallback_count"] += 1
                    raw_scores[batch_start + local_idx] = core.fallback_recent_score(walker, config)
                    continue

                history = walker.get("daily_history_tensors", [])
                if len(history) < 2:
                    score_meta["fallback_count"] += 1
                    raw_scores[batch_start + local_idx] = core.fallback_recent_score(walker, config)
                    continue

                valid_positions.append(local_idx)
                valid_items.append(np.concatenate([history[-2], history[-1]], axis=0))

            if not valid_items:
                continue

            base_tensors = torch.tensor(np.stack(valid_items), dtype=torch.float32, device=device)
            normalized_history = core.normalize_history_pair(base_tensors, data_mean, data_std)

            member_scores = np.zeros((len(valid_items), config["M_members"]), dtype=np.float64)

            for member_start in range(0, config["M_members"], member_microbatch_size):
                member_end = min(member_start + member_microbatch_size, config["M_members"])
                chunk_members = member_end - member_start
                current_tensor = normalized_history.repeat_interleave(chunk_members, dim=0)
                chunk_score_sum = np.zeros((len(valid_items), chunk_members), dtype=np.float64)
                chunk_invalid = np.zeros((len(valid_items), chunk_members), dtype=bool)

                for f_step in range(forecast_horizon):
                    pred = _emulator_step(model, current_tensor)
                    if target_start <= f_step < target_end:
                        phys_pred = pred * (data_std + 1e-6) + data_mean
                        region_temps = (
                            torch.mean(phys_pred[region_slice], dim=(1, 2))
                            .detach()
                            .cpu()
                            .numpy()
                            .reshape(len(valid_items), chunk_members)
                        )
                        nonfinite = ~np.isfinite(region_temps)
                        chunk_invalid |= nonfinite
                        chunk_score_sum += np.where(nonfinite, 0.0, region_temps)
                    current_tensor = torch.cat([current_tensor[:, 14:, :, :], pred], dim=1)

                chunk_scores = chunk_score_sum / target_window_len
                chunk_scores[chunk_invalid] = np.nan
                member_scores[:, member_start:member_end] = chunk_scores

            invalid_member_mask = ~np.isfinite(member_scores)
            score_meta["nonfinite_member_score_count"] += int(np.count_nonzero(invalid_member_mask))
            member_scores = np.where(np.isfinite(member_scores), member_scores, np.nan)
            walker_scores = np.nanmean(member_scores, axis=1)

            fallback_scores = np.array(
                [core.fallback_recent_score(batch_walkers[pos], config) for pos in valid_positions],
                dtype=np.float64,
            )
            walker_scores[~np.isfinite(walker_scores)] = fallback_scores[~np.isfinite(walker_scores)]

            for pos, score in zip(valid_positions, walker_scores):
                raw_scores[batch_start + pos] = float(score)

    return raw_scores, score_meta


# ---------------------------------------------------------------------------
# AI+RES experiment loop
# ---------------------------------------------------------------------------

def run_ai_res_scheme(config: dict, scheme: str, target_region_name: str | None = None) -> None:
    """Run a QDMC AI+RES experiment with the old fuxiens emulator.

    See driver_new.run_ai_res_scheme for the full description; this function is
    identical except that it loads and rolls out the old emulator.

    Outputs are written to outputs/ai_res_{region}_{scheme}_output/.
    """
    schedule_key = f"C_schedule_{scheme}"
    if schedule_key not in config:
        available = sorted(k[len("C_schedule_"):] for k in config if k.startswith("C_schedule_"))
        raise ValueError(f"Unknown scheme '{scheme}'. Available: {', '.join(available)}.")

    config = core.resolve_target_region(config, target_region_name=target_region_name)
    core.validate_config(config, schedule_key=schedule_key)

    base_dir = core.get_base_dir()
    out_dir = os.path.join(base_dir, "outputs", f"ai_res_{config['REGION_NAME']}_{scheme}_output")
    tmp_dir = os.path.join(base_dir, "tmp", f"ai_res_{config['REGION_NAME']}_{scheme}_tmp")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    checkpoint_path, checkpoint_meta_path = core.ai_checkpoint_paths(out_dir)

    max_workers = core.resolve_physics_worker_count(config)
    pool_chunksize = core.resolve_physics_pool_chunksize(config)
    maxtasksperchild = core.resolve_physics_maxtasksperchild(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visible_gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0

    print(
        f"\n=== AI+RES [{scheme}] | emulator={EMULATOR_NAME} | region={config['REGION_LABEL']} | "
        f"N={config['N_walkers']} | M={config['M_members']} | device={device} | "
        f"gpus={visible_gpu_count} | workers={max_workers} ==="
    )
    print(
        f"Pool: chunksize={pool_chunksize}, maxtasksperchild={maxtasksperchild or 'unbounded'}, "
        f"walker_batch={core.resolve_ai_scoring_walker_batch_size(config, int(config['N_walkers']))}, "
        f"member_microbatch={core.resolve_ai_member_microbatch_size(config)}",
        flush=True,
    )

    np.random.seed(int(config.get("experiment_seed", 314159)))
    model, data_mean, data_std = load_model(device)
    c_schedule = config[schedule_key]
    checkpoint_payload = core.load_ai_checkpoint(checkpoint_path)

    if checkpoint_payload is not None:
        if checkpoint_payload.get("scheme_name") != scheme:
            raise RuntimeError(
                f"Checkpoint mismatch: belongs to scheme "
                f"'{checkpoint_payload.get('scheme_name')}', not '{scheme}'."
            )
        if checkpoint_payload.get("region_name") != config["REGION_NAME"]:
            raise RuntimeError(
                f"Checkpoint mismatch: belongs to region "
                f"'{checkpoint_payload.get('region_name')}', not '{config['REGION_NAME']}'."
            )
        walkers = checkpoint_payload["walkers"]
        diagnostics = checkpoint_payload["diagnostics"]
        np.random.set_state(checkpoint_payload["numpy_random_state"])
        start_step_idx = int(checkpoint_payload.get("next_step_idx", 0))
        resume_phase = checkpoint_payload.get("phase", "resampling")
        print(f"Resuming from checkpoint: phase='{resume_phase}', next_step_idx={start_step_idx}.", flush=True)
    else:
        walkers = core.build_initial_walkers(config)
        diagnostics = core.build_diagnostics(config, scheme, c_schedule, emulator=EMULATOR_NAME)
        start_step_idx = 0
        resume_phase = "resampling"

    num_resampling_steps = config["total_days"] // config["resample_interval"]
    step_range = range(start_step_idx, num_resampling_steps) if resume_phase == "resampling" else range(0)

    # Keep one pool alive across all steps to amortise process-spawn cost.
    with core.make_spawn_pool(max_workers, maxtasksperchild) as pool:
        for step_idx in step_range:
            current_day = (step_idx + 1) * config["resample_interval"]
            c_k = c_schedule[step_idx]

            print(
                f"\n--- Step {step_idx + 1}/{num_resampling_steps} | "
                f"days {current_day - config['resample_interval']}→{current_day} | c_k={c_k} ---"
            )

            args_list = core.build_worker_args(walkers, config["resample_interval"], tmp_dir)
            completed_ids = core.run_pool_map(pool, core.step_physics_chunk_ai, args_list, pool_chunksize)
            core.collect_worker_outputs(tmp_dir, completed_ids, walkers)

            walkers = core.replace_exploded_walkers(
                walkers, step_label=f"step_{step_idx + 1}_post_chunk", diagnostics=diagnostics
            )

            days_remaining = config["total_days"] - current_day
            if c_k > 0:
                print(
                    f"Scoring via AI ensemble (M={config['M_members']}, "
                    f"horizon={days_remaining + config['L']} days)..."
                )
                raw_scores, score_meta = score_walkers_with_ai(
                    walkers, config, days_remaining, model, data_mean, data_std, device
                )
                v_k = core.compute_splitting_function(raw_scores, c_k)
                walkers, clones, ancestry, incr_weights, w_bar_k = core.resample_walkers(
                    walkers, v_k, config
                )
                core.refresh_branch_perturbation_seeds(walkers, config, branch_generation=step_idx + 1)

                diagnostics["raw_scores_per_step"].append(raw_scores.copy())
                diagnostics["V_k_per_step"].append(v_k.copy())
                diagnostics["clone_counts_per_step"].append(clones.copy())
                diagnostics["ancestry_per_step"].append(ancestry)
                diagnostics["per_walker_weights_per_step"].append(incr_weights.copy())
                diagnostics["score_meta_per_step"].append(score_meta)
                diagnostics["global_weights"].append(w_bar_k)
                print(
                    f"Resampling complete. Population={len(walkers)}, w_bar_k={w_bar_k:.4f}. "
                    f"Fallbacks={score_meta['fallback_count']}."
                )
            else:
                core.reset_previous_potential(walkers, value=0.0)
                for key in (
                    "raw_scores_per_step", "V_k_per_step", "clone_counts_per_step",
                    "ancestry_per_step", "per_walker_weights_per_step",
                ):
                    diagnostics[key].append(None)
                diagnostics["score_meta_per_step"].append({"fallback_count": 0, "v_k_reset_to_zero": True})
                diagnostics["global_weights"].append(1.0)

            next_phase = "resampling" if step_idx + 1 < num_resampling_steps else "final_chunk"
            core.save_ai_checkpoint(
                checkpoint_path,
                checkpoint_meta_path,
                {
                    "scheme_name": scheme,
                    "region_name": config["REGION_NAME"],
                    "phase": next_phase,
                    "next_step_idx": step_idx + 1,
                    "walkers": walkers,
                    "diagnostics": diagnostics,
                    "numpy_random_state": np.random.get_state(),
                },
            )

        print(f"\n--- Final step: integrating {config['L']} more days past t_f ---")
        args_list = core.build_worker_args(walkers, config["L"], tmp_dir)
        completed_ids = core.run_pool_map(pool, core.step_physics_chunk_ai, args_list, pool_chunksize)
        core.collect_worker_outputs(tmp_dir, completed_ids, walkers)

    core.finalize_and_save_airres(
        walkers, config, diagnostics, out_dir, checkpoint_path, checkpoint_meta_path
    )
