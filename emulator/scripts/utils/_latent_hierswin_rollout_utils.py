from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from upd_emulator.data import GCMSequenceDataset, load_stats
from upd_emulator.models.fuxi_ens_v3_latent_hierswin import FuXiENSModelV3LatentHierSwin


def load_checkpoint_state(path: Path, device: torch.device) -> tuple[dict[str, torch.Tensor], dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    return state_dict, metadata


def build_latent_hierswin_model(
    checkpoint_path: Path,
    device: torch.device,
    use_checkpoint: bool = False,
) -> FuXiENSModelV3LatentHierSwin:
    state_dict, metadata = load_checkpoint_state(checkpoint_path, device)
    model = FuXiENSModelV3LatentHierSwin(
        use_checkpoint=use_checkpoint,
        latent_channels=int(metadata.get("latent_channels", 128)),
        perturb_stage_dims=tuple(metadata.get("perturb_stage_dims", [96, 128, 128, 160, 192])),
        perturb_stage_heads=tuple(metadata.get("perturb_stage_heads", [8, 8, 8, 8, 8])),
        perturb_encoder_depths=tuple(metadata.get("perturb_encoder_depths", [1, 1, 2, 2, 2])),
        perturb_decoder_depths=tuple(metadata.get("perturb_decoder_depths", [1, 1, 1, 2])),
        perturb_window_size=int(metadata.get("perturb_window_size", 8)),
        perturb_shift_size=int(metadata.get("perturb_shift_size", 4)),
        perturb_drop_path_rate=float(metadata.get("perturb_drop_path_rate", 0.05)),
        initial_logvar_bias=float(metadata.get("initial_logvar_bias", -4.0)),
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def denormalize_ensemble(states: torch.Tensor, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    mean = stats["physical_mean"].view(1, 1, -1, 1, 1)
    std = stats["physical_std"].view(1, 1, -1, 1, 1)
    return (states.cpu() * (std + 1e-6)) + mean


def denormalize_states(states: torch.Tensor, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    mean = stats["physical_mean"].view(1, -1, 1, 1)
    std = stats["physical_std"].view(1, -1, 1, 1)
    return (states.cpu() * (std + 1e-6)) + mean


def compute_acc(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    climatology: torch.Tensor,
    latitudes: np.ndarray,
) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes))
    weights = torch.from_numpy(weights).view(1, 1, -1, 1).float()
    climatology = climatology.unsqueeze(0)
    forecast_anomaly = predictions - climatology
    analysis_anomaly = targets - climatology
    covariance = (weights * forecast_anomaly * analysis_anomaly).sum(dim=(2, 3))
    forecast_var = (weights * forecast_anomaly**2).sum(dim=(2, 3))
    analysis_var = (weights * analysis_anomaly**2).sum(dim=(2, 3))
    return (covariance / torch.sqrt(forecast_var * analysis_var)).numpy()


def rollout_ensemble(
    model: FuXiENSModelV3LatentHierSwin,
    input_window: torch.Tensor,
    lead_steps: int,
    ensemble_members: int,
    device: torch.device,
    bf16_enabled: bool,
) -> torch.Tensor:
    current_members = input_window.unsqueeze(0).to(device)
    current_members = current_members.repeat_interleave(ensemble_members, dim=0)
    forecasts: list[torch.Tensor] = []

    with torch.no_grad():
        for _ in range(lead_steps):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=bf16_enabled and device.type == "cuda",
            ):
                prediction = model.sample_prior(current_members, num_samples=1)
            forecasts.append(prediction.float().cpu())
            previous_state = current_members[:, prediction.shape[1] :, :, :]
            current_members = torch.cat([previous_state, prediction], dim=1)

    return torch.stack(forecasts, dim=1)


def rollout_no_perturb(
    model: FuXiENSModelV3LatentHierSwin,
    input_window: torch.Tensor,
    lead_steps: int,
    device: torch.device,
    bf16_enabled: bool,
) -> torch.Tensor:
    current = input_window.unsqueeze(0).to(device)
    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(lead_steps):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=bf16_enabled and device.type == "cuda",
            ):
                prediction = model.forecast_model(current)
            outputs.append(prediction.float().cpu())
            current = torch.cat([current[:, prediction.shape[1] :, :, :], prediction], dim=1)
    return torch.stack(outputs, dim=0).squeeze(1)


def evaluate_rollout_acc(
    model: FuXiENSModelV3LatentHierSwin,
    dataset: GCMSequenceDataset,
    sample_indices: list[int],
    lead_steps: int,
    ensemble_members: int,
    stats: dict[str, torch.Tensor],
    climatology: torch.Tensor,
    device: torch.device,
    bf16_enabled: bool,
) -> dict[str, np.ndarray]:
    model.eval()
    ensemble_overall: list[np.ndarray] = []
    no_perturb_overall: list[np.ndarray] = []

    for sample_index in sample_indices:
        if sample_index < 0 or sample_index >= len(dataset):
            raise IndexError(f"Sample index {sample_index} is outside [0, {len(dataset) - 1}].")
        sample = dataset[sample_index]
        predictions_norm = rollout_ensemble(
            model=model,
            input_window=sample["input_window"],
            lead_steps=lead_steps,
            ensemble_members=ensemble_members,
            device=device,
            bf16_enabled=bf16_enabled,
        )
        no_perturb_norm = rollout_no_perturb(
            model=model,
            input_window=sample["input_window"],
            lead_steps=lead_steps,
            device=device,
            bf16_enabled=bf16_enabled,
        )
        targets_norm = sample["future_states"][:lead_steps].cpu()

        predictions_physical = denormalize_ensemble(predictions_norm, stats)
        mean_physical = predictions_physical.mean(dim=0)
        no_perturb_physical = denormalize_states(no_perturb_norm, stats)
        targets_physical = denormalize_states(targets_norm, stats)

        acc_ens = compute_acc(
            predictions=mean_physical,
            targets=targets_physical,
            climatology=climatology,
            latitudes=dataset.latitudes.cpu().numpy(),
        )
        acc_det = compute_acc(
            predictions=no_perturb_physical,
            targets=targets_physical,
            climatology=climatology,
            latitudes=dataset.latitudes.cpu().numpy(),
        )

        ensemble_overall.append(np.mean(acc_ens, axis=1))
        no_perturb_overall.append(np.mean(acc_det, axis=1))

    ensemble_mean = np.mean(np.stack(ensemble_overall, axis=0), axis=0)
    no_perturb_mean = np.mean(np.stack(no_perturb_overall, axis=0), axis=0)
    return {
        "ensemble_overall": ensemble_mean,
        "no_perturb_overall": no_perturb_mean,
    }


def load_stats_and_climatology(
    stats_path: Path,
    climatology_path: Path,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    stats = load_stats(stats_path)
    climatology = torch.load(climatology_path, map_location="cpu", weights_only=False)["mean_map"]
    return stats, climatology
