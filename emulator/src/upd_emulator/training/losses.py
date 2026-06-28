from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from upd_emulator.grid import gaussian_latitudes, normalized_latitude_loss_weights


def _resolve_latitudes(latitudes, height: int) -> np.ndarray:
    if latitudes is None:
        return gaussian_latitudes(height)

    if isinstance(latitudes, torch.Tensor):
        latitudes = latitudes.detach().cpu().numpy()

    latitudes = np.asarray(latitudes, dtype=np.float32)
    if latitudes.ndim != 1:
        raise ValueError("latitudes must be a 1D array")
    if latitudes.shape[0] != height:
        raise ValueError(f"Expected {height} latitudes, got {latitudes.shape[0]}")
    return latitudes


def latitude_weight_tensor(
    height: int,
    latitudes=None,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    resolved_latitudes = _resolve_latitudes(latitudes, height)
    weights = torch.from_numpy(normalized_latitude_loss_weights(resolved_latitudes)).view(
        1, 1, height, 1
    )
    return weights.to(device=device, dtype=dtype)


class LatitudeWeightedL1(nn.Module):
    """Latitude-weighted L1 loss from the FuXi/FuXi-ENS training recipe."""

    def __init__(self, height: int = 64, latitudes=None) -> None:
        super().__init__()
        self.register_buffer("weights", latitude_weight_tensor(height, latitudes=latitudes))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(prediction - target) * self.weights)


def reshape_member_batch(tensor: torch.Tensor, batch_size: int, local_members: int) -> torch.Tensor:
    """Convert [batch * members, ...] to [members, batch, ...]."""

    trailing_shape = tensor.shape[1:]
    return tensor.view(batch_size, local_members, *trailing_shape).transpose(0, 1).contiguous()


def gather_ensemble_predictions(local_preds: torch.Tensor) -> torch.Tensor:
    """Gather local ensemble predictions across ranks without building cross-rank autograd."""

    local_preds = local_preds.detach()
    if not dist.is_available() or not dist.is_initialized():
        return local_preds

    gathered = [torch.zeros_like(local_preds) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local_preds)
    return torch.cat(gathered, dim=0)


def _pairwise_abs_mean_chunked(
    local_preds: torch.Tensor,
    global_preds: torch.Tensor,
    row_weights: torch.Tensor | None = None,
    chunk_size: int = 8,
) -> torch.Tensor:
    total_sum = local_preds.new_zeros(())
    total_count = 0

    for start in range(0, global_preds.shape[0], chunk_size):
        chunk = global_preds[start : start + chunk_size]
        diffs = torch.abs(local_preds.unsqueeze(1) - chunk.unsqueeze(0)).float()
        if row_weights is not None:
            total_sum = total_sum + (diffs * row_weights).sum()
        else:
            total_sum = total_sum + diffs.sum()
        total_count += diffs.numel()

    return total_sum / float(total_count)


def distributed_crps_local(
    local_preds: torch.Tensor,
    target: torch.Tensor,
    global_preds_detached: torch.Tensor,
    row_weights: torch.Tensor | None = None,
    chunk_size: int = 8,
) -> torch.Tensor:
    """Local CRPS contribution for ensemble-parallel DDP.

    Each rank owns gradients for its local members only. DDP gradient averaging over
    ranks turns the local contribution into the full global-member CRPS gradient.
    """

    local_preds = local_preds.float()
    target = target.float()
    global_preds_detached = global_preds_detached.float()

    abs_err = torch.abs(local_preds - target.unsqueeze(0))
    term1 = torch.mean(abs_err * row_weights) if row_weights is not None else torch.mean(abs_err)
    term2 = 0.5 * _pairwise_abs_mean_chunked(
        local_preds,
        global_preds_detached,
        row_weights=row_weights,
        chunk_size=chunk_size,
    )
    return term1 - term2


def gaussian_kl_divergence(
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    row_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """KL(q || p) between diagonal Gaussians."""

    kl = 0.5 * (
        logvar_p
        - logvar_q
        + (torch.exp(logvar_q) + (mu_q - mu_p).pow(2)) / torch.exp(logvar_p)
        - 1.0
    )
    if row_weights is not None:
        return torch.mean(kl * row_weights)
    return kl.mean()
