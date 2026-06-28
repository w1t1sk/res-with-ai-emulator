#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import random
import sys
import time
import warnings

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from upd_emulator.constants import DEFAULT_DATA_ROOT, DEFAULT_TRAIN_RUNS, DEFAULT_VAL_RUNS, PHYSICAL_CHANNELS
from upd_emulator.data import GCMSequenceDataset
from upd_emulator.models.fuxi_ens_v3_latent_hierswin import FuXiENSModelV3LatentHierSwin
from upd_emulator.training import latitude_weight_tensor

warnings.filterwarnings(
    "ignore",
    message="TypedStorage is deprecated.*",
    category=UserWarning,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step-based stage-2 latent HierSwin training: keep the strong V3 deterministic "
            "backbone frozen in eval mode and learn uncertainty as a latent-space residual model."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=REPO_ROOT / "resources" / "stats" / "exac_input_stats.pt",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "cache",
    )
    parser.add_argument(
        "--forecast-checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "stage1" / "forecast_stage1_best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "latent_hierswin_stage2",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=REPO_ROOT / "outputs" / "latent_hierswin_stage2" / "training_log.csv",
    )
    parser.add_argument("--save-name", default="latent_hierswin_stage2_best.pt")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--val-batches", type=int, default=128)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--memory-cache-size", type=int, default=8)
    parser.add_argument("--ensemble-size", type=int, default=4)
    parser.add_argument("--kl-weight", type=float, default=3e-4)
    parser.add_argument("--lr", type=float, default=4.5e-6)
    parser.add_argument("--eta-min", type=float, default=1e-7)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--latent-channels", type=int, default=128)
    parser.add_argument("--perturb-stage-dims", nargs="+", type=int, default=[96, 128, 128, 160, 192])
    parser.add_argument("--perturb-stage-heads", nargs="+", type=int, default=[8, 8, 8, 8, 8])
    parser.add_argument("--perturb-encoder-depths", nargs="+", type=int, default=[1, 1, 2, 2, 2])
    parser.add_argument("--perturb-decoder-depths", nargs="+", type=int, default=[1, 1, 1, 2])
    parser.add_argument("--perturb-window-size", type=int, default=8)
    parser.add_argument("--perturb-shift-size", type=int, default=4)
    parser.add_argument("--perturb-drop-path-rate", type=float, default=0.05)
    parser.add_argument("--initial-logvar-bias", type=float, default=-4.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-runs", nargs="+", default=DEFAULT_TRAIN_RUNS)
    parser.add_argument("--val-runs", nargs="+", default=DEFAULT_VAL_RUNS)
    parser.add_argument("--disable-bf16", action="store_true")
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    return parser.parse_args()


def setup_distributed_env() -> tuple[bool, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))

    if distributed:
        os.environ.setdefault("WORLD_SIZE", str(world_size))
        os.environ.setdefault("RANK", os.environ.get("SLURM_PROCID", "0"))
        os.environ.setdefault("LOCAL_RANK", str(local_rank))
        os.environ.setdefault("MASTER_ADDR", os.environ.get("MASTER_ADDR", "127.0.0.1"))
        os.environ.setdefault("MASTER_PORT", os.environ.get("MASTER_PORT", "29500"))

    return distributed, world_size, local_rank


def _visible_cuda_devices() -> list[str] | None:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        return None
    devices = [device.strip() for device in visible_devices.split(",") if device.strip()]
    if devices == ["NoDevFiles"]:
        return []
    return devices


def _slurm_requested_gpus() -> bool:
    slurm_gpu_vars = (
        "SLURM_STEP_GPUS",
        "SLURM_JOB_GPUS",
        "SLURM_GPUS",
        "SLURM_GPUS_ON_NODE",
        "SLURM_GPUS_PER_NODE",
    )
    return any(os.environ.get(name) for name in slurm_gpu_vars)


def _cuda_expected() -> bool:
    visible_devices = _visible_cuda_devices()
    return (visible_devices is not None and len(visible_devices) > 0) or _slurm_requested_gpus()


def select_device(local_rank: int) -> torch.device:
    visible_devices = _visible_cuda_devices()
    if _cuda_expected():
        if visible_devices is not None and len(visible_devices) == 1:
            cuda_index = 0
        else:
            cuda_index = local_rank
            if visible_devices is not None and cuda_index >= len(visible_devices):
                raise RuntimeError(
                    f"Local rank {local_rank} cannot use CUDA device {cuda_index}; "
                    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}."
                )
        torch.cuda.set_device(cuda_index)
        return torch.device(f"cuda:{cuda_index}")

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        return torch.device("cuda:0")

    return torch.device("cpu")


def initialize_distributed(distributed: bool, device: torch.device) -> None:
    if distributed:
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)


def set_seed(seed: int, use_cuda: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def is_main_process(distributed: bool) -> bool:
    return not distributed or dist.get_rank() == 0


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location=device, weights_only=False)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    return state_dict


def save_checkpoint(
    model: torch.nn.Module,
    save_path: Path,
    step: int,
    best_val: float,
    stage1_checkpoint: Path,
    args: argparse.Namespace,
) -> None:
    payload = {
        "model": unwrap_model(model).state_dict(),
        "step": step,
        "best_val": best_val,
        "metadata": {
            "forecast_frozen": True,
            "forecast_eval_mode_during_stage2": True,
            "stage1_checkpoint": str(stage1_checkpoint),
            "prior_model": "latent_hierswin_128_to_128",
            "posterior_model": "latent_hierswin_256_to_128",
            "latent_channels": int(args.latent_channels),
            "perturb_stage_dims": [int(v) for v in args.perturb_stage_dims],
            "perturb_stage_heads": [int(v) for v in args.perturb_stage_heads],
            "perturb_encoder_depths": [int(v) for v in args.perturb_encoder_depths],
            "perturb_decoder_depths": [int(v) for v in args.perturb_decoder_depths],
            "perturb_window_size": int(args.perturb_window_size),
            "perturb_shift_size": int(args.perturb_shift_size),
            "perturb_drop_path_rate": float(args.perturb_drop_path_rate),
            "initial_logvar_bias": float(args.initial_logvar_bias),
            "training_mode": "iteration_based",
        },
    }
    torch.save(payload, save_path)


def compute_stage2_loss(
    model_ref: FuXiENSModelV3LatentHierSwin,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    ensemble_size: int,
    row_weights: torch.Tensor,
    kl_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = inputs.shape[0]
    truth_window = torch.cat([inputs[:, PHYSICAL_CHANNELS:, :, :], targets], dim=1)

    with torch.no_grad():
        encoded_current = model_ref.forecast_model.encode(inputs).detach()
        encoded_truth = model_ref.forecast_model.encode(truth_window).detach()

    mu_p, logvar_p = model_ref.prior_model(encoded_current)
    posterior_input = torch.cat([encoded_current, encoded_truth], dim=1)
    mu_q, logvar_q = model_ref.posterior_model(posterior_input)

    encoded_expanded = (
        encoded_current.unsqueeze(0)
        .expand(ensemble_size, -1, -1, -1, -1)
        .reshape(ensemble_size * batch_size, encoded_current.shape[1], encoded_current.shape[2], encoded_current.shape[3])
    )
    mu_q_expanded = (
        mu_q.unsqueeze(0)
        .expand(ensemble_size, -1, -1, -1, -1)
        .reshape(ensemble_size * batch_size, mu_q.shape[1], mu_q.shape[2], mu_q.shape[3])
    )
    logvar_q_expanded = (
        logvar_q.unsqueeze(0)
        .expand(ensemble_size, -1, -1, -1, -1)
        .reshape(ensemble_size * batch_size, logvar_q.shape[1], logvar_q.shape[2], logvar_q.shape[3])
    )

    perturbation = model_ref.reparameterize(mu_q_expanded, logvar_q_expanded)
    preds_flat = model_ref.forecast_model.forecast_from_encoded(encoded_expanded + perturbation)
    preds = preds_flat.view(ensemble_size, batch_size, -1, targets.shape[-2], targets.shape[-1])
    targets_expanded = targets.unsqueeze(0)

    abs_err = torch.abs(preds - targets_expanded)
    term1 = torch.mean(abs_err, dim=0)
    pairwise_diff = torch.abs(preds.unsqueeze(1) - preds.unsqueeze(0))
    term2 = 0.5 * torch.mean(pairwise_diff, dim=(0, 1))
    crps = torch.mean((term1 - term2) * row_weights)

    pointwise_kl = 0.5 * (
        logvar_p
        - logvar_q
        + (torch.exp(logvar_q) + (mu_q - mu_p).pow(2)) / torch.exp(logvar_p)
        - 1.0
    )
    kl_div = torch.mean(pointwise_kl * row_weights)
    return crps + (kl_weight * kl_div), crps, kl_div


def evaluate(
    model_ref: FuXiENSModelV3LatentHierSwin,
    data_loader: DataLoader,
    device: torch.device,
    ensemble_size: int,
    row_weights: torch.Tensor,
    kl_weight: float,
    distributed: bool,
    max_batches: int,
    bf16_enabled: bool,
) -> float:
    model_ref.eval()
    total_loss = torch.zeros((), device=device)
    count = torch.zeros((), device=device)

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            inputs = batch["input_window"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=bf16_enabled and device.type == "cuda",
            ):
                total_batch_loss, _, _ = compute_stage2_loss(
                    model_ref=model_ref,
                    inputs=inputs,
                    targets=targets,
                    ensemble_size=ensemble_size,
                    row_weights=row_weights,
                    kl_weight=kl_weight,
                )
            total_loss += total_batch_loss.float()
            count += 1

    metrics = torch.stack([total_loss, count])
    if distributed:
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    if metrics[1].item() == 0:
        return float("inf")
    return (metrics[0] / metrics[1]).item()


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be >= 1.")
    if args.val_every < 1:
        raise ValueError("--val-every must be >= 1.")
    if args.save_every < 0:
        raise ValueError("--save-every must be >= 0.")

    distributed, world_size, local_rank = setup_distributed_env()
    device = select_device(local_rank)
    initialize_distributed(distributed, device)
    set_seed(args.seed, use_cuda=device.type == "cuda")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    train_dataset = GCMSequenceDataset(
        data_root=args.data_root,
        stats_path=args.stats_path,
        runs=args.train_runs,
        target_steps=1,
        cache_dir=args.cache_dir,
        memory_cache_size=args.memory_cache_size,
    )
    val_dataset = GCMSequenceDataset(
        data_root=args.data_root,
        stats_path=args.stats_path,
        runs=args.val_runs,
        target_steps=1,
        cache_dir=args.cache_dir,
        memory_cache_size=max(2, args.memory_cache_size // 2),
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    val_workers = max(0, args.num_workers // 2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=val_workers > 0,
    )

    model = FuXiENSModelV3LatentHierSwin(
        use_checkpoint=not args.disable_gradient_checkpointing,
        latent_channels=args.latent_channels,
        perturb_stage_dims=args.perturb_stage_dims,
        perturb_stage_heads=args.perturb_stage_heads,
        perturb_encoder_depths=args.perturb_encoder_depths,
        perturb_decoder_depths=args.perturb_decoder_depths,
        perturb_window_size=args.perturb_window_size,
        perturb_shift_size=args.perturb_shift_size,
        perturb_drop_path_rate=args.perturb_drop_path_rate,
        initial_logvar_bias=args.initial_logvar_bias,
    ).to(device)
    forecast_state = load_state_dict(args.forecast_checkpoint, device)
    model.forecast_model.load_state_dict(forecast_state, strict=False)
    for parameter in model.forecast_model.parameters():
        parameter.requires_grad_(False)
    model.forecast_model.eval()

    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
        )
    model_ref = unwrap_model(model)

    row_weights = latitude_weight_tensor(
        height=int(train_dataset.latitudes.numel()),
        latitudes=train_dataset.latitudes,
        device=device,
    )

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.steps,
        eta_min=args.eta_min,
    )

    best_val_loss = float("inf")
    start_time = time.monotonic()
    bf16_enabled = not args.disable_bf16

    if is_main_process(distributed):
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.log_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoints_dir = args.output_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        with open(args.log_path, "w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(["step", "train_loss", "val_loss"])
        print(
            f"Stage 2 latent HierSwin starting | steps={args.steps} | val_every={args.val_every} | "
            f"train_samples={len(train_dataset)} | batch_size={args.batch_size} | forecast_frozen=True | "
            f"forecast_eval_mode=True | latent_channels={args.latent_channels} | "
            f"stage_dims={args.perturb_stage_dims} | encoder_depths={args.perturb_encoder_depths} | "
            f"decoder_depths={args.perturb_decoder_depths} | kl_weight={args.kl_weight:.1e} | "
            f"world_size={world_size} | device={device.type}",
            flush=True,
        )

    data_epoch = 0
    if train_sampler is not None:
        train_sampler.set_epoch(data_epoch)
    train_iterator = iter(train_loader)
    running_train_loss_sum = 0.0
    running_train_count = 0

    for step in range(1, args.steps + 1):
        try:
            batch = next(train_iterator)
        except StopIteration:
            data_epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(data_epoch)
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        model.train()
        model_ref.forecast_model.eval()
        inputs = batch["input_window"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=bf16_enabled and device.type == "cuda",
        ):
            total_loss, _, _ = compute_stage2_loss(
                model_ref=model_ref,
                inputs=inputs,
                targets=targets,
                ensemble_size=args.ensemble_size,
                row_weights=row_weights,
                kl_weight=args.kl_weight,
            )
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        running_train_loss_sum += float(total_loss.item())
        running_train_count += 1

        if is_main_process(distributed) and args.log_every > 0 and (step == 1 or step % args.log_every == 0):
            elapsed = format_duration(time.monotonic() - start_time)
            print(
                f"Step {step:05d}/{args.steps:05d} | lr={optimizer.param_groups[0]['lr']:.7f} | "
                f"train_loss={total_loss.item():.5f} | elapsed {elapsed}",
                flush=True,
            )

        should_validate = step % args.val_every == 0 or step == args.steps
        if not should_validate:
            continue

        val_loss = evaluate(
            model_ref=model_ref,
            data_loader=val_loader,
            device=device,
            ensemble_size=args.ensemble_size,
            row_weights=row_weights,
            kl_weight=args.kl_weight,
            distributed=distributed,
            max_batches=args.val_batches,
            bf16_enabled=bf16_enabled,
        )

        train_metrics = torch.tensor(
            [running_train_loss_sum, float(running_train_count)],
            device=device,
            dtype=torch.float64,
        )
        if distributed:
            dist.all_reduce(train_metrics, op=dist.ReduceOp.SUM)

        avg_train_loss = train_metrics[0].item() / max(train_metrics[1].item(), 1.0)
        running_train_loss_sum = 0.0
        running_train_count = 0

        if is_main_process(distributed):
            current_lr = optimizer.param_groups[0]["lr"]
            elapsed = format_duration(time.monotonic() - start_time)
            print(
                f"Step {step:05d}/{args.steps:05d} | lr={current_lr:.7f} | "
                f"train_loss={avg_train_loss:.5f} | val_loss={val_loss:.5f} | elapsed {elapsed}",
                flush=True,
            )

            with open(args.log_path, "a", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow([step, avg_train_loss, val_loss])

            latest_path = args.output_dir / "checkpoints" / f"{Path(args.save_name).stem}_last.pt"
            save_checkpoint(
                model=model,
                save_path=latest_path,
                step=step,
                best_val=best_val_loss,
                stage1_checkpoint=args.forecast_checkpoint,
                args=args,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = args.output_dir / args.save_name
                save_checkpoint(
                    model=model,
                    save_path=best_path,
                    step=step,
                    best_val=best_val_loss,
                    stage1_checkpoint=args.forecast_checkpoint,
                    args=args,
                )
                print(f"  -> saved new best checkpoint to {best_path}", flush=True)

            if args.save_every > 0 and step % args.save_every == 0:
                periodic_path = args.output_dir / "checkpoints" / f"{Path(args.save_name).stem}_step_{step:05d}.pt"
                save_checkpoint(
                    model=model,
                    save_path=periodic_path,
                    step=step,
                    best_val=best_val_loss,
                    stage1_checkpoint=args.forecast_checkpoint,
                    args=args,
                )

    if is_main_process(distributed):
        elapsed = format_duration(time.monotonic() - start_time)
        print(f"Stage 2 latent HierSwin complete | elapsed {elapsed}", flush=True)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
