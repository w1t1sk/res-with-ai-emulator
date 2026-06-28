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

from upd_emulator.constants import DEFAULT_DATA_ROOT, DEFAULT_TRAIN_RUNS, DEFAULT_VAL_RUNS
from upd_emulator.data import GCMSequenceDataset
from upd_emulator.models import FuXiForecastModelV3
from upd_emulator.training import LatitudeWeightedL1

warnings.filterwarnings(
    "ignore",
    message="TypedStorage is deprecated.*",
    category=UserWarning,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-1 deterministic V3 forecast-model training.")
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
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "forecast_stage1",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=REPO_ROOT / "outputs" / "forecast_stage1" / "training_log.csv",
    )
    parser.add_argument("--save-name", default="forecast_stage1_best.pt")
    parser.add_argument("--steps", type=int, default=60_000)
    parser.add_argument("--val-every", type=int, default=1_000)
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Archive a checkpoint every N steps. Defaults to --val-every.",
    )
    parser.add_argument("--val-batches", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--memory-cache-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
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

        try:
            torch.cuda.set_device(cuda_index)
        except Exception as exc:
            raise RuntimeError(
                "Slurm exposed GPUs for this job, but PyTorch could not initialize CUDA. "
                "Check that the allocated node has healthy A100 devices and NVIDIA Fabric Manager is running."
            ) from exc

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


def reduce_mean(value: torch.Tensor, distributed: bool) -> torch.Tensor:
    if not distributed:
        return value
    value = value.clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value /= dist.get_world_size()
    return value


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


def print_progress(
    step: int,
    total_steps: int,
    train_loss: float,
    start_time: float,
) -> None:
    elapsed = time.monotonic() - start_time
    seconds_per_step = elapsed / max(step, 1)
    eta = seconds_per_step * max(total_steps - step, 0)
    percent = 100.0 * step / max(total_steps, 1)
    print(
        f"Step {step:06d}/{total_steps:06d} ({percent:5.1f}%) | "
        f"Train L1 {train_loss:.5f} | "
        f"{seconds_per_step:.2f}s/step | elapsed {format_duration(elapsed)} | "
        f"ETA {format_duration(eta)}",
        flush=True,
    )


def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: LatitudeWeightedL1,
    device: torch.device,
    distributed: bool,
    bf16_enabled: bool,
    max_batches: int,
) -> float:
    model.eval()
    total = torch.zeros((), device=device)
    count = torch.zeros((), device=device)

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= max_batches:
                break

            inputs = batch["input_window"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=bf16_enabled and device.type == "cuda",
            ):
                predictions = model(inputs)
                loss = criterion(predictions, targets)
            total += loss.detach().float()
            count += 1

    if count.item() == 0:
        return float("inf")

    metrics = torch.stack([total, count])
    if distributed:
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    return (metrics[0] / metrics[1]).item()


def save_checkpoint(
    model: torch.nn.Module,
    save_path: Path,
    step: int,
    best_val: float,
) -> None:
    payload = {
        "model": unwrap_model(model).state_dict(),
        "step": step,
        "best_val": best_val,
    }
    torch.save(payload, save_path)


def main() -> None:
    args = parse_args()
    distributed, world_size, local_rank = setup_distributed_env()
    device = select_device(local_rank)
    initialize_distributed(distributed, device)
    set_seed(args.seed, use_cuda=device.type == "cuda")

    if is_main_process(distributed):
        node_list = os.environ.get("SLURM_JOB_NODELIST", "local")
        print(
            f"Stage 1 starting | steps={args.steps} | world_size={world_size} | "
            f"device={device.type} | nodes={node_list} | log_every={args.log_every}",
            flush=True,
        )

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

    if is_main_process(distributed):
        print(
            f"Datasets ready | train_samples={len(train_dataset)} | "
            f"val_samples={len(val_dataset)} | batch_size_per_rank={args.batch_size}",
            flush=True,
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

    model = FuXiForecastModelV3(
        use_checkpoint=not args.disable_gradient_checkpointing,
    ).to(device)
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
        )

    criterion = LatitudeWeightedL1(
        height=int(train_dataset.latitudes.numel()),
        latitudes=train_dataset.latitudes,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    bf16_enabled = not args.disable_bf16
    best_val = float("inf")
    save_every = args.val_every if args.save_every is None else args.save_every
    current_epoch = 0
    if train_sampler is not None:
        train_sampler.set_epoch(current_epoch)
    train_iterator = iter(train_loader)
    train_start_time = time.monotonic()

    if is_main_process(distributed):
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.log_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoints_dir = args.output_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        with open(args.log_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["step", "train_l1", "val_l1"])

    for step in range(1, args.steps + 1):
        try:
            batch = next(train_iterator)
        except StopIteration:
            current_epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(current_epoch)
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        inputs = batch["input_window"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=bf16_enabled and device.type == "cuda",
        ):
            predictions = model(inputs)
            loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        if is_main_process(distributed) and save_every > 0 and step % save_every == 0:
            periodic_path = checkpoints_dir / f"{Path(args.save_name).stem}_step_{step:06d}.pt"
            save_checkpoint(model, periodic_path, step=step, best_val=best_val)

        train_loss = reduce_mean(loss.detach().float(), distributed).item()
        should_log = args.log_every > 0 and (
            step == 1 or step % args.log_every == 0 or step == args.steps
        )
        if is_main_process(distributed) and should_log:
            print_progress(
                step=step,
                total_steps=args.steps,
                train_loss=train_loss,
                start_time=train_start_time,
            )

        if step % args.val_every == 0 or step == args.steps:
            if is_main_process(distributed):
                print(
                    f"Step {step:06d}/{args.steps:06d} | starting validation "
                    f"for up to {args.val_batches} batches",
                    flush=True,
                )
            val_loss = evaluate(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                device=device,
                distributed=distributed,
                bf16_enabled=bf16_enabled,
                max_batches=args.val_batches,
            )

            if is_main_process(distributed):
                with open(args.log_path, "a", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow([step, train_loss, val_loss])

                latest_path = args.output_dir / "forecast_stage1_last.pt"
                save_checkpoint(model, latest_path, step=step, best_val=best_val)
                if val_loss < best_val:
                    best_val = val_loss
                    best_path = args.output_dir / args.save_name
                    save_checkpoint(model, best_path, step=step, best_val=best_val)
                    print(
                        f"Step {step:06d} | Train L1 {train_loss:.5f} | "
                        f"Val L1 {val_loss:.5f} | saved {best_path}",
                        flush=True,
                    )
                else:
                    print(
                        f"Step {step:06d} | Train L1 {train_loss:.5f} | "
                        f"Val L1 {val_loss:.5f}",
                        flush=True,
                    )

    if is_main_process(distributed):
        latest_path = args.output_dir / "forecast_stage1_last.pt"
        save_checkpoint(model, latest_path, step=args.steps, best_val=best_val)
        print(
            f"Stage 1 complete | elapsed {format_duration(time.monotonic() - train_start_time)}",
            flush=True,
        )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
