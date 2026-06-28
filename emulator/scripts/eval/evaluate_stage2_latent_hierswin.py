#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_MPLCONFIGDIR = Path(os.environ.get("TMPDIR", "/tmp")) / f"upd_emulator_mpl_{os.getuid()}"
DEFAULT_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
UTILS_ROOT = REPO_ROOT / "scripts" / "utils"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(UTILS_ROOT) not in sys.path:
    sys.path.insert(0, str(UTILS_ROOT))

from upd_emulator.constants import DEFAULT_DATA_ROOT, PHYSICAL_CHANNEL_NAMES
from upd_emulator.data import GCMSequenceDataset

from _latent_hierswin_rollout_utils import (
    build_latent_hierswin_model,
    compute_acc,
    denormalize_ensemble,
    denormalize_states,
    load_stats_and_climatology,
    rollout_ensemble,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the latent-HierSwin stage-2 checkpoint on a single rollout seed."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=REPO_ROOT / "resources" / "stats" / "exac_input_stats.pt",
    )
    parser.add_argument(
        "--climatology-path",
        type=Path,
        default=REPO_ROOT / "resources" / "stats" / "climatology_run16.pt",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "stage2" / "latent_hierswin_rollout_best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "latent_hierswin_eval_sample0",
    )
    parser.add_argument("--lead-steps", type=int, default=30)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--ensemble-members", type=int, default=6)
    parser.add_argument("--disable-bf16", action="store_true")
    return parser.parse_args()


def plot_acc_comparison(acc_ens: np.ndarray, output_path: Path) -> None:
    lead_steps = acc_ens.shape[0]
    leads = np.arange(0, lead_steps + 1)
    acc_ens_plot = np.concatenate([np.ones((1, acc_ens.shape[1])), acc_ens], axis=0)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    axes = axes.flatten()
    groups = {
        "Surface": [0, 1],
        "Air Temperature": [2, 3, 4],
        "Specific Humidity": [5, 6, 7],
        "Wind": [8, 9, 10, 11, 12, 13],
    }

    for ax, (group_name, channel_indices) in zip(axes, groups.items()):
        for idx in channel_indices:
            name = PHYSICAL_CHANNEL_NAMES[idx]
            ax.plot(leads, acc_ens_plot[:, idx], "--", alpha=0.85, label=f"{name} ensemble")
        ax.axhline(y=0.6, color="r", linestyle="--", alpha=0.55)
        ax.set_title(f"ACC 0-{lead_steps} Days: {group_name}")
        ax.set_xlabel("Lead time (days)")
        ax.set_ylabel("ACC")
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(0, lead_steps)
        ax.grid(True)
        ax.legend(fontsize=7)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_acc_csv(acc_ens: np.ndarray, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        header = ["lead"]
        for name in PHYSICAL_CHANNEL_NAMES:
            header.extend([f"ensemble__{name}"])
        handle.write(",".join(header) + "\n")
        row = ["0"]
        for _ in range(acc_ens.shape[1]):
            row.extend(["1.0"])
        handle.write(",".join(row) + "\n")
        for lead_idx in range(acc_ens.shape[0]):
            row = [str(lead_idx + 1)]
            for channel_idx in range(acc_ens.shape[1]):
                row.append(str(float(acc_ens[lead_idx, channel_idx])))
            handle.write(",".join(row) + "\n")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    bf16_enabled = not args.disable_bf16

    stats, climatology = load_stats_and_climatology(args.stats_path, args.climatology_path)
    model = build_latent_hierswin_model(args.checkpoint_path, device=device, use_checkpoint=False)
    model.eval()

    dataset = GCMSequenceDataset(
        data_root=args.data_root,
        stats_path=args.stats_path,
        runs=["run16"],
        target_steps=args.lead_steps,
        cache_dir=REPO_ROOT / "outputs" / "cache",
    )
    sample = dataset[args.sample_index]

    print(
        f"Evaluating latent-HierSwin checkpoint | lead_steps={args.lead_steps} | "
        f"ensemble_members={args.ensemble_members} | sample_index={args.sample_index} | device={device.type}",
        flush=True,
    )

    predictions_norm = rollout_ensemble(
        model=model,
        input_window=sample["input_window"],
        lead_steps=args.lead_steps,
        ensemble_members=args.ensemble_members,
        device=device,
        bf16_enabled=bf16_enabled,
    )

    targets_norm = sample["future_states"][: args.lead_steps].cpu()

    predictions_physical = denormalize_ensemble(predictions_norm, stats)
    mean_physical = predictions_physical.mean(dim=0)

    targets_physical = denormalize_states(targets_norm, stats)

    acc_ens = compute_acc(
        predictions=mean_physical,
        targets=targets_physical,
        climatology=climatology,
        latitudes=dataset.latitudes.cpu().numpy(),
    )


    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_acc_comparison(acc_ens, args.output_dir / f"acc_{args.lead_steps:02d}_days.png")
    write_acc_csv(acc_ens, args.output_dir / f"acc_{args.lead_steps:02d}_days.csv")

    overall_ens = np.mean(acc_ens, axis=1)
    for lead in [1, 3, 5, 10, 15, 20, 30]:
        if lead <= args.lead_steps:
            print(
                f"Lead {lead:02d} | ensemble={overall_ens[lead - 1]:.4f}",
                flush=True,
            )

    print(f"Saved evaluation to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
