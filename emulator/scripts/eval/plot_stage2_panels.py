#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_MPLCONFIGDIR = Path(os.environ.get("TMPDIR", "/tmp")) / f"upd_emulator_mpl_{os.getuid()}"
DEFAULT_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
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

from upd_emulator.constants import DEFAULT_DATA_ROOT
from upd_emulator.data import GCMSequenceDataset
from upd_emulator.data.dataset import load_stats

from _latent_hierswin_rollout_utils import (
    build_latent_hierswin_model,
    denormalize_ensemble,
    denormalize_states,
    rollout_ensemble,
)


CHANNEL_NAMES = [
    "surface_air_temperature",
    "surface_air_pressure",
    "air_temperature_860hPa",
    "air_temperature_500hPa",
    "air_temperature_260hPa",
    "specific_humidity_860hPa",
    "specific_humidity_500hPa",
    "specific_humidity_260hPa",
    "eastward_wind_860hPa",
    "eastward_wind_500hPa",
    "eastward_wind_260hPa",
    "northward_wind_860hPa",
    "northward_wind_500hPa",
    "northward_wind_260hPa",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ground-truth / ensemble-mean / difference field panels for the latent-HierSwin emulator."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run", default="run16")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "stage2" / "latent_hierswin_rollout_best.pt",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=REPO_ROOT / "resources" / "stats" / "exac_input_stats.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "latent_hierswin_field_panels_sample0",
    )
    parser.add_argument("--lead-times", nargs="+", type=int, default=[3, 7, 11])
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--ensemble-members", type=int, default=6)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--disable-bf16", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available.")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot_field_panel(
    prediction_mean: np.ndarray,
    target: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    lead: int,
    channel_name: str,
    output_path: Path,
) -> None:
    error_np = prediction_mean - target
    value_min = float(np.nanmin([prediction_mean.min(), target.min()]))
    value_max = float(np.nanmax([prediction_mean.max(), target.max()]))
    error_limit = float(np.nanmax(np.abs(error_np)))
    if error_limit == 0.0:
        error_limit = 1.0

    projection = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.4),
        subplot_kw={"projection": projection},
        constrained_layout=True,
    )
    panels = [
        ("Ground truth", target, "turbo", value_min, value_max),
        ("Ensemble mean", prediction_mean, "turbo", value_min, value_max),
        ("Mean - truth", error_np, "coolwarm", -error_limit, error_limit),
    ]

    for axis, (title, data, cmap, vmin, vmax) in zip(axes, panels, strict=True):
        levels = np.linspace(vmin, vmax, 64)
        data_cyclic, lon_cyclic = add_cyclic_point(data, coord=longitudes)
        image = axis.contourf(
            lon_cyclic,
            latitudes,
            data_cyclic,
            levels=levels,
            cmap=cmap,
            extend="both",
            transform=projection,
        )
        axis.coastlines(linewidth=0.8, color="k")
        axis.set_title(title)
        fig.colorbar(image, ax=axis, shrink=0.82)

    fig.suptitle(f"latent_hierswin | {channel_name} | lead {lead} days")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    plot_leads = sorted(set(args.lead_times))
    print(
        f"Rendering latent-HierSwin panels | run={args.run} | sample_index={args.sample_index} | "
        f"lead_times={plot_leads} | ensemble_members={args.ensemble_members} | device={device.type}",
        flush=True,
    )

    max_lead = max(plot_leads)
    stats = load_stats(args.stats_path)
    model = build_latent_hierswin_model(args.checkpoint_path, device=device, use_checkpoint=False)
    model.eval()

    dataset = GCMSequenceDataset(
        data_root=args.data_root,
        stats_path=args.stats_path,
        runs=[args.run],
        target_steps=max_lead,
        cache_dir=REPO_ROOT / "outputs" / "cache",
    )


    sample = dataset[args.sample_index]
    predictions_norm = rollout_ensemble(
        model=model,
        input_window=sample["input_window"],
        lead_steps=max_lead,
        ensemble_members=args.ensemble_members,
        device=device,
        bf16_enabled=(not args.disable_bf16 and device.type == "cuda"),
    )
    targets_norm = sample["future_states"][:max_lead].cpu()

    ensemble_mean = denormalize_ensemble(predictions_norm, stats).mean(dim=0).numpy()
    ground_truth = denormalize_states(targets_norm, stats).numpy()

    file_idx, start_idx = dataset.sample_index[args.sample_index]
    entry = dataset.file_entries[file_idx]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "backend": "latent_hierswin",
                "run": entry["run"],
                "year": entry["year"],
                "sample_index": args.sample_index,
                "start_time_index": int(start_idx),
                "checkpoint_path": str(args.checkpoint_path),
                "stats_path": str(args.stats_path),
                "ensemble_members": args.ensemble_members,
                "lead_times": plot_leads,
                "channel_names": CHANNEL_NAMES,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    latitudes = dataset.latitudes.cpu().numpy()
    longitudes = dataset.longitudes.cpu().numpy()
    for lead in plot_leads:
        lead_idx = lead - 1
        lead_dir = args.output_dir / f"lead_{lead:02d}"
        lead_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            lead_dir / f"lead_{lead:02d}_fields.npz",
            ensemble_mean=ensemble_mean[lead_idx].astype(np.float32),
            ground_truth=ground_truth[lead_idx].astype(np.float32),
            difference=(ensemble_mean[lead_idx] - ground_truth[lead_idx]).astype(np.float32),
            latitudes=latitudes.astype(np.float32),
            longitudes=longitudes.astype(np.float32),
            channel_names=np.array(CHANNEL_NAMES, dtype=object),
        )

        for channel_idx, channel_name in enumerate(CHANNEL_NAMES):
            plot_field_panel(
                prediction_mean=ensemble_mean[lead_idx, channel_idx],
                target=ground_truth[lead_idx, channel_idx],
                latitudes=latitudes,
                longitudes=longitudes,
                lead=lead,
                channel_name=channel_name,
                output_path=lead_dir / f"{channel_idx:02d}_{channel_name}.png",
            )

    print(f"Saved field panels to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
