#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
import sys

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

from _latent_hierswin_rollout_utils import (
    build_latent_hierswin_model,
    evaluate_rollout_acc,
    load_stats_and_climatology,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rescore latent HierSwin stage-2 checkpoints by rollout ACC instead of one-step loss."
        )
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
        "--checkpoint-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "latent_hierswin_stage2" / "checkpoints",
    )
    parser.add_argument(
        "--best-checkpoint",
        type=Path,
        default=REPO_ROOT / "outputs" / "latent_hierswin_stage2" / "latent_hierswin_stage2_best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "latent_hierswin_stage2_rescored",
    )
    parser.add_argument(
        "--promote-best-path",
        type=Path,
        default=REPO_ROOT / "outputs" / "latent_hierswin_stage2_rescored" / "latent_hierswin_rollout_best.pt",
    )
    parser.add_argument("--lead-steps", type=int, default=20)
    parser.add_argument("--ensemble-members", type=int, default=6)
    parser.add_argument("--sample-indices", nargs="+", type=int, default=[0])
    parser.add_argument("--score-start-lead", type=int, default=10)
    parser.add_argument("--score-end-lead", type=int, default=20)
    parser.add_argument("--rollout-runs", nargs="+", default=["run16"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--disable-bf16", action="store_true")
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def summarize_rollout(
    overall_ensemble: np.ndarray,
    overall_no_perturb: np.ndarray,
    score_start_lead: int,
    score_end_lead: int,
) -> dict[str, float]:
    score = float(np.mean(overall_ensemble[score_start_lead - 1 : score_end_lead]))
    summary = {
        "rollout_score": score,
        "rollout_no_perturb_score": float(np.mean(overall_no_perturb[score_start_lead - 1 : score_end_lead])),
    }
    for lead in [10, 15, 20, 30]:
        if lead <= overall_ensemble.shape[0]:
            summary[f"ensemble_day{lead}"] = float(overall_ensemble[lead - 1])
            summary[f"no_perturb_day{lead}"] = float(overall_no_perturb[lead - 1])
        else:
            summary[f"ensemble_day{lead}"] = float("nan")
            summary[f"no_perturb_day{lead}"] = float("nan")
    return summary


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    marker = "_step_"
    if marker in stem:
        tail = stem.split(marker, 1)[1]
        digits = "".join(ch for ch in tail if ch.isdigit())
        if digits:
            return int(digits), path.name
    if "last" in stem:
        return 10**9, path.name
    return -1, path.name


def main() -> None:
    args = parse_args()
    if args.lead_steps < 1:
        raise ValueError("--lead-steps must be >= 1.")
    if args.ensemble_members < 1:
        raise ValueError("--ensemble-members must be >= 1.")
    if args.score_start_lead < 1 or args.score_end_lead < args.score_start_lead or args.score_end_lead > args.lead_steps:
        raise ValueError("Invalid rollout score lead window.")

    device = select_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    stats, climatology = load_stats_and_climatology(args.stats_path, args.climatology_path)
    rollout_dataset = GCMSequenceDataset(
        data_root=args.data_root,
        stats_path=args.stats_path,
        runs=args.rollout_runs,
        target_steps=args.lead_steps,
        cache_dir=REPO_ROOT / "outputs" / "cache",
        memory_cache_size=2,
    )

    checkpoint_paths = sorted(args.checkpoint_dir.glob("*.pt"), key=checkpoint_sort_key)
    if args.best_checkpoint.exists():
        checkpoint_paths.append(args.best_checkpoint)
    checkpoint_paths = list(dict.fromkeys(checkpoint_paths))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found in {args.checkpoint_dir}.")

    bf16_enabled = not args.disable_bf16
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "rollout_rescore_summary.csv"

    rows: list[dict[str, object]] = []
    best_row: dict[str, object] | None = None

    print(
        f"Rescoring {len(checkpoint_paths)} latent HierSwin checkpoints | lead_steps={args.lead_steps} | "
        f"score_window=[{args.score_start_lead},{args.score_end_lead}] | "
        f"samples={args.sample_indices} | ensemble_members={args.ensemble_members} | device={device.type}",
        flush=True,
    )

    for checkpoint_path in checkpoint_paths:
        model = build_latent_hierswin_model(checkpoint_path, device=device, use_checkpoint=False)
        rollout_metrics = evaluate_rollout_acc(
            model=model,
            dataset=rollout_dataset,
            sample_indices=args.sample_indices,
            lead_steps=args.lead_steps,
            ensemble_members=args.ensemble_members,
            stats=stats,
            climatology=climatology,
            device=device,
            bf16_enabled=bf16_enabled,
        )
        summary = summarize_rollout(
            overall_ensemble=rollout_metrics["ensemble_overall"],
            overall_no_perturb=rollout_metrics["no_perturb_overall"],
            score_start_lead=args.score_start_lead,
            score_end_lead=args.score_end_lead,
        )
        row = {
            "checkpoint_path": str(checkpoint_path),
            **summary,
        }
        rows.append(row)
        if best_row is None or float(row["rollout_score"]) > float(best_row["rollout_score"]):
            best_row = row
        print(
            f"{checkpoint_path.name} | rollout_score={row['rollout_score']:.4f} | "
            f"day10={row['ensemble_day10']:.4f} | day15={row['ensemble_day15']:.4f} | "
            f"day20={row['ensemble_day20']:.4f}",
            flush=True,
        )

    assert best_row is not None
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "checkpoint_path",
                "rollout_score",
                "rollout_no_perturb_score",
                "ensemble_day10",
                "no_perturb_day10",
                "ensemble_day15",
                "no_perturb_day15",
                "ensemble_day20",
                "no_perturb_day20",
                "ensemble_day30",
                "no_perturb_day30",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    best_source = Path(str(best_row["checkpoint_path"]))
    args.promote_best_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_source, args.promote_best_path)

    best_report = args.output_dir / "best_rollout_checkpoint.txt"
    best_report.write_text(
        "\n".join(
            [
                f"best_checkpoint={best_source}",
                f"promoted_copy={args.promote_best_path}",
                f"rollout_score={best_row['rollout_score']}",
                f"ensemble_day10={best_row['ensemble_day10']}",
                f"ensemble_day15={best_row['ensemble_day15']}",
                f"ensemble_day20={best_row['ensemble_day20']}",
                f"ensemble_day30={best_row['ensemble_day30']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"Best rollout checkpoint: {best_source} | score={best_row['rollout_score']:.4f} | "
        f"copied to {args.promote_best_path}",
        flush=True,
    )
    print(f"Saved summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
