#!/usr/bin/env python3
"""
run.py — Single command-line entry point for the QDMC rare event sampling pipeline.

Usage
-----
    python run.py spinup
        Generate the aquaplanet spinup state (run once, before any experiment).

    python run.py dns [--walkers N] [--members M] [--region NAME]
        Run the Direct Numerical Simulation baseline (no resampling).

    python run.py airres --emulator {new,old} [--scheme NAME]
                         [--walkers N] [--members M] [--region NAME]
        Run an AI-emulator-guided rare event sampling (AI+RES) experiment.

Examples
--------
    # Paper-scale run with the new emulator and the default 'ck' schedule:
    python run.py airres --emulator new

    # Reproduce the report figures (old lightweight emulator):
    python run.py airres --emulator old --scheme ck

    # Quick smoke test (tiny ensemble):
    python run.py dns    --walkers 4
    python run.py airres --emulator new --walkers 4 --members 2

The --walkers / --members flags override config.json for quick tests, so a
separate "small" config file is not needed.
"""

import argparse
import sys

import core


def _apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply optional CLI overrides (walkers / members) onto the loaded config."""
    if getattr(args, "walkers", None) is not None:
        config["N_walkers"] = int(args.walkers)
    if getattr(args, "members", None) is not None:
        config["M_members"] = int(args.members)
    return config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="QDMC rare event sampling: spinup, DNS baseline, and AI+RES experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="{spinup,dns,airres}")

    # --- spinup ---
    sub.add_parser("spinup", help="Generate the aquaplanet spinup state (run once).")

    # --- dns ---
    p_dns = sub.add_parser("dns", help="Run the DNS baseline (no resampling).")
    p_dns.add_argument("--walkers", type=int, default=None, help="Override N_walkers.")
    p_dns.add_argument("--members", type=int, default=None, help="Override M_members (unused by DNS).")
    p_dns.add_argument("--region", type=str, default=None, help="Target region name.")

    # --- airres ---
    p_air = sub.add_parser("airres", help="Run an AI+RES experiment.")
    p_air.add_argument(
        "--emulator", choices=["new", "old"], default="new",
        help="Which emulator to use (default: new latent-hierswin).",
    )
    p_air.add_argument(
        "--scheme", type=str, default="ck",
        help="C_k schedule name (a 'C_schedule_<scheme>' key in config.json; default: ck).",
    )
    p_air.add_argument("--walkers", type=int, default=None, help="Override N_walkers.")
    p_air.add_argument("--members", type=int, default=None, help="Override M_members.")
    p_air.add_argument("--region", type=str, default=None, help="Target region name.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    config = _apply_overrides(core.load_config(), args)

    if args.command == "spinup":
        core.run_spinup(config)

    elif args.command == "dns":
        core.run_dns_baseline(config, target_region_name=args.region)

    elif args.command == "airres":
        # Import the chosen driver lazily so a DNS/spinup run never needs torch
        # or the emulator packages on the path.
        if args.emulator == "new":
            import driver_new as driver
        else:
            import driver_old as driver
        driver.run_ai_res_scheme(config, scheme=args.scheme, target_region_name=args.region)

    else:  # pragma: no cover — argparse enforces a valid command
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
