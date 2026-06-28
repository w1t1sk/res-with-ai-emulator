from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from upd_emulator.constants import GRID_SHAPE, SURFACE_VARS, UPPER_AIR_VARS
from upd_emulator.grid import gaussian_latitudes, regular_longitudes


def year_sort_key(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("year"):
        raise ValueError(f"Expected a year*.nc file, got {path.name}")

    try:
        return int(stem.replace("year", "", 1))
    except ValueError as exc:
        raise ValueError(f"Could not parse year index from {path.name}") from exc


def sorted_year_files(run_path: Path) -> list[Path]:
    if not run_path.exists():
        raise FileNotFoundError(f"Missing run directory: {run_path}")

    year_files = sorted(run_path.glob("year*.nc"), key=year_sort_key)
    if not year_files:
        raise FileNotFoundError(f"No year*.nc files found in {run_path}")
    return year_files


def resolve_year_files(run_path: Path, requested_years: list[str] | None = None) -> list[Path]:
    if requested_years is None:
        return sorted_year_files(run_path)

    if not run_path.exists():
        raise FileNotFoundError(f"Missing run directory: {run_path}")

    year_files: list[Path] = []
    missing_years: list[str] = []
    for year_name in requested_years:
        year_path = run_path / year_name
        if year_path.exists():
            year_files.append(year_path)
        else:
            missing_years.append(year_name)

    if missing_years:
        missing = ", ".join(missing_years)
        raise FileNotFoundError(f"Missing year files in {run_path}: {missing}")

    return sorted(year_files, key=year_sort_key)


def dataset_coordinates(ds: xr.Dataset) -> tuple[np.ndarray | None, np.ndarray | None]:
    latitudes = None
    longitudes = None
    if "lat" in ds.coords:
        latitudes = np.asarray(ds["lat"].values, dtype=np.float32)
    if "lon" in ds.coords:
        longitudes = np.asarray(ds["lon"].values, dtype=np.float32)
    return latitudes, longitudes


def load_grid_coordinates(
    data_root: Path | str,
    runs: list[str] | tuple[str, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    data_root = Path(data_root)
    candidate_runs = (
        list(runs)
        if runs is not None
        else sorted(path.name for path in data_root.glob("run*") if path.is_dir())
    )

    latitudes = None
    longitudes = None
    for run in candidate_runs:
        run_path = data_root / run
        if not run_path.exists():
            continue

        try:
            year_files = sorted_year_files(run_path)
        except FileNotFoundError:
            continue

        for year_path in year_files:
            with xr.open_dataset(year_path) as ds:
                ds_latitudes, ds_longitudes = dataset_coordinates(ds)

            if latitudes is None and ds_latitudes is not None:
                latitudes = ds_latitudes
            if longitudes is None and ds_longitudes is not None:
                longitudes = ds_longitudes
            if latitudes is not None and longitudes is not None:
                return latitudes, longitudes

    if latitudes is None:
        latitudes = gaussian_latitudes(GRID_SHAPE[0])
    if longitudes is None:
        longitudes = regular_longitudes(GRID_SHAPE[1])
    return latitudes, longitudes


def stack_physical_channels(ds: xr.Dataset) -> np.ndarray:
    vars_2d = [
        ds[var_name].values[:, np.newaxis, :, :].astype(np.float32) for var_name in SURFACE_VARS
    ]
    vars_3d = [ds[var_name].values.astype(np.float32) for var_name in UPPER_AIR_VARS]
    return np.concatenate(vars_2d + vars_3d, axis=1)
