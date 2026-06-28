from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import os
import socket
import time
import uuid

import numpy as np
import torch
import torch.distributed as dist
import xarray as xr
from torch.utils.data import Dataset

from upd_emulator.constants import CONTEXT_STEPS, PHYSICAL_CHANNELS
from upd_emulator.data.io import dataset_coordinates, resolve_year_files, stack_physical_channels
from upd_emulator.grid import grid_scheme

CACHE_LOCK_POLL_SECONDS = 2.0
CACHE_LOCK_STALE_SECONDS = 30 * 60


def load_stats(path: str | Path) -> dict[str, torch.Tensor]:
    """Load the combined physical/helper normalization statistics."""
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def build_truth_windows(states: torch.Tensor) -> torch.Tensor:
    """Build paper-style posterior windows [x(t), x(t+1)] from a full state sequence."""

    windows = []
    for offset in range(states.shape[0] - CONTEXT_STEPS):
        windows.append(torch.cat([states[offset + 1], states[offset + 2]], dim=0))
    return torch.stack(windows, dim=0)


class GCMSequenceDataset(Dataset):
    """Lazy dataset for FuXi-ENS-style autoregressive training windows."""

    def __init__(
        self,
        data_root: str | Path,
        stats_path: str | Path,
        runs: list[str],
        years: list[str] | None = None,
        target_steps: int = 1,
        cache_dir: str | Path | None = None,
        memory_cache_size: int = 8,
    ) -> None:

        self.data_root = Path(data_root)
        self.stats_path = Path(stats_path)
        self.runs = list(runs)
        self.years = list(years) if years is not None else None
        self.target_steps = target_steps
        self.memory_cache_size = memory_cache_size
        self.sequence_length = CONTEXT_STEPS + self.target_steps

        self.cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else Path(os.environ.get("UPD_EMULATOR_CACHE_DIR", self.data_root / ".cache"))
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        stats = load_stats(self.stats_path)
        self.physical_mean = stats["physical_mean"].view(-1, 1, 1)
        self.physical_std = stats["physical_std"].view(-1, 1, 1)
        self.helper_mean = stats["helper_mean"].view(-1, 1, 1)
        self.helper_std = stats["helper_std"].view(-1, 1, 1)
        self.stats_token = f"{self.stats_path.stem}_{self.stats_path.stat().st_mtime_ns}"

        year_token = (
            "all"
            if self.years is None
            else f"{Path(self.years[0]).stem}_{Path(self.years[-1]).stem}_{len(self.years)}"
        )
        manifest_name = f"manifest_{self.runs[0]}_{self.runs[-1]}_{year_token}.pt"
        self.manifest_path = self.cache_dir / manifest_name
        self._memory_cache: OrderedDict[int, torch.Tensor] = OrderedDict()

        self.manifest = self._load_or_create_manifest()
        self.lat_map = self.manifest["lat_map"].float()
        self.lon_map = self.manifest["lon_map"].float()
        self.latitudes = self.manifest["latitudes"].float()
        self.longitudes = self.manifest["longitudes"].float()
        self.zero_map = torch.zeros_like(self.lat_map)
        self.file_entries = self.manifest["files"]

        self.sample_index = []
        for file_idx, entry in enumerate(self.file_entries):
            max_start = entry["n_time"] - self.sequence_length + 1
            for start in range(max(0, max_start)):
                self.sample_index.append((file_idx, start))

    def _manifest_is_current(self, manifest: dict) -> bool:
        if not isinstance(manifest, dict):
            return False

        required = {
            "files",
            "lat_map",
            "lon_map",
            "latitudes",
            "longitudes",
            "grid_scheme",
            "provenance",
            "requested_years",
            "stats_token",
        }
        if not required.issubset(manifest):
            return False

        expected_years = None if self.years is None else list(self.years)
        return (
            manifest.get("requested_years") == expected_years
            and manifest.get("stats_token") == self.stats_token
        )

    def _load_or_create_manifest(self) -> dict:
        manifest = None
        if self.manifest_path.exists():
            manifest = torch.load(self.manifest_path, map_location="cpu", weights_only=False)
            if self._manifest_is_current(manifest):
                return manifest

        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:
                torch.save(self._build_manifest(), self.manifest_path)
            dist.barrier()
        else:
            torch.save(self._build_manifest(), self.manifest_path)

        return torch.load(self.manifest_path, map_location="cpu", weights_only=False)

    def _build_manifest(self) -> dict:
        files = []
        latitudes = None
        longitudes = None
        source_files = {}

        for run in self.runs:
            year_paths = resolve_year_files(self.data_root / run, self.years)
            source_files[run] = [path.name for path in year_paths]
            for raw_path in year_paths:
                cache_stub = f"{run}_{raw_path.stem}_{self.stats_token}_physical.pt"
                cache_path = self.cache_dir / cache_stub

                with xr.open_dataset(raw_path) as ds:
                    times = ds["time"]
                    day_of_year = torch.from_numpy(times.dt.dayofyear.values.astype(np.int16))
                    hour_of_day = torch.from_numpy(times.dt.hour.values.astype(np.int16))
                    n_time = int(ds.sizes["time"])

                    ds_latitudes, ds_longitudes = dataset_coordinates(ds)
                    if latitudes is None and ds_latitudes is not None:
                        latitudes = ds_latitudes
                    if longitudes is None and ds_longitudes is not None:
                        longitudes = ds_longitudes

                files.append(
                    {
                        "run": run,
                        "year": raw_path.name,
                        "raw_path": str(raw_path),
                        "cache_path": str(cache_path),
                        "day_of_year": day_of_year,
                        "hour_of_day": hour_of_day,
                        "n_time": n_time,
                    }
                )



        lat_map_np = np.repeat(latitudes[:, None], longitudes.shape[0], axis=1)
        lon_map_np = np.repeat(longitudes[None, :], latitudes.shape[0], axis=0)

        return {
            "files": files,
            "lat_map": torch.from_numpy(lat_map_np).float(),
            "lon_map": torch.from_numpy(lon_map_np).float(),
            "latitudes": torch.from_numpy(latitudes).float(),
            "longitudes": torch.from_numpy(longitudes).float(),
            "grid_scheme": grid_scheme(latitudes, longitudes),
            "requested_years": None if self.years is None else list(self.years),
            "stats_token": self.stats_token,
            "provenance": {
                "data_root": str(self.data_root.resolve()),
                "stats_path": str(self.stats_path.resolve()),
                "runs": list(self.runs),
                "year_files": source_files,
            },
        }

    def _get_year_tensor(self, file_idx: int) -> torch.Tensor:
        if file_idx in self._memory_cache:
            self._memory_cache.move_to_end(file_idx)
            return self._memory_cache[file_idx]

        entry = self.file_entries[file_idx]
        cache_path = Path(entry["cache_path"])
        if cache_path.exists():
            try:
                year_tensor = self._load_year_cache(cache_path, entry)
            except Exception:
                year_tensor = self._load_or_rebuild_year_cache(cache_path, entry)
        else:
            year_tensor = self._load_or_rebuild_year_cache(cache_path, entry)

        self._memory_cache[file_idx] = year_tensor
        while len(self._memory_cache) > self.memory_cache_size:
            self._memory_cache.popitem(last=False)
        return year_tensor

    def _load_year_cache(self, cache_path: Path, entry: dict) -> torch.Tensor:
        year_tensor = torch.load(cache_path, map_location="cpu", weights_only=True)
        expected_shape = (int(entry["n_time"]), PHYSICAL_CHANNELS, *self.lat_map.shape)
        if tuple(year_tensor.shape) != expected_shape:
            raise ValueError(
                f"Cache file {cache_path} has shape {tuple(year_tensor.shape)}, "
                f"expected {expected_shape}."
            )
        return year_tensor

    def _load_or_rebuild_year_cache(self, cache_path: Path, entry: dict) -> torch.Tensor:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = cache_path.with_name(f"{cache_path.name}.lock")

        while True:
            lock_fd = self._try_acquire_cache_lock(lock_path)
            if lock_fd is None:
                if cache_path.exists():
                    try:
                        return self._load_year_cache(cache_path, entry)
                    except Exception:
                        pass
                self._remove_stale_cache_lock(lock_path)
                time.sleep(CACHE_LOCK_POLL_SECONDS)
                continue

            try:
                if cache_path.exists():
                    try:
                        return self._load_year_cache(cache_path, entry)
                    except Exception:
                        cache_path.unlink(missing_ok=True)

                year_tensor = self._build_physical_tensor(Path(entry["raw_path"]))
                self._atomic_save_year_cache(year_tensor, cache_path)
                return year_tensor
            finally:
                os.close(lock_fd)
                lock_path.unlink(missing_ok=True)

    def _try_acquire_cache_lock(self, lock_path: Path) -> int | None:
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            lock_fd = os.open(lock_path, flags, 0o664)
        except FileExistsError:
            return None

        metadata = f"host={socket.gethostname()} pid={os.getpid()} time={time.time()}\n"
        os.write(lock_fd, metadata.encode("utf-8"))
        return lock_fd

    def _remove_stale_cache_lock(self, lock_path: Path) -> None:
        try:
            age_seconds = time.time() - lock_path.stat().st_mtime
        except FileNotFoundError:
            return

        if age_seconds > CACHE_LOCK_STALE_SECONDS:
            lock_path.unlink(missing_ok=True)

    def _atomic_save_year_cache(self, year_tensor: torch.Tensor, cache_path: Path) -> None:
        temp_name = f".{cache_path.name}.{socket.gethostname()}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        temp_path = cache_path.with_name(temp_name)
        try:
            torch.save(year_tensor, temp_path)
            os.replace(temp_path, cache_path)
        finally:
            temp_path.unlink(missing_ok=True)

    def _build_physical_tensor(self, raw_path: Path) -> torch.Tensor:
        with xr.open_dataset(raw_path) as ds:
            stacked = stack_physical_channels(ds)
        tensor = torch.from_numpy(stacked)
        return (tensor - self.physical_mean) / (self.physical_std + 1e-6)

    def _build_helper_map(self, day_of_year: int, hour_of_day: int, step: int) -> torch.Tensor:
        helper_raw = torch.stack(
            [
                self.zero_map,
                self.zero_map,
                self.lat_map,
                self.lon_map,
                torch.full_like(self.lat_map, float(hour_of_day)),
                torch.full_like(self.lat_map, float(day_of_year)),
                torch.full_like(self.lat_map, float(step)),
            ],
            dim=0,
        )
        return (helper_raw - self.helper_mean) / (self.helper_std + 1e-6)

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        file_idx, start = self.sample_index[idx]
        year_entry = self.file_entries[file_idx]
        year_tensor = self._get_year_tensor(file_idx)
        sequence = year_tensor[start : start + self.sequence_length]

        input_window = torch.cat([sequence[0], sequence[1]], dim=0)
        future_states = sequence[2:]
        truth_windows = build_truth_windows(sequence)

        helper_maps = []
        for step_idx in range(self.target_steps):
            target_index = start + CONTEXT_STEPS + step_idx
            helper_maps.append(
                self._build_helper_map(
                    int(year_entry["day_of_year"][target_index]),
                    int(year_entry["hour_of_day"][target_index]),
                    step_idx + 1,
                )
            )

        return {
            "input_window": input_window,
            "target": future_states[0],
            "future_states": future_states,
            "helper_sequence": torch.stack(helper_maps, dim=0),
            "truth_windows": truth_windows,
        }
