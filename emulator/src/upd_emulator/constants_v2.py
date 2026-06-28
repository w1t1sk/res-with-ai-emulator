"""High-fidelity V2 constants for the aquaplanet FuXi-ENS emulator."""

from __future__ import annotations

from upd_emulator.constants import (
    CONTEXT_STEPS,
    DEFAULT_DATA_ROOT,
    DEFAULT_TRAIN_RUNS,
    DEFAULT_VAL_RUNS,
    DEFAULT_YEARS,
    FORECAST_HORIZON,
    GRID_SHAPE,
    HELPER_CHANNEL_NAMES,
    HELPER_CHANNELS,
    INPUT_CHANNELS,
    PHYSICAL_CHANNEL_NAMES,
    PHYSICAL_CHANNELS,
    SURFACE_VARS,
    UPPER_AIR_VARS,
)

LATENT_STRIDE = 1
LATENT_SHAPE = GRID_SHAPE
SWIN_WINDOW_SIZE = 8
SWIN_SHIFT_SIZE = 4
