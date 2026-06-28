"""Shared constants for the exact-implementation aquaplanet emulator."""

from pathlib import Path

GRID_SHAPE = (64, 128)
LATENT_SHAPE = (8, 16)
CONTEXT_STEPS = 2
PHYSICAL_CHANNELS = 14
HELPER_CHANNELS = 7
INPUT_CHANNELS = PHYSICAL_CHANNELS * CONTEXT_STEPS
FORECAST_HORIZON = 15
DEFAULT_DATA_ROOT = Path("/home/nishidh/fuxi-ens/data")

SURFACE_VARS = [
    "surface_air_temperature",
    "surface_air_pressure",
]

UPPER_AIR_VARS = [
    "air_temperature",
    "specific_humidity",
    "eastward_wind",
    "northward_wind",
]

PHYSICAL_CHANNEL_NAMES = [
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

HELPER_CHANNEL_NAMES = [
    "orography",
    "land_sea_mask",
    "latitude",
    "longitude",
    "hour_of_day",
    "day_of_year",
    "step",
]

DEFAULT_TRAIN_RUNS = [f"run{i}" for i in range(1, 14)]
DEFAULT_VAL_RUNS = [f"run{i}" for i in range(14, 17)]
DEFAULT_YEARS = [f"year{i}.nc" for i in range(1, 21)]
