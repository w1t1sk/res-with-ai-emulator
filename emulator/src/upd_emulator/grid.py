from __future__ import annotations

import numpy as np


GAUSSIAN_GRID_TOLERANCE = 1e-3


def _as_float_array(values) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64)


def gaussian_latitudes(num_latitudes: int) -> np.ndarray:
    nodes, _ = np.polynomial.legendre.leggauss(num_latitudes)
    return np.degrees(np.arcsin(nodes[::-1])).astype(np.float32)


def gaussian_latitude_weights(num_latitudes: int) -> np.ndarray:
    _, weights = np.polynomial.legendre.leggauss(num_latitudes)
    return weights[::-1].astype(np.float64)


def regular_longitudes(num_longitudes: int) -> np.ndarray:
    return np.linspace(0.0, 360.0, num_longitudes, endpoint=False, dtype=np.float32)


def is_uniform_spacing(values, atol: float = 1e-6) -> bool:
    array = _as_float_array(values)
    if array.size < 2:
        return True
    diffs = np.diff(array)
    return bool(np.allclose(diffs, diffs[0], atol=atol, rtol=0.0))


def is_gaussian_latitudes(latitudes, atol: float = GAUSSIAN_GRID_TOLERANCE) -> bool:
    latitudes = _as_float_array(latitudes)
    if latitudes.ndim != 1 or latitudes.size == 0:
        return False

    expected = gaussian_latitudes(latitudes.size).astype(np.float64)
    return bool(np.allclose(latitudes, expected, atol=atol, rtol=0.0))


def grid_scheme(latitudes=None, longitudes=None) -> str:
    if latitudes is None or longitudes is None:
        return "unknown"

    if is_gaussian_latitudes(latitudes) and is_uniform_spacing(longitudes):
        return "regular_gaussian_latlon"
    if is_uniform_spacing(latitudes) and is_uniform_spacing(longitudes):
        return "equiangular_latlon"
    return "custom_latlon"


def latitude_row_area_weights(latitudes) -> np.ndarray:
    latitudes = _as_float_array(latitudes)
    if latitudes.ndim != 1 or latitudes.size == 0:
        raise ValueError("latitudes must be a non-empty 1D array")

    if is_gaussian_latitudes(latitudes):
        return gaussian_latitude_weights(latitudes.size)
    return np.cos(np.deg2rad(latitudes)).astype(np.float64)


def normalized_latitude_loss_weights(latitudes) -> np.ndarray:
    row_weights = latitude_row_area_weights(latitudes)
    return (row_weights.size * row_weights / row_weights.sum()).astype(np.float32)
