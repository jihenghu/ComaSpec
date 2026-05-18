"""Temperature profile models."""

from __future__ import annotations

import numpy as np

from exceptions import ComartError


def temperature_constant(r: np.ndarray, temperature: float) -> np.ndarray:
    """Return a constant temperature profile."""
    if temperature <= 0:
        raise ComartError("Temperature must be positive.")
    return np.ones_like(r, dtype=np.float64) * temperature


def temperature_inverse_distance(
    r: np.ndarray | float,
    *,
    a: float = 1.0,
    b: float = 0.0,
    t0: float = 150.0,
    r0: float = 2e3,
) -> np.ndarray | float:
    """Return an inverse-distance temperature law."""
    if t0 <= 0:
        raise ComartError("`t0` must be positive.")
    if r0 <= 0:
        raise ComartError("`r0` must be positive.")

    r_array = np.atleast_1d(r).astype(np.float64)
    if np.any(r_array <= 0):
        raise ComartError("Radial distance must be positive.")

    values = t0 * (a * r0 / r_array + b)
    return float(values[0]) if np.isscalar(r) else values
