"""Velocity profile models."""

from __future__ import annotations

import numpy as np

from exceptions import ComartError


def velocity_tanh(r: np.ndarray | float, *, v0: float = 800.0, c: float = 1e5) -> np.ndarray | float:
    """Return a smooth radial outflow profile using a hyperbolic tangent law."""
    if v0 < 0:
        raise ComartError(f"`v0` must be non-negative, got {v0}.")
    if c <= 0:
        raise ComartError(f"`c` must be positive, got {c}.")

    r_array = np.atleast_1d(r).astype(np.float64)
    if np.any(r_array < 0):
        raise ComartError("Radial distance must be non-negative.")

    values = v0 * np.tanh((r_array) / c)
    return float(values[0]) if np.isscalar(r) else values