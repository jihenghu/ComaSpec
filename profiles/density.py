"""Density models for a cometary coma."""

from __future__ import annotations

import numpy as np

from exceptions import ComartError


def _isotropic(q: float, r: np.ndarray, v: np.ndarray) -> np.ndarray:
    if q <= 0:
        raise ComartError("Production rate must be positive.")
    if np.any(r <= 0):
        raise ComartError("Radial distance must be positive.")
    if np.any(v <= 0):
        raise ComartError("Velocity must be positive.")
    return q / (4.0 * np.pi * r**2 * v)


def calc_density(
    q: float,
    r: np.ndarray,
    v: np.ndarray,
    *,
    r_h: float = 1.0,
    model_type: str = "isotropic",
    photo_dissociation: bool = True,
    beta: float = 1.042e-5,
) -> np.ndarray:
    """Return the neutral density profile for the requested outgassing model."""
    if model_type != "isotropic":
        raise ComartError("Only the isotropic density model is implemented in the clean rebuild.")

    density = _isotropic(q, np.asarray(r, dtype=np.float64), np.asarray(v, dtype=np.float64))
    if not photo_dissociation:
        return density
    return density * np.exp(-beta / (r_h**2) * np.asarray(r) / np.asarray(v))
