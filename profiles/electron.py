"""Electron density and temperature models."""

from __future__ import annotations

import numpy as np

from exceptions import ComartError


def electron_biver_1997(
    q: float,
    *,
    r_h: float,
    r: np.ndarray,
    v: np.ndarray,
    t_kin: np.ndarray,
    t_emax: float = 10000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return electron density and temperature following the legacy Biver-style fit."""
    if r_h <= 0:
        raise ComartError("`r_h` must be positive.")
    if q < 0:
        raise ComartError("`q` must be non-negative.")
    if np.any(r < 0):
        raise ComartError("Radial distance must be non-negative.")
    if np.any(v <= 0):
        raise ComartError("Velocity must be positive.")
    if np.any(t_kin < 0):
        raise ComartError("Kinetic temperature must be non-negative.")
    if t_emax <= 0:
        raise ComartError("`t_emax` must be positive.")

    q29 = 1e-29 * q
    r_cs = 1.125e6 * q29**0.75
    r_rec = 3.2e6 * q29**0.5

    t_e = np.zeros_like(r, dtype=np.float64)
    mask_inner = r < r_cs
    mask_mid = (r >= r_cs) & (r < 2 * r_cs)
    mask_outer = r >= 2 * r_cs

    t_e[mask_inner] = t_kin[mask_inner]
    t_e[mask_mid] = t_kin[mask_mid] + (t_emax - t_kin[mask_mid]) * (r[mask_mid] - r_cs) / r_cs
    t_e[mask_outer] = t_emax

    k_ion = 4.1e-7 * r_h**-2
    k_rec = 7.0e-13 * (300.0 / t_e) ** 0.5

    term1 = q * k_ion / (v * k_rec * r_h**2)
    term2 = t_e / 300.0
    term3 = r_rec / r**2
    term4 = 1.0 - np.exp(-r / r_rec)
    term5 = 5.0e6 * r_h**-2

    n_e = np.sqrt(term1) * term2**0.15 * term3 * term4 + term5
    return n_e, t_e
