"""Utilities for scaling solar pumping rates with heliocentric distance."""

from __future__ import annotations

import numpy as np


def scale_g_factors_from_1au(
    heliocentric_distance_AU: float | np.ndarray = 1.0,
    G_lu_1AU_s_1: float | np.ndarray = 1.654e-5,
    G_ul_1AU_s_1: float | np.ndarray = 1.423e-5,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Scale solar pumping rates from their 1 AU values.

    Parameters
    ----------
    heliocentric_distance_AU
        Heliocentric distance in AU.
    G_lu_1AU_s_1
        Pumping or radiative coupling rate for the ``l -> u`` transition at 1 AU.
        The default value is ``1.654e-5 s^-1`` for the ``101 -> 110`` pair.
    G_ul_1AU_s_1
        Pumping or radiative coupling rate for the ``u -> l`` transition at 1 AU.
        The default value is ``1.423e-5 s^-1`` for the ``101 -> 110`` pair.

    Returns
    -------
    tuple[float | np.ndarray, float | np.ndarray]
        ``(G_ul_s_1, G_lu_s_1)`` scaled as ``1 / r_h^2``.
    """

    G_ul_1AU_s_1 = np.asarray(G_ul_1AU_s_1, dtype=np.float64)
    G_lu_1AU_s_1 = np.asarray(G_lu_1AU_s_1, dtype=np.float64)
    heliocentric_distance_AU = np.asarray(
        heliocentric_distance_AU, dtype=np.float64
    )

    if np.any(heliocentric_distance_AU <= 0.0):
        raise ValueError("heliocentric_distance_AU must be positive")

    scale = 1.0 / heliocentric_distance_AU**2
    G_ul_s_1 = G_ul_1AU_s_1 * scale
    G_lu_s_1 = G_lu_1AU_s_1 * scale

    if G_ul_s_1.ndim == 0 and G_lu_s_1.ndim == 0:
        return float(G_ul_s_1), float(G_lu_s_1)

    return G_ul_s_1, G_lu_s_1
