"""Utilities for electron-molecule collisional rates."""

from __future__ import annotations

import numpy as np
from scipy.special import k0

from Constants import (
    BOLTZMANN_CONSTANT as BOLTZMANN_C,
    ELEMENTARY_CHARGE,
    ELECTRON_MASS,
    PLANCK_CONSTANT as PLANCK_C,
    SPEED_OF_LIGHT as LIGHT_SPEED,
    VACUUM_PERMITTIVITY as EPSILON_0,
)


def _electron_cross_section(A_ul: float, nu_Hz: float) -> float:
    if A_ul <= 0 or nu_Hz <= 0:
        raise ValueError("A coefficient and frequency must be positive.")
    denom = 16.0 * np.pi**2 * PLANCK_C**2 * nu_Hz**4 * EPSILON_0
    return ELECTRON_MASS * ELEMENTARY_CHARGE**2 * LIGHT_SPEED**3 * A_ul / denom


def electron_collision_rate(
    n_e: float,
    t_e: float,
    nu_Hz: float = 556.936e9,
    A_ul_s_1: float = 3.456e-3,
    g_u: float = 9.0,
    g_l: float = 9.0,
) -> tuple[float, float]:
    """Return the electron de-excitation and excitation rates."""

    if n_e < 0 or t_e <= 0:
        raise ValueError(
            "Electron density must be non-negative and temperature positive."
        )

    # for de-excitation Ce_ul_s_1
    aul = PLANCK_C * nu_Hz / (2.0 * BOLTZMANN_C * t_e)
    v_th = np.sqrt(8.0 * BOLTZMANN_C * t_e / (np.pi * ELECTRON_MASS))
    sigma_ul = _electron_cross_section(A_ul_s_1, nu_Hz)
    Ce_ul_s_1 = float(n_e * v_th * sigma_ul * 2.0 * aul * np.exp(aul) * k0(aul))

    # for excitation Ce_lu_s_1
    Ce_lu_s_1 = float(
        n_e
        * v_th
        * g_u
        / g_l
        * sigma_ul
        * 2.0
        * aul
        * np.exp(-aul)
        * k0(aul)
    )

    return Ce_ul_s_1, Ce_lu_s_1
