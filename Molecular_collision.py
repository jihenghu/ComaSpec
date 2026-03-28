"""Utilities for molecule-molecule collisional rates."""

from __future__ import annotations

import numpy as np

from Constants import ATOMIC_MASS_UNIT, BOLTZMANN_CONSTANT


def _thermal_velocity(
    temperature_K: float | np.ndarray,
    *,
    mass1_amu: float = 18.01528,
    mass2_amu: float = 18.01528,
) -> float | np.ndarray:
    """Return the mean thermal relative velocity for a molecular pair.

    Parameters
    ----------
    temperature_K
        Gas temperature in K.
    mass1_amu
        Molecular mass of the first collision partner in amu.
    mass2_amu
        Molecular mass of the second collision partner in amu.
    """

    temperature_K = np.asarray(temperature_K, dtype=np.float64)

    if np.any(temperature_K < 0.0):
        raise ValueError("temperature_K must be non-negative")
    if mass1_amu <= 0.0 or mass2_amu <= 0.0:
        raise ValueError("mass1_amu and mass2_amu must be positive")

    reduced_mass_kg = (
        (mass1_amu * mass2_amu) / (mass1_amu + mass2_amu) * ATOMIC_MASS_UNIT
    )
    velocity_m_s = np.sqrt(
        8.0 * BOLTZMANN_CONSTANT * temperature_K / (np.pi * reduced_mass_kg)
    )

    if velocity_m_s.ndim == 0:
        return float(velocity_m_s)

    return velocity_m_s


def molecular_collision_rate(
    number_density_m_3: float | np.ndarray,
    temperature_K: float | np.ndarray,
    *,
    sigma_ul_m2: float | np.ndarray = 2.924e-18,
    mass1_amu: float = 18.01528,
    mass2_amu: float = 18.01528,
) -> float | np.ndarray:
    """Compute the molecular collisional rate for a transition.

    Parameters
    ----------
    sigma_ul_m2
        Collision cross section for the ``u -> l`` transition in m^2.
        The default value is ``2.924e-18 m^2`` for the ``110 -> 101`` de-excitation.
    number_density_m_3
        Number density of the colliding partner in m^-3.
    temperature_K
        Gas temperature in K used to compute the thermal relative velocity.
    mass1_amu
        Molecular mass of the first collision partner in amu.
    mass2_amu
        Molecular mass of the second collision partner in amu.

    Returns
    -------
    float | np.ndarray
        ``Cm_ul_s_1 = number_density_m_3 * sigma_ul_m2 *
        _thermal_velocity(temperature_K)``.
    """

    sigma_ul_m2 = np.asarray(sigma_ul_m2, dtype=np.float64)
    number_density_m_3 = np.asarray(number_density_m_3, dtype=np.float64)

    if np.any(sigma_ul_m2 < 0.0):
        raise ValueError("sigma_ul_m2 must be non-negative")
    if np.any(number_density_m_3 < 0.0):
        raise ValueError("number_density_m_3 must be non-negative")

    thermal_velocity_m_s = _thermal_velocity(
        temperature_K, mass1_amu=mass1_amu, mass2_amu=mass2_amu
    )
    k_ul_m3_s_1 = sigma_ul_m2 * thermal_velocity_m_s
    Cm_ul_s_1 = number_density_m_3 * k_ul_m3_s_1

    if np.ndim(Cm_ul_s_1) == 0:
        return float(Cm_ul_s_1)

    return Cm_ul_s_1
