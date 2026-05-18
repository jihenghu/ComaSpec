"""Utilities for molecule-molecule collisional rates."""

from __future__ import annotations

import numpy as np

from Constants import ATOMIC_MASS_UNIT, BOLTZMANN_CONSTANT, PLANCK_CONSTANT


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


# implement the sigma temperature dependence in the future if needed, e.g., sigma_ul_m2 = alpa * temperature_K^beta
def ww_sigma_ul_m2(
    temperature_K: float | np.ndarray,
    *,
    alpha: float = 2.924e-18, # sigma_ul_m2 at 100 K for the 110 -> 101 de-excitation  Buffa+, 2000
    beta: float = -0.6,
) -> float | np.ndarray:
    """Return the collision cross section for a transition in m^2.

    Parameters
    ----------
    temperature_K
        Gas temperature in K used to compute the thermal relative velocity.
    alpha
        Pre-factor for the collision cross section in m^2. The default value is
        ``2.924e-18 m^2`` for the ``110 -> 101`` de-excitation.
    beta
        Power-law index for the temperature dependence of the collision cross
        section. The default value is ``-0.2``, i.e., temperature dependence.
    """

    temperature_K = np.asarray(temperature_K, dtype=np.float64)

    if np.any(temperature_K < 0.0):
        raise ValueError("temperature_K must be non-negative")
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")

    sigma_ul_m2 = alpha * (temperature_K/100.0)**beta

    if sigma_ul_m2.ndim == 0:
        return float(sigma_ul_m2)

    return sigma_ul_m2


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


def detailed_balance_excitation_rate(
    Cm_ul_s_1: float | np.ndarray,
    temperature_K: float | np.ndarray,
    *,
    nu_Hz: float = 556.936e9,
    g_u: float = 9.0,
    g_l: float = 9.0,
) -> float | np.ndarray:
    """Compute ``l -> u`` collisional excitation from ``u -> l`` detailed balance."""

    Cm_ul_s_1 = np.asarray(Cm_ul_s_1, dtype=np.float64)
    temperature_K = np.asarray(temperature_K, dtype=np.float64)

    if np.any(Cm_ul_s_1 < 0.0):
        raise ValueError("Cm_ul_s_1 must be non-negative")
    if np.any(temperature_K <= 0.0):
        raise ValueError("temperature_K must be positive")
    if nu_Hz <= 0.0:
        raise ValueError("nu_Hz must be positive")
    if g_u <= 0.0 or g_l <= 0.0:
        raise ValueError("g_u and g_l must be positive")

    boltzmann_factor = np.exp(
        -(PLANCK_CONSTANT * float(nu_Hz)) / (BOLTZMANN_CONSTANT * temperature_K)
    )
    Cm_lu_s_1 = Cm_ul_s_1 * (float(g_u) / float(g_l)) * boltzmann_factor

    if Cm_lu_s_1.ndim == 0:
        return float(Cm_lu_s_1)

    return Cm_lu_s_1
