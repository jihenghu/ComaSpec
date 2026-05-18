"""Non-LTE line optical properties for coma-grid radiative transfer."""

from __future__ import annotations

import numpy as np

import Constants as constants
from ComaGrid import ComaGrid
from Einstein_coeff import einstein_coeffs_B


def thermal_broadening_gaussian(
    nu_eval_Hz: float | np.ndarray,
    nu0_Hz: float,
    temperature: np.ndarray,
    absorber_mass_kg: float = 18.01528 * constants.ATOMIC_MASS_UNIT,
) -> np.ndarray:
    """Return the thermal Doppler Gaussian line profile in ``Hz^-1``."""

    nu0_Hz = float(nu0_Hz)
    absorber_mass_kg = float(absorber_mass_kg)
    nu_eval_array = np.asarray(nu_eval_Hz, dtype=np.float64)
    temperature_array = np.asarray(temperature, dtype=np.float64)

    if nu0_Hz <= 0.0:
        raise ValueError("nu0_Hz must be positive")
    if np.any(nu_eval_array <= 0.0):
        raise ValueError("nu_eval_Hz must be positive")
    if np.any(temperature_array < 0.0):
        raise ValueError("temperature must be non-negative")
    if absorber_mass_kg <= 0.0:
        raise ValueError("absorber_mass_kg must be positive")

    nu_eval_array, temperature_array = np.broadcast_arrays(
        nu_eval_array, temperature_array
    )
    u_th = np.sqrt(
        2.0 * constants.BOLTZMANN_CONSTANT * temperature_array / absorber_mass_kg
    )
    delta_nu_D = nu0_Hz * u_th / constants.SPEED_OF_LIGHT
    delta_nu_D = np.maximum(delta_nu_D, np.finfo(np.float64).tiny)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        x = (nu_eval_array - nu0_Hz) / delta_nu_D
        profile = np.exp(-(x**2)) / (np.sqrt(np.pi) * delta_nu_D)

    return np.array(profile, dtype=np.float64, copy=False)



def _resolve_phi_nu(
    coma: ComaGrid,
    *,
    lower_level: int,
    upper_level: int,
    nu_Hz: float,
    nu0_Hz: float,
    absorber_mass_kg: float = 18.01528 * constants.ATOMIC_MASS_UNIT,
) -> np.ndarray:
    """Return the per-layer thermal Doppler line profile."""

    if not 0 <= lower_level < coma.nlevels:
        raise ValueError("lower_level is out of bounds")
    if not 0 <= upper_level < coma.nlevels:
        raise ValueError("upper_level is out of bounds")
    if lower_level == upper_level:
        raise ValueError("lower_level and upper_level must be different")
    if nu_Hz <= 0.0:
        raise ValueError("nu_Hz must be positive")
    if nu0_Hz <= 0.0:
        raise ValueError("nu0_Hz must be positive")
    if absorber_mass_kg <= 0.0:
        raise ValueError("absorber_mass_kg must be positive")

    n_lower = np.asarray(coma.density[:, lower_level], dtype=np.float64)
    n_upper = np.asarray(coma.density[:, upper_level], dtype=np.float64)
    if n_lower.shape != n_upper.shape:
        raise ValueError("Lower and upper level populations must have the same shape")

    if coma.temperature.shape != n_lower.shape:
        raise ValueError("coma.temperature must match the population shape")

    phi_nu = thermal_broadening_gaussian(
        nu_eval_Hz=nu_Hz,
        nu0_Hz=nu0_Hz,
        temperature=np.asarray(coma.temperature, dtype=np.float64),
        absorber_mass_kg=absorber_mass_kg,
    )
    return np.array(phi_nu, dtype=np.float64, copy=False)



def nonlte_tau(
    coma: ComaGrid,
    *,
    path_length_m: float | np.ndarray,
    lower_level: int = 0,
    upper_level: int = 1,
    nu_Hz: float = 556.936e9,
    A_ul_s_1: float = 3.456e-3,
    g_u: float = 9.0,
    g_l: float = 9.0,
    nu0_Hz: float=556.936e9,
    absorber_mass_kg: float = 18.01528 * constants.ATOMIC_MASS_UNIT,
) -> np.ndarray:
    """Return the per-layer optical depth from non-LTE level populations."""

    if coma.nlevels < 2:
        raise ValueError("coma must contain at least two levels")

    n_lower = np.asarray(coma.density[:, lower_level], dtype=np.float64)
    n_upper = np.asarray(coma.density[:, upper_level], dtype=np.float64)
    if np.any(n_lower < 0.0):
        raise ValueError("coma.density[:, lower_level] must be non-negative")
    if np.any(n_upper < 0.0):
        raise ValueError("coma.density[:, upper_level] must be non-negative")

    ds_m = np.asarray(path_length_m, dtype=np.float64)
    if ds_m.ndim == 0:
        ds_m = np.full(coma.ngrid, float(ds_m), dtype=np.float64)
    elif ds_m.shape != (coma.ngrid,):
        raise ValueError(f"path_length_m must be scalar or have shape ({coma.ngrid},)")
    if np.any(ds_m < 0.0):
        raise ValueError("path_length_m must be non-negative")

    phi_nu = _resolve_phi_nu(
        coma,
        lower_level=lower_level,
        upper_level=upper_level,
        nu_Hz=nu_Hz,
        nu0_Hz=nu0_Hz,
        absorber_mass_kg=absorber_mass_kg,
    )

    if A_ul_s_1 < 0.0:
        raise ValueError("A_ul_s_1 must be non-negative")
    if g_l <= 0.0 or g_u <= 0.0:
        raise ValueError("g_l and g_u must be positive")
    if nu_Hz <= 0.0:
        raise ValueError("nu_Hz must be positive")
    if np.any(phi_nu < 0.0):
        raise ValueError("phi_nu must be non-negative")

    return (
        constants.SPEED_OF_LIGHT**2
        / (8.0 * np.pi * nu_Hz**2)
        * A_ul_s_1
        * phi_nu
        * ((g_u / g_l) * (n_lower * ds_m) - n_upper * ds_m)
    )


def nonlte_excitation_temperature(
    coma: ComaGrid,
    *,
    lower_level: int = 0,
    upper_level: int = 1,
    nu_Hz: float = 556.936e9,
    g_u: float = 9.0,
    g_l: float = 9.0,
) -> np.ndarray:
    """Return the per-layer excitation temperature from the non-LTE populations."""

    if coma.nlevels < 2:
        raise ValueError("coma must contain at least two levels")
    if not 0 <= lower_level < coma.nlevels:
        raise ValueError("lower_level is out of bounds")
    if not 0 <= upper_level < coma.nlevels:
        raise ValueError("upper_level is out of bounds")
    if lower_level == upper_level:
        raise ValueError("lower_level and upper_level must be different")
    if nu_Hz <= 0.0:
        raise ValueError("nu_Hz must be positive")
    if g_l <= 0.0 or g_u <= 0.0:
        raise ValueError("g_l and g_u must be positive")

    n_lower = np.asarray(coma.density[:, lower_level], dtype=np.float64)
    n_upper = np.asarray(coma.density[:, upper_level], dtype=np.float64)
    if np.any(n_lower < 0.0) or np.any(n_upper < 0.0):
        raise ValueError("Level populations must be non-negative")

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (g_u * n_lower) / (g_l * n_upper)
        tex = (
            constants.PLANCK_CONSTANT
            * nu_Hz
            / constants.BOLTZMANN_CONSTANT
            / np.log(ratio)
        )

    return np.asarray(tex, dtype=np.float64)


def nonlte_source_function(
    coma: ComaGrid,
    *,
    lower_level: int = 0,
    upper_level: int = 1,
    nu_Hz: float = 556.936e9,
    g_u: float = 9.0,
    g_l: float = 9.0,
) -> np.ndarray:
    """Return the per-layer source function from the non-LTE populations."""

    if coma.nlevels < 2:
        raise ValueError("coma must contain at least two levels")
    if not 0 <= lower_level < coma.nlevels:
        raise ValueError("lower_level is out of bounds")
    if not 0 <= upper_level < coma.nlevels:
        raise ValueError("upper_level is out of bounds")
    if lower_level == upper_level:
        raise ValueError("lower_level and upper_level must be different")
    if nu_Hz <= 0.0:
        raise ValueError("nu_Hz must be positive")
    if g_l <= 0.0 or g_u <= 0.0:
        raise ValueError("g_l and g_u must be positive")

    n_lower = np.asarray(coma.density[:, lower_level], dtype=np.float64)
    n_upper = np.asarray(coma.density[:, upper_level], dtype=np.float64)
    if np.any(n_lower < 0.0) or np.any(n_upper < 0.0):
        raise ValueError("Level populations must be non-negative")

    prefactor = 2.0 * constants.PLANCK_CONSTANT * nu_Hz**3 / constants.SPEED_OF_LIGHT**2
    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = (g_u * n_lower) / (g_l * n_upper) - 1.0
        s_nu = prefactor / denominator

    return np.asarray(s_nu, dtype=np.float64)


def nonlte_optical_coefficients(
    coma: ComaGrid,
    *,
    lower_level: int = 0,
    upper_level: int = 1,
    nu_Hz: float = 556.936e9,
    A_ul_s_1: float = 3.456e-3,
    B_ul_SI: float | None = None,
    B_lu_SI: float | None = None,
    g_u: float = 9.0,
    g_l: float = 9.0,
    absorber_mass_kg: float = 18.01528 * constants.ATOMIC_MASS_UNIT,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the line absorption and emission coefficients at frequency ``nu``."""

    if coma.nlevels < 2:
        raise ValueError("coma must contain at least two levels")
    if not 0 <= lower_level < coma.nlevels:
        raise ValueError("lower_level is out of bounds")
    if not 0 <= upper_level < coma.nlevels:
        raise ValueError("upper_level is out of bounds")
    if lower_level == upper_level:
        raise ValueError("lower_level and upper_level must be different")

    n_lower = np.asarray(coma.density[:, lower_level], dtype=np.float64)
    n_upper = np.asarray(coma.density[:, upper_level], dtype=np.float64)

    if np.any(n_lower < 0.0):
        raise ValueError("coma.density[:, lower_level] must be non-negative")
    if np.any(n_upper < 0.0):
        raise ValueError("coma.density[:, upper_level] must be non-negative")
    if nu_Hz <= 0.0:
        raise ValueError("nu_Hz must be positive")
    if A_ul_s_1 < 0.0:
        raise ValueError("A_ul_s_1 must be non-negative")

    phi_nu = _resolve_phi_nu(
        coma,
        lower_level=lower_level,
        upper_level=upper_level,
        nu_Hz=nu_Hz,
        absorber_mass_kg=absorber_mass_kg,
    )

    if B_ul_SI is None or B_lu_SI is None:
        B_ul_SI, B_lu_SI = einstein_coeffs_B(
            nu_Hz=nu_Hz,
            A_ul_s_1=A_ul_s_1,
            g_u=g_u,
            g_l=g_l,
        )
    else:
        B_ul_SI = float(B_ul_SI)
        B_lu_SI = float(B_lu_SI)

    if B_ul_SI < 0.0:
        raise ValueError("B_ul_SI must be non-negative")
    if B_lu_SI < 0.0:
        raise ValueError("B_lu_SI must be non-negative")

    prefactor = constants.PLANCK_CONSTANT * nu_Hz / (4.0 * np.pi)
    alpha_nu_m_1 = prefactor * (n_lower * B_lu_SI - n_upper * B_ul_SI) * phi_nu
    j_nu_SI = prefactor * n_upper * A_ul_s_1 * phi_nu

    return alpha_nu_m_1, j_nu_SI


# calculate tau and source function for a single transition
