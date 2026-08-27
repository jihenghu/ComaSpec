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
    """Return a thermal Gaussian line profile in ``Hz^-1``.

    This is written in the same convention as pythonradex: a Gaussian profile is
    normalised as ``1 / (sigma_nu * sqrt(2*pi))`` after converting the velocity
    FWHM to a frequency FWHM.  For thermal broadening the equivalent velocity
    FWHM is ``sqrt(8 ln(2) kT / m)``.
    """

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
    width_v_fwhm = thermal_fwhm_velocity(
        temperature_array,
        absorber_mass_kg=absorber_mass_kg,
    )
    return gaussian_line_profile_phi_nu(
        nu_eval_Hz=nu_eval_array,
        nu0_Hz=nu0_Hz,
        width_v_fwhm_m_s=width_v_fwhm,
    )


def thermal_fwhm_velocity(
    temperature: float | np.ndarray,
    absorber_mass_kg: float = 18.01528 * constants.ATOMIC_MASS_UNIT,
) -> np.ndarray:
    """Return the thermal Doppler FWHM velocity in ``m s^-1``."""

    temperature_array = np.asarray(temperature, dtype=np.float64)
    absorber_mass_kg = float(absorber_mass_kg)
    if np.any(temperature_array < 0.0):
        raise ValueError("temperature must be non-negative")
    if absorber_mass_kg <= 0.0:
        raise ValueError("absorber_mass_kg must be positive")

    width_v = np.sqrt(
        8.0
        * np.log(2.0)
        * constants.BOLTZMANN_CONSTANT
        * temperature_array
        / absorber_mass_kg
    )
    return np.asarray(width_v, dtype=np.float64)


def _velocity_fwhm_to_frequency_fwhm(
    width_v_fwhm_m_s: float | np.ndarray,
    nu0_Hz: float,
) -> np.ndarray:
    """Convert velocity FWHM to frequency FWHM following pythonradex."""

    width_v = np.asarray(width_v_fwhm_m_s, dtype=np.float64)
    if np.any(width_v < 0.0):
        raise ValueError("width_v_fwhm_m_s must be non-negative")
    return nu0_Hz * width_v / constants.SPEED_OF_LIGHT


def gaussian_line_profile_phi_nu(
    nu_eval_Hz: float | np.ndarray,
    nu0_Hz: float,
    width_v_fwhm_m_s: float | np.ndarray,
) -> np.ndarray:
    """Return a pythonradex-style normalised Gaussian ``phi_nu`` in ``Hz^-1``."""

    nu0_Hz = float(nu0_Hz)
    nu_eval_array = np.asarray(nu_eval_Hz, dtype=np.float64)

    if nu0_Hz <= 0.0:
        raise ValueError("nu0_Hz must be positive")
    if np.any(nu_eval_array <= 0.0):
        raise ValueError("nu_eval_Hz must be positive")

    width_nu_fwhm = _velocity_fwhm_to_frequency_fwhm(width_v_fwhm_m_s, nu0_Hz)
    width_nu_fwhm, nu_eval_array = np.broadcast_arrays(
        width_nu_fwhm,
        nu_eval_array,
    )
    sigma_nu = width_nu_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_nu = np.maximum(sigma_nu, np.finfo(np.float64).tiny)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        profile = (
            np.exp(-((nu_eval_array - nu0_Hz) ** 2) / (2.0 * sigma_nu**2))
            / (sigma_nu * np.sqrt(2.0 * np.pi))
        )

    return np.array(profile, dtype=np.float64, copy=False)


def rectangular_line_profile_phi_nu(
    nu_eval_Hz: float | np.ndarray,
    nu0_Hz: float,
    width_v_fwhm_m_s: float | np.ndarray,
) -> np.ndarray:
    """Return a pythonradex-style rectangular ``phi_nu`` in ``Hz^-1``."""

    nu0_Hz = float(nu0_Hz)
    nu_eval_array = np.asarray(nu_eval_Hz, dtype=np.float64)
    if nu0_Hz <= 0.0:
        raise ValueError("nu0_Hz must be positive")
    if np.any(nu_eval_array <= 0.0):
        raise ValueError("nu_eval_Hz must be positive")

    width_nu = _velocity_fwhm_to_frequency_fwhm(width_v_fwhm_m_s, nu0_Hz)
    width_nu, nu_eval_array = np.broadcast_arrays(width_nu, nu_eval_array)
    width_nu = np.maximum(width_nu, np.finfo(np.float64).tiny)
    inside_line = np.abs(nu_eval_array - nu0_Hz) <= 0.5 * width_nu
    return np.where(inside_line, 1.0 / width_nu, 0.0).astype(np.float64, copy=False)



def _resolve_phi_nu(
    coma: ComaGrid,
    *,
    lower_level: int,
    upper_level: int,
    nu_Hz: float,
    nu0_Hz: float,
    absorber_mass_kg: float = 18.01528 * constants.ATOMIC_MASS_UNIT,
    line_profile_type: str = "thermal Gaussian",
    width_v_fwhm_m_s: float | np.ndarray | None = None,
) -> np.ndarray:
    """Return the per-layer line profile in the pythonradex convention."""

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

    normalized_profile = line_profile_type.lower().replace("_", " ").replace("-", " ")
    if width_v_fwhm_m_s is None:
        width_v_fwhm_m_s = thermal_fwhm_velocity(
            np.asarray(coma.temperature, dtype=np.float64),
            absorber_mass_kg=absorber_mass_kg,
        )

    if normalized_profile in {"thermal gaussian", "gaussian"}:
        phi_nu = gaussian_line_profile_phi_nu(
            nu_eval_Hz=nu_Hz,
            nu0_Hz=nu0_Hz,
            width_v_fwhm_m_s=width_v_fwhm_m_s,
        )
    elif normalized_profile == "rectangular":
        phi_nu = rectangular_line_profile_phi_nu(
            nu_eval_Hz=nu_Hz,
            nu0_Hz=nu0_Hz,
            width_v_fwhm_m_s=width_v_fwhm_m_s,
        )
    else:
        raise ValueError(
            "line_profile_type must be 'thermal Gaussian', 'Gaussian', or "
            "'rectangular'"
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
    line_profile_type: str = "thermal Gaussian",
    width_v_fwhm_m_s: float | np.ndarray | None = None,
) -> np.ndarray:
    """Return the per-layer optical depth from non-LTE level populations.

    The tau expression follows pythonradex's ``atomic_transition.tau``.  The
    level densities are first converted to layer columns by multiplying by
    ``path_length_m``.
    """

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
        line_profile_type=line_profile_type,
        width_v_fwhm_m_s=width_v_fwhm_m_s,
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
    nu0_Hz: float=556.936e9,
    absorber_mass_kg: float = 18.01528 * constants.ATOMIC_MASS_UNIT,
    line_profile_type: str = "thermal Gaussian",
    width_v_fwhm_m_s: float | np.ndarray | None = None,
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
        nu0_Hz=nu0_Hz,
        absorber_mass_kg=absorber_mass_kg,
        line_profile_type=line_profile_type,
        width_v_fwhm_m_s=width_v_fwhm_m_s,
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
