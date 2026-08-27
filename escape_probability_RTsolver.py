"""Escape-probability radiative transfer for local uniform layers."""

from __future__ import annotations

import numpy as np

import Constants as constants
from ComaGrid import ComaGrid
from NonLTE_optics import nonlte_excitation_temperature, nonlte_source_function, nonlte_tau


MIN_RELIABLE_NEGATIVE_TAU = -1.0


def layer_thicknesses(coma: ComaGrid) -> np.ndarray:
    """Return positive radial layer thicknesses from a monotonic ``coma.xf``."""

    xf = np.asarray(coma.xf, dtype=np.float64)
    if xf.shape != (coma.ngrid + 1,):
        raise ValueError("coma.xf must have shape (coma.ngrid + 1,)")
    dxf = np.diff(xf)
    if not (np.all(dxf > 0.0) or np.all(dxf < 0.0)):
        raise ValueError("coma.xf must be strictly monotonic")
    return np.abs(dxf)


def _clip_probability(probability: np.ndarray) -> np.ndarray:
    """Clip escape probabilities into the physical interval."""

    return np.clip(probability, 0.0, 1.0)


def _as_finite_tau(tau: float | np.ndarray) -> np.ndarray:
    """Return finite optical depth as a float array."""

    tau_array = np.asarray(tau, dtype=np.float64)
    if np.any(~np.isfinite(tau_array)):
        raise ValueError("tau must be finite")
    return tau_array


def _stable_tau_for_large_negative(tau: np.ndarray) -> np.ndarray:
    """Stabilize strongly inverted optical depths like pythonradex does."""

    return np.where(tau < MIN_RELIABLE_NEGATIVE_TAU, np.abs(tau), tau)


def _one_minus_exp_minus_tau(tau: np.ndarray) -> np.ndarray:
    """Stable evaluation of ``1 - exp(-tau)``."""

    return np.where(np.abs(tau) < 1.0e-5, tau, -np.expm1(-tau))


def beta_static_sphere(tau_diameter: float | np.ndarray) -> np.ndarray:
    """Return the escape probability for a homogeneous static sphere.

    The input optical depth is measured along the diameter of the sphere.  The
    expression follows the Osterbrock uniform-sphere formula used by
    pythonradex, with a small-|tau| Taylor expansion for numerical stability.
    """

    tau = _stable_tau_for_large_negative(_as_finite_tau(tau_diameter))

    beta = np.empty_like(tau, dtype=np.float64)
    small = np.abs(tau) <= 0.05
    normal = ~small

    beta[small] = (
        1.0
        - 0.375 * tau[small]
        + 0.1 * tau[small] ** 2
        - 0.0208333 * tau[small] ** 3
    )

    tau_normal = tau[normal]
    with np.errstate(over="ignore", invalid="ignore"):
        beta[normal] = (
            1.5
            / tau_normal
            * (
                1.0
                - 2.0 / tau_normal**2
                + (2.0 / tau_normal + 2.0 / tau_normal**2)
                * np.exp(-tau_normal)
            )
        )

    return _clip_probability(beta)


def beta_LVG_sphere(tau: float | np.ndarray) -> np.ndarray:
    """Return the LVG sphere escape probability ``(1-exp(-tau))/tau``."""

    tau_array = _stable_tau_for_large_negative(_as_finite_tau(tau))
    beta = np.empty_like(tau_array, dtype=np.float64)
    small = np.abs(tau_array) <= 0.05
    normal = ~small

    tau_small = tau_array[small]
    beta[small] = (
        1.0
        - 0.5 * tau_small
        + tau_small**2 / 6.0
        - tau_small**3 / 24.0
    )
    beta[normal] = _one_minus_exp_minus_tau(tau_array[normal]) / tau_array[normal]
    return _clip_probability(beta)


def beta_LVG_slab(tau: float | np.ndarray) -> np.ndarray:
    """Return the LVG slab escape probability ``(1-exp(-3 tau))/(3 tau)``."""

    tau_array = _stable_tau_for_large_negative(_as_finite_tau(tau))
    beta = np.empty_like(tau_array, dtype=np.float64)
    small = np.abs(tau_array) <= 0.05
    normal = ~small

    tau_small = tau_array[small]
    beta[small] = (
        1.0
        - 1.5 * tau_small
        + 1.5 * tau_small**2
        - 1.125 * tau_small**3
    )
    beta[normal] = _one_minus_exp_minus_tau(3.0 * tau_array[normal]) / (
        3.0 * tau_array[normal]
    )
    return _clip_probability(beta)


def beta_static_slab(tau: float | np.ndarray, *, n_mu: int = 200) -> np.ndarray:
    """Return the static slab escape probability by angular quadrature."""

    if n_mu < 8:
        raise ValueError("n_mu must be at least 8")

    tau_array = _stable_tau_for_large_negative(_as_finite_tau(tau))
    flat_tau = tau_array.reshape(-1)
    mu, weights = np.polynomial.legendre.leggauss(n_mu)
    mu = 0.5 * (mu + 1.0)
    weights = 0.5 * weights

    beta = np.empty_like(flat_tau, dtype=np.float64)
    small = np.abs(flat_tau) <= 1.0e-5
    beta[small] = 1.0
    for index, tau_value in enumerate(flat_tau):
        if small[index]:
            continue
        integrand = (1.0 - np.exp(-tau_value / mu)) * mu
        beta[index] = np.sum(weights * integrand) / tau_value

    return _clip_probability(beta.reshape(tau_array.shape))


def beta_LVG_sphere_RADEX(tau: float | np.ndarray) -> np.ndarray:
    """Return the de Jong/RADEX LVG sphere approximation."""

    tau_array = _stable_tau_for_large_negative(_as_finite_tau(tau))
    beta = np.empty_like(tau_array, dtype=np.float64)
    tau_radius = tau_array / 2.0

    small = np.abs(tau_array) < 1.0e-3
    less7 = (~small) & (tau_array < 7.0)
    gtr7 = tau_array >= 7.0

    beta[small] = 1.0
    beta[less7] = (
        2.0
        * (1.0 - np.exp(-2.34 * tau_radius[less7]))
        / (4.68 * tau_radius[less7])
    )
    beta[gtr7] = 2.0 / (
        tau_radius[gtr7]
        * 4.0
        * np.sqrt(np.log(tau_radius[gtr7] / np.sqrt(np.pi)))
    )
    return _clip_probability(beta)


def escape_probability(
    tau: float | np.ndarray,
    *,
    geometry: str = "static sphere",
    static_slab_n_mu: int = 200,
) -> np.ndarray:
    """Return escape probability for a selected local geometry."""

    normalized_geometry = geometry.lower().replace("_", " ").replace("-", " ")
    if normalized_geometry in {"static sphere", "uniform sphere"}:
        return beta_static_sphere(tau)
    if normalized_geometry in {"lvg sphere"}:
        return beta_LVG_sphere(tau)
    if normalized_geometry in {"lvg slab"}:
        return beta_LVG_slab(tau)
    if normalized_geometry in {"static slab"}:
        return beta_static_slab(tau, n_mu=static_slab_n_mu)
    if normalized_geometry in {"lvg sphere radex", "radex lvg sphere"}:
        return beta_LVG_sphere_RADEX(tau)
    raise ValueError(
        "geometry must be one of: static sphere, static slab, "
        "LVG sphere, LVG slab, LVG sphere RADEX"
    )


def uniform_sphere_specific_intensity(
    tau_diameter: float | np.ndarray,
    source_function: float | np.ndarray,
) -> np.ndarray:
    """Return the unresolved mean emerging intensity of a uniform sphere."""

    tau, source = np.broadcast_arrays(
        np.asarray(tau_diameter, dtype=np.float64),
        np.asarray(source_function, dtype=np.float64),
    )
    if np.any(~np.isfinite(tau)) or np.any(~np.isfinite(source)):
        raise ValueError("tau_diameter and source_function must be finite")

    intensity = np.empty_like(tau, dtype=np.float64)
    small = np.abs(tau) <= 1.0e-2
    normal = ~small

    tau_normal = tau[normal]
    with np.errstate(over="ignore", invalid="ignore"):
        flux_surface = (
            2.0
            * np.pi
            * source[normal]
            / tau_normal**2
            * (
                tau_normal**2 / 2.0
                - 1.0
                + (tau_normal + 1.0) * np.exp(-tau_normal)
            )
        )
    intensity[normal] = flux_surface / np.pi

    tau_small = tau[small]
    flux_surface_taylor = (
        2.0
        * np.pi
        * source[small]
        * (
            tau_small / 3.0
            - tau_small**2 / 8.0
            + tau_small**3 / 30.0
            - tau_small**4 / 144.0
        )
    )
    intensity[small] = flux_surface_taylor / np.pi

    return np.asarray(intensity, dtype=np.float64)


def flux_1d_specific_intensity(
    tau: float | np.ndarray,
    source_function: float | np.ndarray,
) -> np.ndarray:
    """Return one-direction emergent intensity ``S * (1-exp(-tau))``."""

    tau_array, source = np.broadcast_arrays(
        _as_finite_tau(tau),
        np.asarray(source_function, dtype=np.float64),
    )
    if np.any(~np.isfinite(source)):
        raise ValueError("source_function must be finite")
    return source * _one_minus_exp_minus_tau(tau_array)


def solve_escape_probability_layers(
    coma: ComaGrid,
    *,
    lower_level: int = 0,
    upper_level: int = 1,
    nu_Hz: float = 556.936e9,
    A_ul_s_1: float = 3.456e-3,
    g_u: float = 9.0,
    g_l: float = 9.0,
    absorber_mass_kg: float = 18.01528 * constants.ATOMIC_MASS_UNIT,
    background_intensity: float | np.ndarray = 0.0,
    geometry: str = "static sphere",
    static_slab_n_mu: int = 200,
) -> dict[str, np.ndarray | float | str]:
    """Solve one escape-probability RT update for the current coma populations.

    Each layer is approximated as a local uniform escape-probability element.
    The local mean intensity used by the statistical-equilibrium update is

    ``J_nu = (1 - beta) * S_nu + beta * J_background``.
    """

    if np.any(np.asarray(background_intensity, dtype=np.float64) < 0.0):
        raise ValueError("background_intensity must be non-negative")

    ds_m = layer_thicknesses(coma)
    tau_profile = nonlte_tau(
        coma,
        path_length_m=ds_m,
        lower_level=lower_level,
        upper_level=upper_level,
        nu_Hz=nu_Hz,
        A_ul_s_1=A_ul_s_1,
        g_u=g_u,
        g_l=g_l,
        nu0_Hz=nu_Hz,
        absorber_mass_kg=absorber_mass_kg,
    )
    source_profile = nonlte_source_function(
        coma,
        lower_level=lower_level,
        upper_level=upper_level,
        nu_Hz=nu_Hz,
        g_u=g_u,
        g_l=g_l,
    )
    tex_profile = nonlte_excitation_temperature(
        coma,
        lower_level=lower_level,
        upper_level=upper_level,
        nu_Hz=nu_Hz,
        g_u=g_u,
        g_l=g_l,
    )

    background = np.asarray(background_intensity, dtype=np.float64)
    if background.ndim == 0:
        background = np.full(coma.ngrid, float(background), dtype=np.float64)
    elif background.shape != (coma.ngrid,):
        raise ValueError("background_intensity must be scalar or shape (coma.ngrid,)")

    beta = escape_probability(
        tau_profile,
        geometry=geometry,
        static_slab_n_mu=static_slab_n_mu,
    )
    j_nu = (1.0 - beta) * source_profile + beta * background
    normalized_geometry = geometry.lower().replace("_", " ").replace("-", " ")
    if normalized_geometry in {"static sphere", "uniform sphere"}:
        emergent_intensity = uniform_sphere_specific_intensity(
            tau_profile,
            source_profile,
        )
    else:
        emergent_intensity = flux_1d_specific_intensity(tau_profile, source_profile)

    return {
        "geometry": normalized_geometry,
        "path_length_m": np.array(ds_m, copy=True),
        "tau_profile": np.array(tau_profile, copy=True),
        "source_profile": np.array(source_profile, copy=True),
        "excitation_temperature_K": np.array(tex_profile, copy=True),
        "escape_probability": np.array(beta, copy=True),
        "background_intensity": np.array(background, copy=True),
        "J_nu": np.array(j_nu, copy=True),
        "J_v_stream_average": np.array(j_nu, copy=True),
        "emergent_intensity": np.array(emergent_intensity, copy=True),
    }
