"""Radiative-transfer solver for layer-by-layer non-LTE coma calculations."""

from __future__ import annotations

import numpy as np

import Constants as constants
from ComaGrid import ComaGrid
from NonLTE_optics import nonlte_excitation_temperature, nonlte_tau


def layer_thicknesses(coma: ComaGrid) -> np.ndarray:
    """Return positive path lengths for each layer from ``coma.xf``."""

    xf = np.asarray(coma.xf, dtype=np.float64)
    if xf.shape != (coma.ngrid + 1,):
        raise ValueError(
            "coma.xf must have shape (coma.ngrid + 1,) to derive layer thicknesses"
        )

    ds_m = np.abs(np.diff(xf))
    if np.any(ds_m <= 0.0):
        raise ValueError("All coma layer thicknesses must be positive")
    return ds_m


def planck_intensity(nu_Hz: float, temperature_K: float | np.ndarray) -> np.ndarray:
    """Return the Planck specific intensity for a temperature profile."""

    nu_Hz = float(nu_Hz)
    temperature = np.asarray(temperature_K, dtype=np.float64)

    if nu_Hz <= 0.0:
        raise ValueError("nu_Hz must be positive")
    if np.any(temperature < 0.0):
        raise ValueError("temperature_K must be non-negative")

    prefactor = 2.0 * constants.PLANCK_CONSTANT * nu_Hz**3 / constants.SPEED_OF_LIGHT**2
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        exponent = (
            constants.PLANCK_CONSTANT
            * nu_Hz
            / (constants.BOLTZMANN_CONSTANT * temperature)
        )
        intensity = prefactor / np.expm1(exponent)

    return np.asarray(intensity, dtype=np.float64)


def formal_solution(
    tau_profile: float | np.ndarray,
    source_profile: float | np.ndarray,
    *,
    background_intensity: float = 0.0,
    inward: bool = False,
) -> dict[str, np.ndarray | float]:
    """Propagate specific intensity through a stack of layers.

    The formal solution in each layer is

    ``I_out = I_in * exp(-tau) + S_nu * (1 - exp(-tau))``.
    """

    if background_intensity < 0.0:
        raise ValueError("background_intensity must be non-negative")

    tau = np.asarray(tau_profile, dtype=np.float64)
    source = np.asarray(source_profile, dtype=np.float64)

    if tau.ndim == 0:
        tau = tau[np.newaxis]
    if source.ndim == 0:
        source = np.full(tau.shape, float(source), dtype=np.float64)
    if tau.shape != source.shape:
        raise ValueError("tau_profile and source_profile must have the same shape")

    layer_indices = np.arange(tau.size - 1, -1, -1) if inward else np.arange(tau.size)
    intensity_in = np.empty(tau.size, dtype=np.float64)
    intensity_out = np.empty(tau.size, dtype=np.float64)

    intensity = float(background_intensity)
    for layer_index in layer_indices:
        intensity_in[layer_index] = intensity
        transmission = np.exp(-tau[layer_index])
        intensity = transmission * intensity + source[layer_index] * (1.0 - transmission)
        intensity_out[layer_index] = intensity

    return {
        "layer_indices_traversed": layer_indices,
        "tau_profile": tau,
        "source_profile": source,
        "intensity_in": intensity_in,
        "intensity_out": intensity_out,
        "I_nu_emergent": float(intensity),
        "cumulative_tau_traversed": np.cumsum(tau[layer_indices]),
    }


def solve_nlte_layers(
    coma: ComaGrid,
    *,
    background_intensity: float = 0.0,
    lower_level: int = 0,
    upper_level: int = 1,
    nu_Hz: float = 556.936e9,
    A_ul_s_1: float = 3.456e-3,
    g_u: float = 9.0,
    g_l: float = 9.0,
    absorber_mass_kg: float = 18.01528 * constants.ATOMIC_MASS_UNIT,
    inward: bool = False,
) -> dict[str, np.ndarray | float]:
    """Compute tau, excitation temperature, source function, and intensity."""


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
        absorber_mass_kg=absorber_mass_kg,
    )
    tex_profile = nonlte_excitation_temperature(
        coma,
        lower_level=lower_level,
        upper_level=upper_level,
        nu_Hz=nu_Hz,
        g_u=g_u,
        g_l=g_l,
    )
    source_profile = planck_intensity(nu_Hz, tex_profile)

    result = formal_solution(
        tau_profile,
        source_profile,
        background_intensity=background_intensity,
        inward=inward,
    )
    # result["tau"] = tau_profile
    # result["source"] = source_profile
    result["path_length_m"] = ds_m
    result["excitation_temperature_K"] = tex_profile
    return result
