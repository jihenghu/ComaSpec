"""PYDISORT-based radiative-transfer solver for non-LTE coma layers."""

from __future__ import annotations

import numpy as np
import torch

import Constants as constants
from ComaGrid import ComaGrid
from NonLTE_optics import nonlte_excitation_temperature, nonlte_tau
from RTsolver import layer_thicknesses

try:
    import pydisort
except ImportError as exc:  # pragma: no cover - import guard for optional dependency
    raise ImportError(
        "disort_solver.py requires the optional 'pydisort' package."
    ) from exc


DEFAULT_DISORT_FLAGS = "lamber,planck,usrang,usrtau"
DEFAULT_BACKGROUND_TEMPERATURE_K = 2.73
DEFAULT_SURFACE_ALBEDO = 0# 1.0
DEFAULT_TOP_EMISSIVITY = 0.0
DEFAULT_TOP_ISOTROPIC_ILLUMINATION = 0.0


def gaussian_quadrature_streams(nstreams: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss-Legendre stream cosines and weights on ``[-1, 1]``."""

    if nstreams < 2:
        raise ValueError("nstreams must be at least 2")

    mu, weights = np.polynomial.legendre.leggauss(nstreams)
    return (
        np.asarray(mu, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
    )


def _layer_boundary_temperatures(layer_temperatures_K: np.ndarray) -> np.ndarray:
    """Interpolate layer-centered temperatures onto layer boundaries."""

    temperatures = np.asarray(layer_temperatures_K, dtype=np.float64)
    if temperatures.ndim != 1:
        raise ValueError("layer_temperatures_K must be a 1D array")
    if temperatures.size == 0:
        raise ValueError("layer_temperatures_K must not be empty")

    boundaries = np.empty(temperatures.size + 1, dtype=np.float64)
    boundaries[0] = temperatures[0]
    boundaries[-1] = temperatures[-1]
    if temperatures.size > 1:
        boundaries[1:-1] = 0.5 * (temperatures[:-1] + temperatures[1:])
    return boundaries


def _cumulative_user_tau(layer_tau: np.ndarray) -> np.ndarray:
    """Build monotonically increasing user optical-depth locations."""

    tau = np.asarray(layer_tau, dtype=np.float64)
    if tau.ndim != 1:
        raise ValueError("layer_tau must be a 1D array")
    if np.any(tau < 0.0):
        raise ValueError(
            "DISORT requires non-negative layer optical depths; "
            "the current tau profile contains negative values."
        )

    cumulative_tau = np.empty(tau.size + 1, dtype=np.float64)
    cumulative_tau[0] = 0.0
    cumulative_tau[1:] = np.cumsum(tau)
    return cumulative_tau


def _build_disort_options(
    *,
    nlayer: int,
    user_tau: np.ndarray,
    user_mu: np.ndarray,
    nu_Hz: float,
    flags: str = DEFAULT_DISORT_FLAGS,
    nwave: int = 1,
    ncol: int = 1,
) -> pydisort.DisortOptions:
    """Create a PYDISORT options object for a 1-band, 1-column run."""

    nu_Hz = float(nu_Hz)
    if nu_Hz <= 0.0:
        raise ValueError("nu_Hz must be positive")

    wavenumber_cm_1 = nu_Hz / (constants.SPEED_OF_LIGHT * 100.0)

    options = pydisort.DisortOptions().flags(flags).nwave(nwave).ncol(ncol)
    options.wave_lower(np.array([wavenumber_cm_1], dtype=np.float64))
    options.wave_upper(np.array([wavenumber_cm_1], dtype=np.float64))

    state = options.ds()
    state.nlyr = int(nlayer)
    state.nstr = int(user_mu.size)
    state.nmom = int(user_mu.size)
    state.nphase = 1

    options.user_tau(np.asarray(user_tau, dtype=np.float64))
    options.user_mu(np.asarray(user_mu, dtype=np.float64))
    options.user_phi(np.array([0.0], dtype=np.float64))
    return options


def _build_optical_properties(
    tau_profile: np.ndarray,
    *,
    nmom: int,
) -> torch.Tensor:
    """Build the DISORT optical-properties tensor ``(nwave, ncol, nlyr, nprop)``."""

    tau = np.asarray(tau_profile, dtype=np.float64)
    if tau.ndim != 1:
        raise ValueError("tau_profile must be a 1D array")

    nlayer = tau.size
    nprop = 2 + nmom

    properties = torch.zeros((1, 1, nlayer, nprop), dtype=torch.float64)
    properties[0, 0, :, 0] = torch.from_numpy(tau)
    properties[0, 0, :, 1] = 0.0
    properties[0, 0, :, 2:] = pydisort.scattering_moments(nmom, "isotropic")
    return properties


def solve_disort_layers(
    coma: ComaGrid,
    *,
    lower_level: int = 0,
    upper_level: int = 1,
    nu_Hz: float = 556.936e9,
    A_ul_s_1: float = 3.456e-3,
    g_u: float = 9.0,
    g_l: float = 9.0,
    absorber_mass_kg: float = 18.01528 * constants.ATOMIC_MASS_UNIT,
    nstreams: int = 16,
    flags: str = DEFAULT_DISORT_FLAGS,
    surface_albedo: float = DEFAULT_SURFACE_ALBEDO,
    background_temperature_K: float = DEFAULT_BACKGROUND_TEMPERATURE_K,
    temis: float = DEFAULT_TOP_EMISSIVITY,
    fisot: float = DEFAULT_TOP_ISOTROPIC_ILLUMINATION,
    apply_spherical_dilution: bool = False,
    dilution_reference_radius_m: float | None = None,
) -> dict[str, np.ndarray | float | str]:
    """Solve the non-LTE transfer problem with PYDISORT.

    The run uses 16 Gauss-Legendre user streams by default and returns
    layer-centered stream intensities together with a Gauss-quadrature mean
    intensity ``J_v``.
    """

    if surface_albedo < 0.0 or surface_albedo > 1.0:
        raise ValueError("surface_albedo must lie in [0, 1]")
    if background_temperature_K < 0.0:
        raise ValueError("background_temperature_K must be non-negative")
    if temis < 0.0:
        raise ValueError("temis must be non-negative")
    if fisot < 0.0:
        raise ValueError("fisot must be non-negative")
    if dilution_reference_radius_m is not None and dilution_reference_radius_m <= 0.0:
        raise ValueError("dilution_reference_radius_m must be positive")

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
    tex_profile = nonlte_excitation_temperature(
        coma,
        lower_level=lower_level,
        upper_level=upper_level,
        nu_Hz=nu_Hz,
        g_u=g_u,
        g_l=g_l,
    )
    if np.any(~np.isfinite(tex_profile)):
        raise ValueError("Excitation temperature profile contains non-finite values")
    if np.any(tex_profile < 0.0):
        raise ValueError(
            "DISORT Planck source requires non-negative excitation temperatures"
        )

    mu, weights = gaussian_quadrature_streams(nstreams=nstreams)
    user_tau = _cumulative_user_tau(tau_profile)
    options = _build_disort_options(
        nlayer=coma.ngrid,
        user_tau=user_tau,
        user_mu=mu,
        nu_Hz=nu_Hz,
        flags=flags,
    )

    solver = pydisort.Disort(options)
    properties = _build_optical_properties(tau_profile, nmom=mu.size)
    temf = torch.from_numpy(_layer_boundary_temperatures(tex_profile)).reshape(
        1, coma.ngrid + 1
    )

    print("....temf...", temf)
    solver.forward(
        properties,
        temf=temf,
        umu0=torch.tensor([1.0], dtype=torch.float64),
        phi0=torch.tensor([0.0], dtype=torch.float64),
        fbeam=torch.tensor([[0.0]], dtype=torch.float64),
        albedo=torch.tensor([[surface_albedo]], dtype=torch.float64),
        fluor=torch.tensor([[0.0]], dtype=torch.float64),
        fisot=torch.tensor([[fisot]], dtype=torch.float64),
        temis=torch.tensor([[temis]], dtype=torch.float64),
        ttemp=torch.tensor([background_temperature_K], dtype=torch.float64),
        btemp=torch.tensor([background_temperature_K], dtype=torch.float64),
    )

    intensity_levels = np.asarray(solver.gather_rad().detach().cpu().numpy(), dtype=np.float64)
    intensity_levels = intensity_levels[0, 0, 0, :, :]
    intensity_layers = 0.5 * (intensity_levels[:-1, :] + intensity_levels[1:, :])

    # j_v = 0.5 * np.sum(intensity_layers * weights[np.newaxis, :], axis=1)
    stream_average = np.mean(intensity_layers, axis=1)

    j_nu_hz_undiluted = stream_average / (constants.SPEED_OF_LIGHT * 100.0)
    spherical_dilution = np.ones_like(j_nu_hz_undiluted)
    if apply_spherical_dilution:
        r_m = np.asarray(coma.xc, dtype=np.float64)
        if r_m.shape != (coma.ngrid,):
            raise ValueError(
                "coma.xc must have shape (coma.ngrid,) to apply spherical dilution"
            )
        if np.any(r_m <= 0.0):
            raise ValueError("coma.xc must be positive to apply spherical dilution")

        if dilution_reference_radius_m is None:
            dilution_reference_radius_m = float(np.min(r_m))

        spherical_dilution = (float(dilution_reference_radius_m) / r_m) ** 2
        spherical_dilution = np.minimum(spherical_dilution, 1.0)

    j_nu_hz = j_nu_hz_undiluted * spherical_dilution

    # rad = solver.gather_rad() #/ c_cm_s
    # weights = np.ones_like(mu) / len(mu)
    # stream_average = np.sum(rad.numpy() * weights, axis=-1) / 4 / np.pi

    return {
        "flags": flags,
        "nstreams": int(mu.size),
        "stream_mu": np.array(mu, copy=True),
        "stream_weights": np.array(weights, copy=True),
        "path_length_m": np.array(ds_m, copy=True),
        "tau_profile": np.array(tau_profile, copy=True),
        "user_tau": np.array(user_tau, copy=True),
        "excitation_temperature_K": np.array(tex_profile, copy=True),
        # "boundary_temperature_K": np.array(temf[0].detach().cpu().numpy(), copy=True),
        # "surface_albedo": float(surface_albedo),
        # "background_temperature_K": float(background_temperature_K),
        # "temis": float(temis),
        # "fisot": float(fisot),
        # "intensity_levels": np.array(intensity_levels, copy=True),
        # "intensity_streams": np.array(intensity_layers, copy=True),
        # "J_v": np.array(j_v, copy=True),
        # "J_v_stream_average": np.array(stream_average, copy=True),
        "J_nu_undiluted": np.array(j_nu_hz_undiluted, copy=True),
        "spherical_dilution": np.array(spherical_dilution, copy=True),
        "J_v_stream_average": np.array(j_nu_hz, copy=True),
    }
