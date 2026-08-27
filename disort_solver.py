"""Spherical-characteristic PYDISORT solver for non-LTE coma line transfer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import Constants as constants
from ComaGrid import ComaGrid
from NonLTE_optics import nonlte_excitation_temperature

try:
    import pydisort
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "disort_solver.py requires the optional 'pydisort' package."
    ) from exc


DEFAULT_DISORT_FLAGS = "lamber,planck,usrang,quiet"
DEFAULT_BACKGROUND_TEMPERATURE_K = 2.725
DEFAULT_SURFACE_ALBEDO = 0.0
DEFAULT_TOP_EMISSIVITY = 0.0
DEFAULT_TOP_ISOTROPIC_ILLUMINATION = 0.0
DEFAULT_FREQUENCY_POINTS = 7
DEFAULT_SUBSEGMENTS_PER_SHELL = 4

_C_CM_S = constants.SPEED_OF_LIGHT * 100.0


@dataclass(frozen=True)
class _RaySegments:
    length_m: np.ndarray
    shell_index: np.ndarray
    projected_velocity_factor: np.ndarray
    enters_from_nucleus: bool


def gaussian_quadrature_streams(nstreams: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss-Legendre direction cosines and weights on [-1, 1]."""

    if nstreams < 2:
        raise ValueError("nstreams must be at least 2")
    mu, weights = np.polynomial.legendre.leggauss(nstreams)
    return np.asarray(mu, dtype=np.float64), np.asarray(weights, dtype=np.float64)


def _cumulative_user_tau(layer_tau: np.ndarray) -> np.ndarray:
    tau = np.asarray(layer_tau, dtype=np.float64)
    cumulative_tau = np.empty(tau.size + 1, dtype=np.float64)
    cumulative_tau[0] = 0.0
    cumulative_tau[1:] = np.cumsum(tau)
    return cumulative_tau


def _radial_grid(
    coma: ComaGrid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers = np.asarray(coma.xc, dtype=np.float64)
    faces = np.asarray(coma.xf, dtype=np.float64)
    order = np.argsort(centers)
    inverse_order = np.argsort(order)
    centers = centers[order]
    faces = np.sort(faces)
    return centers, faces, order, inverse_order


def _radial_path_lengths(
    coma: ComaGrid, path_length_m: float | np.ndarray | None
) -> np.ndarray:
    if path_length_m is not None:
        raise ValueError("path_length_m cannot override spherical chord lengths")
    return np.abs(np.diff(np.asarray(coma.xf, dtype=np.float64)))


def _doppler_width_hz(
    temperature_K: np.ndarray,
    nu_Hz: float,
    absorber_mass_kg: float,
    width_v_fwhm_m_s: float | np.ndarray | None,
) -> np.ndarray:
    if width_v_fwhm_m_s is None:
        velocity_width = np.sqrt(
            2.0 * constants.BOLTZMANN_CONSTANT * temperature_K / absorber_mass_kg
        )
    else:
        fwhm = np.asarray(width_v_fwhm_m_s, dtype=np.float64)
        fwhm = np.broadcast_to(fwhm, temperature_K.shape)
        velocity_width = fwhm / (2.0 * np.sqrt(np.log(2.0)))
    return nu_Hz * velocity_width / constants.SPEED_OF_LIGHT


def _ordered_ray_segments(
    faces_m: np.ndarray,
    target_radius_m: float,
    mu: float,
    subsegments_per_shell: int,
) -> _RaySegments:
    inner_m = float(faces_m[0])
    outer_m = float(faces_m[-1])
    impact_m = target_radius_m * np.sqrt(max(0.0, 1.0 - mu**2))
    target_z_m = target_radius_m * mu
    enters_from_nucleus = bool(mu >= 0.0 and impact_m < inner_m)
    if enters_from_nucleus:
        start_z_m = np.sqrt(max(0.0, inner_m**2 - impact_m**2))
    else:
        start_z_m = -np.sqrt(max(0.0, outer_m**2 - impact_m**2))

    crossings = [start_z_m, target_z_m]
    if start_z_m < 0.0 < target_z_m:
        crossings.append(0.0)
    for face_m in faces_m:
        if face_m <= impact_m:
            continue
        face_z_m = np.sqrt(max(0.0, face_m**2 - impact_m**2))
        for signed_z_m in (-face_z_m, face_z_m):
            if start_z_m < signed_z_m < target_z_m:
                crossings.append(signed_z_m)
    z_points_m = np.unique(np.asarray(sorted(crossings), dtype=np.float64))

    lengths = []
    shell_indices = []
    projection_factors = []
    for left_z_m, right_z_m in zip(z_points_m[:-1], z_points_m[1:]):
        sub_edges_m = np.linspace(
            left_z_m,
            right_z_m,
            subsegments_per_shell + 1,
            dtype=np.float64,
        )
        for sub_left_m, sub_right_m in zip(sub_edges_m[:-1], sub_edges_m[1:]):
            length_m = sub_right_m - sub_left_m
            if length_m <= 0.0:
                continue
            midpoint_z_m = 0.5 * (sub_left_m + sub_right_m)
            midpoint_radius_m = np.hypot(impact_m, midpoint_z_m)
            shell_index = np.searchsorted(faces_m, midpoint_radius_m, side="right") - 1
            lengths.append(length_m)
            shell_indices.append(int(np.clip(shell_index, 0, faces_m.size - 2)))
            projection_factors.append(midpoint_z_m / midpoint_radius_m)

    return _RaySegments(
        length_m=np.asarray(lengths, dtype=np.float64),
        shell_index=np.asarray(shell_indices, dtype=np.int64),
        projected_velocity_factor=np.asarray(projection_factors, dtype=np.float64),
        enters_from_nucleus=enters_from_nucleus,
    )


class _SphericalDisort:
    def __init__(
        self,
        *,
        radius_m: np.ndarray,
        faces_m: np.ndarray,
        velocity_m_s: np.ndarray,
        doppler_width_hz: np.ndarray,
        nu_Hz: float,
        A_ul_s_1: float,
        g_u: float,
        g_l: float,
        nstreams: int,
        nfrequency: int,
        subsegments_per_shell: int,
        flags: str,
        surface_albedo: float,
        background_temperature_K: float,
        nucleus_temperature_K: float,
    ) -> None:
        self.radius_m = radius_m
        self.faces_m = faces_m
        self.velocity_m_s = velocity_m_s
        self.doppler_width_hz = doppler_width_hz
        self.nu_Hz = nu_Hz
        self.A_ul_s_1 = A_ul_s_1
        self.g_ratio = g_u / g_l
        self.flags = flags
        self.surface_albedo = surface_albedo
        self.background_temperature_K = background_temperature_K
        self.nucleus_temperature_K = nucleus_temperature_K
        self.mu, self.mu_weights = gaussian_quadrature_streams(nstreams)
        self.frequency_x, frequency_weights = np.polynomial.hermite.hermgauss(
            nfrequency
        )
        self.frequency_weights = frequency_weights / np.sqrt(np.pi)
        self.subsegments_per_shell = subsegments_per_shell
        self._build_characteristics()
        self._build_solver()

    def _build_characteristics(self) -> None:
        rays = []
        maximum_layers = 0
        for radius_index, radius_m in enumerate(self.radius_m):
            for mu_index, mu in enumerate(self.mu):
                ray = _ordered_ray_segments(
                    self.faces_m,
                    float(radius_m),
                    float(mu),
                    self.subsegments_per_shell,
                )
                rays.append((radius_index, mu_index, ray))
                maximum_layers = max(maximum_layers, ray.length_m.size)

        ncolumns = len(rays) * self.frequency_x.size
        self.ncolumns = ncolumns
        self.nlayers = maximum_layers
        self.shell_index = np.zeros((ncolumns, maximum_layers), dtype=np.int64)
        self.valid = np.zeros((ncolumns, maximum_layers), dtype=bool)
        self.opacity_kernel = np.zeros((ncolumns, maximum_layers), dtype=np.float64)
        self.bottom_temperature_K = np.empty(ncolumns, dtype=np.float64)
        self.bottom_albedo = np.zeros(ncolumns, dtype=np.float64)

        column = 0
        opacity_scale = (
            constants.SPEED_OF_LIGHT**2 * self.A_ul_s_1 / (8.0 * np.pi * self.nu_Hz**2)
        )
        for radius_index, mu_index, ray in rays:
            target_center_hz = self.nu_Hz / (
                1.0
                - self.velocity_m_s[radius_index]
                * self.mu[mu_index]
                / constants.SPEED_OF_LIGHT
            )
            for frequency_x in self.frequency_x:
                shell = ray.shell_index[::-1]
                length_m = ray.length_m[::-1]
                projection = ray.projected_velocity_factor[::-1]
                count = shell.size
                self.shell_index[column, :count] = shell
                self.shell_index[column, count:] = shell[-1]
                self.valid[column, :count] = True

                evaluation_hz = (
                    target_center_hz + frequency_x * self.doppler_width_hz[radius_index]
                )
                local_center_hz = self.nu_Hz / (
                    1.0
                    - self.velocity_m_s[shell] * projection / constants.SPEED_OF_LIGHT
                )
                local_width_hz = self.doppler_width_hz[shell]
                profile_hz_1 = np.exp(
                    -(((evaluation_hz - local_center_hz) / local_width_hz) ** 2)
                ) / (np.sqrt(np.pi) * local_width_hz)
                self.opacity_kernel[column, :count] = (
                    opacity_scale * profile_hz_1 * length_m
                )
                if ray.enters_from_nucleus:
                    self.bottom_temperature_K[column] = self.nucleus_temperature_K
                    self.bottom_albedo[column] = self.surface_albedo
                else:
                    self.bottom_temperature_K[column] = self.background_temperature_K
                column += 1

    def _build_solver(self) -> None:
        options = (
            pydisort.DisortOptions().flags(self.flags).nwave(1).ncol(self.ncolumns)
        )
        wavenumber_cm_1 = self.nu_Hz / _C_CM_S
        options.wave_lower(np.array([wavenumber_cm_1], dtype=np.float64))
        options.wave_upper(np.array([wavenumber_cm_1], dtype=np.float64))
        options.user_mu(np.array([1.0], dtype=np.float64))
        options.user_phi(np.array([0.0], dtype=np.float64))
        if "usrtau" in {flag.strip() for flag in self.flags.split(",")}:
            options.user_tau(np.array([0.0], dtype=np.float64))
        options.accur(0.0)
        options.ds().nlyr = self.nlayers
        options.ds().nstr = 4
        options.ds().nmom = 4
        options.ds().nphase = 1
        self.solver = pydisort.Disort(options)
        self.moments = pydisort.scattering_moments(4, "isotropic")

    def mean_intensity(
        self,
        n_lower_m_3: np.ndarray,
        n_upper_m_3: np.ndarray,
        excitation_temperature_K: np.ndarray,
    ) -> np.ndarray:
        population_difference = self.g_ratio * n_lower_m_3 - n_upper_m_3
        tau = self.opacity_kernel * population_difference[self.shell_index]
        tau[~self.valid] = 0.0
        layer_temperature = excitation_temperature_K[self.shell_index]
        boundary_temperature = np.empty((self.ncolumns, self.nlayers + 1))
        boundary_temperature[:, 0] = layer_temperature[:, 0]
        boundary_temperature[:, -1] = layer_temperature[:, -1]
        boundary_temperature[:, 1:-1] = 0.5 * (
            layer_temperature[:, :-1] + layer_temperature[:, 1:]
        )

        properties = torch.zeros(
            (1, self.ncolumns, self.nlayers, 6), dtype=torch.float64
        )
        properties[0, :, :, 0] = torch.from_numpy(tau)
        properties[0, :, :, 2:] = self.moments
        zeros_wave_column = torch.zeros((1, self.ncolumns), dtype=torch.float64)
        zeros_column = torch.zeros(self.ncolumns, dtype=torch.float64)
        self.solver.forward(
            properties,
            temf=torch.from_numpy(boundary_temperature),
            umu0=torch.ones(self.ncolumns, dtype=torch.float64),
            phi0=zeros_column,
            fbeam=zeros_wave_column,
            albedo=torch.from_numpy(self.bottom_albedo).reshape(1, self.ncolumns),
            fluor=zeros_wave_column,
            fisot=zeros_wave_column,
            temis=zeros_wave_column,
            ttemp=zeros_column,
            btemp=torch.from_numpy(self.bottom_temperature_K),
        )

        intensity_per_hz = (
            self.solver.gather_rad().detach().cpu().numpy()[0, :, 0, 0, 0] / _C_CM_S
        )
        intensity = intensity_per_hz.reshape(
            self.radius_m.size, self.mu.size, self.frequency_x.size
        )
        frequency_average = np.sum(
            intensity * self.frequency_weights[np.newaxis, np.newaxis, :],
            axis=2,
        )
        return 0.5 * np.sum(frequency_average * self.mu_weights[np.newaxis, :], axis=1)


def _cache_key(*values: object) -> tuple[object, ...]:
    key = []
    for value in values:
        if isinstance(value, np.ndarray):
            key.append((value.shape, value.dtype.str, value.tobytes()))
        else:
            key.append(value)
    return tuple(key)


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
    line_profile_type: str = "thermal Gaussian",
    width_v_fwhm_m_s: float | np.ndarray | None = None,
    path_length_m: float | np.ndarray | None = None,
    nstreams: int = 16,
    flags: str = DEFAULT_DISORT_FLAGS,
    surface_albedo: float = DEFAULT_SURFACE_ALBEDO,
    background_temperature_K: float = DEFAULT_BACKGROUND_TEMPERATURE_K,
    temis: float = DEFAULT_TOP_EMISSIVITY,
    fisot: float = DEFAULT_TOP_ISOTROPIC_ILLUMINATION,
    nfrequency: int = DEFAULT_FREQUENCY_POINTS,
    subsegments_per_shell: int = DEFAULT_SUBSEGMENTS_PER_SHELL,
    nucleus_temperature_K: float = 0.0,
) -> dict[str, np.ndarray | float | str]:
    """Solve line transfer along finite spherical chords."""

    if line_profile_type.strip().lower() not in {"thermal gaussian", "gaussian"}:
        raise ValueError("Only Gaussian line profiles are supported")
    if background_temperature_K < 0.0 or nucleus_temperature_K < 0.0:
        raise ValueError("Boundary temperatures must be non-negative")
    if temis != 0.0 or fisot != 0.0:
        raise ValueError("temis and fisot are not physical boundaries of a chord")
    if nfrequency < 1 or subsegments_per_shell < 1:
        raise ValueError("Quadrature and ray subdivision counts must be positive")
    flag_set = {flag.strip() for flag in flags.split(",")}
    if not {"lamber", "planck", "usrang"}.issubset(flag_set):
        raise ValueError("flags must include lamber, planck, and usrang")

    radius_m, faces_m, order, inverse_order = _radial_grid(coma)
    path_lengths_m = _radial_path_lengths(coma, path_length_m)
    temperature_K = np.asarray(coma.temperature, dtype=np.float64)
    velocity_m_s = np.asarray(coma.velocity, dtype=np.float64)
    doppler_width_hz = _doppler_width_hz(
        temperature_K,
        nu_Hz,
        absorber_mass_kg,
        width_v_fwhm_m_s,
    )
    transfer_key = _cache_key(
        radius_m,
        faces_m,
        velocity_m_s[order],
        doppler_width_hz[order],
        nu_Hz,
        A_ul_s_1,
        g_u,
        g_l,
        nstreams,
        nfrequency,
        subsegments_per_shell,
        flags,
        surface_albedo,
        background_temperature_K,
        nucleus_temperature_K,
    )
    cache = getattr(coma, "_spherical_disort_cache", None)
    if cache is None or cache[0] != transfer_key:
        transfer = _SphericalDisort(
            radius_m=radius_m,
            faces_m=faces_m,
            velocity_m_s=velocity_m_s[order],
            doppler_width_hz=doppler_width_hz[order],
            nu_Hz=nu_Hz,
            A_ul_s_1=A_ul_s_1,
            g_u=g_u,
            g_l=g_l,
            nstreams=nstreams,
            nfrequency=nfrequency,
            subsegments_per_shell=subsegments_per_shell,
            flags=flags,
            surface_albedo=surface_albedo,
            background_temperature_K=background_temperature_K,
            nucleus_temperature_K=nucleus_temperature_K,
        )
        coma._spherical_disort_cache = (transfer_key, transfer)
    else:
        transfer = cache[1]

    n_lower_m_3 = np.asarray(coma.density[:, lower_level], dtype=np.float64)
    n_upper_m_3 = np.asarray(coma.density[:, upper_level], dtype=np.float64)
    excitation_temperature_K = nonlte_excitation_temperature(
        coma,
        lower_level=lower_level,
        upper_level=upper_level,
        nu_Hz=nu_Hz,
        g_u=g_u,
        g_l=g_l,
    )
    population_difference = (g_u / g_l) * n_lower_m_3 - n_upper_m_3
    if np.any(population_difference < 0.0) or np.any(
        ~np.isfinite(excitation_temperature_K) | (excitation_temperature_K < 0.0)
    ):
        raise ValueError("Population inversions are not supported")
    phi0_hz_1 = 1.0 / (np.sqrt(np.pi) * doppler_width_hz)
    tau_profile = (
        constants.SPEED_OF_LIGHT**2
        / (8.0 * np.pi * nu_Hz**2)
        * A_ul_s_1
        * phi0_hz_1
        * population_difference
        * path_lengths_m
    )
    mean_intensity_sorted = transfer.mean_intensity(
        n_lower_m_3[order],
        n_upper_m_3[order],
        excitation_temperature_K[order],
    )
    mean_intensity = mean_intensity_sorted[inverse_order]

    return {
        "flags": flags,
        "nstreams": int(transfer.mu.size),
        "stream_mu": np.array(transfer.mu, copy=True),
        "stream_weights": np.array(transfer.mu_weights, copy=True),
        "path_length_m": np.array(path_lengths_m, copy=True),
        "tau_profile": np.array(tau_profile, copy=True),
        "user_tau": _cumulative_user_tau(tau_profile),
        "excitation_temperature_K": np.array(excitation_temperature_K, copy=True),
        "J_v_stream_average": np.array(mean_intensity, copy=True),
    }
