"""Spherical one-dimensional radiative-transfer solver for coma shells."""

from __future__ import annotations

import numpy as np

import Constants as constants
from ComaGrid import ComaGrid
from NonLTE_optics import nonlte_excitation_temperature, nonlte_source_function, nonlte_tau


def layer_thicknesses(coma: ComaGrid) -> np.ndarray:
    """Return positive radial shell thicknesses from ``coma.xf``."""

    xf = np.asarray(coma.xf, dtype=np.float64)
    if xf.shape != (coma.ngrid + 1,):
        raise ValueError("coma.xf must have shape (coma.ngrid + 1,)")
    dxf = np.diff(xf)
    if not (np.all(dxf > 0.0) or np.all(dxf < 0.0)):
        raise ValueError("coma.xf must be strictly monotonic")
    return np.abs(dxf)


def _validate_shell_grid(
    coma: ComaGrid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return shell geometry ordered from inner radius to outer radius."""

    faces_m = np.asarray(coma.xf, dtype=np.float64)
    centers_m = np.asarray(coma.xc, dtype=np.float64)

    if faces_m.shape != (coma.ngrid + 1,):
        raise ValueError("coma.xf must have shape (coma.ngrid + 1,)")
    if centers_m.shape != (coma.ngrid,):
        raise ValueError("coma.xc must have shape (coma.ngrid,)")
    if np.any(~np.isfinite(faces_m)) or np.any(~np.isfinite(centers_m)):
        raise ValueError("coma.xf and coma.xc must be finite")
    if np.any(faces_m < 0.0) or np.any(centers_m <= 0.0):
        raise ValueError("spherical radii must be non-negative with positive centers")
    face_steps_m = np.diff(faces_m)
    if np.all(face_steps_m > 0.0):
        sorted_to_original = np.arange(coma.ngrid)
        faces_inner_to_outer_m = faces_m
        centers_inner_to_outer_m = centers_m
    elif np.all(face_steps_m < 0.0):
        sorted_to_original = np.arange(coma.ngrid - 1, -1, -1)
        faces_inner_to_outer_m = faces_m[::-1]
        centers_inner_to_outer_m = centers_m[::-1]
    else:
        raise ValueError("coma.xf must be strictly monotonic")

    shell_inner_m = np.minimum(faces_m[:-1], faces_m[1:])
    shell_outer_m = np.maximum(faces_m[:-1], faces_m[1:])
    if np.any(centers_m <= shell_inner_m) or np.any(centers_m >= shell_outer_m):
        raise ValueError("each coma.xc value must lie inside its shell faces")

    return (
        faces_inner_to_outer_m,
        centers_inner_to_outer_m,
        np.diff(faces_inner_to_outer_m),
        sorted_to_original,
    )


def _ray_shell_path_lengths(
    faces_m: np.ndarray,
    radius_m: float,
    mu: float,
) -> tuple[np.ndarray, int]:
    """Return path lengths crossed by a ray before it reaches ``(radius, mu)``.

    ``mu`` is the cosine between the outward radial direction and the photon
    direction at the evaluation point.  The returned boundary flag is ``0`` for
    rays entering from the inner boundary and ``-1`` for rays entering from the
    outer boundary.
    """

    impact_m = radius_m * np.sqrt(max(0.0, 1.0 - mu * mu))
    z_end_m = radius_m * mu
    inner_m = faces_m[0]
    outer_m = faces_m[-1]

    if mu >= 0.0 and impact_m < inner_m:
        z_start_m = np.sqrt(inner_m * inner_m - impact_m * impact_m)
        boundary_index = 0
    else:
        z_start_m = -np.sqrt(outer_m * outer_m - impact_m * impact_m)
        boundary_index = -1

    crossings = [z_start_m, z_end_m]
    for face_m in faces_m:
        if face_m <= impact_m:
            continue
        z_face_m = np.sqrt(face_m * face_m - impact_m * impact_m)
        for signed_z_m in (-z_face_m, z_face_m):
            if z_start_m < signed_z_m < z_end_m:
                crossings.append(signed_z_m)

    z_points_m = np.array(sorted(crossings), dtype=np.float64)
    path_lengths_m = np.zeros(faces_m.size - 1, dtype=np.float64)

    for z_left_m, z_right_m in zip(z_points_m[:-1], z_points_m[1:]):
        ds_m = z_right_m - z_left_m
        if ds_m <= 0.0:
            continue
        z_mid_m = 0.5 * (z_left_m + z_right_m)
        r_mid_m = np.sqrt(impact_m * impact_m + z_mid_m * z_mid_m)
        shell_index = int(np.searchsorted(faces_m, r_mid_m, side="right") - 1)
        if 0 <= shell_index < path_lengths_m.size:
            path_lengths_m[shell_index] += ds_m

    return path_lengths_m, boundary_index


def _radial_tau_to_centers(
    tau_profile: np.ndarray,
    centers_m: np.ndarray,
    faces_m: np.ndarray,
    shell_thickness_m: np.ndarray,
) -> np.ndarray:
    """Return radial optical depth from the inner face to each shell center."""

    alpha_m_1 = tau_profile / shell_thickness_m
    tau_inner_faces = np.concatenate(([0.0], np.cumsum(tau_profile[:-1])))
    return tau_inner_faces + alpha_m_1 * (centers_m - faces_m[:-1])


def spherical_mean_intensity(
    tau_profile: np.ndarray,
    source_profile: np.ndarray,
    coma: ComaGrid,
    *,
    n_mu: int = 32,
    background_intensity_outer: float = 0.0,
    background_intensity_inner: float = 0.0,
    central_flux_nu_reference: float = 0.0,
    central_flux_reference_radius_m: float | None = None,
) -> dict[str, np.ndarray | float]:
    """Compute spherical ``J_nu`` from shell emission plus optional beam flux."""

    if n_mu < 2:
        raise ValueError("n_mu must be at least 2")
    if background_intensity_outer < 0.0 or background_intensity_inner < 0.0:
        raise ValueError("background intensities must be non-negative")
    if central_flux_nu_reference < 0.0:
        raise ValueError("central_flux_nu_reference must be non-negative")

    tau = np.asarray(tau_profile, dtype=np.float64)
    source = np.asarray(source_profile, dtype=np.float64)
    if tau.shape != (coma.ngrid,) or source.shape != (coma.ngrid,):
        raise ValueError("tau_profile and source_profile must have shape (coma.ngrid,)")
    if np.any(~np.isfinite(tau)) or np.any(~np.isfinite(source)):
        raise ValueError("tau_profile and source_profile must be finite")

    (
        faces_m,
        centers_m,
        shell_thickness_m,
        sorted_to_original,
    ) = _validate_shell_grid(coma)
    original_to_sorted = np.empty_like(sorted_to_original)
    original_to_sorted[sorted_to_original] = np.arange(coma.ngrid)

    tau_sorted = tau[sorted_to_original]
    source_sorted = source[sorted_to_original]
    alpha_m_1 = tau_sorted / shell_thickness_m
    mu, weights = np.polynomial.legendre.leggauss(n_mu)
    intensities = np.empty((coma.ngrid, n_mu), dtype=np.float64)

    for radius_index, radius_m in enumerate(centers_m):
        for mu_index, mu_value in enumerate(mu):
            path_lengths_m, boundary_index = _ray_shell_path_lengths(
                faces_m,
                float(radius_m),
                float(mu_value),
            )
            intensity = (
                background_intensity_inner
                if boundary_index == 0
                else background_intensity_outer
            )

            for shell_index, ds_m in enumerate(path_lengths_m):
                if ds_m == 0.0:
                    continue
                delta_tau = alpha_m_1[shell_index] * ds_m
                transmission = np.exp(-delta_tau)
                intensity = intensity * transmission + source_sorted[shell_index] * (
                    1.0 - transmission
                )

            intensities[radius_index, mu_index] = intensity

    j_diffuse = 0.5 * np.sum(intensities * weights[np.newaxis, :], axis=1)
    flux_diffuse = 2.0 * np.pi * np.sum(
        intensities * (weights * mu)[np.newaxis, :],
        axis=1,
    )

    tau_radial = _radial_tau_to_centers(
        tau_sorted,
        centers_m,
        faces_m,
        shell_thickness_m,
    )
    flux_beam = np.zeros(coma.ngrid, dtype=np.float64)
    j_beam = np.zeros(coma.ngrid, dtype=np.float64)

    if central_flux_nu_reference > 0.0:
        if central_flux_reference_radius_m is None:
            central_flux_reference_radius_m = float(max(faces_m[0], centers_m[0]))
        if central_flux_reference_radius_m <= 0.0:
            raise ValueError("central_flux_reference_radius_m must be positive")
        flux_beam = (
            float(central_flux_nu_reference)
            * (float(central_flux_reference_radius_m) / centers_m) ** 2
            * np.exp(-tau_radial)
        )
        j_beam = flux_beam / (4.0 * np.pi)

    j_diffuse_original = j_diffuse[original_to_sorted]
    j_beam_original = j_beam[original_to_sorted]
    flux_diffuse_original = flux_diffuse[original_to_sorted]
    flux_beam_original = flux_beam[original_to_sorted]

    return {
        "J_nu": np.asarray(j_diffuse_original + j_beam_original, dtype=np.float64),
        "J_nu_diffuse": np.asarray(j_diffuse_original, dtype=np.float64),
        "J_nu_beam": np.asarray(j_beam_original, dtype=np.float64),
        "F_nu": np.asarray(flux_diffuse_original + flux_beam_original, dtype=np.float64),
        "F_nu_diffuse": np.asarray(flux_diffuse_original, dtype=np.float64),
        "F_nu_beam": np.asarray(flux_beam_original, dtype=np.float64),
        "I_nu_mu": np.asarray(intensities[original_to_sorted], dtype=np.float64),
        "mu": np.asarray(mu, dtype=np.float64),
        "mu_weights": np.asarray(weights, dtype=np.float64),
        "tau_radial_to_center": np.asarray(
            tau_radial[original_to_sorted],
            dtype=np.float64,
        ),
    }


def solve_rt_spherical(
    coma: ComaGrid,
    *,
    lower_level: int = 0,
    upper_level: int = 1,
    nu_Hz: float = 556.936e9,
    A_ul_s_1: float = 3.456e-3,
    g_u: float = 9.0,
    g_l: float = 9.0,
    absorber_mass_kg: float = 18.01528 * constants.ATOMIC_MASS_UNIT,
    n_mu: int = 32,
    background_intensity_outer: float = 0.0,
    background_intensity_inner: float = 0.0,
    central_flux_nu_reference: float = 0.0,
    central_flux_reference_radius_m: float | None = None,
) -> dict[str, np.ndarray | float]:
    """Solve one spherical RT update for the current non-LTE coma populations."""

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
    excitation_temperature = nonlte_excitation_temperature(
        coma,
        lower_level=lower_level,
        upper_level=upper_level,
        nu_Hz=nu_Hz,
        g_u=g_u,
        g_l=g_l,
    )

    result = spherical_mean_intensity(
        tau_profile,
        source_profile,
        coma,
        n_mu=n_mu,
        background_intensity_outer=background_intensity_outer,
        background_intensity_inner=background_intensity_inner,
        central_flux_nu_reference=central_flux_nu_reference,
        central_flux_reference_radius_m=central_flux_reference_radius_m,
    )
    result["path_length_m"] = np.array(ds_m, copy=True)
    result["tau_profile"] = np.array(tau_profile, copy=True)
    result["source_profile"] = np.array(source_profile, copy=True)
    result["excitation_temperature_K"] = np.array(excitation_temperature, copy=True)
    result["J_v_stream_average"] = np.array(result["J_nu"], copy=True)
    return result
