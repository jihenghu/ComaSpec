"""Iterative non-LTE two-level solver for coma grids."""

from __future__ import annotations

import numpy as np

import Constants as constants
from ComaGrid import ComaGrid
from Einstein_coeff import einstein_coeffs_B
from Electron_collision import electron_collision_rate
from Molecular_collision import (
    detailed_balance_excitation_rate,
    molecular_collision_rate,
)
from NonLTE_optics import nonlte_source_function, nonlte_tau
from RTsolver import formal_solution, layer_thicknesses
from Solar_pumping import scale_g_factors_from_1au
from Statistical_equilibrium import monte_carlo_two_level_populations


def _as_layer_array(
    values: float | np.ndarray,
    nlayer: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        return np.full(nlayer, float(array), dtype=np.float64)
    if array.shape != (nlayer,):
        raise ValueError(f"{name} must be scalar or have shape ({nlayer},)")
    return np.array(array, dtype=np.float64, copy=False)


def _default_collision_and_pumping_rates(
    coma: ComaGrid,
    *,
    heliocentric_distance_AU: float,
    nu_Hz: float,
    A_ul_s_1: float,
    g_u: float,
    g_l: float,
    molecular_Cm_lu_s_1: float | np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build default two-level rate profiles from the coma state."""

    nlayer = coma.ngrid
    Cm_ul_s_1 = _as_layer_array(
        molecular_collision_rate(coma.total_number_density, coma.temperature),
        nlayer,
        "Cm_ul_s_1",
    )
    if molecular_Cm_lu_s_1 is None:
        Cm_lu_s_1 = _as_layer_array(
            detailed_balance_excitation_rate(
                Cm_ul_s_1,
                coma.temperature,
                nu_Hz=nu_Hz,
                g_u=g_u,
                g_l=g_l,
            ),
            nlayer,
            "Cm_lu_s_1",
        )
    else:
        Cm_lu_s_1 = _as_layer_array(molecular_Cm_lu_s_1, nlayer, "Cm_lu_s_1")

    electron_rates = np.array(
        [
            electron_collision_rate(
                n_e=float(n_e),
                t_e=float(t_e),
                nu_Hz=nu_Hz,
                A_ul_s_1=A_ul_s_1,
                g_u=g_u,
                g_l=g_l,
            )
            for n_e, t_e in zip(coma.nelec, coma.telec)
        ],
        dtype=np.float64,
    )
    Ce_ul_s_1 = electron_rates[:, 0]
    Ce_lu_s_1 = electron_rates[:, 1]

    G_ul_s_1, G_lu_s_1 = scale_g_factors_from_1au(
        heliocentric_distance_AU=heliocentric_distance_AU
    )

    return {
        "Cm_ul_s_1": Cm_ul_s_1,
        "Cm_lu_s_1": Cm_lu_s_1,
        "Ce_ul_s_1": _as_layer_array(Ce_ul_s_1, nlayer, "Ce_ul_s_1"),
        "Ce_lu_s_1": _as_layer_array(Ce_lu_s_1, nlayer, "Ce_lu_s_1"),
        "G_ul_s_1": _as_layer_array(G_ul_s_1, nlayer, "G_ul_s_1"),
        "G_lu_s_1": _as_layer_array(G_lu_s_1, nlayer, "G_lu_s_1"),
    }


def mean_intensity_two_stream(
    tau_profile: np.ndarray,
    source_profile: np.ndarray,
    *,
    background_intensity_outer: float = 0.0,
    background_intensity_inner: float = 0.0,
) -> dict[str, np.ndarray | float]:
    """Estimate layer-centered ``J_nu`` from inward and outward formal solutions."""

    outward = formal_solution(
        tau_profile,
        source_profile,
        background_intensity=background_intensity_outer,
        inward=False,
    )
    inward = formal_solution(
        tau_profile,
        source_profile,
        background_intensity=background_intensity_inner,
        inward=True,
    )

    outward_center = 0.5 * (outward["intensity_in"] + outward["intensity_out"])
    inward_center = 0.5 * (inward["intensity_in"] + inward["intensity_out"])
    j_nu = 0.5 * (outward_center + inward_center)

    return {
        "J_nu": np.asarray(j_nu, dtype=np.float64),
        "outward_intensity_in": np.asarray(outward["intensity_in"], dtype=np.float64),
        "outward_intensity_out": np.asarray(outward["intensity_out"], dtype=np.float64),
        "inward_intensity_in": np.asarray(inward["intensity_in"], dtype=np.float64),
        "inward_intensity_out": np.asarray(inward["intensity_out"], dtype=np.float64),
        "I_nu_emergent_outer": float(outward["I_nu_emergent"]),
        "I_nu_emergent_inner": float(inward["I_nu_emergent"]),
    }


def iterate_two_level_nlte(
    coma: ComaGrid,
    *,
    heliocentric_distance_AU: float,
    lower_level: int = 0,
    upper_level: int = 1,
    nu_Hz: float = 556.936e9,
    A_ul_s_1: float = 3.456e-3,
    g_u: float = 9.0,
    g_l: float = 9.0,
    absorber_mass_kg: float = 18.01528 * constants.ATOMIC_MASS_UNIT,
    background_intensity_outer: float = 0.0,
    background_intensity_inner: float = 0.0,
    B_ul_SI: float | None = None,
    B_lu_SI: float | None = None,
    Cm_ul_s_1: float | np.ndarray | None = None,
    Cm_lu_s_1: float | np.ndarray | None = None,
    Ce_ul_s_1: float | np.ndarray | None = None,
    Ce_lu_s_1: float | np.ndarray | None = None,
    G_ul_s_1: float | np.ndarray | None = None,
    G_lu_s_1: float | np.ndarray | None = None,
    max_iterations: int = 100,
    j_nu_rtol: float = 1.0e-6,
    j_nu_atol: float = 1.0e-20,
    population_max_iterations: int = 20000,
    population_tolerance: float = 1.0e-10,
    packet_fraction_range: tuple[float, float] = (0.05, 0.35),
    random_seed: int | None = None,
) -> dict[str, np.ndarray | list[dict[str, np.ndarray | float | int]] | int | bool]:
    """Iterate RT and statistical equilibrium until ``J_nu`` converges.

    The input ``coma`` is updated in place with the final two-level populations.
    """

    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    if j_nu_rtol < 0.0 or j_nu_atol < 0.0:
        raise ValueError("j_nu_rtol and j_nu_atol must be non-negative")

    nlayer = coma.ngrid
    ds_m = layer_thicknesses(coma)

    if B_ul_SI is None or B_lu_SI is None:
        B_ul_default, B_lu_default = einstein_coeffs_B(
            nu_Hz=nu_Hz,
            A_ul_s_1=A_ul_s_1,
            g_u=g_u,
            g_l=g_l,
        )
        if B_ul_SI is None:
            B_ul_SI = B_ul_default
        if B_lu_SI is None:
            B_lu_SI = B_lu_default

    if any(rate is None for rate in (Cm_ul_s_1, Ce_ul_s_1, Ce_lu_s_1, G_ul_s_1, G_lu_s_1)):
        default_rates = _default_collision_and_pumping_rates(
            coma,
            heliocentric_distance_AU=heliocentric_distance_AU,
            nu_Hz=nu_Hz,
            A_ul_s_1=A_ul_s_1,
            g_u=g_u,
            g_l=g_l,
            molecular_Cm_lu_s_1=Cm_lu_s_1,
        )
        if Cm_ul_s_1 is None:
            Cm_ul_s_1 = default_rates["Cm_ul_s_1"]
        if Ce_ul_s_1 is None:
            Ce_ul_s_1 = default_rates["Ce_ul_s_1"]
        if Ce_lu_s_1 is None:
            Ce_lu_s_1 = default_rates["Ce_lu_s_1"]
        if G_ul_s_1 is None:
            G_ul_s_1 = default_rates["G_ul_s_1"]
        if G_lu_s_1 is None:
            G_lu_s_1 = default_rates["G_lu_s_1"]

    rates = {
        "A_ul_s_1": _as_layer_array(A_ul_s_1, nlayer, "A_ul_s_1"),
        "B_ul_SI": _as_layer_array(B_ul_SI, nlayer, "B_ul_SI"),
        "B_lu_SI": _as_layer_array(B_lu_SI, nlayer, "B_lu_SI"),
        "Cm_ul_s_1": _as_layer_array(Cm_ul_s_1, nlayer, "Cm_ul_s_1"),
        "Cm_lu_s_1": _as_layer_array(Cm_lu_s_1, nlayer, "Cm_lu_s_1"),
        "Ce_ul_s_1": _as_layer_array(Ce_ul_s_1, nlayer, "Ce_ul_s_1"),
        "Ce_lu_s_1": _as_layer_array(Ce_lu_s_1, nlayer, "Ce_lu_s_1"),
        "G_ul_s_1": _as_layer_array(G_ul_s_1, nlayer, "G_ul_s_1"),
        "G_lu_s_1": _as_layer_array(G_lu_s_1, nlayer, "G_lu_s_1"),
    }

    history: list[dict[str, np.ndarray | float | int | bool]] = []
    previous_j_nu: np.ndarray | None = None
    converged = False

    for iteration in range(1, max_iterations + 1):
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

        rt_result = mean_intensity_two_stream(
            tau_profile,
            source_profile,
            background_intensity_outer=background_intensity_outer,
            background_intensity_inner=background_intensity_inner,
        )
        j_nu = np.asarray(rt_result["J_nu"], dtype=np.float64)

        if previous_j_nu is None:
            delta_j_nu = np.full_like(j_nu, np.inf)
            relative_change = np.full_like(j_nu, np.inf)
        else:
            delta_j_nu = np.abs(j_nu - previous_j_nu)
            scale = np.maximum(np.abs(previous_j_nu), j_nu_atol)
            relative_change = delta_j_nu / scale
            converged = bool(
                np.all(delta_j_nu <= j_nu_atol + j_nu_rtol * np.abs(previous_j_nu))
            )
            if converged:
                history.append(
                    {
                        "iteration": iteration,
                        "tau_profile": np.array(tau_profile, copy=True),
                        "source_profile": np.array(source_profile, copy=True),
                        "J_nu": np.array(j_nu, copy=True),
                        "delta_J_nu": np.array(delta_j_nu, copy=True),
                        "relative_change": np.array(relative_change, copy=True),
                        "n_lower": np.array(coma.density[:, lower_level], copy=True),
                        "n_upper": np.array(coma.density[:, upper_level], copy=True),
                        "population_iterations_used": 0,
                        "population_converged": True,
                        "I_nu_emergent_outer": float(rt_result["I_nu_emergent_outer"]),
                        "I_nu_emergent_inner": float(rt_result["I_nu_emergent_inner"]),
                    }
                )
                break

        pop_result = monte_carlo_two_level_populations(
            coma,
            j_nu,
            A_ul_s_1=rates["A_ul_s_1"],
            B_ul_SI=rates["B_ul_SI"],
            B_lu_SI=rates["B_lu_SI"],
            Cm_ul_s_1=rates["Cm_ul_s_1"],
            Cm_lu_s_1=rates["Cm_lu_s_1"],
            Ce_ul_s_1=rates["Ce_ul_s_1"],
            Ce_lu_s_1=rates["Ce_lu_s_1"],
            G_ul_s_1=rates["G_ul_s_1"],
            G_lu_s_1=rates["G_lu_s_1"],
            nu_Hz=nu_Hz,
            g_u=g_u,
            g_l=g_l,
            lower_level=lower_level,
            upper_level=upper_level,
            max_iterations=population_max_iterations,
            tolerance=population_tolerance,
            packet_fraction_range=packet_fraction_range,
            random_seed=None if random_seed is None else random_seed + iteration - 1,
        )

        history.append(
            {
                "iteration": iteration,
                "tau_profile": np.array(tau_profile, copy=True),
                "source_profile": np.array(source_profile, copy=True),
                "J_nu": np.array(j_nu, copy=True),
                "delta_J_nu": np.array(delta_j_nu, copy=True),
                "relative_change": np.array(relative_change, copy=True),
                "n_lower": np.array(pop_result["n_lower"], copy=True),
                "n_upper": np.array(pop_result["n_upper"], copy=True),
                "population_iterations_used": int(pop_result["iterations_used"]),
                "population_converged": bool(np.all(pop_result["converged"])),
                "I_nu_emergent_outer": float(rt_result["I_nu_emergent_outer"]),
                "I_nu_emergent_inner": float(rt_result["I_nu_emergent_inner"]),
            }
        )
        previous_j_nu = j_nu

    final_state = history[-1]
    return {
        "converged": converged,
        "iterations_used": len(history),
        "J_nu": np.array(final_state["J_nu"], copy=True),
        "delta_J_nu": np.array(final_state["delta_J_nu"], copy=True),
        "relative_change": np.array(final_state["relative_change"], copy=True),
        "tau_profile": np.array(final_state["tau_profile"], copy=True),
        "source_profile": np.array(final_state["source_profile"], copy=True),
        "path_length_m": np.array(ds_m, copy=True),
        "n_lower": np.array(coma.density[:, lower_level], copy=True),
        "n_upper": np.array(coma.density[:, upper_level], copy=True),
        "rates": {name: np.array(values, copy=True) for name, values in rates.items()},
        "history": history,
    }
