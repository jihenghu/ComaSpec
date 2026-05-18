"""Monte Carlo iterator for two-level statistical equilibrium."""

from __future__ import annotations

import numpy as np

from ComaGrid import ComaGrid


def _broadcast_layer_parameter(
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


def monte_carlo_two_level_populations(
    coma_grid: ComaGrid,
    j_nu: float | np.ndarray,
    *,
    A_ul_s_1: float | np.ndarray,
    B_ul_SI: float | np.ndarray,
    B_lu_SI: float | np.ndarray,
    Cm_ul_s_1: float | np.ndarray,
    Ce_ul_s_1: float | np.ndarray,
    Ce_lu_s_1: float | np.ndarray,
    G_ul_s_1: float | np.ndarray,
    G_lu_s_1: float | np.ndarray,
    nu_Hz: float=556.936e9,
    g_u: float=9.0,
    g_l: float=9.0,
    lower_level: int = 0,
    upper_level: int = 1,
    Cm_lu_s_1: float | np.ndarray = 0.0,
    max_iterations: int = 20000,
    tolerance: float = 1.0e-10,
    packet_fraction_range: tuple[float, float] = (0.05, 0.35),
    random_seed: int | None = None,
) -> dict[str, np.ndarray | int | float]:
    """Iterate two-level populations in each layer toward statistical equilibrium.

    The per-layer equilibrium condition is

    ``n_u * (A_ul + B_ul J_nu + Cm_ul + Ce_ul + G_ul)``
    ``= n_l * (B_lu J_nu + Cm_lu + Ce_lu + G_lu)``.

    A random transfer packet is used at each iteration to move population between
    the lower and upper levels. Every transfer is clipped so that no negative
    number density can be created.
    """

    if coma_grid.nlevels < 2:
        raise ValueError("coma_grid must contain at least two levels")
    if not 0 <= lower_level < coma_grid.nlevels:
        raise ValueError("lower_level is out of bounds")
    if not 0 <= upper_level < coma_grid.nlevels:
        raise ValueError("upper_level is out of bounds")
    if lower_level == upper_level:
        raise ValueError("lower_level and upper_level must be different")
    if nu_Hz <= 0.0:
        raise ValueError("nu_Hz must be positive")
    if g_u <= 0.0 or g_l <= 0.0:
        raise ValueError("g_u and g_l must be positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")

    min_packet_fraction, max_packet_fraction = packet_fraction_range
    if not 0.0 < min_packet_fraction <= max_packet_fraction <= 1.0:
        raise ValueError(
            "packet_fraction_range must satisfy 0 < min <= max <= 1"
        )

    nlayer = coma_grid.ngrid
    j_nu = _broadcast_layer_parameter(j_nu, nlayer, "j_nu")
    A_ul_s_1 = _broadcast_layer_parameter(A_ul_s_1, nlayer, "A_ul_s_1")
    B_ul_SI = _broadcast_layer_parameter(B_ul_SI, nlayer, "B_ul_SI")
    B_lu_SI = _broadcast_layer_parameter(B_lu_SI, nlayer, "B_lu_SI")
    Cm_ul_s_1 = _broadcast_layer_parameter(Cm_ul_s_1, nlayer, "Cm_ul_s_1")
    Ce_ul_s_1 = _broadcast_layer_parameter(Ce_ul_s_1, nlayer, "Ce_ul_s_1")
    Ce_lu_s_1 = _broadcast_layer_parameter(Ce_lu_s_1, nlayer, "Ce_lu_s_1")
    G_ul_s_1 = _broadcast_layer_parameter(G_ul_s_1, nlayer, "G_ul_s_1")
    G_lu_s_1 = _broadcast_layer_parameter(G_lu_s_1, nlayer, "G_lu_s_1")
    Cm_lu_s_1 = _broadcast_layer_parameter(Cm_lu_s_1, nlayer, "Cm_lu_s_1")

    for name, values in (
        ("j_nu", j_nu),
        ("A_ul_s_1", A_ul_s_1),
        ("B_ul_SI", B_ul_SI),
        ("B_lu_SI", B_lu_SI),
        ("Cm_ul_s_1", Cm_ul_s_1),
        ("Cm_lu_s_1", Cm_lu_s_1),
        ("Ce_ul_s_1", Ce_ul_s_1),
        ("Ce_lu_s_1", Ce_lu_s_1),
        ("G_ul_s_1", G_ul_s_1),
        ("G_lu_s_1", G_lu_s_1),
    ):
        if np.any(values < 0.0):
            raise ValueError(f"{name} must be non-negative")

    downward_rate_s_1 = A_ul_s_1 + B_ul_SI * j_nu + Cm_ul_s_1 + Ce_ul_s_1 + G_ul_s_1
    upward_rate_s_1 = B_lu_SI * j_nu + Cm_lu_s_1 + Ce_lu_s_1 + G_lu_s_1
    total_rate_s_1 = downward_rate_s_1 + upward_rate_s_1

    if np.any(total_rate_s_1 <= 0.0):
        raise ValueError("Each layer must have a positive total transition rate")

    n_l = np.array(coma_grid.density[:, lower_level], dtype=np.float64, copy=True)
    n_u = np.array(coma_grid.density[:, upper_level], dtype=np.float64, copy=True)
    n_total = n_l + n_u

    target_n_u = n_total * upward_rate_s_1 / total_rate_s_1
    target_n_l = n_total - target_n_u

    rng = np.random.default_rng(random_seed)
    residual = np.abs(n_u * downward_rate_s_1 - n_l * upward_rate_s_1)
    scale = np.maximum(n_total * total_rate_s_1, np.finfo(np.float64).tiny)
    converged = residual / scale <= tolerance

    iterations_used = 0
    for iteration in range(1, max_iterations + 1):
        if np.all(converged):
            iterations_used = iteration - 1
            break

        active_layers = np.where(~converged)[0]
        packet_fraction = rng.uniform(
            min_packet_fraction,
            max_packet_fraction,
            size=active_layers.shape[0],
        )
        delta_to_target = target_n_u[active_layers] - n_u[active_layers]
        transfer = packet_fraction * delta_to_target

        upward_mask = transfer > 0.0
        if np.any(upward_mask):
            upward_layers = active_layers[upward_mask]
            upward_transfer = np.minimum(
                transfer[upward_mask],
                n_l[upward_layers],
            )
            n_l[upward_layers] -= upward_transfer
            n_u[upward_layers] += upward_transfer

        downward_mask = transfer < 0.0
        if np.any(downward_mask):
            downward_layers = active_layers[downward_mask]
            downward_transfer = np.minimum(
                -transfer[downward_mask],
                n_u[downward_layers],
            )
            n_u[downward_layers] -= downward_transfer
            n_l[downward_layers] += downward_transfer

        n_l = np.maximum(n_l, 0.0)
        n_u = np.maximum(n_u, 0.0)

        residual = np.abs(n_u * downward_rate_s_1 - n_l * upward_rate_s_1)
        converged = residual / scale <= tolerance
        iterations_used = iteration
    else:
        iterations_used = max_iterations

    coma_grid.density[:, lower_level] = n_l
    coma_grid.density[:, upper_level] = n_u

    if coma_grid.nlevels == 2 and lower_level == 0 and upper_level == 1:
        total_density = coma_grid.total_number_density
        safe_total_density = np.maximum(total_density, np.finfo(np.float64).tiny)
        fractions = (n_u / safe_total_density)[:, np.newaxis]
        coma_grid.update_density_from_fractions(fractions)

    return {
        "n_lower": np.array(n_l, copy=True),
        "n_upper": np.array(n_u, copy=True),
        "target_n_lower": target_n_l,
        "target_n_upper": target_n_u,
        "upward_rate_s_1": upward_rate_s_1,
        "downward_rate_s_1": downward_rate_s_1,
        "relative_residual": residual / scale,
        "converged": converged,
        "iterations_used": iterations_used,
    }
