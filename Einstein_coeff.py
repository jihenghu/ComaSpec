"""Utilities for Einstein radiative coefficients.

This module uses the convention where radiative rates are written directly as
``B * J_nu`` with the mean intensity ``J_nu``.
"""

from __future__ import annotations

from Constants import PLANCK_CONSTANT, SPEED_OF_LIGHT


def einstein_coeffs_B(
    nu_Hz: float = 556.936e9,
    A_ul_s_1: float = 3.456e-3,
    g_u: float = 9.0,
    g_l: float = 9.0,
) -> tuple[float, float]:
    """Compute Einstein coefficients for a transition between levels u and l.

    Parameters
    ----------
    nu_Hz
        Transition frequency in Hz.
    A_ul_s_1
        Spontaneous emission coefficient from upper to lower level in s^-1.
    g_u
        Degeneracy of the upper level.
    g_l
        Degeneracy of the lower level.
    Returns
    -------
    tuple[float, float]
        ``(B_ul_SI, B_lu_SI)`` for the ``J_nu`` convention.
    """

    A_ul_s_1 = float(A_ul_s_1)
    g_u = float(g_u)
    g_l = float(g_l)
    nu_Hz = float(nu_Hz)

    if A_ul_s_1 < 0.0:
        raise ValueError("A_ul_s_1 must be non-negative")
    if g_u <= 0.0:
        raise ValueError("g_u must be positive")
    if g_l <= 0.0:
        raise ValueError("g_l must be positive")
    if nu_Hz <= 0.0:
        raise ValueError("nu_Hz must be positive")

    B_ul_SI = SPEED_OF_LIGHT**2 * A_ul_s_1 / (
        2.0 * PLANCK_CONSTANT * nu_Hz**3      # unit m2 J-1 S-1
    )
    B_lu_SI = (g_u / g_l) * B_ul_SI

    return B_ul_SI, B_lu_SI
