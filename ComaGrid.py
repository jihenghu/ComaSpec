"""Grid container for coma physical properties and level populations."""

from __future__ import annotations

import numpy as np

from Index_map import IDEN, ILEV0, ITEMP, IVEL


class ComaGrid:
    """Store coma grid properties and non-LTE level populations.

    Parameters
    ----------
    temperature
        Grid temperature profile with shape ``(ngrid,)``.
    total_number_density
        Total number density profile with shape ``(ngrid,)``.
    velocity
        Velocity profile with shape ``(ngrid,)``.
    fractions
        Excited-state fractional populations with shape ``(ngrid, nlevels - 1)``.
        Level 0 is treated as the ground state and is filled by normalization:
        ``fraction_0 = 1 - sum(fraction_1 ... fraction_n-1)``.
    """

    def __init__(
        self,
        temperature: np.ndarray,
        total_number_density: np.ndarray,
        velocity: np.ndarray,
        fractions: np.ndarray,
    ) -> None:
        temperature = self._as_profile_array(temperature, "temperature")
        total_number_density = self._as_profile_array(
            total_number_density, "total_number_density"
        )
        velocity = self._as_profile_array(velocity, "velocity")
        fractions = self._as_fraction_array(fractions)

        ngrid = temperature.shape[0]
        self._validate_same_length(total_number_density, ngrid, "total_number_density")
        self._validate_same_length(velocity, ngrid, "velocity")
        self._validate_same_length(fractions, ngrid, "fractions")

        self.ngrid = ngrid
        self.nprops = 3
        self.nlevels = fractions.shape[1] + 1

        self.props = np.empty((self.nprops, self.ngrid), dtype=np.float64, order="C")
        self.props[ITEMP, :] = temperature
        self.props[IDEN, :] = total_number_density
        self.props[IVEL, :] = velocity

        self.density = np.empty(
            (self.ngrid, self.nlevels), dtype=np.float64, order="C"
        )
        self.update_density_from_fractions(fractions)

    @property
    def temperature(self) -> np.ndarray:
        return self.props[ITEMP]

    @property
    def total_number_density(self) -> np.ndarray:
        return self.props[IDEN]

    @property
    def velocity(self) -> np.ndarray:
        return self.props[IVEL]

    def update_density_from_fractions(self, fractions: np.ndarray) -> None:
        """Update ``density`` from excited-state fractions.

        Parameters
        ----------
        fractions
            Fractional populations for levels ``1..nlevels-1`` with shape
            ``(ngrid, nlevels - 1)``.
        """

        fractions = self._as_fraction_array(fractions)
        expected_shape = (self.ngrid, self.nlevels - 1)
        if fractions.shape != expected_shape:
            raise ValueError(
                "fractions must have shape "
                f"{expected_shape}, got {fractions.shape}"
            )

        excited_fraction_sum = np.sum(fractions, axis=1)
        if np.any(excited_fraction_sum > 1.0):
            raise ValueError(
                "fractions are not physically valid: "
                "sum of excited-state fractions exceeds 1 in at least one grid cell"
            )

        total_number_density = self.total_number_density
        self.density[:, ILEV0 + 1 :] = (
            total_number_density[:, np.newaxis] * fractions
        )
        self.density[:, ILEV0] = (
            total_number_density - np.sum(self.density[:, ILEV0 + 1 :], axis=1)
        )

    @staticmethod
    def _as_profile_array(values: np.ndarray, name: str) -> np.ndarray:
        array = np.array(values, dtype=np.float64, order="C", copy=False)
        if array.ndim != 1:
            raise ValueError(f"{name} must be a 1D float64 array")
        return array

    @staticmethod
    def _as_fraction_array(values: np.ndarray) -> np.ndarray:
        array = np.array(values, dtype=np.float64, order="C", copy=False)
        if array.ndim != 2:
            raise ValueError("fractions must be a 2D float64 array")
        if array.shape[1] < 1:
            raise ValueError("fractions must contain at least one excited level")
        if np.any(array < 0.0):
            raise ValueError("fractions must be non-negative")
        return array

    @staticmethod
    def _validate_same_length(array: np.ndarray, expected_length: int, name: str) -> None:
        if array.shape[0] != expected_length:
            raise ValueError(
                f"{name} must have length {expected_length}, got {array.shape[0]}"
            )
