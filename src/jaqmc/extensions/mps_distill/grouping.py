"""Group repeated spin determinants in a selected-CI expansion.

No assumptions about the coefficient normalization or the MPS retained weight
are made here. Occupied orbital indices must be in the same fixed MO ordering.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpinGroupedCI:
    alpha_occ: np.ndarray
    beta_occ: np.ndarray
    alpha_index: np.ndarray
    beta_index: np.ndarray
    coefficients: np.ndarray

    @property
    def ndeterminants(self) -> int:
        return int(self.coefficients.size)

    @property
    def unique_counts(self) -> tuple[int, int]:
        return len(self.alpha_occ), len(self.beta_occ)


def group_spin_determinants(
    alpha_occ: np.ndarray,
    beta_occ: np.ndarray,
    coefficients: np.ndarray,
) -> SpinGroupedCI:
    """Convert K occupation pairs into unique alpha/beta occupation tables.

    The ordering of each occupation tuple is significant and is preserved.
    Do not sort occupation indices here: sorting can change determinant signs.
    """
    alpha = np.asarray(alpha_occ, dtype=np.int32)
    beta = np.asarray(beta_occ, dtype=np.int32)
    coeff = np.asarray(coefficients, dtype=np.complex128)
    if alpha.ndim != 2 or beta.ndim != 2:
        raise ValueError("Occupation arrays must have shape [K, nspin].")
    if alpha.shape[0] != beta.shape[0] or alpha.shape[0] != coeff.size:
        raise ValueError("alpha, beta and coefficient counts differ.")
    if coeff.ndim != 1 or not np.all(np.isfinite(coeff)):
        raise ValueError("Coefficients must be a finite vector.")
    if np.any(alpha < 0) or np.any(beta < 0):
        raise ValueError("Occupied orbital indices must be nonnegative.")
    for occ in (alpha, beta):
        if occ.shape[1] and np.any(np.diff(np.sort(occ, axis=1), axis=1) == 0):
            raise ValueError("Repeated orbital within a Slater determinant.")
    def encode(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        unique = []
        mapping = {}
        indices = np.empty(len(rows), dtype=np.int32)
        for k, row in enumerate(rows):
            key = tuple(int(x) for x in row)
            if key not in mapping:
                mapping[key] = len(unique)
                unique.append(row.copy())
            indices[k] = mapping[key]
        return (
            np.stack(unique) if unique else rows[:0].copy(),
            indices,
        )
    au, ai = encode(alpha)
    bu, bi = encode(beta)
    return SpinGroupedCI(au, bu, ai, bi, coeff.copy())


def grouped_amplitude(
    grouped: SpinGroupedCI, alpha_values: np.ndarray, beta_values: np.ndarray
) -> np.ndarray:
    """Contract precomputed unique determinants, with trailing batch dimensions."""
    a = np.asarray(alpha_values)
    b = np.asarray(beta_values)
    if a.shape[0] != len(grouped.alpha_occ) or b.shape[0] != len(grouped.beta_occ):
        raise ValueError("Unique determinant evaluation count mismatch.")
    return np.sum(
        grouped.coefficients.reshape((-1,) + (1,) * (a.ndim - 1))
        * a[grouped.alpha_index]
        * b[grouped.beta_index],
        axis=0,
    )
