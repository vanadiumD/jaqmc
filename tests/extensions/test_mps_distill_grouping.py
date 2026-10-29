"""Unit tests for CI spin grouping, independent of block2."""
import numpy as np
import pytest

from jaqmc.extensions.mps_distill.grouping import (
    group_spin_determinants,
    grouped_amplitude,
)


def test_grouping_preserves_complex_amplitude():
    alpha = np.array([[0, 2], [0, 2], [1, 3], [0, 2]])
    beta = np.array([[0, 1], [1, 2], [0, 1], [0, 1]])
    coeff = np.array([1 + 2j, -0.5j, 3 - 1j, 0.1])
    grouped = group_spin_determinants(alpha, beta, coeff)
    assert grouped.unique_counts == (2, 2)
    rng = np.random.default_rng(19)
    a = rng.normal(size=(2, 7)) + 1j * rng.normal(size=(2, 7))
    b = rng.normal(size=(2, 7)) + 1j * rng.normal(size=(2, 7))
    expected = sum(
        coeff[k] * a[grouped.alpha_index[k]] * b[grouped.beta_index[k]]
        for k in range(len(coeff))
    )
    np.testing.assert_allclose(grouped_amplitude(grouped, a, b), expected)


def test_reject_repeated_orbital():
    with pytest.raises(ValueError, match="Repeated"):
        group_spin_determinants(
            np.array([[1, 1]]), np.array([[0, 2]]), np.array([1.0])
        )
