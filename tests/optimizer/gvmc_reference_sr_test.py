# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from jaqmc.data import BatchedData, Data
from jaqmc.optimizer.gvmc_reference_sr import (
    GVMCReferenceSROptimizer,
    minsr_solve,
    minsr_solve_gradient,
    minsr_solve_kacz,
)


def _reference_matrix(jacobian, lam0, lam1):
    n = jacobian.shape[0]
    scale = jnp.linalg.norm(jacobian) ** 2 / n
    return (
        jacobian @ jnp.conj(jacobian.T)
        + scale * (lam1 / n)
        + lam0 * jnp.eye(n)
    )


def test_minsr_solve_matches_gvmc_source_equation():
    jacobian = jnp.array(
        [[1.0 + 0.2j, 0.3], [-0.4j, 1.2], [-1.0 + 0.2j, -1.5]],
        dtype=jnp.complex128,
    )
    force = jnp.array([0.4 + 0.1j, -0.2j, -0.4 + 0.1j])
    lam0, lam1 = 2e-3, 0.7

    actual = minsr_solve(jacobian, force, lam0=lam0, lam1=lam1)
    expected = jnp.conj(jacobian.T) @ jnp.linalg.solve(
        _reference_matrix(jacobian, lam0, lam1), force
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)


def test_minsr_kacz_matches_reference_continuation():
    jacobian = jnp.array(
        [[0.5, -0.2, 0.4], [1.0, 0.3, -0.1], [-1.5, -0.1, -0.3]],
        dtype=jnp.float64,
    )
    force = jnp.array([0.2, -0.5, 0.3], dtype=jnp.float64)
    previous = jnp.array([0.1, -0.2, 0.05], dtype=jnp.float64)
    mu = 0.8

    actual = minsr_solve_kacz(
        jacobian, force, previous, lam0=1e-3, lam1=1.0, mu=mu
    )
    residual = force - mu * jacobian @ previous
    expected = (
        jnp.conj(jacobian.T)
        @ jnp.linalg.solve(_reference_matrix(jacobian, 1e-3, 1.0), residual)
        + mu * previous
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_gradient_form_is_equivalent_for_centered_force():
    key_j, key_b = jax.random.split(jax.random.key(3))
    jacobian = jax.random.normal(key_j, (6, 4), dtype=jnp.float64)
    jacobian -= jnp.mean(jacobian, axis=0, keepdims=True)
    force = jax.random.normal(key_b, (6,), dtype=jnp.float64)
    force -= jnp.mean(force)
    gradient = jnp.conj(jacobian.T) @ force

    direct = minsr_solve(jacobian, force, lam0=3e-3, lam1=1.0)
    from_gradient = minsr_solve_gradient(
        jacobian, gradient, lam0=3e-3, lam1=1.0
    )

    np.testing.assert_allclose(from_gradient, direct, rtol=1e-9, atol=1e-11)


class LinearData(Data):
    features: jax.Array


def test_reference_optimizer_implements_native_optimizer_protocol():
    params = {
        "weights": jnp.array(
            [[0.2, -0.1], [0.4, 0.3]], dtype=jnp.float64
        )
    }
    data = BatchedData(
        LinearData(
            features=jnp.array(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[0.5, 1.0], [-0.5, 0.2]],
                    [[-1.0, 0.5], [1.0, -0.3]],
                    [[0.2, -0.7], [0.4, 0.9]],
                ],
                dtype=jnp.float64,
            )
        ),
        ["features"],
    )
    grads = {"weights": jnp.array([[0.3, -0.2], [0.1, 0.4]])}

    def logpsi(p, sample):
        value = jnp.sum(p["weights"] * sample.features)
        return value + 0.2j * value

    optimizer = GVMCReferenceSROptimizer(
        learning_rate=0.1,
        lam0=1e-2,
        mu=0.0,
        f_log_psi=logpsi,
    )
    state = optimizer.init(params, batched_data=data)
    updates, new_state = jax.jit(optimizer.update)(
        grads, state, params, batched_data=data
    )

    assert jax.tree.structure(updates) == jax.tree.structure(params)
    assert new_state.counter == 1
    assert all(
        np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(updates)
    )
    flat_update, _ = ravel_pytree(updates)
    assert jnp.linalg.norm(flat_update) > 0


def test_reference_optimizer_rejects_multi_device(monkeypatch):
    optimizer = GVMCReferenceSROptimizer(f_log_psi=lambda p, d: p["w"].sum())
    params = {"w": jnp.ones((2, 1))}
    data = BatchedData(LinearData(features=jnp.ones((2, 2, 1))), ["features"])
    monkeypatch.setattr(jax, "device_count", lambda: 2)

    with pytest.raises(NotImplementedError, match="single-device"):
        optimizer.init(params, batched_data=data)
