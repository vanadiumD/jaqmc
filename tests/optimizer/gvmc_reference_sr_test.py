# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from jaqmc.data import BatchedData, Data
from jaqmc.estimator import StreamingLossAndGrad
from jaqmc.optimizer.gvmc_reference_sr import (
    GVMCReferenceSROptimizer,
    _native_vmc_to_gvmc_gradient,
    minsr_solve,
    minsr_solve_gradient,
    minsr_solve_kacz,
)
from jaqmc.utils.func_transform import grad_maybe_complex


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


def test_native_gradient_adapter_removes_vmc_factor_two():
    native_gradient = jnp.array([2.0, -4.0, 0.5], dtype=jnp.float64)

    np.testing.assert_array_equal(
        _native_vmc_to_gvmc_gradient(native_gradient),
        jnp.array([1.0, -2.0, 0.25], dtype=jnp.float64),
    )


class LinearData(Data):
    features: jax.Array


def _linear_problem(n_states=2):
    params = {
        "weights": jnp.array(
            [[0.2, -0.1], [0.4, 0.3]], dtype=jnp.float64
        )[:n_states]
    }
    features = jnp.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 1.0], [-0.5, 0.2]],
            [[-1.0, 0.5], [1.0, -0.3]],
            [[0.2, -0.7], [0.4, 0.9]],
            [[0.8, 0.3], [-0.2, -0.6]],
            [[-0.4, -0.9], [0.7, 0.1]],
            [[1.1, -0.2], [-0.8, 0.5]],
            [[-0.6, 0.8], [0.3, -1.0]],
        ],
        dtype=jnp.float64,
    )[:, :n_states]
    data = BatchedData(LinearData(features=features), ["features"])

    def logpsi(p, sample):
        value = jnp.sum(p["weights"] * sample.features)
        phase = jnp.sum(p["weights"] * sample.features**2)
        return value + 0.2j * phase

    return params, data, logpsi


def _reference_score_and_force(params, data, logpsi, local_trace):
    scores = jax.vmap(lambda sample: grad_maybe_complex(logpsi)(params, sample))(
        data.data
    )
    flat_scores = jax.vmap(lambda tree: ravel_pytree(tree)[0])(scores)
    scale = jnp.sqrt(data.batch_size)
    jacobian = (flat_scores - jnp.mean(flat_scores, axis=0)) / scale
    force = (local_trace - jnp.mean(local_trace)) / scale
    return (
        jnp.concatenate((jnp.real(jacobian), jnp.imag(jacobian)), axis=0),
        jnp.concatenate((jnp.real(force), jnp.imag(force)), axis=0),
    )


def _streaming_native_gradient(params, data, logpsi, local_trace):
    estimator = StreamingLossAndGrad(
        loss_key="local_trace", clip_method="none", f_log_psi=logpsi
    )
    sums, _ = estimator.evaluate_batch_walkers(
        params,
        data,
        {"local_trace": local_trace},
        None,
        jax.random.key(0),
    )
    reduced = estimator.reduce(sums)
    final = estimator.finalize_stats(
        jax.tree.map(lambda x: x[None], reduced), None
    )
    return final["grads"]


def test_reference_optimizer_matches_direct_gvmc_single_step_with_factor_control():
    params, data, logpsi = _linear_problem()
    local_trace = jnp.array(
        [0.8 + 0.1j, -0.4 + 0.7j, 1.3 - 0.2j, -0.1 + 0.5j,
         0.6 - 0.8j, -0.7 + 0.2j, 1.1 + 0.4j, -0.2 - 0.6j],
        dtype=jnp.complex128,
    )
    lam0, lam1, learning_rate = 1e-2, 1.0, 0.1
    jacobian, force = _reference_score_and_force(
        params, data, logpsi, local_trace
    )
    native_grads = _streaming_native_gradient(
        params, data, logpsi, local_trace
    )
    native_gradient, _ = ravel_pytree(native_grads)
    gvmc_gradient = jacobian.T @ force
    direct = minsr_solve(jacobian, force, lam0=lam0, lam1=lam1)
    legacy = minsr_solve_gradient(
        jacobian, native_gradient, lam0=lam0, lam1=lam1
    )
    fixed = minsr_solve_gradient(
        jacobian,
        _native_vmc_to_gvmc_gradient(native_gradient),
        lam0=lam0,
        lam1=lam1,
    )

    optimizer = GVMCReferenceSROptimizer(
        learning_rate=learning_rate,
        lam0=lam0,
        lam1=lam1,
        mu=0.0,
        scale_lr_by_sqrt_n_states=False,
        f_log_psi=logpsi,
    )
    state = optimizer.init(params, batched_data=data)
    updates, new_state = optimizer.update(
        native_grads, state, params, batched_data=data
    )
    flat_updates, _ = ravel_pytree(updates)

    np.testing.assert_allclose(native_gradient, 2.0 * gvmc_gradient, rtol=1e-11)
    np.testing.assert_allclose(legacy, 2.0 * direct, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(fixed, direct, rtol=1e-9, atol=1e-11)
    legacy_ratio = jnp.linalg.norm(legacy) / jnp.linalg.norm(direct)
    fixed_ratio = jnp.linalg.norm(fixed) / jnp.linalg.norm(direct)
    legacy_cosine = jnp.vdot(legacy, direct) / (
        jnp.linalg.norm(legacy) * jnp.linalg.norm(direct)
    )
    fixed_cosine = jnp.vdot(fixed, direct) / (
        jnp.linalg.norm(fixed) * jnp.linalg.norm(direct)
    )
    np.testing.assert_allclose(legacy_ratio, 2.0, rtol=1e-9)
    np.testing.assert_allclose(fixed_ratio, 1.0, rtol=1e-9)
    np.testing.assert_allclose(legacy_cosine, 1.0, rtol=1e-9)
    np.testing.assert_allclose(fixed_cosine, 1.0, rtol=1e-9)
    np.testing.assert_allclose(new_state.previous_delta, direct, rtol=1e-9)
    np.testing.assert_allclose(flat_updates, -learning_rate * direct, rtol=1e-9)


def test_reference_optimizer_matches_direct_gvmc_kacz_trajectory():
    params, data, logpsi = _linear_problem()
    optimizer = GVMCReferenceSROptimizer(
        learning_rate=0.1,
        lam0=3e-3,
        lam1=0.7,
        mu=0.8,
        scale_lr_by_sqrt_n_states=False,
        f_log_psi=logpsi,
    )
    state = optimizer.init(params, batched_data=data)
    direct_previous = jnp.zeros_like(state.previous_delta)
    base_trace = jnp.array(
        [0.8 + 0.1j, -0.4 + 0.7j, 1.3 - 0.2j, -0.1 + 0.5j,
         0.6 - 0.8j, -0.7 + 0.2j, 1.1 + 0.4j, -0.2 - 0.6j]
    )

    for step in range(5):
        local_trace = base_trace + (0.03 * step) * jnp.arange(8) ** 2
        jacobian, force = _reference_score_and_force(
            params, data, logpsi, local_trace
        )
        direct = minsr_solve_kacz(
            jacobian,
            force,
            direct_previous,
            lam0=optimizer.lam0,
            lam1=optimizer.lam1,
            mu=optimizer.mu,
        )
        native_grads = _streaming_native_gradient(
            params, data, logpsi, local_trace
        )
        updates, state = optimizer.update(
            native_grads, state, params, batched_data=data
        )
        flat_updates, _ = ravel_pytree(updates)

        np.testing.assert_allclose(state.previous_delta, direct, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(flat_updates, -0.1 * direct, rtol=1e-9, atol=1e-11)
        np.testing.assert_array_equal(state.counter, step + 1)
        direct_previous = direct


def test_reference_optimizer_m1_matches_direct_sr_and_keeps_learning_rate():
    params, data, logpsi = _linear_problem(n_states=1)
    local_trace = jnp.array(
        [0.2 + 0.3j, -0.6 + 0.1j, 0.8 - 0.4j, 1.1 + 0.2j,
         -0.2 - 0.7j, 0.4 + 0.5j, -0.9 + 0.1j, 0.3 - 0.2j]
    )
    jacobian, force = _reference_score_and_force(
        params, data, logpsi, local_trace
    )
    direct = minsr_solve(jacobian, force, lam0=1e-2, lam1=1.0)
    native_grads = _streaming_native_gradient(
        params, data, logpsi, local_trace
    )
    optimizer = GVMCReferenceSROptimizer(
        learning_rate=0.07,
        lam0=1e-2,
        mu=0.0,
        scale_lr_by_sqrt_n_states=True,
        f_log_psi=logpsi,
    )
    state = optimizer.init(params, batched_data=data)
    updates, state = optimizer.update(
        native_grads, state, params, batched_data=data
    )
    flat_updates, _ = ravel_pytree(updates)

    np.testing.assert_allclose(state.previous_delta, direct, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(flat_updates, -0.07 * direct, rtol=1e-9, atol=1e-11)


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
