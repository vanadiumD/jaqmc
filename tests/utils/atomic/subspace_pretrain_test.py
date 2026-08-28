# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
from jax import numpy as jnp

from jaqmc.data import Data
from jaqmc.utils.atomic.subspace_pretrain import (
    make_subspace_pretrain_loss,
    make_subspace_reference_log_amplitude,
)
from jaqmc.wavefunction.determinant_state import SubspaceSpec


class ReplicaData(Data):
    electrons: jax.Array


class MockStateReference:
    def __init__(self, n_states):
        self.n_states = n_states

    def eval_orbitals(self, state_index, pos, nspins):
        del nspins
        target = pos[0, 0] + state_index
        return target[None, None], jnp.zeros((0, 0))

    def eval_slater(self, state_index, pos, nspins):
        del nspins
        x = pos[0, 0]
        return 0.1 * x + state_index * (0.2 * x**2 + 0.3)


def test_subspace_pretrain_loss_has_finite_statewise_gradients():
    spec = SubspaceSpec(2)
    reference = MockStateReference(2)
    data = ReplicaData(electrons=jnp.array([[[1.0]], [[2.0]]]))
    params = {"w": jnp.array([0.2, -0.4])}

    def orbitals_fn(params_s, physical_data):
        value = params_s["w"] * physical_data.electrons[0, 0]
        return value[None, None]

    estimator = make_subspace_pretrain_loss(
        orbitals_fn,
        reference,
        spec,
        (1, 0),
        full_det=True,
    )
    stats, _ = estimator.evaluate_single_walker(
        params, data, {}, None, jax.random.key(0)
    )

    assert jnp.isscalar(stats["loss"])
    assert jnp.isfinite(stats["loss"])
    assert stats["grads"]["w"].shape == (2,)
    assert jnp.all(jnp.isfinite(stats["grads"]["w"]))
    assert not np.isclose(stats["grads"]["w"][0], stats["grads"]["w"][1])


def test_reference_determinant_log_amplitude_is_replica_permutation_invariant():
    spec = SubspaceSpec(2)
    reference = MockStateReference(2)
    log_amplitude = make_subspace_reference_log_amplitude(
        reference, spec, (1, 0)
    )
    data = ReplicaData(electrons=jnp.array([[[0.4]], [[1.3]]]))
    swapped = ReplicaData(electrons=data.electrons[::-1])

    actual = log_amplitude(data)
    permuted = log_amplitude(swapped)

    assert jnp.isfinite(actual)
    np.testing.assert_allclose(actual, permuted, rtol=1e-6, atol=1e-6)


def test_m1_reference_path_is_finite_without_tda_states():
    spec = SubspaceSpec(1)
    log_amplitude = make_subspace_reference_log_amplitude(
        MockStateReference(1), spec, (1, 0)
    )
    data = ReplicaData(electrons=jnp.array([[[0.7]]]))

    assert jnp.isfinite(log_amplitude(data))
