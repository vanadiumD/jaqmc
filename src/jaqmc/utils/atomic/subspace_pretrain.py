# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""State-aware adapters for native JaQMC orbital pretraining."""

from collections.abc import Mapping
from typing import Any, Protocol

import jax
from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator import FunctionEstimator
from jaqmc.estimator.base import Estimator
from jaqmc.utils import parallel_jax
from jaqmc.utils.subspace_linalg import stable_complex_logdet
from jaqmc.wavefunction.base import NumericWavefunctionEvaluate
from jaqmc.wavefunction.determinant_state import SubspaceSpec, take_replica_dynamic


class StateOrbitalReference(Protocol):
    """Reference that evaluates state-indexed orbitals and Slater amplitudes."""

    n_states: int

    def eval_orbitals(
        self,
        state_index: jax.Array,
        pos: jnp.ndarray,
        nspins: tuple[int, int],
    ) -> tuple[jnp.ndarray, jnp.ndarray]: ...

    def eval_slater(
        self,
        state_index: jax.Array,
        pos: jnp.ndarray,
        nspins: tuple[int, int],
    ) -> jnp.ndarray: ...


def make_subspace_pretrain_loss(
    orbitals_fn: NumericWavefunctionEvaluate,
    orbital_ref: StateOrbitalReference,
    spec: SubspaceSpec,
    nspins: tuple[int, int],
    *,
    full_det: bool,
) -> Estimator:
    """Match each component NN to its diagonal replica orbital reference.

    The loss evaluates state ``s`` only on replica ``s``. Consequently neural
    forward/gradient work scales as ``O(M)`` rather than ``O(M**2)``.
    """
    if not full_det:
        raise NotImplementedError(
            "Subspace TDA pretraining v1 supports full_det wavefunctions only."
        )
    if orbital_ref.n_states != spec.n_states:
        raise ValueError(
            "State reference and subspace sizes differ: "
            f"{orbital_ref.n_states} != {spec.n_states}."
        )
    states = jnp.arange(spec.n_states)

    def one_state_loss(params_s, state_index, determinant_data):
        physical_data = take_replica_dynamic(
            determinant_data, state_index, spec
        )
        target_alpha, target_beta = orbital_ref.eval_orbitals(
            state_index, physical_data.electrons, nspins
        )
        predicted = orbitals_fn(params_s, physical_data)
        na = target_alpha.shape[-2]
        nb = target_beta.shape[-2]
        target = jnp.block(
            [
                [
                    target_alpha,
                    jnp.zeros((na, nb), dtype=target_alpha.dtype),
                ],
                [
                    jnp.zeros((nb, na), dtype=target_beta.dtype),
                    target_beta,
                ],
            ]
        )
        return jnp.mean(jnp.abs(predicted - target) ** 2)

    def loss_fn(stacked_params: Params, determinant_data: Data) -> jnp.ndarray:
        losses = jax.vmap(one_state_loss, in_axes=(0, 0, None))(
            stacked_params, states, determinant_data
        )
        return jnp.mean(losses)

    loss_and_grad_fn = jax.value_and_grad(loss_fn, argnums=0)

    def evaluate(
        params: Params,
        data: Data,
        prev_walker_stats: Mapping[str, Any],
        state: None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], None]:
        del prev_walker_stats, rngs
        loss, grads = loss_and_grad_fn(parallel_jax.pvary(params), data)
        return {"loss": loss, "grads": grads}, state

    return FunctionEstimator(evaluate)


def make_subspace_reference_log_amplitude(
    orbital_ref: StateOrbitalReference,
    spec: SubspaceSpec,
    nspins: tuple[int, int],
):
    """Build the determinant-state HF/TDA reference sampling amplitude."""
    if orbital_ref.n_states != spec.n_states:
        raise ValueError(
            "State reference and subspace sizes differ: "
            f"{orbital_ref.n_states} != {spec.n_states}."
        )
    states = jnp.arange(spec.n_states)

    def one_replica(electrons):
        return jax.vmap(
            lambda state_index: orbital_ref.eval_slater(
                state_index, electrons, nspins
            )
        )(states)

    def reference_log_amplitude(data: Data) -> jnp.ndarray:
        log_matrix = jax.vmap(one_replica)(data.electrons)
        return jnp.real(stable_complex_logdet(log_matrix))

    return reference_log_amplitude
