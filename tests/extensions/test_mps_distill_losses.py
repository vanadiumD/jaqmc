"""Gauge invariance tests for complex logpsi distillation."""
import jax
import jax.numpy as jnp
import numpy as np

from jaqmc.extensions.mps_distill.losses import (
    centered_logpsi_loss,
    teacher_sample_infidelity,
)


def test_centered_loss_global_complex_gauge():
    target = jnp.asarray([[0.2 + 0.1j], [-0.4 - 0.2j], [0.6 + 0.5j]])
    student = target + (1.5 + 0.75j)
    assert np.asarray(centered_logpsi_loss(student, target)) < 1e-6


def test_teacher_fidelity_global_gauge():
    target = jnp.asarray([[0.2 + 0.1j], [-0.4 - 0.2j], [0.6 + 0.5j]])
    student = target + (0.5 + 1.2j)
    assert np.asarray(teacher_sample_infidelity(student, target)) < 1e-6


def test_loss_grad_is_finite():
    target = jnp.asarray([[0.2 + 0.1j], [-0.4 - 0.2j], [0.6 + 0.5j]])
    value, grad = jax.value_and_grad(
        lambda x: centered_logpsi_loss(target + x, target)
    )(jnp.asarray(0.2))
    assert np.isfinite(np.asarray(value))
    assert np.isfinite(np.asarray(grad))
