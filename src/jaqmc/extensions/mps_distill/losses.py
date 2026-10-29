"""Gauge-invariant losses for complex-log-wavefunction distillation.

Input arrays have shape [batch, nstates]. These functions operate on one
global batch; distributed execution must all-reduce the underlying moments.
"""
import jax
import jax.numpy as jnp


def centered_logpsi_loss(
    student_logpsi: jax.Array,
    teacher_logpsi: jax.Array,
    phase_weight: float = 0.2,
) -> jax.Array:
    """Remove a separate global complex scale from each state."""
    target = jax.lax.stop_gradient(teacher_logpsi)
    if student_logpsi.shape != target.shape or student_logpsi.ndim != 2:
        raise ValueError("Expected matching complex logpsi [batch, states].")
    delta = student_logpsi - target
    amplitude = jnp.real(delta)
    amplitude -= jnp.mean(amplitude, axis=0, keepdims=True)
    amp_loss = jnp.mean(amplitude**2)
    phase = jnp.exp(1j * jnp.imag(delta))
    phase_loss = jnp.mean(1.0 - jnp.abs(jnp.mean(phase, axis=0)))
    return amp_loss + phase_weight * phase_loss


def teacher_sample_infidelity(
    student_logpsi: jax.Array,
    teacher_logpsi: jax.Array,
    eps: float = 1e-12,
) -> jax.Array:
    """Estimate 1 - squared overlap from |teacher|^2 samples only.

    Do not use this directly for mixed-sampler data or local multi-GPU batches.
    """
    target = jax.lax.stop_gradient(teacher_logpsi)
    if student_logpsi.shape != target.shape or student_logpsi.ndim != 2:
        raise ValueError("Expected matching complex logpsi [batch, states].")
    delta = student_logpsi - target
    shift = jax.lax.stop_gradient(
        jnp.max(jnp.real(delta), axis=0, keepdims=True)
    )
    ratio = jnp.exp(delta - shift)
    numerator = jnp.abs(jnp.mean(ratio, axis=0)) ** 2
    denominator = jnp.mean(jnp.abs(ratio) ** 2, axis=0)
    return jnp.mean(1.0 - numerator / (denominator + eps))
