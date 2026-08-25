"""Barrier transformations shared by low-speed ISSf-CBF environments."""

import jax.numpy as jnp

from .typing import Array


def compress_safe_barrier(
    alpha: Array,
    alpha_threshold: Array,
    kappa: Array,
) -> Array:
    """Compress only the safe side of ``alpha - alpha_threshold``.

    Let ``q = alpha - alpha_threshold``.  The returned barrier is

    ``q``                              when ``q <= 0``;
    ``kappa * log(1 + q / kappa)``    when ``q > 0``.

    It has the same zero level set and the same unsafe-side values as the
    original barrier.  On the safe side its derivative with respect to alpha
    is ``kappa / (kappa + q)``.  This prevents a distant road boundary's
    heading derivative from growing in direct proportion to its distance.
    The hard minimum used to obtain alpha is deliberately left unchanged.
    """
    q = alpha - alpha_threshold
    kappa = jnp.maximum(jnp.asarray(kappa, dtype=alpha.dtype), 1e-6)
    safe_value = kappa * jnp.log1p(jnp.maximum(q, 0.0) / kappa)
    return jnp.where(q <= 0.0, q, safe_value)


def safe_barrier_derivative(
    alpha: Array,
    alpha_threshold: Array,
    kappa: Array,
) -> Array:
    """Return d(compress_safe_barrier)/d(alpha) on each smooth branch."""
    q = alpha - alpha_threshold
    kappa = jnp.maximum(jnp.asarray(kappa, dtype=alpha.dtype), 1e-6)
    return jnp.where(q <= 0.0, 1.0, kappa / (kappa + q))
