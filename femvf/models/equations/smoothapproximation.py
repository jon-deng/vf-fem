"""
Contains definitions of smooth approximation functions
"""

import jax
from jax import numpy as jnp, scipy as jsp


# @jax.jit
def wavg(s, f, w, axis=-1):
    """
    Return the weighted average of f(s) over s with weights w(s)
    """
    return jsp.integrate.trapezoid(f * w, s, axis=axis) / jsp.integrate.trapezoid(w, s, axis=axis)


# @jax.jit
def smooth_min_weight(f, zeta=1, axis=-1):
    """
    Return a smooth minimum from a set of values f

    This evaluates according to $\exp{-\frac{f}{\zeta}}$ and is normalized so
    that the maximum weight has a value of 1.0.
    """
    # The smooth minimum can be found by negating `zeta` in the soft maximum
    # weighting formula. Use `jax.nn.softmax` since they handle numerical
    # stability issues in the exponential terms.
    return jax.nn.softmax(-f / zeta, axis=-1)
