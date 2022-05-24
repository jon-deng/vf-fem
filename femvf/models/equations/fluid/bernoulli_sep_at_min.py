"""
This module contains the equations defining a quasi-steady Bernoulli fluid with
separation at the minimum area
"""
import numpy as np
from jax import numpy as jnp
import jax

# from blockarray import linalg as bla

# from .base import DynamicalSystem

## To disable jax:
# jnp = np

# pylint: disable=abstract-method
# @jax.jit
def bernoulli_q(asep, psub, psup, rho):
    """
    Return the flow rate based on Bernoulli
    """
    flow_sign = jnp.sign(psub-psup)
    q = flow_sign * (2*jnp.abs(psub-psup)/rho)**0.5 * asep
    return q

# @jax.jit
def bernoulli_p(q, area, psub, psup, rho):
    """
    Return the pressure based on Bernoulli
    """
    p = psub - 1/2*rho*(q/area)**2
    return p

def coeff_sep(s, ssep, zeta_sep):
    # note that jnp.nn.sigmoid is used to model
    # return (1+jnp.exp((s-ssep)/zeta_sep))**-1
    # because of numerical stability
    # Using the normal formula results in nan for large exponents
    return jax.nn.sigmoid(-1*(s-ssep)/zeta_sep)


# @jax.jit
def wavg(s, f, w, axis=-1):
    """
    Return the weighted average of f(s) over s with weights w(s)
    """
    return jnp.trapz(f*w, s, axis=axis)/jnp.trapz(w, s, axis=axis)

# @jax.jit
def smooth_min_weight(f, zeta=1, axis=-1):
    """
    Return a smooth minimum from a set of values f

    This evaluates according to $\exp{-\frac{f}{\zeta}}$ and is normalized so
    that the maximum weight has a value of 1.0.
    """
    # w = jnp.exp(-f/zeta)
    log_w = -f/zeta
    # To prevent overflow for the largest weight, normalize the maximum log-weight to 0
    # This ensures the maximum weight is 1.0 while smaller weights will underflow to 0.0
    log_w = log_w - np.max(log_w, axis=axis, keepdims=True)
    return jnp.exp(log_w)

# @jax.jit
def sigmoid(s, zeta=1):
    """
    Return the sigmoid function evaluated at s

    This function:
    approaches 0 as s -> -inf
    approaches 1 as s -> +inf
    """
    return 1/(1+jnp.exp(-s/zeta))

@jax.jit
def bernoulli_qp(area, s, psub, psup, rho, zeta_min, zeta_sep):
    """
    Return Bernoulli flow and pressure
    """
    wmin = smooth_min_weight(area, zeta_min)
    amin = wavg(s, area, wmin)
    smin = wavg(s, s, wmin)
    # print(wmin, amin, smin)

    asep = amin
    ssep = smin
    q = bernoulli_q(asep, psub, psup, rho)
    p = bernoulli_p(q, area, psub, psup, rho)

    # Separation coefficient ensure pressures tends to zero after separation
    f_sep = coeff_sep(s, ssep, zeta_sep)
    p = f_sep * p
    return q, p

# Use this weird combination of primals/tangent parameters because later we
# want to compute the jacobian wrt area
def dbernoulli_qp(area, s, psub, psup, rho, zeta_min, zeta_sep, tangents):
    """
    Return linearization of Bernoulli flow and pressure
    """
    primals = (area, s, psub, psup, rho, zeta_min, zeta_sep)
    return jax.jvp(bernoulli_qp, primals, tangents)[1]

dbernoulli_qp_darea = jax.jacfwd(bernoulli_qp, argnums=0)
ddbernoulli_qp_darea = jax.jacfwd(dbernoulli_qp, argnums=0)

dbernoulli_qp_dpsub = jax.jacfwd(bernoulli_qp, argnums=2)
ddbernoulli_qp_dpsub = jax.jacfwd(dbernoulli_qp, argnums=2)

dbernoulli_qp_dpsup = jax.jacfwd(bernoulli_qp, argnums=3)
ddbernoulli_qp_dpsup = jax.jacfwd(dbernoulli_qp, argnums=3)
