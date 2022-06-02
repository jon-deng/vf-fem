"""
This module contains the equations defining a quasi-steady Bernoulli fluid with
separation at a fixed location.
"""
import numpy as np
from jax import numpy as jnp
import jax

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

def coeff_sep(idx_sep: int, n: int):
    """
    Return a weighting representing cut-off at the separation point

    Parameters
    ----------
    idx_sep :
        Node where separation occurs
    n :
        Number of nodes in the mesh
    """
    f = np.ones((n,))
    f[np.array(idx_sep)+1:] = 0
    return f

# @jax.jit
def bernoulli_qp(area, s, psub, psup, rho, idx_sep):
    """
    Return Bernoulli flow and pressure
    """
    asep = area[idx_sep]
    # ssep = s[idx_sep]
    q = bernoulli_q(asep, psub, psup, rho)
    p = bernoulli_p(q, area, psub, psup, rho)

    # Separation coefficient ensure pressures tends to zero after separation
    f_sep = coeff_sep(idx_sep, area.size)
    p = f_sep * p
    return q, p

# Use this weird combination of primals/tangent parameters because later we
# want to compute the jacobian wrt area
def dbernoulli_qp(area, s, psub, psup, rho, idx_sep, tangents):
    """
    Return linearization of Bernoulli flow and pressure
    """
    primals = (area, s, psub, psup, rho, idx_sep)
    return jax.jvp(bernoulli_qp, primals, tangents)[1]

dbernoulli_qp_darea = jax.jacfwd(bernoulli_qp, argnums=0)
ddbernoulli_qp_darea = jax.jacfwd(dbernoulli_qp, argnums=0)

dbernoulli_qp_dpsub = jax.jacfwd(bernoulli_qp, argnums=2)
ddbernoulli_qp_dpsub = jax.jacfwd(dbernoulli_qp, argnums=2)

dbernoulli_qp_dpsup = jax.jacfwd(bernoulli_qp, argnums=3)
ddbernoulli_qp_dpsup = jax.jacfwd(dbernoulli_qp, argnums=3)
