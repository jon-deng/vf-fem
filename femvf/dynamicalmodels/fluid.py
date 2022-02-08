"""
Contains definitions of fluid models

The nonlinear dynamical systems here are defined in jax/numpy and augmented a 
bit manually. The basic dynamical system residual has a block form
F(x, xt, g) = [Fq(x, xt, g), Fp(x, xt)]
x = [q, p]
xt = [qt, pt]
and where q and p stand for flow and pressure for a 1D fluid model
"""

import numpy as np
from jax import numpy as jnp
import jax

from blocklinalg import linalg as bla

from .base import DynamicalSystem

# pylint: disable=abstract-method

def bernoulli_q(asep, psub, psup, rho):
    """
    Return the flow rate based on Bernoulli
    """
    q = (2*(psub-psup)/rho)**0.5 * asep
    return q

def bernoulli_p(q, area, ssep, psub, psup, rho):
    """
    Return the pressure based on Bernoulli
    """
    p = psub - 1/2*rho*(q/area)**2
    return p

def wavg(s, f, w):
    """
    Return the weighted average of f(s) over s with weights w(s)
    """
    return jnp.trapz(f*w, s)/jnp.trapz(w, s)

def smooth_min_weight(f, zeta=1):
    """
    Return a smooth minimum from a set of values f
    """
    # w = jnp.exp(-f/zeta)
    log_w = -f/zeta
    log_w = log_w - jnp.min(log_w) # normalize the log weight so the minimum value's weight is 1
    return jnp.exp(log_w)

def sigmoid(s, zeta=1):
    """
    Return the sigmoid function evaluated at s

    This function:
    approaches 0 as s -> -inf
    approaches 1 as s -> +inf
    """
    return 1/(1+jnp.exp(-s/zeta))

def qp(area, s, psub, psup, rho, zeta_min, zeta_sep):
    wmin = smooth_min_weight(area, zeta_min)
    amin = wavg(s, area, wmin)

    asep = amin
    q = bernoulli_q(asep, psub, psup, rho)
    p = bernoulli_p(q, area, ssep, psub, psup, rho)
    return q, p

class FluidDynamicalSystem(DynamicalSystem):

    def __init__(self, s):
        self.s = s

        N = self.s.size
        self.state = bla.BlockVec([np.zeros(1), np.zeros(N)], ['q', 'p'])
        self.statet = bla.BlockVec([np.zeros(1), np.zeros(N)], ['q', 'p'])
        self.icontrol = bla.BlockVec([np.ones(N)], ['area'])

        properties_vec = ['psub', 'psub', 'rho_air', 'zeta_min', 'zeta_sep']
        self.properties = bla.BlockVec(
            [np.ones(1) for i in range(len(properties_vec))], properties_vec)

