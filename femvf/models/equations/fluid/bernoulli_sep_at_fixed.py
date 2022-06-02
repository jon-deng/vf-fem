"""
This module contains the equations defining a quasi-steady Bernoulli fluid with
separation at a fixed location.
"""
import numpy as np
from jax import numpy as jnp
import jax


def BernoulliFixedSep(s: jnp.ndarray, idx_sep: int):
    """
    Return quantities defining a fixed-separation point Bernoulli model
    """

    N = s.size

    # Separation coefficient to ensure pressures after the separation point
    # match the supraglottal pressure
    f = np.ones((N,))
    f[np.array(idx_sep)+1:] = 0

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

    # @jax.jit
    def bernoulli_qp(area, psub, psup, rho):
        """
        Return Bernoulli flow and pressure
        """
        asep = area[idx_sep]
        # ssep = s[idx_sep]
        q = bernoulli_q(asep, psub, psup, rho)
        p = bernoulli_p(q, area, psub, psup, rho)

        # Separation coefficient ensure pressures tends to zero after separation
        p = f * p + (1-f)*psup
        return q, p

    # Use this weird combination of primals/tangent parameters because later we
    # want to compute the jacobian wrt area
    def dbernoulli_qp(area, psub, psup, rho, tangents):
        """
        Return linearization of Bernoulli flow and pressure
        """
        primals = (area, psub, psup, rho)
        return jax.jvp(bernoulli_qp, primals, tangents)[1]


    # Key functions/variables that have to be exported
    _state = {
        'q': np.ones(1),
        'p': np.ones(N)
    }

    _control = {
        'area': np.ones(N),
        'psub': np.ones(1),
        'psup': np.ones(1)
    }

    _props = {
        'rho': np.ones(N)
    }

    def res(state, control, props):
        q, p = state.values()
        area, psub, psup = control.values()

        q_, p_ = bernoulli_qp(area, psub, psup, props['rho'])
        return {'q': q-q_, 'p': p-p_}

    dres_dstate = jax.jacfwd(res, argnums=0)
    dres_dcontrol = jax.jacfwd(res, argnums=1)
    dres_dprops = jax.jacfwd(res, argnums=2)

    return s, (_state, _control, _props), res
