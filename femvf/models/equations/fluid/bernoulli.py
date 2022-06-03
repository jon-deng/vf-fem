"""
This module contains the equations defining a quasi-steady Bernoulli fluid with
separation at a fixed location.
"""
import numpy as np
from jax import numpy as jnp
import jax

from .smoothapproximation import (wavg, smooth_min_weight)

def BernoulliFixedSep(s: np.ndarray, idx_sep: int=0):
    """
    Return quantities defining a fixed-separation point Bernoulli model
    """

    N = s.size

    # Separation coefficient to ensure pressures after the separation point
    # match the supraglottal pressure
    f = np.ones((N,))
    f[idx_sep+1:] = 0.0

    def bernoulli_q(asep, psub, psup, rho):
        """
        Return the flow rate based on Bernoulli
        """
        # print(asep.shape, psub.shape, psup.shape, rho.shape)
        flow_sign = jnp.sign(psub-psup)
        q = flow_sign * (2*jnp.abs(psub-psup)/rho)**0.5 * asep
        # print(q.shape)
        return q

    # @jax.jit
    def bernoulli_p(q, area, psub, psup, rho):
        """
        Return the pressure based on Bernoulli
        """
        p = psub - 1/2*rho*(q/area)**2
        return p

    def bernoulli_qp(area, psub, psup, rho):
        """
        Return Bernoulli flow and pressure
        """
        # print(idx_sep)
        asep = area[idx_sep]
        # ssep = s[idx_sep]
        q = bernoulli_q(asep, psub, psup, rho)
        p = bernoulli_p(q, area, psub, psup, rho)

        # Separation coefficient ensure pressures tends to zero after separation
        p = f * p + (1-f)*psup
        return q, p


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
        'rho_air': np.ones(1)
    }

    # @jax.jit
    def res(state, control, props):
        q, p = state['q'], state['p']
        area, psub, psup = control['area'], control['psub'], control['psup']

        q_, p_ = bernoulli_qp(area, psub, psup, props['rho_air'])
        # print(q.shape, p.shape, q_.shape, p_.shape)
        return {'q': q-q_, 'p': p-p_}

    return s, (_state, _control, _props), res


def BernoulliSmoothMinSep(s: jnp.ndarray):
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
        """
        Return a weighting representing cut-off at the separation point
        """
        # note that jnp.nn.sigmoid is used to model
        # return (1+jnp.exp((s-ssep)/zeta_sep))**-1
        # because of numerical stability
        # Using the normal formula results in nan for large exponents
        return jax.nn.sigmoid(-1*(s-ssep)/zeta_sep)

    # @jax.jit
    def sigmoid(s, zeta=1):
        """
        Return the sigmoid function evaluated at s

        This function:
        approaches 0 as s -> -inf
        approaches 1 as s -> +inf
        """
        return 1/(1+jnp.exp(-s/zeta))

    # @jax.jit
    def bernoulli_qp(area, psub, psup, rho, zeta_min, zeta_sep):
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

    N = s.size
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
        'rho_air': np.ones(1),
        'zeta_sep': np.ones(1),
        'zeta_min': np.ones(1)
    }

    def res(state, control, props):
        q, p = state['q'], state['p']
        area, psub, psup = control['area'], control['psub'], control['psup']

        q_, p_ = bernoulli_qp(area, psub, psup, props['rho_air'], props['zeta_min'], props['zeta_sep'])
        return {'q': q-q_, 'p': p-p_}

    return s, (_state, _control, _props), res
