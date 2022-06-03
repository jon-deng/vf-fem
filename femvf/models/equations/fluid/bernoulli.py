"""
This module contains the equations defining a quasi-steady Bernoulli fluid with
separation at a fixed location.
"""
import numpy as np
from jax import numpy as jnp
import jax

from .smoothapproximation import (wavg, smooth_min_weight)

## Common bernoulli fluid functions
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



## Fluid model definitions
def BernoulliFixedSep(s: np.ndarray, idx_sep: int=0):
    """
    Return quantities defining a fixed-separation point Bernoulli model
    """

    N = s.size

    # Separation coefficient to ensure pressures after the separation point
    # match the supraglottal pressure
    f = np.ones((N,))
    f[idx_sep+1:] = 0.0

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

    def res(state, control, props):
        q, p = state['q'], state['p']
        area, psub, psup = control['area'], control['psub'], control['psup']

        q_, p_ = bernoulli_qp(area, psub, psup, props['rho_air'])
        # print(q.shape, p.shape, q_.shape, p_.shape)
        return {'q': q-q_, 'p': p-p_}

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

    return s, (_state, _control, _props), res


def BernoulliSmoothMinSep(s: jnp.ndarray):

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

    def res(state, control, props):
        q, p = state['q'], state['p']
        area, psub, psup = control['area'], control['psub'], control['psup']

        q_, p_ = bernoulli_qp(area, psub, psup, props['rho_air'], props['zeta_min'], props['zeta_sep'])
        return {'q': q-q_, 'p': p-p_}

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

    return s, (_state, _control, _props), res


def BernoulliAreaRatioSep(s: jnp.ndarray):
    N = s.size

    s = jnp.array(s)

    def bernoulli_qp(area, psub, psup, rho, r_sep):
        """
        Return Bernoulli flow and pressure
        """
        amin = jnp.min(area)
        idx_min = jnp.min(jnp.argmax(area == amin))
        smin = s[idx_min]

        asep = r_sep*amin
        # To find the separation point, work only with coordinates downstream
        # of the minimum area
        _area = jnp.where(s>=smin, area, jnp.nan)
        idx_sep = jnp.min(jnp.nanargmin(jnp.abs(_area-asep)))
        ssep = s[idx_sep]

        f_sep = jnp.array(s < ssep, dtype=jnp.float64)

        q = bernoulli_q(asep, psub, psup, rho)
        p = bernoulli_p(q, area, psub, psup, rho)

        # Separation coefficient ensure pressures tends to zero after separation
        p = f_sep * p + (1-f_sep)*psup
        return q, p

    def res(state, control, props):
        q, p = state['q'], state['p']
        area, psub, psup = control['area'], control['psub'], control['psup']

        q_, p_ = bernoulli_qp(area, psub, psup, props['rho_air'], props['r_sep'])
        return {'q': q-q_, 'p': p-p_}

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
        'r_sep': np.ones(1)
    }

    return s, (_state, _control, _props), res
