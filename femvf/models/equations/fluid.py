"""
This module contains the equations defining a quasi-steady Bernoulli fluid with
separation at a fixed location.
"""

from typing import Callable, Tuple, Mapping

import numpy as np
from numpy.typing import ArrayLike
from jax import numpy as jnp
import jax

from .smoothapproximation import (wavg, smooth_min_weight)
from . import base

## Common bernoulli fluid functions
def bernoulliq_from_psub_psep(psub, psep, area_sub, area_sep, rho):
    """
    Return the flow rate based on Bernoulli
    """
    flow_sign = jnp.sign(psub-psep)

    q = flow_sign * (2/rho*jnp.abs(psub-psep) / (area_sep**-2 - area_sub**-2))**0.5
    return q

def bernoullip_from_q_psep(qsub, psep, area_sep, area, rho):
    """
    Return the pressure based on Bernoulli
    """
    return psep + 1/2*rho*qsub**2*(area_sep**-2 - area**-2)

ResArgs = Tuple[Mapping[str, ArrayLike], ...]
ResReturn = Mapping[str, ArrayLike]

## Fluid residual classes
class JaxResidual(base.BaseResidual):
    """
    Representation of a (non-linear) residual in `JAX`
    """

    def __init__(
            self,
            res: Callable[[ResArgs], ResReturn],
            res_args: ResArgs
        ):

        self._res = res
        self._res_args = res_args

    @property
    def res(self):
        return self._res

    @property
    def res_args(self):
        return self._res_args

class PredefinedJaxResidual(JaxResidual):
    """
    Predefined `JaxResidual`
    """

    def __init__(
            self,
            mesh: ArrayLike,
            *args, **kwargs
        ):
        res, res_args = self._make_residual(mesh, *args, **kwargs)
        super().__init__(res, res_args)

    def _make_residual(self, mesh, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

class BernoulliFixedSep(PredefinedJaxResidual):

    def _make_residual(self, mesh, idx_sep=0):
        return _BernoulliFixedSep(mesh, idx_sep=idx_sep)

def _BernoulliFixedSep(s: np.ndarray, idx_sep: int=0):
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
        area_sep = area[idx_sep]
        # ssep = s[idx_sep]
        q = bernoulliq_from_psub_psep(psub, psup, jnp.inf, area_sep, rho)
        p = bernoullip_from_q_psep(q, psup, area_sep, area, rho)

        # Separation coefficient ensure pressures tends to zero after separation
        p = f * p + (1-f)*psup
        return q, p

    def res(state, control, prop):
        q, p = state['q'], state['p']
        area, psub, psup = control['area'], control['psub'], control['psup']

        q_, p_ = bernoulli_qp(area, psub, psup, prop['rho_air'])
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

    return res, (_state, _control, _props)

class BernoulliSmoothMinSep(PredefinedJaxResidual):

    def _make_residual(self, mesh):
        return _BernoulliSmoothMinSep(mesh)

def _BernoulliSmoothMinSep(s: jnp.ndarray):

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
        q = bernoulliq_from_psub_psep(psub, psup, jnp.inf, asep, rho)
        p = bernoullip_from_q_psep(q, psup, asep, area, rho)

        # Separation coefficient ensure pressures tends to zero after separation
        f_sep = coeff_sep(s, ssep, zeta_sep)
        p = f_sep * p
        return q, p

    def res(state, control, prop):
        q, p = state['q'], state['p']
        area, psub, psup = control['area'], control['psub'], control['psup']

        q_, p_ = bernoulli_qp(area, psub, psup, prop['rho_air'], prop['zeta_min'], prop['zeta_sep'])
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

    return res, (_state, _control, _props)

class BernoulliAreaRatioSep(PredefinedJaxResidual):

    def _make_residual(self, mesh):
        return _BernoulliAreaRatioSep(mesh)

def _BernoulliAreaRatioSep(s: jnp.ndarray):
    N = s.size

    s = jnp.array(s)

    def bernoulli_qp(area, psub, psup, rho, r_sep, area_lb):
        """
        Return Bernoulli flow and pressure
        """
        area = jnp.maximum(area, area_lb)
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

        q = bernoulliq_from_psub_psep(psub, psup, jnp.inf, asep, rho)
        p = bernoullip_from_q_psep(q, psup, asep, area, rho)

        # Separation coefficient ensure pressures tends to zero after separation
        p = f_sep * p + (1-f_sep)*psup
        return q, p

    def res(state, control, prop):
        q, p = state['q'], state['p']
        area, psub, psup = control['area'], control['psub'], control['psup']

        q_, p_ = bernoulli_qp(area, psub, psup, prop['rho_air'], prop['r_sep'], prop['area_lb'])
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
        'r_sep': np.ones(1),
        'area_lb': np.zeros(1)
    }

    return res, (_state, _control, _props)

class BernoulliFlowFixedSep(PredefinedJaxResidual):

    def _make_residual(self, mesh, idx_sep=0):
        return _BernoulliFlowFixedSep(mesh, idx_sep=idx_sep)

def _BernoulliFlowFixedSep(s: np.ndarray, idx_sep: int=0):
    """
    Return quantities defining a fixed-separation point Bernoulli model with constant flow
    """

    N = s.size

    # Separation coefficient to ensure pressures after the separation point
    # match the supraglottal pressure
    f = np.ones((N,))
    f[idx_sep+1:] = 0.0

    def bernoulli_qp(area, qsub, psup, rho):
        """
        Return Bernoulli pressure
        """
        area_sep = area[idx_sep]
        p = bernoullip_from_q_psep(qsub, psup, area_sep, area, rho)

        # Separation coefficient ensure pressures tends to zero after separation
        p = f * p + (1-f)*psup
        return qsub, p

    def res(state, control, prop):
        q, p = state['q'], state['p']
        area, qsub, psup = control['area'], control['qsub'], control['psup']

        q_, p_ = bernoulli_qp(area, qsub, psup, prop['rho_air'])
        # print(q.shape, p.shape, q_.shape, p_.shape)
        return {'q': q-q_, 'p': p-p_}

    # Key functions/variables that have to be exported
    _state = {
        'q': np.ones(1),
        'p': np.ones(N)
    }

    _control = {
        'area': np.ones(N),
        'qsub': np.ones(1),
        'psup': np.ones(1)
    }

    _props = {
        'rho_air': np.ones(1)
    }

    return res, (_state, _control, _props)
