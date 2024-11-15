"""
This module contains the equations defining a quasi-steady Bernoulli fluid with
separation at a fixed location.
"""

from numpy.typing import NDArray

import numpy as np
from jax import numpy as jnp
import jax

from .equation.smoothapproximation import wavg, smooth_min_weight
from .base import JaxResidual, PredefinedJaxResidual


## Common bernoulli fluid functions
def bernoulliq_from_psub_psep(psub, psep, area_sub, area_sep, rho):
    """
    Return the flow rate based on Bernoulli
    """
    flow_sign = jnp.sign(psub - psep)

    q = (
        flow_sign
        * (2 / rho * jnp.abs(psub - psep) / (area_sep**-2 - area_sub**-2)) ** 0.5
    )
    return q


def bernoullip_from_q_psep(qsub, psep, area_sep, area, rho):
    """
    Return the pressure based on Bernoulli
    """
    return psep + 1 / 2 * rho * qsub**2 * (area_sep**-2 - area**-2)


## Fluid residual classes

class BernoulliFixedSep(PredefinedJaxResidual):

    def _make_residual(self, mesh, idx_sep=0):
        return _BernoulliFixedSep(mesh, idx_sep=idx_sep)


def _BernoulliFixedSep(s: np.ndarray, idx_sep: int = 0):
    """
    Return quantities defining a fixed-separation point Bernoulli model
    """

    # The number of fluid points
    SHAPE_FLUID = s.shape[:-1]
    N_FLUID = int(np.prod(SHAPE_FLUID))
    N_TOTAL = s.size

    def reshape_args(shape_fluid, state, control, prop):
        ret_state = state
        ret_state['q'] = ret_state['q'].reshape(*shape_fluid, 1)
        ret_state['p'] = ret_state['p'].reshape(*shape_fluid, -1)

        ret_control = control
        ret_control['area'] = ret_control['area'].reshape(*shape_fluid, -1)
        ret_control['psub'] = ret_control['psub'].reshape(*shape_fluid, 1)
        ret_control['psup'] = ret_control['psup'].reshape(*shape_fluid, 1)

        ret_prop = prop
        ret_prop['rho_air'] = ret_prop['rho_air'].reshape(*shape_fluid, 1)

        return ret_state, ret_control, ret_prop

    # Separation coefficient to ensure pressures after the separation point
    # match the supraglottal pressure
    f = np.ones(s.shape)
    f[..., idx_sep + 1 :] = 0.0

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
        p = f * p + (1 - f) * psup
        return q, p

    def res(state, control, prop):
        state, control, prop = reshape_args(SHAPE_FLUID, state, control, prop)
        q, p = state['q'], state['p']
        area, psub, psup = control['area'], control['psub'], control['psup']

        q_, p_ = bernoulli_qp(area, psub, psup, prop['rho_air'])
        # print(q.shape, p.shape, q_.shape, p_.shape)
        return {'q': (q - q_).reshape(-1), 'p': (p - p_).reshape(-1)}

    # Key functions/variables that have to be exported
    _state = {'q': np.ones(N_FLUID), 'p': np.ones(N_TOTAL)}

    _control = {
        'area': np.ones(N_TOTAL),
        'psub': np.ones(N_FLUID),
        'psup': np.ones(N_FLUID),
    }

    _props = {'rho_air': np.ones(N_FLUID)}

    return res, (_state, _control, _props)


class BernoulliSmoothMinSep(PredefinedJaxResidual):

    def _make_residual(self, mesh):
        return _BernoulliSmoothMinSep(mesh)


def _BernoulliSmoothMinSep(s: jnp.ndarray):

    # The number of fluid points
    SHAPE_FLUID = s.shape[:-1]
    N_FLUID = int(np.prod(SHAPE_FLUID))
    N_TOTAL = s.size

    def reshape_args(shape_fluid, state, control, prop):
        ret_state = state
        ret_state['q'] = ret_state['q'].reshape(*shape_fluid, 1)
        ret_state['p'] = ret_state['p'].reshape(*shape_fluid, -1)

        ret_control = control
        ret_control['area'] = ret_control['area'].reshape(*shape_fluid, -1)
        ret_control['psub'] = ret_control['psub'].reshape(*shape_fluid, 1)
        ret_control['psup'] = ret_control['psup'].reshape(*shape_fluid, 1)

        ret_prop = prop
        ret_prop['rho_air'] = ret_prop['rho_air'].reshape(*shape_fluid, 1)
        ret_prop['zeta_min'] = ret_prop['zeta_min'].reshape(*shape_fluid, 1)
        ret_prop['zeta_sep'] = ret_prop['zeta_min'].reshape(*shape_fluid, 1)
        # ret_prop['r_sep'] = ret_prop['r_sep'].reshape(*shape_fluid, 1)
        # ret_prop['area_lb'] = ret_prop['area_lb'].reshape(*shape_fluid, 1)

        return ret_state, ret_control, ret_prop

    def coeff_sep(s, ssep, zeta_sep):
        """
        Return a weighting representing cut-off at the separation point
        """
        # note that jnp.nn.sigmoid is used to model
        # return (1+jnp.exp((s-ssep)/zeta_sep))**-1
        # because of numerical stability
        # Using the normal formula results in nan for large exponents
        return jax.nn.sigmoid(-1 * (s - ssep) / zeta_sep)

    # @jax.jit
    def bernoulli_qp(area, psub, psup, rho, zeta_min, zeta_sep):
        """
        Return Bernoulli flow and pressure
        """
        wmin = smooth_min_weight(area, zeta_min, axis=-1)
        # Add empty reduced dimension to ensure axes line up
        amin = wavg(s, area, wmin, axis=-1)[..., None]
        smin = wavg(s, s, wmin, axis=-1)[..., None]
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
        state, control, prop = reshape_args(SHAPE_FLUID, state, control, prop)
        q, p = state['q'], state['p']
        area, psub, psup = control['area'], control['psub'], control['psup']

        q_, p_ = bernoulli_qp(
            area, psub, psup, prop['rho_air'], prop['zeta_min'], prop['zeta_sep']
        )
        return {'q': (q - q_).reshape(-1), 'p': (p - p_).reshape(-1)}

    N = s.size
    # Key functions/variables that have to be exported
    _state = {'q': np.ones(N_FLUID), 'p': np.ones(N_TOTAL)}

    _control = {
        'area': np.ones(N_TOTAL),
        'psub': np.ones(N_FLUID),
        'psup': np.ones(N_FLUID),
    }

    _props = {
        'rho_air': np.ones(N_FLUID),
        'zeta_sep': np.ones(N_FLUID),
        'zeta_min': np.ones(N_FLUID),
    }

    return res, (_state, _control, _props)


class BernoulliAreaRatioSep(PredefinedJaxResidual):

    def _make_residual(self, mesh):
        return _BernoulliAreaRatioSep(mesh)


def _BernoulliAreaRatioSep(s: jnp.ndarray):
    # The number of fluid points
    SHAPE_FLUID = s.shape[:-1]
    N_FLUID = int(np.prod(SHAPE_FLUID))
    N_TOTAL = s.size

    def reshape_args(shape_fluid, state, control, prop):
        ret_state = state
        ret_state['q'] = ret_state['q'].reshape(*shape_fluid, 1)
        ret_state['p'] = ret_state['p'].reshape(*shape_fluid, -1)

        ret_control = control
        ret_control['area'] = ret_control['area'].reshape(*shape_fluid, -1)
        ret_control['psub'] = ret_control['psub'].reshape(*shape_fluid, 1)
        ret_control['psup'] = ret_control['psup'].reshape(*shape_fluid, 1)

        ret_prop = prop
        ret_prop['rho_air'] = ret_prop['rho_air'].reshape(*shape_fluid, 1)
        ret_prop['r_sep'] = ret_prop['r_sep'].reshape(*shape_fluid, 1)
        ret_prop['area_lb'] = ret_prop['area_lb'].reshape(*shape_fluid, 1)

        return ret_state, ret_control, ret_prop

    def bernoulli_qp(
        area: NDArray,
        psub: NDArray,
        psup: NDArray,
        rho: NDArray,
        r_sep: NDArray,
        area_lb: NDArray,
    ):
        """
        Return Bernoulli flow and pressure
        """
        area = jnp.maximum(area, area_lb)
        amin = jnp.min(area, axis=-1, keepdims=True)
        idx_min = jnp.argmax(area == amin, axis=-1, keepdims=True)
        # smin = s[idx_min]
        smin = jnp.take_along_axis(s, idx_min, axis=-1)

        asep = r_sep * amin
        # To find the separation point, work only with coordinates downstream
        # of the minimum area
        _area = jnp.where(s >= smin, area, jnp.nan)
        idx_sep = jnp.nanargmin(jnp.abs(_area - asep), axis=-1, keepdims=True)
        # ssep = s[idx_sep]
        ssep = jnp.take_along_axis(s, idx_sep, axis=-1)

        f_sep = jnp.array(s < ssep, dtype=jnp.float64)

        q = bernoulliq_from_psub_psep(psub, psup, jnp.inf, asep, rho)
        p = bernoullip_from_q_psep(q, psup, asep, area, rho)

        # Separation coefficient ensure pressures tends to zero after separation
        p = f_sep * p + (1 - f_sep) * psup
        return q, p

    def res(state, control, prop):
        state, control, prop = reshape_args(SHAPE_FLUID, state, control, prop)
        q, p = state['q'], state['p']
        area, psub, psup = control['area'], control['psub'], control['psup']

        q_, p_ = bernoulli_qp(
            area, psub, psup, prop['rho_air'], prop['r_sep'], prop['area_lb']
        )
        return {'q': (q - q_).reshape(-1), 'p': (p - p_).reshape(-1)}

    # Key functions/variables that have to be exported
    _state = {'q': np.ones(N_FLUID), 'p': np.ones(N_TOTAL)}

    _control = {
        'area': np.ones(N_TOTAL),
        'psub': np.ones(N_FLUID),
        'psup': np.ones(N_FLUID),
    }

    _props = {
        'rho_air': np.ones(N_FLUID),
        'r_sep': np.ones(N_FLUID),
        'area_lb': np.zeros(N_FLUID),
    }

    return res, (_state, _control, _props)


class BernoulliFlowFixedSep(PredefinedJaxResidual):

    def _make_residual(self, mesh, idx_sep=0):
        return _BernoulliFlowFixedSep(mesh, idx_sep=idx_sep)


def _BernoulliFlowFixedSep(s: np.ndarray, idx_sep: int = 0):
    """
    Return quantities defining a fixed-separation point Bernoulli model with constant flow
    """
    # The number of fluid points
    SHAPE_FLUID = s.shape[:-1]
    N_FLUID = int(np.prod(SHAPE_FLUID))
    N_TOTAL = s.size

    def reshape_args(shape_fluid, state, control, prop):
        ret_state = state
        ret_state['q'] = ret_state['q'].reshape(*shape_fluid, 1)
        ret_state['p'] = ret_state['p'].reshape(*shape_fluid, -1)

        ret_control = control
        ret_control['area'] = ret_control['area'].reshape(*shape_fluid, -1)
        ret_control['qsub'] = ret_control['qsub'].reshape(*shape_fluid, 1)
        ret_control['psup'] = ret_control['psup'].reshape(*shape_fluid, 1)

        ret_prop = prop
        ret_prop['rho_air'] = ret_prop['rho_air'].reshape(*shape_fluid, 1)
        # ret_prop['r_sep'] = ret_prop['r_sep'].reshape(*shape_fluid, 1)
        # ret_prop['area_lb'] = ret_prop['area_lb'].reshape(*shape_fluid, 1)

        return ret_state, ret_control, ret_prop

    # Separation coefficient to ensure pressures after the separation point
    # match the supraglottal pressure
    f = np.ones(s.shape)
    f[..., idx_sep + 1 :] = 0.0

    def bernoulli_qp(area, qsub, psup, rho):
        """
        Return Bernoulli pressure
        """
        area_sep = area[idx_sep]
        p = bernoullip_from_q_psep(qsub, psup, area_sep, area, rho)

        # Separation coefficient ensure pressures tends to zero after separation
        p = f * p + (1 - f) * psup
        return qsub, p

    def res(state, control, prop):
        state, control, prop = reshape_args(SHAPE_FLUID, state, control, prop)
        q, p = state['q'], state['p']
        area, qsub, psup = control['area'], control['qsub'], control['psup']

        q_, p_ = bernoulli_qp(area, qsub, psup, prop['rho_air'])
        # print(q.shape, p.shape, q_.shape, p_.shape)
        return {'q': (q - q_).reshape(-1), 'p': (p - p_).reshape(-1)}

    # Key functions/variables that have to be exported
    _state = {'q': np.ones(N_FLUID), 'p': np.ones(N_TOTAL)}

    _control = {
        'area': np.ones(N_TOTAL),
        'qsub': np.ones(N_FLUID),
        'psup': np.ones(N_FLUID),
    }

    _props = {'rho_air': np.ones(N_FLUID)}

    return res, (_state, _control, _props)
