"""
Contains definitions of fluid models

The nonlinear dynamical systems here are defined in jax/numpy and augmented a
bit manually. The basic dynamical system residual has a block form
F(x, xt, g) = [Fq(x, xt, g), Fp(x, xt)]
x = [q, p]
xt = [qt, pt]
and where q and p stand for flow and pressure for a 1D fluid model
"""

# from typing
from multiprocessing.sharedctypes import Value
import numpy as np
import jax

from blockarray import blockvec as bla


from .base import DynamicalSystem
from ..equations.fluid import bernoulli
from ..jaxutils import (blockvec_to_dict, flatten_nested_dict)

# pylint: disable=missing-docstring

def create_dynamical_residual_class(Parent, res_type):
    class Residual(Parent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            if res_type == 'primal':
                self.__res = self._res
                self.__res_args = self.primals
            elif res_type == 'linearized':
                self.__res = self._dres
                self.__res_args = (*self.primals, self.tangents)
            else:
                raise ValueError(f"Unknown `res_type` of {res_type}")

        def assem_res(self):
            submats = self.__res(*self.__res_args)
            labels = self.state.labels
            submats, shape = flatten_nested_dict(submats, labels)
            return bla.BlockVector(submats, shape, labels)

        def assem_dres_dstate(self):
            submats = jax.jacfwd(self.__res, argnums=0)(*self.__res_args)
            labels = self.state.labels+self.state.labels
            submats, shape = flatten_nested_dict(submats, labels)
            return bla.BlockMatrix(submats, shape, labels)

        def assem_dres_dstatet(self):
            dresq_dq = np.diag(np.zeros(self.state['q'].size))
            dresq_dp = np.zeros((self.state['q'].size, self.state['p'].size))

            dresp_dp = np.diag(np.zeros(self.state['p'].size))
            dresp_dq = np.zeros((self.state['p'].size, self.state['q'].size))
            mats = [
                [dresq_dq, dresq_dp],
                [dresp_dq, dresp_dp]]
            labels = self.state.labels+self.state.labels
            return bla.BlockMatrix(mats, labels=labels)

        def assem_dres_dcontrol(self):
            submats = jax.jacfwd(self.__res, argnums=1)(*self.__res_args)
            labels = self.state.labels+self.control.labels
            submats, shape = flatten_nested_dict(submats, labels)
            return bla.BlockMatrix(submats, shape, labels)

        def assem_dres_dprops(self):
            submats = jax.jacfwd(self.__res, argnums=2)(*self.__res_args)
            labels = self.state.labels + self.props.labels
            submats, shape = flatten_nested_dict(submats, labels)
            return bla.BlockMatrix(submats, shape, labels)

    return Residual

class BaseFluid1DDynamicalSystem(DynamicalSystem):

    def __init__(self, s, res, state, control, props):
        self.s = s

        self._res = jax.jit(res)
        # self._res = res
        self._dres = lambda state, control, props, tangents: jax.jvp(res, (state, control, props), tangents)[1]

        self.state = bla.BlockVector(list(state.values()), labels=[list(state.keys())])
        self.dstate = self.state.copy()

        self.statet = self.state.copy()
        self.dstatet = self.statet.copy()

        self.control = bla.BlockVector(list(control.values()), labels=[list(control.keys())])
        self.dcontrol = self.control.copy()

        self.props = bla.BlockVector(list(props.values()), labels=[list(props.keys())])
        self.dprops = self.props.copy()

        self.primals = (
            blockvec_to_dict(self.state),
            blockvec_to_dict(self.control),
            blockvec_to_dict(self.props)
        )
        self.tangents = (
            blockvec_to_dict(self.dstate),
            blockvec_to_dict(self.dcontrol),
            blockvec_to_dict(self.dprops)
        )

        self.dstate[:] = 0.0
        self.dstatet[:] = 0.0
        self.dcontrol[:] = 0.0
        self.dprops[:] = 0.0

    def set_state(self, state):
        self.state[:] = state

    def set_statet(self, statet):
        self.statet[:] = statet

    def set_control(self, control):
        self.control[:] = control

    def set_props(self, props):
        self.props[:] = props


    def set_dstate(self, dstate):
        self.dstate[:] = dstate

    def set_dstatet(self, dstatet):
        self.dstatet[:] = dstatet

    def set_dcontrol(self, dcontrol):
        self.dcontrol[:] = dcontrol


class BaseBernoulliSmoothMinSep(BaseFluid1DDynamicalSystem):

    def __init__(self, s):
        _, (_state, _control, _props), res = bernoulli.BernoulliSmoothMinSep(s)
        super().__init__(s, res, _state, _control, _props)

BernoulliSmoothMinSep = create_dynamical_residual_class(
    BaseBernoulliSmoothMinSep,
    res_type='primal'
)

LinearizedBernoulliSmoothMinSep = create_dynamical_residual_class(
    BaseBernoulliSmoothMinSep,
    res_type='linearized'
)


class BaseBernoulliFixedSep(BaseFluid1DDynamicalSystem):

    def __init__(self, s, idx_sep=0):
        _, (_state, _control, _props), res = bernoulli.BernoulliFixedSep(s, idx_sep)
        super().__init__(s, res, _state, _control, _props)

BernoulliFixedSep = create_dynamical_residual_class(
    BaseBernoulliFixedSep,
    res_type='primal'
)

LinearizedBernoulliFixedSep = create_dynamical_residual_class(
    BaseBernoulliFixedSep,
    res_type='linearized'
)

class BaseBernoulliAreaRatioSep(BaseFluid1DDynamicalSystem):

    def __init__(self, s):
        _, (_state, _control, _props), res = bernoulli.BernoulliAreaRatioSep(s)
        super().__init__(s, res, _state, _control, _props)

BernoulliAreaRatioSep = create_dynamical_residual_class(
    BaseBernoulliAreaRatioSep,
    res_type='primal'
)

LinearizedBernoulliAreaRatioSep = create_dynamical_residual_class(
    BaseBernoulliAreaRatioSep,
    res_type='linearized'
)
