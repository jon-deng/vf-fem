"""
Contains definitions of fluid models

The nonlinear dynamical systems here are defined in jax/numpy and augmented a
bit manually. The basic dynamical system residual has a block form
F(x, xt, g) = [Fq(x, xt, g), Fp(x, xt)]
x = [q, p]
xt = [qt, pt]
and where q and p stand for flow and pressure for a 1D fluid model
"""

from typing import Tuple, Callable, Mapping, Union, Any, List
from numpy.typing import ArrayLike
import numpy as np
import jax

from blockarray import blockvec as bv


from .base import BaseDynamicalModel
from ..equations import bernoulli
from ..jaxutils import (blockvec_to_dict, flatten_nested_dict)

# pylint: disable=missing-docstring
DictVec = Mapping[str, ArrayLike]

JaxResidualArgs = Tuple[DictVec, DictVec, DictVec]
JaxLinearizedResidualArgs = Tuple[DictVec, DictVec, DictVec, Tuple[DictVec, DictVec, DictVec]]

JaxResidualFunction = Callable[JaxResidualArgs, DictVec]
JaxLinearizedResidualFunction = Callable[JaxLinearizedResidualArgs, DictVec]


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
            return bv.BlockVector(submats, shape, labels)

        def assem_dres_dstate(self):
            submats = jax.jacfwd(self.__res, argnums=0)(*self.__res_args)
            labels = self.state.labels+self.state.labels
            submats, shape = flatten_nested_dict(submats, labels)
            return bv.BlockMatrix(submats, shape, labels)

        def assem_dres_dstatet(self):
            dresq_dq = np.diag(np.zeros(self.state['q'].size))
            dresq_dp = np.zeros((self.state['q'].size, self.state['p'].size))

            dresp_dp = np.diag(np.zeros(self.state['p'].size))
            dresp_dq = np.zeros((self.state['p'].size, self.state['q'].size))
            mats = [
                [dresq_dq, dresq_dp],
                [dresp_dq, dresp_dp]]
            labels = self.state.labels+self.state.labels
            return bv.BlockMatrix(mats, labels=labels)

        def assem_dres_dcontrol(self):
            submats = jax.jacfwd(self.__res, argnums=1)(*self.__res_args)
            labels = self.state.labels+self.control.labels
            submats, shape = flatten_nested_dict(submats, labels)
            return bv.BlockMatrix(submats, shape, labels)

        def assem_dres_dprop(self):
            submats = jax.jacfwd(self.__res, argnums=2)(*self.__res_args)
            labels = self.state.labels + self.prop.labels
            submats, shape = flatten_nested_dict(submats, labels)
            return bv.BlockMatrix(submats, shape, labels)

    return Residual

Residual = Tuple[ArrayLike, Tuple[bv.BlockVector, bv.BlockVector, bv.BlockVector], Callable]

Test = Union[JaxResidualFunction, JaxLinearizedResidualFunction]

class DynamicalFluidModelOps:

    def __init__(self, residual: Residual):
        s, (state, control, prop), _ = residual
        self.s = s

        self.state = bv.BlockVector(list(state.values()), labels=[list(state.keys())])

        self.statet = self.state.copy()

        self.control = bv.BlockVector(list(control.values()), labels=[list(control.keys())])

        self.prop = bv.BlockVector(list(prop.values()), labels=[list(prop.keys())])

    @property
    def residual(self) -> Union[JaxResidualFunction, JaxLinearizedResidualFunction]:
        return self._res

    @property
    def residual_args(self) -> Union[JaxResidualArgs, JaxLinearizedResidualArgs]:
        return self._res_args

    def set_state(self, state):
        self.state[:] = state

    def set_statet(self, statet):
        self.statet[:] = statet

    def set_control(self, control):
        self.control[:] = control

    def set_prop(self, prop):
        self.prop[:] = prop

    def assem_res(self):
        submats = self.residual(*self.residual_args)
        labels = self.state.labels
        submats, shape = flatten_nested_dict(submats, labels)
        return bv.BlockVector(submats, shape, labels)

    def assem_dres_dstate(self):
        submats = jax.jacfwd(self.residual, argnums=0)(*self.residual_args)
        labels = self.state.labels+self.state.labels
        submats, shape = flatten_nested_dict(submats, labels)
        return bv.BlockMatrix(submats, shape, labels)

    def assem_dres_dstatet(self):
        dresq_dq = np.diag(np.zeros(self.state['q'].size))
        dresq_dp = np.zeros((self.state['q'].size, self.state['p'].size))

        dresp_dp = np.diag(np.zeros(self.state['p'].size))
        dresp_dq = np.zeros((self.state['p'].size, self.state['q'].size))
        mats = [
            [dresq_dq, dresq_dp],
            [dresp_dq, dresp_dp]
        ]
        labels = self.state.labels+self.state.labels
        return bv.BlockMatrix(mats, labels=labels)

    def assem_dres_dcontrol(self):
        submats = jax.jacfwd(self.residual, argnums=1)(*self.residual_args)
        labels = self.state.labels+self.control.labels
        submats, shape = flatten_nested_dict(submats, labels)
        return bv.BlockMatrix(submats, shape, labels)

    def assem_dres_dprop(self):
        submats = jax.jacfwd(self.residual, argnums=2)(*self.residual_args)
        labels = self.state.labels + self.prop.labels
        submats, shape = flatten_nested_dict(submats, labels)
        return bv.BlockMatrix(submats, shape, labels)


class Dynamical1DFluid(DynamicalFluidModelOps, BaseDynamicalModel):

    def __init__(self, residual: Residual):

        super().__init__(residual)

        primals = (
            blockvec_to_dict(self.state),
            blockvec_to_dict(self.control),
            blockvec_to_dict(self.prop)
        )
        *_, res = residual

        self._res = jax.jit(res)
        self._res_args = primals

class LinearizedDynamical1DFluid(DynamicalFluidModelOps, BaseDynamicalModel):

    def __init__(self, residual: Residual):

        super().__init__(residual)

        self.dstate = self.state.copy()
        self.dstatet = self.statet.copy()
        self.dcontrol = self.control.copy()
        self.dprop = self.prop.copy()

        self.dstate[:] = 0.0
        self.dstatet[:] = 0.0
        self.dcontrol[:] = 0.0
        self.dprop[:] = 0.0

        primals = (
            blockvec_to_dict(self.state),
            blockvec_to_dict(self.control),
            blockvec_to_dict(self.prop)
        )
        tangents = (
            blockvec_to_dict(self.dstate),
            blockvec_to_dict(self.dcontrol),
            blockvec_to_dict(self.dprop)
        )
        *_, res = residual

        self._res = (
            lambda state, control, prop, tangents:
            jax.jvp(jax.jit(res), (state, control, prop), tangents)[1]
        )
        self._res_args = (*primals, tangents)

    def set_dstate(self, dstate):
        self.dstate[:] = dstate

    def set_dstatet(self, dstatet):
        self.dstatet[:] = dstatet

    def set_dcontrol(self, dcontrol):
        self.dcontrol[:] = dcontrol

    def set_dprop(self, dprop):
        self.dprop[:] = dprop


class PredefinedDynamical1DFluid(Dynamical1DFluid):

    def __init__(self, s: ArrayLike, **kwargs):
        residual = self._make_residual(s, **kwargs)
        super().__init__(residual)

    def _make_residual(self, s, **kwargs):
        raise NotImplementedError()

class PredefinedLinearizedDynamical1DFluid(LinearizedDynamical1DFluid):

    def __init__(self, s: ArrayLike, **kwargs):
        residual = self._make_residual(s, **kwargs)
        super().__init__(residual)

    def _make_residual(self, s: ArrayLike, **kwargs):
        raise NotImplementedError()

## Predefined models
class BernoulliSmoothMinSep(PredefinedDynamical1DFluid):

    def _make_residual(self, s):
        return bernoulli.BernoulliSmoothMinSep(s)

class LinearizedBernoulliSmoothMinSep(PredefinedLinearizedDynamical1DFluid):

    def _make_residual(self, s):
        return bernoulli.BernoulliSmoothMinSep(s)


class BernoulliFixedSep(PredefinedDynamical1DFluid):

    def _make_residual(self, s, idx_sep=0):
        return bernoulli.BernoulliFixedSep(s, idx_sep)

class LinearizedBernoulliFixedSep(PredefinedLinearizedDynamical1DFluid):

    def _make_residual(self, s, idx_sep=0):
        return bernoulli.BernoulliFixedSep(s, idx_sep)


class BernoulliAreaRatioSep(PredefinedDynamical1DFluid):

    def _make_residual(self, s):
        return bernoulli.BernoulliAreaRatioSep(s)

class LinearizedBernoulliAreaRatioSep(PredefinedLinearizedDynamical1DFluid):

    def _make_residual(self, s):
        return bernoulli.BernoulliAreaRatioSep(s)
