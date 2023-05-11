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


from .base import BaseDynamicalModel, BaseLinearizedDynamicalModel
from ..equations import bernoulli
from ..jaxutils import (blockvec_to_dict, flatten_nested_dict)

# pylint: disable=missing-docstring
DictVec = Mapping[str, ArrayLike]

JaxResidualArgs = Tuple[DictVec, DictVec, DictVec]
JaxLinearizedResidualArgs = Tuple[DictVec, DictVec, DictVec, Tuple[DictVec, DictVec, DictVec]]

JaxResidualFunction = Callable[JaxResidualArgs, DictVec]
JaxLinearizedResidualFunction = Callable[JaxLinearizedResidualArgs, DictVec]

Residual = Tuple[ArrayLike, Tuple[bv.BlockVector, bv.BlockVector, bv.BlockVector], Callable]

Test = Union[JaxResidualFunction, JaxLinearizedResidualFunction]

class DynamicalFluidModelInterface:
    _res: JaxResidualFunction
    _res_args: Union[JaxResidualArgs, JaxLinearizedResidualArgs]

    def __init__(self, residual: bernoulli.JaxResidual):

        (state, control, prop) = residual.res_args

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

# NOTE: `Model` and `LinearizedModel` are very similar except for
# the residual functions and arguments (the latter is linearized)
class Model(DynamicalFluidModelInterface, BaseDynamicalModel):
    """
    Representation of a dynamical system model
    """

    def __init__(self, residual: bernoulli.JaxResidual):
        super().__init__(residual)

        self._res = jax.jit(residual.res)
        self._res_args = (
            blockvec_to_dict(self.state),
            blockvec_to_dict(self.control),
            blockvec_to_dict(self.prop)
        )

class LinearizedModel(DynamicalFluidModelInterface, BaseLinearizedDynamicalModel):
    """
    Representation of a linearized dynamical system model
    """

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

        self._res = (
            lambda state, control, prop, tangents:
            jax.jvp(jax.jit(residual.res), (state, control, prop), tangents)[1]
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


class Predefined1DModel(Model):

    def __init__(self, mesh: ArrayLike, **kwargs):
        residual = self._make_residual(mesh, **kwargs)
        super().__init__(residual)

    def _make_residual(self, mesh, **kwargs):
        raise NotImplementedError()

class PredefinedLinearized1DModel(LinearizedModel):

    def __init__(self, mesh: ArrayLike, **kwargs):
        residual = self._make_residual(mesh, **kwargs)
        super().__init__(residual)

    def _make_residual(self, mesh: ArrayLike, **kwargs):
        raise NotImplementedError()

## Predefined models
class BernoulliSmoothMinSep(Predefined1DModel):

    def _make_residual(self, mesh):
        return bernoulli.BernoulliSmoothMinSep(mesh)

class LinearizedBernoulliSmoothMinSep(PredefinedLinearized1DModel):

    def _make_residual(self, mesh):
        return bernoulli.BernoulliSmoothMinSep(mesh)


class BernoulliFixedSep(Predefined1DModel):

    def _make_residual(self, mesh, idx_sep=0):
        return bernoulli.BernoulliFixedSep(mesh, idx_sep)

class LinearizedBernoulliFixedSep(PredefinedLinearized1DModel):

    def _make_residual(self, mesh, idx_sep=0):
        return bernoulli.BernoulliFixedSep(mesh, idx_sep)


class BernoulliAreaRatioSep(Predefined1DModel):

    def _make_residual(self, mesh):
        return bernoulli.BernoulliAreaRatioSep(mesh)

class LinearizedBernoulliAreaRatioSep(PredefinedLinearized1DModel):

    def _make_residual(self, mesh):
        return bernoulli.BernoulliAreaRatioSep(mesh)
