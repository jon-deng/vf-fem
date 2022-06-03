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
import numpy as np
import jax

from blockarray import blockvec as bla


from .base import DynamicalSystem
from ..equations.fluid import bernoulli
from ..jaxutils import (blockvec_to_dict, flatten_nested_dict)

# pylint: disable=missing-docstring

def create_primal_res_class(Parent):
    class PrimalResidual(Parent):
        def assem_res(self):
            submats = self._res(*self.primals)
            labels = self.state.labels
            submats, shape = flatten_nested_dict(submats, labels)
            return bla.BlockVector(submats, shape, labels)

        def assem_dres_dstate(self):
            submats = jax.jacfwd(self._res, argnums=0)(*self.primals)
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
            submats = jax.jacfwd(self._res, argnums=1)(*self.primals)
            labels = self.state.labels+self.control.labels
            submats, shape = flatten_nested_dict(submats, labels)
            return bla.BlockMatrix(submats, shape, labels)

        def assem_dres_dprops(self):
            submats = jax.jacfwd(self._res, argnums=2)(*self.primals)
            labels = self.state.labels + self.props.labels
            submats, shape = flatten_nested_dict(submats, labels)
            return bla.BlockMatrix(submats, shape, labels)

    return PrimalResidual

def create_linearized_res_class(Parent):
    class LinearizedResidual(Parent):
        def assem_res(self):
            submats = self._dres(*self.primals, self.tangents)
            labels = self.state.labels
            submats, shape = flatten_nested_dict(submats, labels)
            return bla.BlockVector(submats, shape, labels)

        def assem_dres_dstate(self):
            submats = jax.jacfwd(self._dres, argnums=0)(*self.primals, self.tangents)
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
            submats = jax.jacfwd(self._dres, argnums=1)(*self.primals, self.tangents)
            labels = self.state.labels+self.control.labels
            submats, shape = flatten_nested_dict(submats, labels)
            return bla.BlockMatrix(submats, shape, labels)

        def assem_dres_dprops(self):
            submats = jax.jacfwd(self._dres, argnums=2)(*self.primals, self.tangents)
            labels = self.state.labels + self.props.labels
            submats, shape = flatten_nested_dict(submats, labels)
            return bla.BlockMatrix(submats, shape, labels)
    return LinearizedResidual



class BaseFluid1DDynamicalSystem(DynamicalSystem):

    def __init__(self, s, res, state, control, props):
        self.s = s

        # self._res = jax.jit(res)
        self._res = res
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


class BaseBernoulliSmoothMinSep(BaseFluid1DDynamicalSystem):

    def __init__(self, s):
        _, (_state, _control, _props), res = bernoulli.BernoulliSmoothMinSep(s)
        super().__init__(s, res, _state, _control, _props)

BernoulliSmoothMinSep = create_primal_res_class(BaseBernoulliSmoothMinSep)

LinearizedBernoulliSmoothMinSep = create_linearized_res_class(BaseBernoulliSmoothMinSep)


class BaseBernoulliFixedSeparation(BaseFluid1DDynamicalSystem):

    def __init__(self, s, idx_sep=0):
        _, (_state, _control, _props), res = bernoulli.BernoulliFixedSep(s, idx_sep)
        super().__init__(s, res, _state, _control, _props)

BernoulliFixedSeparation = create_primal_res_class(BaseBernoulliFixedSeparation)

LinearizedBernoulliFixedSeparation = create_linearized_res_class(BaseBernoulliFixedSeparation)
