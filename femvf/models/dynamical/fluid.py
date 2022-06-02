"""
Contains definitions of fluid models

The nonlinear dynamical systems here are defined in jax/numpy and augmented a
bit manually. The basic dynamical system residual has a block form
F(x, xt, g) = [Fq(x, xt, g), Fp(x, xt)]
x = [q, p]
xt = [qt, pt]
and where q and p stand for flow and pressure for a 1D fluid model
"""

from lib2to3.pytree import Base
import numpy as np
import jax

from blockarray import blockvec as bla
from blockarray.labelledarray import flatten_array

from .base import DynamicalSystem
from ..equations.fluid import bernoulli_sep_at_min as bernmin
from ..equations.fluid import bernoulli_sep_at_fixed as bernfix

# pylint: disable=missing-docstring

def create_primal_res_class(Parent):
    class PrimalResidual(Parent):
        def assem_res(self):
            submats = self._res(*self.primals)
            submats, shape = nested_dict_to_flat_subarrays(submats)
            return bla.BlockVector(submats, shape, labels=self.state.labels)

        def assem_dres_dstate(self):
            submats = jax.jacfwd(self._res, argnums=0)(*self.primals)
            submats, shape = nested_dict_to_flat_subarrays(submats)
            return bla.BlockMatrix(submats, shape, labels=self.state.labels+self.state.labels)

        def assem_dres_dstatet(self):
            dresq_dq = np.diag(np.zeros(self.q.size))
            dresq_dp = np.zeros((self.q.size, self.p.size))

            dresp_dp = np.diag(np.zeros(self.p.size))
            dresp_dq = np.zeros((self.p.size, self.q.size))
            mats = [
                [dresq_dq, dresq_dp],
                [dresp_dq, dresp_dp]]
            return bla.BlockMatrix(mats, labels=self.state.labels+self.state.labels)

        def assem_dres_dcontrol(self):
            submats = jax.jacfwd(self._res, argnums=1)(*self.primals)
            submats, shape = nested_dict_to_flat_subarrays(submats)
            return bla.BlockMatrix(submats, shape, labels=self.state.labels+self.control.labels)

        def assem_dres_dprops(self):
            submats = jax.jacfwd(self._res, argnums=2)(*self.primals)
            submats, shape = nested_dict_to_flat_subarrays(submats)
            return bla.BlockMatrix(submats, shape, labels=(self.state.labels[0], self.props.labels[0]))

    return PrimalResidual

def create_linearized_res_class(Parent):
    class LinearizedResidual(Parent):
        def assem_res(self):
            submats = self._dres(*self.primals, self.tangents)
            submats, shape = nested_dict_to_flat_subarrays(submats)
            return bla.BlockVector(submats, shape, labels=self.state.labels)

        def assem_dres_dstate(self):
            submats = jax.jacfwd(self._dres, argnums=0)(*self.primals, self.tangents)
            submats, shape = nested_dict_to_flat_subarrays(submats)
            return bla.BlockMatrix(submats, shape, labels=self.state.labels+self.state.labels)

        def assem_dres_dstatet(self):
            dresq_dq = np.diag(np.zeros(self.q.size))
            dresq_dp = np.zeros((self.q.size, self.p.size))

            dresp_dp = np.diag(np.zeros(self.p.size))
            dresp_dq = np.zeros((self.p.size, self.q.size))
            mats = [
                [dresq_dq, dresq_dp],
                [dresp_dq, dresp_dp]]
            return bla.BlockMatrix(mats, labels=self.state.labels+self.state.labels)

        def assem_dres_dcontrol(self):
            submats = jax.jacfwd(self._dres, argnums=1)(*self.primals, self.tangents)
            submats, shape = nested_dict_to_flat_subarrays(submats)
            return bla.BlockMatrix(submats, shape, labels=self.state.labels+self.control.labels)

        def assem_dres_dprops(self):
            submats = jax.jacfwd(self._dres, argnums=2)(*self.primals, self.tangents)
            submats, shape = nested_dict_to_flat_subarrays(submats)
            return bla.BlockMatrix(submats, shape, labels=(self.state.labels[0], self.props.labels[0]))
    return LinearizedResidual


class BaseFluid1DDynamicalSystem(DynamicalSystem):

    def __init__(self, s):
        self.s = s


class BaseBernoulliSmoothMinSep(BaseFluid1DDynamicalSystem):

    def __init__(self, s):
        super().__init__(s)

        N = self.s.size
        self.q = np.zeros(1)
        self.p = np.zeros(N)
        self.state = bla.BlockVector([self.q, self.p], labels=[['q', 'p']])

        self.qt = np.zeros(1)
        self.pt = np.zeros(N)
        self.statet = bla.BlockVector([self.qt, self.pt], labels=[['q', 'p']])

        self.control = bla.BlockVector([np.ones(N), np.zeros(1), np.zeros(1)], labels=[['area', 'psub', 'psup']])

        self.dstate = self.state.copy()

        self.dstatet = self.statet.copy()

        self.dcontrol = self.control.copy()

        properties_vec = ['rho_air', 'zeta_min', 'zeta_sep']
        self.props = bla.BlockVector(
            [np.ones(1) for i in range(len(properties_vec))], labels=[properties_vec])

class BernoulliSmoothMinSep(BaseBernoulliSmoothMinSep):
    def assem_res(self):
        # Depack variables from BlockVector for input to Bernoulli functions
        area = self.control['area']
        rho = self.props['rho_air'][0]
        psub = self.control['psub'][0]
        psup = self.control['psup'][0]
        zeta_min = self.props['zeta_min'][0]
        zeta_sep = self.props['zeta_sep'][0]

        primals = (area, self.s, psub, psup, rho, zeta_min, zeta_sep)
        qp_explicit = bernmin.bernoulli_qp(*primals)
        resq = self.q - qp_explicit[0]
        resp = self.p - qp_explicit[1]
        return bla.BlockVector([resq, resp], labels=self.state.labels)

    def assem_dres_dstate(self):
        dresq_dq = np.diag(np.ones(self.q.size))
        dresq_dp = np.zeros((self.q.size, self.p.size))

        dresp_dq = np.zeros((self.p.size, self.q.size))
        dresp_dp = np.diag(np.ones(self.p.size))
        mats = [
            [dresq_dq, dresq_dp],
            [dresp_dq, dresp_dp]]
        return bla.BlockMatrix(mats, labels=self.state.labels+self.state.labels)

    def assem_dres_dstatet(self):
        dresq_dq = np.diag(np.zeros(self.q.size))
        dresq_dp = np.zeros((self.q.size, self.p.size))

        dresp_dp = np.diag(np.zeros(self.p.size))
        dresp_dq = np.zeros((self.p.size, self.q.size))
        mats = [
            [dresq_dq, dresq_dp],
            [dresp_dq, dresp_dp]]
        return bla.BlockMatrix(mats, labels=self.state.labels+self.state.labels)

    def assem_dres_dcontrol(self):
        # Depack variables from BlockVector for input to Bernoulli functions
        area = self.control['area']
        rho = self.props['rho_air'][0]
        psub = self.control['psub']
        psup = self.control['psup']
        zeta_min = self.props['zeta_min'][0]
        zeta_sep = self.props['zeta_sep'][0]
        primals = (area, self.s, psub, psup, rho, zeta_min, zeta_sep)

        dq_darea, dp_darea = bernmin.dbernoulli_qp_darea(*primals)
        dq_dpsub, dp_dpsub = bernmin.dbernoulli_qp_dpsub(*primals)
        dq_dpsup, dp_dpsup = bernmin.dbernoulli_qp_dpsup(*primals)

        mats = [
            [-dq_darea, -dq_dpsub, -dq_dpsup],
            [-dp_darea, -dp_dpsub, -dp_dpsup]]
        return bla.BlockMatrix(mats, labels=self.state.labels+self.control.labels)

    def assem_dres_dprops(self):
        # Depack variables from BlockVector for input to Bernoulli functions
        # area = self.control['area']
        # rho = self.props['rho_air'][0]
        # psub = self.control['psub']
        # psup = self.control['psup']
        # zeta_min = self.props['zeta_min'][0]
        # zeta_sep = self.props['zeta_sep'][0]
        # primals = (area, self.s, psub, psup, rho, zeta_min, zeta_sep)

        mats = [
            [np.zeros((state_subvec.size, prop_subvec.size))
                for prop_subvec in self.props]
            for state_subvec in self.state]
        return bla.BlockMatrix(mats, labels=(self.state.labels[0], self.props.labels[0]))

class LinearizedBernoulliSmoothMinSep(BaseBernoulliSmoothMinSep):
    def assem_res(self):
        # Depack variables from BlockVector for input to Bernoulli functions
        area = self.control['area']
        rho = self.props['rho_air'][0]
        psub = self.control['psub'][0]
        psup = self.control['psup'][0]
        zeta_min = self.props['zeta_min'][0]
        zeta_sep = self.props['zeta_sep'][0]

        primals = (area, self.s, psub, psup, rho, zeta_min, zeta_sep)
        darea = self.dcontrol['area']
        dpsub = self.dcontrol['psub'][0]
        dpsup = self.dcontrol['psup'][0]
        tangents = (darea, np.zeros(self.s.shape), dpsub, dpsup, 0.0, 0.0, 0.0)

        # The linearized residual changes due to linearizations
        resq_bernoulli, resp_bernoulli = bernmin.dbernoulli_qp(*primals, tangents)

        resq = self.dstate['q'] - resq_bernoulli
        resp = self.dstate['p'] - resp_bernoulli

        return bla.BlockVector([resq, resp], labels=self.state.labels)

    def assem_dres_dstate(self):
        dresq_dq = np.diag(np.zeros(self.q.size))
        dresq_dp = np.zeros((self.q.size, self.p.size))

        dresp_dp = np.diag(np.zeros(self.p.size))
        dresp_dq = np.zeros((self.p.size, self.q.size))
        mats = [
            [dresq_dq, dresq_dp],
            [dresp_dq, dresp_dp]]
        return bla.BlockMatrix(mats, labels=self.state.labels+self.state.labels)

    def assem_dres_dstatet(self):
        dresq_dq = np.diag(np.zeros(self.q.size))
        dresq_dp = np.zeros((self.q.size, self.p.size))

        dresp_dp = np.diag(np.zeros(self.p.size))
        dresp_dq = np.zeros((self.p.size, self.q.size))
        mats = [
            [dresq_dq, dresq_dp],
            [dresp_dq, dresp_dp]]
        return bla.BlockMatrix(mats, labels=self.state.labels+self.state.labels)

    def assem_dres_dcontrol(self):
        # Depack variables from BlockVector for input to Bernoulli functions
        area = self.control['area']
        rho = self.props['rho_air'][0]
        psub = self.control['psub']
        psup = self.control['psup']
        zeta_min = self.props['zeta_min'][0]
        zeta_sep = self.props['zeta_sep'][0]

        primals = (area, self.s, psub, psup, rho, zeta_min, zeta_sep)
        darea = self.dcontrol['area']
        dpsub = self.dcontrol['psub']
        dpsup = self.dcontrol['psup']
        tangents = (darea, np.zeros(self.s.shape), dpsub, dpsup, 0.0, 0.0, 0.0)

        dq_darea, dp_darea = bernmin.ddbernoulli_qp_darea(*primals, tangents)
        dq_dpsub, dp_dpsub = bernmin.ddbernoulli_qp_dpsub(*primals, tangents)
        dq_dpsup, dp_dpsup = bernmin.ddbernoulli_qp_dpsup(*primals, tangents)

        mats = [
            [-dq_darea, -dq_dpsub, -dq_dpsup],
            [-dp_darea, -dp_dpsub, -dp_dpsup]]
        return bla.BlockMatrix(mats, labels=(self.state.labels+self.control.labels))

    def assem_dres_dprops(self):
        # Depack variables from BlockVector for input to Bernoulli functions
        # area = self.control['area']
        # rho = self.props['rho_air'][0]
        # psub = self.control['psub']
        # psup = self.control['psup']
        # zeta_min = self.props['zeta_min'][0]
        # zeta_sep = self.props['zeta_sep'][0]
        # primals = (area, self.s, psub, psup, rho, zeta_min, zeta_sep)

        mats = [
            [np.zeros((state_subvec.size, prop_subvec.size))
                for prop_subvec in self.props]
            for state_subvec in self.state]
        return bla.BlockMatrix(mats, labels=(self.state.labels+self.props.labels))

def blockvec_to_dict(blockvec):
    return {key: subvec for key, subvec in blockvec.items()}

def nest_dict_values(dict_array):
    return tuple([
        nest_dict_values(value) if isinstance(value, dict) else value
        for value in dict_array.values()
    ])

def nested_dict_to_flat_subarrays(dict_array):
    nested_array = nest_dict_values(dict_array)
    return flatten_array(nested_array)

class BaseBernoulliFixedSeparation(BaseFluid1DDynamicalSystem):

    def __init__(self, s, idx_sep=0):
        super().__init__(s)
        _, (_state, _control, _props), res = bernfix.BernoulliFixedSep(s, idx_sep)

        self._res = res
        self._dres = lambda state, control, props, tangents: jax.jvp(res, (state, control, props), tangents)[1]

        self.state = bla.BlockVector(list(_state.values()), labels=[list(_state.keys())])
        self.dstate = self.state.copy()

        self.statet = self.state.copy()
        self.dstatet = self.statet.copy()

        self.control = bla.BlockVector(list(_control.values()), labels=[list(_control.keys())])
        self.dcontrol = self.control.copy()

        self.props = bla.BlockVector(list(_props.values()), labels=[list(_props.keys())])
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

BernoulliFixedSeparation = create_primal_res_class(BaseBernoulliFixedSeparation)

LinearizedBernoulliFixedSeparation = create_linearized_res_class(BaseBernoulliFixedSeparation)
