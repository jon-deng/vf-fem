"""
Contains definitions of fluid models

The nonlinear dynamical systems here are defined in jax/numpy and augmented a
bit manually. The basic dynamical system residual has a block form
F(x, xt, g) = [Fq(x, xt, g), Fp(x, xt)]
x = [q, p]
xt = [qt, pt]
and where q and p stand for flow and pressure for a 1D fluid model
"""

import numpy as np

from blocktensor import vec as bla

from .base import DynamicalSystem
from ..equations.fluid.bernoulli_sep_at_min import (
    bernoulli_qp,
    dbernoulli_qp,
    dbernoulli_qp_darea,
    dbernoulli_qp_dpsub,
    dbernoulli_qp_dpsup,
    ddbernoulli_qp_darea,
    ddbernoulli_qp_dpsub,
    ddbernoulli_qp_dpsup
)

class BaseFluid1DDynamicalSystem(DynamicalSystem):

    def __init__(self, s):
        self.s = s

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

        properties_vec = ['psub', 'psup', 'rho_air', 'zeta_min', 'zeta_sep']
        self.properties = bla.BlockVector(
            [np.ones(1) for i in range(len(properties_vec))], labels=[properties_vec])

class Bernoulli1DDynamicalSystem(BaseFluid1DDynamicalSystem):
    def assem_res(self):
        # Depack variables from BlockVector for input to Bernoulli functions
        area = self.control['area']
        rho = self.properties['rho_air'][0]
        psub = self.control['psub'][0]
        psup = self.control['psup'][0]
        zeta_min = self.properties['zeta_min'][0]
        zeta_sep = self.properties['zeta_sep'][0]

        primals = (area, self.s, psub, psup, rho, zeta_min, zeta_sep)
        qp_explicit = bernoulli_qp(*primals)
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
        return bla.BlockMatrix(mats, labels=(self.state.keys, self.state.keys))

    def assem_dres_dstatet(self):
        dresq_dq = np.diag(np.zeros(self.q.size))
        dresq_dp = np.zeros((self.q.size, self.p.size))

        dresp_dp = np.diag(np.zeros(self.p.size))
        dresp_dq = np.zeros((self.p.size, self.q.size))
        mats = [
            [dresq_dq, dresq_dp],
            [dresp_dq, dresp_dp]]
        return bla.BlockMatrix(mats, labels=(self.state.keys, self.state.keys))

    def assem_dres_dcontrol(self):
        # Depack variables from BlockVector for input to Bernoulli functions
        area = self.control['area']
        rho = self.properties['rho_air'][0]
        psub = self.control['psub']
        psup = self.control['psup']
        zeta_min = self.properties['zeta_min'][0]
        zeta_sep = self.properties['zeta_sep'][0]
        primals = (area, self.s, psub, psup, rho, zeta_min, zeta_sep)

        dq_darea, dp_darea = dbernoulli_qp_darea(*primals)
        dq_dpsub, dp_dpsub = dbernoulli_qp_dpsub(*primals)
        dq_dpsup, dp_dpsup = dbernoulli_qp_dpsup(*primals)

        mats = [
            [-dq_darea, -dq_dpsub, -dq_dpsup],
            [-dp_darea, -dp_dpsub, -dp_dpsup]]
        return bla.BlockMatrix(mats, labels=(self.state.keys, self.control.keys))

class LinearizedBernoulli1DDynamicalSystem(BaseFluid1DDynamicalSystem):
    def assem_res(self):
        # Depack variables from BlockVector for input to Bernoulli functions
        area = self.control['area']
        rho = self.properties['rho_air'][0]
        psub = self.control['psub'][0]
        psup = self.control['psup'][0]
        zeta_min = self.properties['zeta_min'][0]
        zeta_sep = self.properties['zeta_sep'][0]

        primals = (area, self.s, psub, psup, rho, zeta_min, zeta_sep)
        darea = self.dcontrol['area']
        dpsub = self.dcontrol['psub'][0]
        dpsup = self.dcontrol['psup'][0]
        tangents = (darea, np.zeros(self.s.shape), dpsub, dpsup, 0.0, 0.0, 0.0)

        # The linearized residual changes due to linearizations
        resq_bernoulli, resp_bernoulli = dbernoulli_qp(*primals, tangents)

        resq = self.dstate['q'] - resq_bernoulli
        resp = self.dstate['p'] - resp_bernoulli

        return bla.BlockVector([resq, resp], labels=[self.state.keys])

    def assem_dres_dstate(self):
        dresq_dq = np.diag(np.zeros(self.q.size))
        dresq_dp = np.zeros((self.q.size, self.p.size))

        dresp_dp = np.diag(np.zeros(self.p.size))
        dresp_dq = np.zeros((self.p.size, self.q.size))
        mats = [
            [dresq_dq, dresq_dp],
            [dresp_dq, dresp_dp]]
        return bla.BlockMatrix(mats, labels=(self.state.keys, self.state.keys))

    def assem_dres_dstatet(self):
        dresq_dq = np.diag(np.zeros(self.q.size))
        dresq_dp = np.zeros((self.q.size, self.p.size))

        dresp_dp = np.diag(np.zeros(self.p.size))
        dresp_dq = np.zeros((self.p.size, self.q.size))
        mats = [
            [dresq_dq, dresq_dp],
            [dresp_dq, dresp_dp]]
        return bla.BlockMatrix(mats, labels=(self.state.keys, self.state.keys))

    def assem_dres_dcontrol(self):
        # Depack variables from BlockVector for input to Bernoulli functions
        area = self.control['area']
        rho = self.properties['rho_air'][0]
        psub = self.control['psub']
        psup = self.control['psup']
        zeta_min = self.properties['zeta_min'][0]
        zeta_sep = self.properties['zeta_sep'][0]

        primals = (area, self.s, psub, psup, rho, zeta_min, zeta_sep)
        darea = self.dcontrol['area']
        dpsub = self.dcontrol['psub']
        dpsup = self.dcontrol['psup']
        tangents = (darea, np.zeros(self.s.shape), dpsub, dpsup, 0.0, 0.0, 0.0)

        dq_darea, dp_darea = ddbernoulli_qp_darea(*primals, tangents)
        dq_dpsub, dp_dpsub = ddbernoulli_qp_dpsub(*primals, tangents)
        dq_dpsup, dp_dpsup = ddbernoulli_qp_dpsup(*primals, tangents)

        mats = [
            [-dq_darea, -dq_dpsub, -dq_dpsup],
            [-dp_darea, -dp_dpsub, -dp_dpsup]]
        return bla.BlockMatrix(mats, labels=(self.state.keys, self.control.keys))
