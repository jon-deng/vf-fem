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
from jax import numpy as jnp
import jax

from blocktensor import linalg as bla

from .base import DynamicalSystem

## To disable jax:
# jnp = np

# pylint: disable=abstract-method
# @jax.jit
def bernoulli_q(asep, psub, psup, rho):
    """
    Return the flow rate based on Bernoulli
    """
    if psub >= psup:
        q = (2*(psub-psup)/rho)**0.5 * asep
    else: 
        q = -(2*(psup-psub)/rho)**0.5 * asep
    return q

# @jax.jit
def bernoulli_p(q, area, ssep, psub, psup, rho):
    """
    Return the pressure based on Bernoulli
    """
    p = psub - 1/2*rho*(q/area)**2
    return p

def coeff_sep(s, ssep, zeta_sep):
    # note that jnp.nn.sigmoid is used to model
    # return (1+jnp.exp((s-ssep)/zeta_sep))**-1
    # because of numerical stability
    # Using the normal formula results in nan for large exponents
    return jax.nn.sigmoid(-1*(s-ssep)/zeta_sep)
    

# @jax.jit
def wavg(s, f, w):
    """
    Return the weighted average of f(s) over s with weights w(s)
    """
    return jnp.trapz(f*w, s)/jnp.trapz(w, s)

# @jax.jit
def smooth_min_weight(f, zeta=1):
    """
    Return a smooth minimum from a set of values f

    This evaluates according to $\exp{-\frac{f}{\zeta}}$ and is normalized so
    that the maximum weight has a value of 1.0.
    """
    # w = jnp.exp(-f/zeta)
    log_w = -f/zeta
    # To prevent overflow for the largest weight, normalize the maximum log-weight to 0
    # This ensures the maximum weight is 1.0 while smaller weights will underflow to 0.0
    log_w = log_w - np.max(log_w)
    return jnp.exp(log_w)

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
def bernoulli_qp(area, s, psub, psup, rho, zeta_min, zeta_sep):
    wmin = smooth_min_weight(area, zeta_min)
    amin = wavg(s, area, wmin)
    smin = wavg(s, s, wmin)
    # print(wmin, amin, smin)

    asep = amin
    ssep = smin
    q = bernoulli_q(asep, psub, psup, rho)
    p = bernoulli_p(q, area, ssep, psub, psup, rho)

    # Separation coefficient ensure pressures tends to zero after separation
    f_sep = coeff_sep(s, ssep, zeta_sep)
    p = f_sep * p
    return q, p

# Use this weird combination of primals/tangent parameters because later we 
# want to compute the jacobian wrt area
def dbernoulli_qp(area, s, psub, psup, rho, zeta_min, zeta_sep, tangents):
    primals = (area, s, psub, psup, rho, zeta_min, zeta_sep)
    return jax.jvp(bernoulli_qp, primals, tangents)[1]

dbernoulli_qp_darea = jax.jacfwd(bernoulli_qp, argnums=0)
ddbernoulli_qp_darea = jax.jacfwd(dbernoulli_qp, argnums=0)

dbernoulli_qp_dpsub = jax.jacfwd(bernoulli_qp, argnums=2)
ddbernoulli_qp_dpsub = jax.jacfwd(dbernoulli_qp, argnums=2)

dbernoulli_qp_dpsup = jax.jacfwd(bernoulli_qp, argnums=3)
ddbernoulli_qp_dpsup = jax.jacfwd(dbernoulli_qp, argnums=3)

class BaseFluid1DDynamicalSystem(DynamicalSystem):

    def __init__(self, s):
        self.s = s

        N = self.s.size
        self.q = np.zeros(1)
        self.p = np.zeros(N)
        self.state = bla.BlockVec([self.q, self.p], ['q', 'p'])

        self.qt = np.zeros(1)
        self.pt = np.zeros(N)
        self.statet = bla.BlockVec([self.qt, self.pt], ['q', 'p'])

        self.control = bla.BlockVec([np.ones(N), np.zeros(1), np.zeros(1)], ['area', 'psub', 'psup'])

        self.dq = np.zeros(1)
        self.dp = np.zeros(N)
        self.dstate = bla.BlockVec([self.q, self.p], ['q', 'p'])

        self.dqt = np.zeros(1)
        self.dpt = np.zeros(N)
        self.dstatet = bla.BlockVec([self.qt, self.pt], ['q', 'p'])

        self.dcontrol = self.control.copy()

        properties_vec = ['psub', 'psup', 'rho_air', 'zeta_min', 'zeta_sep']
        self.properties = bla.BlockVec(
            [np.ones(1) for i in range(len(properties_vec))], properties_vec)

class Bernoulli1DDynamicalSystem(BaseFluid1DDynamicalSystem):
    def assem_res(self):
        # Depack variables from BlockVec for input to Bernoulli functions
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
        return bla.BlockVec([resq, resp], self.state.keys)

    def assem_dres_dstate(self):
        dresq_dq = np.diag(np.ones(self.q.size))
        dresq_dp = np.zeros((self.q.size, self.p.size))

        dresp_dq = np.zeros((self.p.size, self.q.size))
        dresp_dp = np.diag(np.ones(self.p.size))
        mats = [
            [dresq_dq, dresq_dp],
            [dresp_dq, dresp_dp]]
        return bla.BlockMat(mats, (self.state.keys, self.state.keys))

    def assem_dres_dstatet(self):
        dresq_dq = np.diag(np.zeros(self.q.size))
        dresq_dp = np.zeros((self.q.size, self.p.size))

        dresp_dp = np.diag(np.zeros(self.p.size))
        dresp_dq = np.zeros((self.p.size, self.q.size))
        mats = [
            [dresq_dq, dresq_dp],
            [dresp_dq, dresp_dp]]
        return bla.BlockMat(mats, (self.state.keys, self.state.keys))

    def assem_dres_dcontrol(self):
        # Depack variables from BlockVec for input to Bernoulli functions
        area = self.control['area']
        rho = self.properties['rho_air'][0]
        psub = self.control['psub']
        psup = self.control['psup']
        zeta_min = self.properties['zeta_min'][0]
        zeta_sep = self.properties['zeta_sep'][0]
        primals = (area, self.s, psub, psup, rho, zeta_min, zeta_sep)
        # breakpoint()
        dq_darea, dp_darea = dbernoulli_qp_darea(*primals)
        dq_dpsub, dp_dpsub = dbernoulli_qp_dpsub(*primals)
        dq_dpsup, dp_dpsup = dbernoulli_qp_dpsup(*primals)

        mats = [
            [-dq_darea, -dq_dpsub, -dq_dpsup],
            [-dp_darea, -dp_dpsub, -dp_dpsup]]
        return bla.BlockMat(mats, (self.state.keys, self.control.keys))

class LinearStateBernoulli1DDynamicalSystem(BaseFluid1DDynamicalSystem):
    def assem_res(self):
        resq = self.dq
        resp = self.dp
        return bla.BlockVec([resq, resp], self.state.keys)

    def assem_dres_dstate(self):
        dresq_dq = np.diag(np.zeros(self.q.size))
        dresq_dp = np.zeros((self.q.size, self.p.size))

        dresp_dp = np.diag(np.zeros(self.p.size))
        dresp_dq = np.zeros((self.p.size, self.q.size))
        mats = [
            [dresq_dq, dresq_dp],
            [dresp_dq, dresp_dp]]
        return bla.BlockMat(mats, (self.state.keys, self.state.keys))

    def assem_dres_dstatet(self):
        dresq_dq = np.diag(np.zeros(self.q.size))
        dresq_dp = np.zeros((self.q.size, self.p.size))

        dresp_dp = np.diag(np.zeros(self.p.size))
        dresp_dq = np.zeros((self.p.size, self.q.size))
        mats = [
            [dresq_dq, dresq_dp],
            [dresp_dq, dresp_dp]]
        return bla.BlockMat(mats, (self.state.keys, self.state.keys))

    def assem_dres_dcontrol(self):
        dresq_darea = np.zeros((self.q.size, self.s.size))
        dresp_darea = np.zeros((self.p.size, self.s.size))

        dresq_dpsub = np.zeros((self.q.size, 1))
        dresp_dpsub = np.zeros((self.p.size, 1))

        dresq_dpsup = np.zeros((self.q.size, 1))
        dresp_dpsup = np.zeros((self.p.size, 1))
        mats = [
            [dresq_darea, dresq_dpsub, dresq_dpsup],
            [dresp_darea, dresp_dpsub, dresp_dpsup]]
        return bla.BlockMat(mats, (self.state.keys, self.control.keys))

class LinearStatetBernoulli1DDynamicalSystem(BaseFluid1DDynamicalSystem):
    """
    This linearized residual works out to be zero because the Bernoulli model
    doesn't have any transient effects
    """
    def assem_res(self):
        resq = np.zeros(1)
        resp = np.zeros(self.p.size)
        return bla.BlockVec([resq, resp], self.state.keys)

    def assem_dres_dstate(self):
        dresq_dq = np.diag(np.zeros(self.q.size))
        dresq_dp = np.zeros((self.q.size, self.p.size))

        dresp_dp = np.diag(np.zeros(self.p.size))
        dresp_dq = np.zeros((self.p.size, self.q.size))
        mats = [
            [dresq_dq, dresq_dp],
            [dresp_dq, dresp_dp]]
        return bla.BlockMat(mats, (self.state.keys, self.state.keys))

    def assem_dres_dstatet(self):
        dresq_dq = np.diag(np.zeros(self.q.size))
        dresq_dp = np.zeros((self.q.size, self.p.size))

        dresp_dp = np.diag(np.zeros(self.p.size))
        dresp_dq = np.zeros((self.p.size, self.q.size))
        mats = [
            [dresq_dq, dresq_dp],
            [dresp_dq, dresp_dp]]
        return bla.BlockMat(mats, (self.state.keys, self.state.keys))

    def assem_dres_dcontrol(self):
        dresq_darea = np.zeros((self.q.size, self.s.size))
        dresp_darea = np.zeros((self.p.size, self.s.size))

        dresq_dpsub = np.zeros((self.q.size, 1))
        dresp_dpsub = np.zeros((self.p.size, 1))

        dresq_dpsup = np.zeros((self.q.size, 1))
        dresp_dpsup = np.zeros((self.p.size, 1))
        mats = [
            [dresq_darea, dresq_dpsub, dresq_dpsup],
            [dresp_darea, dresp_dpsub, dresp_dpsup]]
        return bla.BlockMat(mats, (self.state.keys, self.control.keys))

class LinearControlBernoulli1DDynamicalSystem(BaseFluid1DDynamicalSystem):
    def assem_res(self):
        # Depack variables from BlockVec for input to Bernoulli functions
        area = self.control['area']
        rho = self.properties['rho_air'][0]
        psub = self.control['psub'][0]
        psup = self.control['psup'][0]
        zeta_min = self.properties['zeta_min'][0]
        zeta_sep = self.properties['zeta_sep'][0]
        
        primals = (area, self.s, rho, psub, psup, zeta_min, zeta_sep)
        tangents = (self.dcontrol['area'], 0, 0, 0, 0, 0, 0)
        dqp = dbernoulli_qp(*primals, tangents)
        resq = -dqp[0]
        resp = -dqp[1]
        return bla.BlockVec([resq, resp], self.state.keys)

    def assem_dres_dstate(self):
        dresq_dq = np.diag(np.zeros(self.q.size))
        dresq_dp = np.zeros((self.q.size, self.p.size))

        dresp_dp = np.diag(np.zeros(self.p.size))
        dresp_dq = np.zeros((self.p.size, self.q.size))
        mats = [
            [dresq_dq, dresq_dp],
            [dresp_dq, dresp_dp]]
        return bla.BlockMat(mats, (self.state.keys, self.state.keys))

    def assem_dres_dstatet(self):
        dresq_dq = np.diag(np.zeros(self.q.size))
        dresq_dp = np.zeros((self.q.size, self.p.size))

        dresp_dp = np.diag(np.zeros(self.p.size))
        dresp_dq = np.zeros((self.p.size, self.q.size))
        mats = [
            [dresq_dq, dresq_dp],
            [dresp_dq, dresp_dp]]
        return bla.BlockMat(mats, (self.state.keys, self.state.keys))

    def assem_dres_dcontrol(self):
        # Depack variables from BlockVec for input to Bernoulli functions
        area = self.control['area']
        rho = self.properties['rho_air'][0]
        psub = self.control['psub'][0]
        psup = self.control['psup'][0]
        zeta_min = self.properties['zeta_min'][0]
        zeta_sep = self.properties['zeta_sep'][0]
        
        primals = (area, self.s, rho, psub, psup, zeta_min, zeta_sep)
        tangents = (self.dcontrol['area'], 0, self.dcontrol['psub'][0], self.dcontrol['psup'], 0, 0, 0)

        ddq_darea, ddp_darea = ddbernoulli_qp_darea(*primals, tangents)
        ddq_dpsub, ddp_dpsub = ddbernoulli_qp_dpsub(*primals, tangents)
        ddq_dpsup, ddp_dpsup = ddbernoulli_qp_dpsup(*primals, tangents)
        dresq_darea = -ddq_darea
        dresp_darea = -ddp_darea

        dresq_dpsub = -ddq_dpsub
        dresp_dpsub = -ddp_dpsub

        dresq_dpsup = -ddq_dpsup
        dresp_dpsup = -ddp_dpsup
        mats = [
            [dresq_darea, dresq_dpsub, dresq_dpsup],
            [dresp_darea, dresp_dpsub, dresp_dpsup]]
        return bla.BlockMat(mats, (self.state.keys, self.control.keys))

