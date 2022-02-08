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

from blocklinalg import linalg as bla

from .base import DynamicalSystem

# pylint: disable=abstract-method

def bernoulli_q(asep, psub, psup, rho):
    """
    Return the flow rate based on Bernoulli
    """
    q = (2*(psub-psup)/rho)**0.5 * asep
    return q

def bernoulli_p(q, area, ssep, psub, psup, rho):
    """
    Return the pressure based on Bernoulli
    """
    p = psub - 1/2*rho*(q/area)**2
    return p

def wavg(s, f, w):
    """
    Return the weighted average of f(s) over s with weights w(s)
    """
    return jnp.trapz(f*w, s)/jnp.trapz(w, s)

def smooth_min_weight(f, zeta=1):
    """
    Return a smooth minimum from a set of values f
    """
    # w = jnp.exp(-f/zeta)
    log_w = -f/zeta
    log_w = log_w - jnp.min(log_w) # normalize the log weight so the minimum value's weight is 1
    return jnp.exp(log_w)

def sigmoid(s, zeta=1):
    """
    Return the sigmoid function evaluated at s

    This function:
    approaches 0 as s -> -inf
    approaches 1 as s -> +inf
    """
    return 1/(1+jnp.exp(-s/zeta))

@jax.jit
def bernoulli_qp(area, s, psub, psup, rho, zeta_min, zeta_sep):
    wmin = smooth_min_weight(area, zeta_min)
    amin = wavg(s, area, wmin)
    smin = wavg(s, s, wmin)

    asep = amin
    ssep = smin
    q = bernoulli_q(asep, psub, psup, rho)
    p = bernoulli_p(q, area, ssep, psub, psup, rho)
    return q, p

# Use this weird combination of primals/tangent parameters because later we 
# want to compute the jacobian wrt area
def dbernoulli_qp(area, s, psub, psup, rho, zeta_min, zeta_sep, tangents):
    primals = (area, s, psub, psup, rho, zeta_min, zeta_sep)
    return jax.jvp(bernoulli_qp, primals, tangents)

ddbernoulli_qp_darea = jax.jacfwd(dbernoulli_qp, argnums=0)

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

        self.icontrol = bla.BlockVec([np.ones(N)], ['area'])

        self.dq = np.zeros(1)
        self.dp = np.zeros(N)
        self.dstate = bla.BlockVec([self.q, self.p], ['q', 'p'])

        self.dqt = np.zeros(1)
        self.dpt = np.zeros(N)
        self.dstatet = bla.BlockVec([self.qt, self.pt], ['q', 'p'])

        self.dicontrol = bla.BlockVec([np.ones(N)], ['area'])

        properties_vec = ['psub', 'psub', 'rho_air', 'zeta_min', 'zeta_sep']
        self.properties = bla.BlockVec(
            [np.ones(1) for i in range(len(properties_vec))], properties_vec)

class Bernoulli1DDynamicalSystem(BaseFluid1DDynamicalSystem):
    def assem_res(self):
        # Depack variables from BlockVec for input to Bernoulli functions
        area = self.icontrol['area']
        rho = self.properties['rho_air'][0]
        psub = self.properties['psub'][0]
        psup = self.properties['psup'][0]
        zeta_min = self.properties['zeta_min'][0]
        zeta_sep = self.properties['zeta_sep'][0]

        qp_explicit = qp(area, self.s, psub, psup, rho, zeta_min, zeta_sep)
        resq = self.q - qp_explicit[0]
        resp = self.p - qp_explicit[1]
        return bla.BlockVec([resq, resp], self.state.keys)

    def assem_dres_dstate(self):
        dresq_dq = np.diag(np.ones(self.q.size))
        dresq_dp = np.diag(np.zeros(self.q.size, self.p.size))

        dresp_dp = np.diag(np.ones(self.p.size))
        dresp_dq = np.diag(np.zeros(self.p.size, self.q.size))
        mats = [
            [dresq_dq, dresq_dp],
            [dresp_dq, dresp_dp]]
        return bla.BlockMat(mats, row_keys=self.state.keys, col_keys=self.state.keys)

class LinearStateBernoulli1DDynamicalSystem(BaseFluid1DDynamicalSystem):
    def assem_res(self):
        resq = np.ones(1)
        resp = np.ones(self.p.size)
        return bla.BlockVec([resq, resp], self.state.keys)

class LinearStatetBernoulli1DDynamicalSystem(BaseFluid1DDynamicalSystem):
    def assem_res(self):
        resq = np.zeros(1)
        resp = np.zeros(self.p.size)
        return bla.BlockVec([resq, resp], self.state.keys)

class LinearIcontrolBernoulli1DDynamicalSystem(BaseFluid1DDynamicalSystem):
    def assem_res(self):
        # Depack variables from BlockVec for input to Bernoulli functions
        area = self.icontrol['area']
        rho = self.properties['rho_air'][0]
        psub = self.properties['psub'][0]
        psup = self.properties['psup'][0]
        zeta_min = self.properties['zeta_min'][0]
        zeta_sep = self.properties['zeta_sep'][0]
        
        primals = (area, self.s, rho, psub, psup, zeta_min, zeta_sep)
        tangents = (self.dicontrol['area'], 0, 0, 0, 0, 0)
        dqp = dbernoulli_qp(*primals, tangents)
        resq = -dqp[0]
        resp = -dqp[1]
        return bla.BlockVec([resq, resp], self.state.keys)

    def assem_dres_dstate(self):
        dresq_dq = np.diag(np.zeros(self.q.size))
        dresq_dp = np.diag(np.zeros(self.q.size, self.p.size))

        dresp_dp = np.diag(np.zeros(self.p.size))
        dresp_dq = np.diag(np.zeros(self.p.size, self.q.size))
        mats = [
            [dresq_dq, dresq_dp],
            [dresp_dq, dresp_dp]]
        return bla.BlockMat(mats, row_keys=self.state.keys, col_keys=self.state.keys)

    def assem_dres_dstatet(self):
        dresq_dq = np.diag(np.zeros(self.q.size))
        dresq_dp = np.diag(np.zeros(self.q.size, self.p.size))

        dresp_dp = np.diag(np.zeros(self.p.size))
        dresp_dq = np.diag(np.zeros(self.p.size, self.q.size))
        mats = [
            [dresq_dq, dresq_dp],
            [dresp_dq, dresp_dp]]
        return bla.BlockMat(mats, row_keys=self.state.keys, col_keys=self.state.keys)

    def assem_dres_dicontrol(self):
        # Depack variables from BlockVec for input to Bernoulli functions
        area = self.icontrol['area']
        rho = self.properties['rho_air'][0]
        psub = self.properties['psub'][0]
        psup = self.properties['psup'][0]
        zeta_min = self.properties['zeta_min'][0]
        zeta_sep = self.properties['zeta_sep'][0]
        
        primals = (area, rho, psub, psup, zeta_min, zeta_sep)
        tangents = (self.dicontrol['area'], 0, 0, 0, 0, 0)

        ddq_darea, ddp_darea = ddbernoulli_qp_darea(*primals, tangents)
        dresq_darea = -ddq_darea
        dresp_darea = -ddp_darea
        mats = [
            [dresq_darea],
            [dresp_darea]]
        return bla.BlockMat(mats, row_keys=self.state.keys, col_keys=['area'])

