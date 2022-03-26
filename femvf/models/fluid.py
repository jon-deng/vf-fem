"""
Functionality related to fluids

TODO: Change smoothing parameters to all be expressed in units of length 
(all the smoothing parameters have a smoothing effect that occurs over a small length. 
The smaller the length, the sharper the smoothing. )
"""
import warnings

import numpy as np
# import autograd
# import autograd.numpy as np

import dolfin as dfn
from petsc4py import PETSc

from . import base
from ..parameters.properties import property_vecs
from ..constants import PASCAL_TO_CGS, SI_DENSITY_TO_CGS
from blocktensor.linalg import BlockVec

## 1D Bernoulli approximation codes

class QuasiSteady1DFluid(base.Model):
    """
    This class represents a 1D fluid model

    Parameters
    ----------
    s: np.ndarray
        streamwise channel centreline coordinates
    area: np.ndarray
        channel cross-sectional areas
    """
    def __init__(self, s, area):
        NVERT = s.size
        # the 'mesh' (also the x coordinates in the reference configuration)
        self.s_vertices = s

        p0 = np.zeros(NVERT)
        q0 = np.zeros((1,))
        self.state0 = BlockVec((q0, p0), ('q', 'p'))

        p1 = np.zeros(NVERT)
        q1 = np.zeros(1)
        self.state1 = BlockVec((q1, p1), ('q', 'p'))

        # form type quantities associated with the mesh
        # displacement and velocity along the surface at state 0 and 1
        area = np.zeros(NVERT)
        psub, psup = np.zeros(1), np.zeros(1)
        self.control = BlockVec((area, psub, psup), ('area', 'psub', 'psup'))

        self.properties = self.get_properties_vec(set_default=True)

        self._dt = 1.0

    @property
    def fluid(self):
        return self
        
    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        self._dt = value

    ## Parameter setting functions
    def set_ini_state(self, state):
        """
        Set the initial fluid state
        """
        self.state0[:] = state

    def set_fin_state(self, state):
        """
        Set the final fluid state
        """
        self.state1[:] = state

    def set_control(self, control):
        """
        Set the final surface displacement and velocity
        """
        self.control[:] = control

    def set_properties(self, props):
        """
        Set the fluid properties
        """
        self.properties[:] = props

    ## Get empty vectors
    def get_state_vec(self):
        """
        Return empty flow speed and pressure state vectors
        """
        q, p = np.zeros((1,)), np.zeros(self.x_vertices.size)
        return BlockVec((q, p), ('q', 'p'))

    def get_properties_vec(self, set_default=True):
        """
        Return a BlockVec representing the properties of the fluid
        """
        field_size = 1
        prop_defaults = None
        if set_default:
            prop_defaults = self.PROPERTY_DEFAULTS
        vecs, labels = property_vecs(field_size, self.PROPERTY_TYPES, prop_defaults)

        return BlockVec(vecs, labels)

    def get_control_vec(self):
        ret = self.control.copy()
        ret.set(0.0)
        return ret

    ## Residual and sensitivity methods
    # These are all specific to the Bernoulli model below?
    def apply_dres_dstate0(self, x):
        dres = self.get_state_vec()
        dres.set(0.0)
        return dres

class Bernoulli(QuasiSteady1DFluid):
    """
    Represents the Bernoulli fluid model

    TODO : Refactor this to behave similar to the Solid model. A mesh should be used, corresponding
    to the reference configuration of the fluid (conformal with reference configuration of solid?)
    One of the properties should be the mapping from the reference configuration to the current
    configuration that would be used in ALE.

    Note there are a total of 4 smoothing properties that are meant to approximate the
    ad-hoc separation criteria used in the literature.
        zeta_amin :
            Factor controlling the smoothness of the minimum area function.
            As zeta_amin->0, the exact minimum area is produced (Should behave similarly as a true 
            min function)
        zeta_ainv :
            Approximates smoothing for an inverse area function; given an area, return the surface
            coordinate. As zeta_ainv->0, the function selects the closest surface coordinate at
            where the area matches some target area.
        zeta_sep :
            Controls the sharpness of the cutoff that models separation. As zeta_sep->0,
            the sharpness will approach an instantaneous jump from 1 to 0
        zeta_lb : 
            Controls the smoothness of the lower bound function that restricts glottal area to be
            larger than a lower bound

    Properties
    ----------
    
    """
    PROPERTY_TYPES = {
        'a_sub': ('const', ()),
        'a_sup': ('const', ()),
        'rho_air': ('const', ()),
        'r_sep': ('const', ()),
        'zeta_amin': ('const', ()),
        'zeta_sep': ('const', ()),
        'zeta_ainv': ('const', ()),
        'zeta_lb': ('const', ()),
        'area_lb': ('const', ())}

    PROPERTY_DEFAULTS = {
        'a_sub': 100000,
        'a_sup': 0.6,
        'r_sep': 1.0,
        'rho_air': 1.225 * SI_DENSITY_TO_CGS,
        'zeta_amin': 0.002/3,
        'zeta_sep': 0.002/3,
        'zeta_ainv': 2.5*0.002,
        'zeta_lb': 0.002/3,
        'area_lb': 0.001}

    ## Model res sensitivity interface
    def res(self):
        return self.state1 - self.solve_state1(self.state0)[0]

    def solve_state1(self, state1):
        """
        Return the final flow state
        """
        return self.fluid_pressure(self.s_vertices, *self.control.vecs, self.properties)

    def solve_dres_dstate1(self, b):
        return b

    def solve_dres_dstate1_adj(self, x):
        return x
        
    def apply_dres_dp_adj(self, x):
        b = self.get_properties_vec()
        b.set(0.0)
        return b

    def apply_dres_dcontrol(self, x):
        raise NotImplementedError()
        
    def apply_dres_dcontrol_adj(self, x):
        raise NotImplementedError()


## Bernoulli fluid pressure
def pbernoulli(qsqr, pref, aref, area, rho):
    return pref + 1/2*rho*qsqr*(aref**-2 - area**-2)

def dpbernoulli(qsqr, pref, aref, area, rho):
    dpbern_dsqsr = 1/2*rho*(aref**-2 - area**-2)
    dpbern_dpref = 1.0
    dpbern_daref = 1/2*rho*qsqr*-2*aref**-3
    dpbern_darea = 1/2*rho*qsqr*2*area**-3
    return dpbern_dsqsr, dpbern_dpref, dpbern_daref, dpbern_darea

def flow_rate_sqr(pref, aref, psep, asep, rho):
    qsqr = 0.0
    with np.errstate(divide='raise'):
        qsqr = 2/rho*(psep - pref)/(aref**-2 - asep**-2)
    return qsqr

def dflow_rate_sqr(pref, aref, psep, asep, rho):
    with np.errstate(divide='raise'):
        qsqr = 2/rho*(psep - pref)/(aref**-2 - asep**-2)
        dqsqr_dpref = -2/rho/(aref**-2 - asep**-2)
        dqsqr_dpsep = 2/rho/(aref**-2 - asep**-2)
        dqsqr_dasep = -2/rho*(psep - pref)/(aref**-2 - asep**-2)**2 * (2/asep**3)
        return dqsqr_dpref, dqsqr_dpsep, dqsqr_dasep

def fluid_pressure(s, area, psub, psup, fluid_props):
    """
    Computes the pressure loading at a series of surface nodes according to Pelorson (1994)

    TODO: I think it would make more sense to treat this as a generic Bernoulli pressure
    calculator. It could be refactored to not use an self variable, instead it would pass a
    reference surface mesh, and current surface mesh (x, y coordinates of vertices in increasing
    streamwise direction).

    Parameters
    ----------
    surface_state : tuple of (u, v) each of (NUM_VERTICES, GEOMETRIC_DIM) np.ndarray
        States of the surface vertices, ordered following the flow (increasing x coordinate).
    fluid_props : BlockVec
        A dictionary of fluid properties.

    Returns
    -------
    q : float
        The flow rate
    p : np.ndarray
        An array of pressure vectors for each each vertex
    xy_min, xy_sep: (2,) np.ndarray
        The coordinates of the vertices at minimum and separation areas
    """
    flow_sign = np.sign(psub-psup)
    rho = fluid_props['rho_air']
    asub = fluid_props['a_sub']
    asup = fluid_props['a_sup']

    a = area
    asafe = smoothlb(a, fluid_props['area_lb'], fluid_props['zeta_lb'])

    # Calculate minimum and separation areas/locations
    smin, amin = smooth_min_area(s, asafe, fluid_props['zeta_amin'])

    # Bound all areas to the 'smooth' minimum area above
    # This is done because of the weighting scheme; the smooth min is always slightly larger
    # then the actual min, which leads to problems in the calculated pressures at very small
    # areas where small areas can have huge area ratios leading to weird Bernoulli behaviour
    asafe = smoothlb(asafe, amin, fluid_props['zeta_lb'])

    asep = fluid_props['r_sep'] * amin
    zeta_sep, zeta_ainv = fluid_props['zeta_sep'], fluid_props['zeta_ainv']
    ssep = smooth_separation_point(s, smin, asafe, asep, zeta_sep, zeta_ainv, flow_sign)
    
    # Compute the flow rate based on pressure drop from "reference" to 
    # separation location
    pref, psep, aref = None, None, None
    if flow_sign >= 0:
        pref, aref = psub, asub
        psep = psup
    else:
        pref, aref = psup, asup
        psep = psub

    qsqr = flow_rate_sqr(pref, aref, psep, asep, rho)
    pbern = pbernoulli(qsqr, pref, aref, asafe, rho)

    sepmult = -(flow_sign-1)/2 + flow_sign*smoothstep(s, ssep, alpha=fluid_props['zeta_sep'])
    p = sepmult * pbern + (1-sepmult)*psep
    q = flow_sign*qsqr**0.5 *fluid_props['vf_length']

    # These are not really used anywhere, mainly output for debugging purposes
    idx_min = np.argmax(s>smin)
    idx_sep = np.argmax(s>ssep)
    
    info = {'flow_rate': q,
            'idx_sep': idx_sep,
            'idx_min': idx_min,
            's_sep': ssep,
            's_min': smin,
            'a_min': amin,
            'a_sep': asep,
            'area': asafe,
            'pressure': p}
    return BlockVec((np.array(q), p), ('q', 'p')), info

def flow_sensitivity(s, area, psub, psup, fluid_props):
    """
    Return the sensitivities of pressure and flow rate to the surface state.

    Parameters
    ----------
    surface_state : tuple of (u, v, a) each of (NUM_VERTICES, GEOMETRIC_DIM) np.ndarray
        States of the surface vertices, ordered following the flow (increasing x coordinate).
    fluid_props : BlockVec
        A dictionary of fluid property keyword arguments.
    """
    flow_sign = np.sign(psub-psup)
    rho = fluid_props['rho_air']
    asub = fluid_props['a_sub']
    asup = fluid_props['a_sup']

    a = area
    # da_dy = -2

    asafe = smoothlb(a, fluid_props['area_lb'], fluid_props['zeta_lb'])
    dasafe_da = dsmoothlb_df(a, fluid_props['area_lb'], fluid_props['zeta_lb'])

    smin, amin = smooth_min_area(s, asafe, fluid_props['zeta_amin'])
    dsmin_dasafe, damin_dasafe = dsmooth_min_area(s, asafe, fluid_props['zeta_amin'])
    dsmin_da = dsmin_dasafe*dasafe_da
    damin_da = damin_dasafe*dasafe_da

    asep = fluid_props['r_sep'] * amin
    dasep_damin = fluid_props['r_sep']
    dasep_da = dasep_damin * damin_da

    zeta_sep, zeta_ainv = fluid_props['zeta_sep'], fluid_props['zeta_ainv']
    ssep = smooth_separation_point(s, smin, asafe, asep, zeta_sep, zeta_ainv, flow_sign)
    dssep_dsmin, dssep_dasafe, dssep_dasep = \
        dsmooth_separation_point(s, smin, asafe, asep, zeta_sep, zeta_ainv, flow_sign)
    dssep_da = (dssep_dsmin*dsmin_da + dssep_dasafe*dasafe_da
                + dssep_dasep*dasep_da) #* da_dy

    # Compute the flow rate based on pressure drop from "reference" to 
    # separation location
    pref, psep, aref = None, None, None
    if flow_sign >= 0:
        pref, aref = psub, asub
        psep = psup
    else:
        pref, aref = psup, asup
        psep = psub

    qsqr = flow_rate_sqr(pref, aref, psep, asep, rho)
    dqsqr_dpref, dqsqr_dpsep, dqsqr_dasep = dflow_rate_sqr(pref, aref, psep, asep, rho)
    dqsqr_da = dqsqr_dasep*dasep_da

    # Find Bernoulli pressure
    pbern = pbernoulli(qsqr, pref, aref, asafe, rho)
    dpbern_dqsqr, dpbern_dpref, dpbern_daref, dpbern_dasafe = dpbernoulli(qsqr, pref, aref, asafe, rho)
    dpbern_da = dpbern_dasafe*dasafe_da + dpbern_dqsqr*dqsqr_da
    dpbern_dpref = dpbern_dpref + dpbern_dqsqr*dqsqr_dpref
    dpbern_dpsep = dpbern_dqsqr*dqsqr_dpsep

    # Correct Bernoulli pressure by applying a smooth mask after separation
    zeta_sep = fluid_props['zeta_sep']
    sepmult = -(flow_sign-1)/2 + flow_sign*smoothstep(s, ssep, zeta_sep)
    dsepmult_da = flow_sign*dsmoothstep_dx0(s, ssep, zeta_sep)[:, None] * dssep_da

    # p = sepmult * pbern + (1-sep_multiplier)*psep
    dp_da = sepmult[:, None]*dpbern_da + dsepmult_da*pbern[:, None]
    dp_dpref = sepmult*dpbern_dpref
    dp_dpsep = sepmult*dpbern_dpsep + (1-sepmult)

    # q = flow_sign*qsqr**0.5
    dq_da = 0.5*flow_sign*qsqr**-0.5 * dqsqr_da
    dq_dpref = 0.5*flow_sign*qsqr**-0.5 * dqsqr_dpref*fluid_props['vf_length']
    dq_dpsep = 0.5*flow_sign*qsqr**-0.5 * dqsqr_dpsep*fluid_props['vf_length']

    dq_dpsub, dp_dpsub, dq_dpsup, dp_dpsup = None, None, None, None
    if flow_sign >= 0:
        dq_dpsub, dp_dpsub = dq_dpref, dp_dpref
        dq_dpsup, dp_dpsup = dq_dpsep, dp_dpsep
    else:
        dq_dpsub, dp_dpsub = dq_dpsep, dp_dpsep
        dq_dpsup, dp_dpsup = dq_dpref, dp_dpref

    return dq_da, dp_da, dq_dpsub, dp_dpsub, dq_dpsup, dp_dpsup

## Bernoulli minimum/separation area functions
def min_area(s, area):
    # find the smallest, minimum area index (if there's multiple)
    idx_min = np.min(np.argmin(area))
    return s[idx_min], area[idx_min]

def smooth_min_area(s, area, zeta_amin):
    if zeta_amin == 0:
        return min_area(s, area)
    else:
        wmin = expweight(area, zeta_amin)
        amin = wavg(s, area, wmin)
        smin = wavg(s, s, wmin)
        return smin, amin

def dsmooth_min_area(s, area, zeta_amin):
    wmin = expweight(area, zeta_amin)
    dwmin_darea = dexpweight_df(area, zeta_amin)
    
    # amin = wavg(s, area, wmin)
    # smin = wavg(s, s, wmin)
    damin_darea = dwavg_dw(s, area, wmin)*dwmin_darea + dwavg_df(s, area, wmin)
    dsmin_darea = dwavg_dw(s, s, wmin) * dwmin_darea
    return dsmin_darea, damin_darea

def separation_point(s, smin, area, asep):
    # Find the first index where the area function is larger than the separation area
    bool_seps = np.logical_and(s[:-1]>=smin, np.logical_and(area[:-1] <= asep, area[1:] > asep))
    idx_seps = np.arange(s.size-1)[bool_seps]
    idx_sep = np.min(idx_seps)

    ssep = s[idx_sep]
    return ssep

def smooth_separation_point(s, smin, area, asep, zeta_sep, zeta_ainv, flow_sign):
    """
    flow_dir : float (1 or -1)
    """
    if zeta_ainv == 0:
        return separation_point(s, smin, area, asep)
    else:
        # This ensures the separation area is selected at a point past the minimum area
        log_wsep = None
        with np.errstate(divide='ignore'):
            # this applies the condition where the separation point is selected only after the 
            # minimum area
            # this should represent area/hh, while handling the divide by zero errors
            hh = (1+flow_sign)/2-flow_sign*smoothstep(s, smin, zeta_sep)
            farea = area/hh
            log_wsep = log_gaussian(farea, asep, zeta_ainv)
        wsep = np.exp(log_wsep - log_wsep.max())
        ssep = wavg(s, s, wsep)

        assert not np.isnan(ssep)

        return ssep

def dsmooth_separation_point(s, smin, area, asep, zeta_sep, zeta_ainv, flow_sign):
    # caculate sensitivity of the separation coordinate `ssep` (see `smooth_separation_point`)
    log_wsep = None
    farea, dfarea_dsmin, dfarea_darea = None, None, None
    with np.errstate(divide='ignore', invalid='raise'):
        hh = (1+flow_sign)/2-flow_sign*smoothstep(s, smin, zeta_sep)
        dhh_dsmin = -flow_sign*dsmoothstep_dx0(s, smin, zeta_sep)

        farea = area/hh
        dfarea_dsmin = np.zeros(area.size)
        idx = (hh != 0) # this is needed because the below product is indeterminate sometimes
        dfarea_dsmin[idx] = -area[idx]/hh[idx]**2 * dhh_dsmin[idx]
        dfarea_darea = 1/hh

        log_wsep = log_gaussian(farea, asep, zeta_ainv)
    wsep = np.exp(log_wsep - log_wsep.max())
    dwsep_dfarea = dgaussian_dx(farea, asep, zeta_ainv)
    dwsep_dasep = dgaussian_dx0(farea, asep, zeta_ainv)
    dwsep_darea = np.zeros(wsep.shape)
    idx_valid = dwsep_dfarea != 0
    dwsep_darea[idx_valid] = dwsep_dfarea[idx_valid]*dfarea_darea[idx_valid]
    dwsep_dsmin = dwsep_dfarea*dfarea_dsmin
    # ssep = wavg(s, s, wsep)

    dssep_dwsep = dwavg_dw(s, s, wsep)*np.exp(-log_wsep.max())
    dssep_darea = dssep_dwsep*dwsep_darea
    dssep_dasep = np.dot(dssep_dwsep, dwsep_dasep)
    dssep_dsmin = np.dot(dssep_dwsep, dwsep_dsmin)

    return dssep_dsmin, dssep_darea, dssep_dasep

## Weighted average function
def wavg(s, f, w):
    """
    Return the weighted average of 'f(s)' with weight 'w(s)'
    """
    # This handles the special case where the w is non-zero at one index and zero
    # everywhere else. If this isn't done, floating point errors will make it
    # so that wavg(s, f, w) doesn't equal the non-zero weight location of f
    favg = None
    if np.sum(w != 0) == 1: 
        idx_nonzero = np.argmax(w != 0)
        favg = f[idx_nonzero]
    else:
        favg = trapz(f*w, s) / trapz(w, s)
    return favg

def dwavg_df(s, f, w):
    # trapz(f*w, s) / trapz(w, s)

    # num = trapz(f*w, s)
    den = trapz(w, s)

    dnum_df = dtrapz_df(f*w, s)*w
    # dden_df = 0.0

    return dnum_df/den

def dwavg_dw(s, f, w):
    # trapz(f*w, s) / trapz(w, s)

    num = trapz(f*w, s)
    den = trapz(w, s)

    dnum_dw = dtrapz_df(f*w, s)*f
    dden_dw = dtrapz_df(w, s)

    return (dnum_dw*den - num*dden_dw)/den**2

## Smoothed lower bound function
def smoothlb(f, f_lb, alpha=1.0):
    """
    Return the value of `f` subject to a smooth lower bound `f_lb`

    Function is based on a scaled and shifted version of the 'SoftPlus' function. This function
    smoothly blends a constant function when f<f_lb with a linear function when f>f_lb.

    The 'region' of smoothness is roughly characterized by 'df = f-f_lb', where the function is 95%
    a straight line when `df/alpha = 3`.

    Parameters
    ----------
    f : array_like
    f_lb : float
        The minimum possible value of `f`
    alpha : float
        The level of smoothness of the bounded function. This quantity has units of [cm^-1] if `f`
        has units of [cm]. Larger values of alpha increase the sharpness of the bound.
    """
    out = np.zeros(f.shape)
    if alpha == 0.0:
        out[f < f_lb] = f_lb
        out[f >= f_lb] = f[f >= f_lb]
    else:
        # Manually return 1 if the exponent is large enough to cause overflow
        exponent = (f-f_lb)/alpha
        idx_underflow = exponent <= -50.0
        idx_normal = np.logical_and(exponent > -50.0, exponent <= 50.0)
        idx_overflow = exponent > 50.0

        out[idx_underflow] = f_lb
        out[idx_normal] = alpha*np.log(1 + np.exp(exponent[idx_normal])) + f_lb
        out[idx_overflow] = f[idx_overflow]
    return out

def dsmoothlb_df(f, f_lb, alpha=1.0):
    """
    Return the sensitivity of `smooth_lower_bound` to `f`

    Parameters
    ----------
    f : array_like
    f_lb : float
        The minimum possible value of `f`
    alpha : float
        The level of smoothness of the bounded function. This quantity has units of [cm^-1] if `f`
        has units of [cm]. Larger values of alpha increase the sharpness of the bound.
    """
    out = np.zeros(f.shape)
    if alpha == 0.0:
        out[f >= f_lb] = 1.0
    else:
        # Manually return limiting values if the exponents are large enough to cause overflow
        exponent = (f-f_lb)/alpha
        # idx_underflow = exponent <= -50.0
        idx_normal = np.logical_and(exponent > -50.0, exponent <= 50.0)
        idx_overflow = exponent > 50.0

        # out[idx_underflow] = 0
        out[idx_normal] = np.exp(exponent[idx_normal]) / (1+np.exp(exponent[idx_normal]))
        out[idx_overflow] = 1.0
    return out

## Exponential weighting function (for smooth min)
def expweight(f, alpha=1.0):
    """
    Return exponential weights as exp(-1*f/alpha) 
    """
    w = np.zeros(f.shape)
    if alpha == 0:
        w[f == np.min(f)] = 1.0
    else:
        # For numerical stability subtract a judicious constant from `alpha*x` to prevent exponents
        # being too large (overflow). This constant factors when you weights in an average
        K_STABILITY = np.max(-f/alpha)
        w[:] = np.exp(-f/alpha - K_STABILITY)
    return w

def dexpweight_df(f, alpha=1.0):
    dw_df = np.zeros(f.shape)
    if alpha == 0:
        dw_df[np.argmin(f)] = 1.0
    else:
        K_STABILITY = np.max(-f/alpha)
        dw_df[:] = -1/alpha*np.exp(-f/alpha - K_STABILITY)
    return dw_df

## Trapezoidal integration rule
def trapz(f, s):
    """
    Return the integral of `f` over `s` using the trapezoidal rule
    """
    assert len(f.shape) == 1
    assert len(s.shape) == 1

    return np.sum((s[1:]-s[:-1])*(f[1:]+f[:-1])/2)

def dtrapz_df(f, s):
    """
    Return the sensitivity of `trapz` to `f`
    """
    out = np.zeros(f.size)
    out[:-1] += (s[1:]-s[:-1]) / 2
    out[1:] += (s[1:]-s[:-1]) / 2
    return out

## Smoothed Heaviside cutoff function
def sigmoid(x):
    """
    Return the sigmoid function evaluated at `x`
    """
    exponent = -x
    idx_underflow = exponent <= -50.0
    idx_normal = np.logical_and(exponent > -50.0, exponent < 50.0)
    idx_overflow = exponent >= 50.0

    out = np.zeros(x.shape)
    out[idx_underflow] = 1.0
    out[idx_normal] = 1/(1+np.exp(exponent[idx_normal]))
    out[idx_overflow] = 0.0

    return out

def dsigmoid_dx(x):
    """
    Return the sensitivity of `sigmoid` to `x`

    This returns a scalar representing the diagonal of the sensitivity matrix
    """
    sig = sigmoid(x)
    return sig * (1-sig)

def smoothstep(x, x0, alpha=1.0):
    """
    Return the mirrored logistic function evaluated at x-x0

    This steps from 1.0 when x << x0 to 0.0 when x >> x0.

    The 'region' of smoothness is roughly characterized by dx. If x = x0 + dx, the cutoff function
    will drop to just 5% if dx/alpha = 3.
    """
    arg = np.zeros(x.shape)
    if alpha == 0.0:
        arg[x == x0] = 0.0
        arg[x > x0] = -np.inf
        arg[x < x0] = np.inf
    else:
        arg[:] = -(x-x0)/alpha
    return sigmoid(arg)

def dsmoothstep_dx(x, x0, alpha=1.0):
    """
    Return the logistic function evaluated at x-xref
    """
    if alpha == 0:
        return np.zeros(x.shape)
    else:
        arg = -(x-x0)/alpha
        darg_dx = -1/alpha
        return dsigmoid_dx(arg) * darg_dx

def dsmoothstep_dx0(x, x0, alpha=1.0):
    """
    Return the logistic function evaluated at x-xref
    """
    if alpha == 0:
        return np.zeros(x.shape)
    else:
        arg = -(x-x0)/alpha
        darg_dx0 = 1/alpha
        return dsigmoid_dx(arg) * darg_dx0

## Smoothed gaussian selection function
def log_gaussian(x, x0, alpha=1.0):
    """
    Return the log of the gaussian with mean `x0` and variance `alpha`
    """
    out = np.zeros(x.shape)
    if alpha == 0.0:
        out[:] = -np.inf
        arg = np.abs(x-x0)
        out[arg == np.min(arg)] = 0.0
    else:
        out[:] = -((x-x0)/alpha)**2
    return out

def gaussian(x, x0, alpha=1.0):
    """
    Return the 'gaussian' with mean `x0` and variance `alpha`

    The gaussian is scaled by an arbitrary factor, C, so that the output has a maximum value of 1.0.
    This is needed for numerical stability, otherwise for small `alpha` the gaussian will simply be 
    zero everywhere. If the `gaussian` weights are used in an averaging scheme then any constant
    factor will not matter since they cancel out in the ratio.
    """
    out = np.zeros(x.shape)
    if alpha == 0.0:
        arg = np.abs(x-x0)
        out[arg == np.min(arg)] = 1.0
    else:
        arg = ((x-x0)/alpha)**2
        K_STABILITY = np.min(arg)
        out[:] = np.exp(-arg+K_STABILITY)
    return out

def dgaussian_dx(x, x0, alpha=1.0):
    """
    Return the sensitivity of `gaussian` to `x`
    """
    if alpha == 0:
        return np.zeros(x.shape)
    else:
        # indexes are used here because the expression is indeterminate
        g = gaussian(x, x0, alpha)
        dg_dx = np.zeros(g.shape)
        idx = g != 0.0
        dg_dx[idx] = g[idx] * -2*((x[idx]-x0)/alpha) / alpha
        return dg_dx

def dgaussian_dx0(x, x0, alpha=1.0):
    """
    Return the sensitivity of `gaussian` to `x0`
    """
    return -dgaussian_dx(x, x0, alpha)
