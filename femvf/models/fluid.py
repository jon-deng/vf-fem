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
from ..linalg import BlockVec, general_vec_set

## 1D Bernoulli approximation codes

class QuasiSteady1DFluid(base.Model):
    """
    This class represents a 1D fluid
    """
    def __init__(self, x_vertices, y_surface):
        """

        Parameters
        ----------
        x_vertices: np.ndarray
            Array of x vertex locations numbered in steamwise increasing order.
        y_surface: np.ndarray
            Array of y surface locations numbered in steamwise increasing order.
        """

        NVERT = x_vertices.size
        # the 'mesh' (also the x coordinates in the reference configuration)
        self.x_vertices = x_vertices

        # the surface y coordinates of the solid
        self.y_surface = y_surface

        # Calculate surface coordinates which are needed to compute surface integrals
        dx = self.x_vertices[1:] - self.x_vertices[:-1]
        dy = self.y_surface[1:] - self.y_surface[:-1]
        ds = (dx**2+dy**2)**0.5
        self.s_vertices = np.concatenate(([0.0], np.cumsum(ds)))

        p0 = np.zeros(NVERT)
        q0 = np.zeros((1,))
        self.state0 = BlockVec((q0, p0), ('q', 'p'))

        p1 = np.zeros(NVERT)
        q1 = np.zeros(1)
        self.state1 = BlockVec((q1, p1), ('q', 'p'))

        # form type quantities associated with the mesh
        # displacement and velocity along the surface at state 0 and 1
        usurf = np.zeros(2*NVERT)
        vsurf = np.zeros(2*NVERT)
        psub, psup = np.zeros(1), np.zeros(1)
        self.control = BlockVec((usurf, vsurf, psub, psup), ('usurf', 'vsurf', 'psub', 'psup'))

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
            Factor controlling the smoothness of the approximation of minimum area.
            A value of 0 weights areas of all nodes evenly, while a value of -inf
            returns the exact minimum area. Large negative values return an
            average of the smallest areas.
        zeta_ainv :
            Approximates smoothing for an inverse area function; given an area, return the surface
            coordinate
        zeta_sep :
            Controls the sharpness of the cutoff that models separation. as zeta_sep approaches inf,
            the sharpness will approach an instantaneous jump from 1 to 0
        zeta_lb : 
            Controls the smoothness of a lower bound function that prevents negative glottal areas

    Properties
    ----------
    
    """
    PROPERTY_TYPES = {
        'y_midline': ('const', ()),
        'a_sub': ('const', ()),
        'a_sup': ('const', ()),
        'vf_length': ('const', ()),
        'rho_air': ('const', ()),
        'r_sep': ('const', ()),
        'zeta_amin': ('const', ()),
        'zeta_sep': ('const', ()),
        'zeta_ainv': ('const', ()),
        'zeta_lb': ('const', ()),
        'ygap_lb': ('const', ())}

    PROPERTY_DEFAULTS = {
        'y_midline': 1e6,
        'a_sub': 100000,
        'a_sup': 0.6,
        'vf_length': 1.0,
        'r_sep': 1.0,
        'rho_air': 1.225 * SI_DENSITY_TO_CGS,
        'zeta_amin': 0.002/3,
        'zeta_sep': 0.002/3,
        'zeta_ainv': 2.5*0.002,
        'zeta_lb': 0.002/3,
        'ygap_lb': 0.001}

    # TODO: Refactor as solve_dres_dcontrol
    def solve_dqp1_du1(self, adjoint=False):
        """
        Return the final flow state
        """
        return self.flow_sensitivity(*self.control.vecs, self.properties)

    # TODO: Remove this. Coupling to the solid should be done in a coupling model
    def solve_dqp1_du1_solid(self, model, adjoint=False):
        """
        Return the final flow state
        """
        return self.flow_sensitivity_solid(model, *self.control.vecs, self.properties,
                                           adjoint)

    ## internal methods
    def separation_point(self, s, smin, area, asep, zeta_sep, zeta_ainv, flow_sign):
        """
        flow_dir : float (1 or -1)
        """
        # This ensures the separation area is selected at a point past the minimum area
        log_wsep = None
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'divide by zero encountered in log')
            hh = (1+flow_sign)/2-flow_sign*smoothstep(s, smin, zeta_sep)

            # this applies the condition where the separation point is selected only after the 
            # minimum area
            # this should represent area/hh, while handling the divide by zero errors
            filtered_area = np.zeros(area.shape)
            idx = hh == 0
            filtered_area[idx] = np.sign(area[idx])*np.inf
            idx = hh != 0
            filtered_area[idx] = area[idx]/hh[idx]

            log_wsep = log_gaussian(filtered_area, asep, zeta_ainv)
        wsep = np.exp(log_wsep - log_wsep.max())
        ssep = wavg(s, s, wsep)

        assert not np.isnan(ssep)

        return ssep

    def dseparation_point(self, s, amin, smin, a, fluid_props):
        
        asep = fluid_props['r_sep'] * amin
        dasep_damin = fluid_props['r_sep']
        dasep_dsmin = 0.0 
        dasep_da = 0.0

        # This ensures the separation area is selected at a point past the minimum area
        wpostsep = 1-smoothstep(s, smin, fluid_props['zeta_sep'])
        dwpostsep_dsmin = -dsmoothstep_dx0(s, smin, fluid_props['zeta_sep'])

        wgaussian = gaussian(a, asep, fluid_props['zeta_ainv'])
        dwgaussian_da = dgaussian_dx(a, asep, fluid_props['zeta_ainv'])
        dwgaussian_damin = dgaussian_dx0(a, asep, fluid_props['zeta_ainv']) * dasep_damin

        # wsep = wpostsep * wgaussian
        log_wsep = None
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'divide by zero encountered in log')
            log_wsep = np.log(wpostsep) +  log_gaussian(a, asep, fluid_props['zeta_ainv'])
        wsep = np.exp(log_wsep - log_wsep.max())

        dwsep_dsmin = dwpostsep_dsmin * wgaussian
        dwsep_damin = wpostsep * dwgaussian_damin 
        dwsep_da = wpostsep * dwgaussian_da

        # ssep = wavg(s, s, wsep)
        # the factor below is because a scaled wsep was used for numerical stability
        # TODO: Make weighted averages accept log weights to automatically account for numerical stability
        # issues?
        dssep_dwsep = dwavg_dw(s, s, wsep)*np.exp(-log_wsep.max())
        dssep_da = dssep_dwsep*dwsep_da
        dssep_damin = np.dot(dssep_dwsep, dwsep_damin)
        dssep_dsmin = np.dot(dssep_dwsep, dwsep_dsmin)

        return dssep_damin, dssep_dsmin, dssep_da, dasep_damin, dasep_dsmin, dasep_da

    def min_area(self, s, a, fluid_props):
        wmin = expweight(a, fluid_props['zeta_amin'])
        amin = wavg(s, a, wmin)
        return amin

    def fluid_pressure(self, usurf, vsurf, psub, psup, fluid_props):
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
        s = self.s_vertices

        # x = surface_state[0][:-1:2]
        y = usurf[1::2]

        a = 2 * (fluid_props['y_midline'] - y) * fluid_props['vf_length']
        asafe = smoothlb(a, 2*fluid_props['ygap_lb']*fluid_props['vf_length'], fluid_props['zeta_lb'])

        # Calculate minimum and separation areas/locations
        wmin = expweight(asafe, fluid_props['zeta_amin'])
        # special case due to floating point errors when wmin has only 1 weighted point
        amin, smin = np.nan, np.nan
        if np.sum(wmin == 0) == wmin.size-1:
            idx_min = np.argmin(asafe)
            amin = asafe[idx_min]
            smin = s[idx_min]
        else:
            amin = wavg(s, asafe, wmin)
            smin = wavg(s, s, wmin)

        # Bound all areas to the 'smooth' minimum area above
        # This is done because of the weighting scheme; the smooth min is always slightly larger
        # then the actual min, which leads to problems in the calculated pressures at very small
        # areas where small areas can have huge area ratios leading to weird Bernoulli behaviour
        asafe = smoothlb(asafe, amin, fluid_props['zeta_lb'])

        asep = fluid_props['r_sep'] * amin
        zeta_sep, zeta_ainv = fluid_props['zeta_sep'], fluid_props['zeta_ainv']
        ssep = self.separation_point(s, smin, asafe, asep, zeta_sep, zeta_ainv, flow_sign)
        
        # Compute the flow rate based on pressure drop from "reference" to 
        # separation location
        pref, aref = (psub, asub) if flow_sign >= 0 else (psup, asup)
        psep = psup if flow_sign >= 0 else psub
        qsqr = 0.0
        with np.errstate(divide='raise'):
            qsqr = 2/rho*(psep - pref)/(aref**-2 - asep**-2)
        assert qsqr >= 0.0

        pbern = pref + 1/2*rho*qsqr*(aref**-2 - asafe**-2)
        sep_multiplier = -(flow_sign-1)/2 + flow_sign*smoothstep(self.s_vertices, ssep, alpha=fluid_props['zeta_sep'])
        p = sep_multiplier * pbern + (1-sep_multiplier)*psep
        q = flow_sign*qsqr**0.5

        # These are not really used anywhere, mainly output for debugging purposes
        idx_min = np.argmax(s>smin)
        idx_sep = np.argmax(s>ssep)
        xy_min = usurf.reshape(-1, 2)[idx_min]
        xy_sep = usurf.reshape(-1, 2)[idx_sep]
        
        info = {'flow_rate': q,
                'idx_sep': idx_sep,
                'idx_min': idx_min,
                's_sep': ssep,
                's_min': smin,
                'xy_min': xy_min,
                'xy_sep': xy_sep,
                'a_min': amin,
                'a_sep': asep,
                'area': asafe,
                'pressure': p}
        return BlockVec((np.array(q), p), ('q', 'p')), info

    def flow_sensitivity(self, usurf, vsurf, psub, psup, fluid_props):
        """
        Return the sensitivities of pressure and flow rate to the surface state.

        Parameters
        ----------
        surface_state : tuple of (u, v, a) each of (NUM_VERTICES, GEOMETRIC_DIM) np.ndarray
            States of the surface vertices, ordered following the flow (increasing x coordinate).
        fluid_props : BlockVec
            A dictionary of fluid property keyword arguments.
        """
        assert psub > psup
        assert usurf.size%2 == 0
        rho = fluid_props['rho_air']
        asub = fluid_props['a_sub']
        s = self.s_vertices

        # x = surface_state[0][:-1:2]
        y = usurf[1::2]

        a = 2 * (fluid_props['y_midline'] - y)
        da_dy = -2

        asafe = smoothlb(a, 2*fluid_props['ygap_lb'], fluid_props['zeta_lb'])
        dasafe_da = dsmoothlb_df(a, 2*fluid_props['ygap_lb'], fluid_props['zeta_lb'])

        wmin = expweight(asafe, fluid_props['zeta_amin'])
        dwmin_da = dexpweight_df(asafe, fluid_props['zeta_amin']) * dasafe_da

        amin = wavg(s, asafe, wmin)
        damin_da = dwavg_df(s, asafe, wmin)*dasafe_da + \
                   dwavg_dw(s, asafe, wmin)*dwmin_da
        smin = wavg(s, s, wmin)
        dsmin_da = dwavg_dw(s, s, wmin)*dwmin_da

        ssep, asep = self.separation_point(s, amin, smin, asafe, fluid_props)
        dssep_damin, dssep_dsmin, dssep_dasafe, dasep_damin, dasep_dsmin, dasep_dasafe = \
            self.dseparation_point(s, amin, smin, asafe, fluid_props)
        dssep_dy = (dssep_damin*damin_da + dssep_dsmin*dsmin_da + dssep_dasafe*dasafe_da) * da_dy
        dasep_dy = (dasep_damin*damin_da + dasep_dsmin*dsmin_da + dasep_dasafe*dasafe_da) * da_dy

        # Calculate the flow rate using Bernoulli
        coeff = 2/rho*(psup - psub)
        dcoeff_dpsub = -2/rho
        dcoeff_dpsup = 2/rho

        qsqr = coeff/(asub**-2 - asep**-2)
        dqsqr_dasep = -coeff/(asub**-2 - asep**-2)**2 * (2/asep**3)
        dqsqr_dpsub = dcoeff_dpsub/(asub**-2 - asep**-2)
        dqsqr_dpsup = dcoeff_dpsup/(asub**-2 - asep**-2)

        # Find Bernoulli pressure
        pbern = psub + 1/2*rho*qsqr*(asub**-2 - asafe**-2)
        dpbern_da = 1/2*rho*qsqr*(2/asafe**3) * dasafe_da
        dpbern_dasep = 1/2*rho*(asub**-2 - asafe**-2) * dqsqr_dasep
        dpbern_dy = np.diag(dpbern_da*da_dy) + dpbern_dasep[:, None]*dasep_dy
        dpbern_dpsub = 1.0 + 1/2*rho*dqsqr_dpsub*(asub**-2 - asafe**-2)
        dpbern_dpsup = 1/2*rho*dqsqr_dpsup*(asub**-2 - asafe**-2)

        # Correct Bernoulli pressure by applying a smooth mask after separation
        sepweight = smoothstep(self.s_vertices, ssep, fluid_props['zeta_sep'])
        dsepweight_dy = dsmoothstep_dx0(self.s_vertices, ssep, fluid_props['zeta_sep'])[:, None] * dssep_dy

        # p = sepweight * pbern
        dp_dy = sepweight[:, None]*dpbern_dy + dsepweight_dy*pbern[:, None]

        dp_dpsub = sepweight*dpbern_dpsub
        dp_dpsup = sepweight*dpbern_dpsup

        dq_dpsub = 0.5*qsqr**-0.5 * dqsqr_dpsub
        dq_dpsup = 0.5*qsqr**-0.5 * dqsqr_dpsup

        dp_du = np.zeros((usurf.size//2, usurf.size))
        dp_du[:, :-1:2] = 0
        dp_du[:, 1::2] = dp_dy

        ## Calculate the flow rate sensitivity
        dq_du = np.zeros(usurf.size)
        dq_du[1::2] = dqsqr_dasep/(2*qsqr**(1/2)) * dasep_dy

        return dq_du, dp_du, dq_dpsub, dp_dpsub, dq_dpsup, dp_dpsup

    def dres_dcontrol(self):
        pass

    def flow_sensitivity_solid(self, model, usurf, vsurf, psub, psup, fluid_props, adjoint=False):
        """
        Returns sparse mats/vecs for the sensitivity of pressure and flow rate to displacement.

        TODO: This function is a bit weird as a dense block in a sparse matrix is set

        Parameters
        ----------
        model
        surface_state : tuple of (u, v, a) each of (NUM_VERTICES, GEOMETRIC_DIM) np.ndarray
            States of the surface vertices, ordered following the flow (increasing x coordinate).

        Returns
        -------
        dp_du : PETSc.Mat
            Sensitivity of pressure with respect to displacement
        dq_du : PETSc.Vec
            Sensitivity of flow rate with respect to displacement
        """
        _dq_du, _dp_du, *_ = self.flow_sensitivity(usurf, vsurf, psub, psup, fluid_props)

        dp_du = PETSc.Mat().create(PETSc.COMM_SELF)
        dp_du.setType('aij')

        shape = None
        if not adjoint:
            shape = (self.state1['p'].size, model.solid.vert_to_vdof.size)
        else:
            shape = (model.solid.vert_to_vdof.size, self.state1['p'].size)
        dp_du.setSizes(shape)
        dp_du.setUp()

        pressure_vertices = model.fsi_verts
        solid_dofs, fluid_dofs = model.get_fsi_vector_dofs()
        # ()
        rows, cols = None, None
        if not adjoint:
            rows = np.arange(self.state1['p'].size, dtype=np.int32)
            cols = solid_dofs
        else:
            rows = solid_dofs
            cols = np.arange(self.state1['p'].size, dtype=np.int32)

        nnz = np.zeros(dp_du.size[0], dtype=np.int32)
        nnz[rows] = cols.size

        dp_du.setPreallocationNNZ(nnz)

        vals = None
        if not adjoint:
            vals = _dp_du
        else:
            vals = _dp_du.T

        # I think the rows have to be in increasing order for setValues when you set multiple rows
        # at once. This is not true for assembling the adjoint version so I set one row at a time in
        # a loop. Otherwise you could do it this way
        # dp_du.setValues(rows, cols, vals)
        for ii, row in enumerate(rows):
            # Pressure dofs are ordered from 0 to #dofs so that's why the index is `i` on `vals`
            dp_du.setValues(row, cols, vals[ii, :])

        dp_du.assemblyBegin()
        dp_du.assemblyEnd()

        dq_du = dfn.Function(model.solid.vector_fspace).vector()
        dq_du[model.solid.vert_to_vdof.reshape(-1, 2)[pressure_vertices].flat] = _dq_du

        return dq_du, dp_du

    ## Model res sensitivity interface
    def res(self):
        return self.state1 - self.solve_state1(self.state0)[0]

    def solve_state1(self, state1):
        """
        Return the final flow state
        """
        return self.fluid_pressure(*self.control.vecs, self.properties)

    def solve_dres_dstate1(self, b):
        return b

    def solve_dres_dstate1_adj(self, x):
        return x
        
    def apply_dres_dp_adj(self, x):
        b = self.get_properties_vec()
        b.set(0.0)
        return b

## Weighted average function
def wavg(s, f, w):
    """
    Return the weighted average of 'f(s)' with weight 'w(s)'
    """
    return trapz(f*w, s) / trapz(w, s)

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
    K_STABILITY = np.max(-f/alpha)
    dw_df = np.zeros(f.shape)
    if alpha == 0:
        dw_df[np.argmin(f)] = 1.0
    else:
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
    if alpha == 0:
        return np.zeros(x.shape)
    else:
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
        return gaussian(x, x0, alpha) * -2*((x-x0)/alpha) / alpha

def dgaussian_dx0(x, x0, alpha=1.0):
    """
    Return the sensitivity of `gaussian` to `x0`
    """
    if alpha == 0:
        return np.zeros(x.shape)
    else:
        return gaussian(x, x0, alpha) * -2*((x-x0)/alpha) / -alpha
