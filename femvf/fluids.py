"""
Functionality related to fluids
"""

import numpy as np
# import autograd
import autograd.numpy as np

import dolfin as dfn
from petsc4py import PETSc

from .parameters.properties import property_vecs
from .constants import PASCAL_TO_CGS, SI_DENSITY_TO_CGS
from .linalg import BlockVec, general_vec_set

## 1D Bernoulli approximation codes

class QuasiSteady1DFluid:
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

        # self.p1 = np.zeros(x_vertices.shape)
        # self.p0 = np.zeros(x_vertices.shape)

        # self.q0 = np.zeros((1,))
        # self.q1 = np.zeros((1,))

        # Calculate surface coordinates which are needed to compute surface integrals
        dx = self.x_vertices[1:] - self.x_vertices[:-1]
        dy = self.y_surface[1:] - self.y_surface[:-1]
        ds = (dx**2+dy**2)**0.5
        self.s_vertices = np.concatenate(([0.0], np.cumsum(ds)))

        p0 = np.zeros(NVERT)
        q0 = np.zeros((1,))
        self.state0 = BlockVec((q0, p0), ('q', 'p'))

        p1 = np.zeros(NVERT)
        q1 = np.zeros((1,))
        self.state1 = BlockVec((q1, p1), ('q', 'p'))

        # form type quantities associated with the mesh
        # displacement and velocity along the surface at state 0 and 1
        u0surf = np.zeros(2*NVERT)
        v0surf = np.zeros(2*NVERT)
        self.control0 = BlockVec((u0surf, v0surf), ('usurf', 'vsurf'))

        u1surf = np.zeros(2*NVERT)
        v1surf = np.zeros(2*NVERT)
        self.control1 = BlockVec((u1surf, v1surf), ('usurf', 'vsurf'))

        self.properties = self.get_properties_vec(set_default=True)

    def set_ini_state(self, qp0):
        """
        Set the initial fluid state
        """
        self.state0['q'][()] = qp0[0]
        self.state0['p'][:] = qp0[1]

    def set_fin_state(self, qp1):
        """
        Set the final fluid state
        """
        self.state1['q'][()] = qp1[0]
        self.state1['p'][:] = qp1[1]

    def set_ini_control(self, uv0):
        """
        Set the initial surface displacement and velocity
        """
        self.control0['usurf'][:] = uv0[0]
        self.control0['vsurf'][:] = uv0[1]

    def set_fin_control(self, uv1):
        """
        Set the final surface displacement and velocity
        """
        self.control1['usurf'][:] = uv1[0]
        self.control1['vsurf'][:] = uv1[1]

    def set_time_step(self, dt):
        """
        Set the time step
        """
        # This is a quasi-steady fluid so time doesn't matter. I put this here for consistency if
        # you were to use a unsteady fluid.
        pass

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

    def set_properties(self, props):
        """
        Set the fluid properties
        """
        for key in props.keys:
            vec = self.properties[key]
            general_vec_set(vec, props[key])
            # if vec.shape == ():
            #     vec[()] = props[key]
            # else:
            #     vec[:] = props[key]

    def get_properties(self):
        """
        Return the fluid properties
        """
        return self.properties.copy()

    def get_ini_control(self):
        """
        Return the initial surface displacement and velocity
        """
        return self.control0.copy()

    def get_fin_control(self):
        """
        Return the final surface displacement and velocity
        """
        return self.control1.copy()

    def get_state_vec(self):
        """
        Return empty flow speed and pressure state vectors
        """
        q, p = np.zeros((1,)), np.zeros(self.x_vertices.size)
        return BlockVec((q, p), ('q', 'p'))

    def get_surf_vector(self):
        """
        Return a vector representing vector data on the FSI surface
        """
        return np.zeros(self.x_vertices.shape[0]*2)

    def get_surf_scalar(self):
        """
        Return a vector representing scalar data on the FSI surface
        """
        return np.zeros(self.x_vertices.shape[0])

    def set_iter_params(self, qp0, uvsurf0, qp1, uvsurf1, fluid_props):
        """
        Set all parameters needed to define an iteration/time step of the model
        """
        self.set_ini_state(qp0)
        self.set_fin_state(qp1)

        self.set_ini_control(uvsurf0)
        self.set_fin_control(uvsurf1)

        self.set_properties(fluid_props)

    ## Methods that subclasses must implement
    def res(self):
        pass

    def solve_qp1(self):
        """
        Return the final flow and pressure
        """
        raise NotImplementedError("Fluid models have to implement this")

    def solve_qp0(self):
        """
        Return the initial flow and pressure
        """
        raise NotImplementedError("Fluid models have to implement this")

    def solve_dqp1_du1(self, adjoint=False):
        """
        Return sensitivities of the final flow and pressure
        """
        raise NotImplementedError("Fluid models have to implement this")

    def solve_dqp0_du0(self, adjoint=False):
        """
        Return sensitivities of the initial flow and pressure
        """
        raise NotImplementedError("Fluid models have to implement this")

    def solve_dqp1_du1_solid(self, model, adjoint=False):
        """
        Return sensitivities of the final flow and pressure
        """
        raise NotImplementedError("Fluid models have to implement this")

    def solve_dqp0_du0_solid(self, model, adjoint=False):
        """
        Return sensitivities of the initial flow and pressure
        """
        raise NotImplementedError("Fluid models have to implement this")
    
    ## Solver functions
    # def res(self):
    #     """
    #     Return the residual vector, F
    #     """
    #     return res

    # def solve_dres_dstate1(self, b):
    #     """
    #     Solve, dF/du x = f
    #     """
    #     return x

    # def solve_dres_dstate1_adj(self, b):
    #     """
    #     Solve, dF/du^T x = f
    #     """

    #     return x

    # def apply_dres_dstate0_adj(self, x):

    # def apply_dres_dp_adj(self, x):

    # def apply_dres_dcontrol_adj(self, x):


class Bernoulli(QuasiSteady1DFluid):
    """
    Represents the Bernoulli fluid model

    TODO : Refactor this to behave similar to the Solid model. A mesh should be used, corresponding
    to the reference configuration of the fluid (conformal with reference configuration of solid?)
    One of the properties should be the mapping from the reference configuration to the current
    configuration that would be used in ALE.

    Properties
    ----------
    alpha :
        Factor controlling the smoothness of the approximation of minimum area.
        A value of 0 weights areas of all nodes evenly, while a value of -inf
        returns the exact minimum area. Large negative values return an
        average of the smallest areas.
    k :
        Controls the sharpness of the cutoff that models separation. as k approaches inf,
        the sharpness will approach an instantaneous jump from 1 to 0
    """
    PROPERTY_TYPES = {
        'y_midline': ('const', ()),
        'p_sub': ('const', ()),
        'p_sup': ('const', ()),
        'a_sub': ('const', ()),
        'a_sup': ('const', ()),
        'rho_air': ('const', ()),
        'r_sep': ('const', ()),
        'alpha': ('const', ()),
        'k': ('const', ()),
        'sigma': ('const', ()),
        'beta': ('const', ()),
        'y_gap_min': ('const', ())}

    PROPERTY_DEFAULTS = {
        'y_midline': 1e6,
        'p_sub': 800 * PASCAL_TO_CGS,
        'p_sup': 0 * PASCAL_TO_CGS,
        'a_sub': 100000,
        'a_sup': 0.6,
        'r_sep': 1.0,
        'rho_air': 1.1225 * SI_DENSITY_TO_CGS,
        'alpha': -3/0.002,
        'k': 3/0.002,
        'sigma': 2.5*0.002,
        'beta': 3/.002,
        'y_gap_min': 0.001}

    def solve_qp1(self):
        """
        Return the final flow state
        """
        return self.fluid_pressure(self.get_fin_control(), self.properties)

    def solve_qp0(self):
        """
        Return the initial flow state
        """
        return self.fluid_pressure(self.get_ini_control(), self.properties)

    def solve_dqp1_du1(self, adjoint=False):
        """
        Return the final flow state
        """
        return self.flow_sensitivity(self.get_fin_control(), self.properties)

    def solve_dqp0_du0(self, adjoint=False):
        """
        Return the final flow state
        """
        return self.flow_sensitivity(self.get_ini_control(), self.properties)

    # TODO: Refactor to use the DOF map from solid to fluid rather than `ForwardModel`
    def solve_dqp1_du1_solid(self, model, adjoint=False):
        """
        Return the final flow state
        """
        return self.flow_sensitivity_solid(model, self.get_fin_control(), self.properties,
                                           adjoint)

    def solve_dqp0_du0_solid(self, model, adjoint=False):
        """
        Return the final flow state
        """
        return self.flow_sensitivity_solid(model, self.get_ini_control(), self.properties,
                                           adjoint)

    ## internal methods
    def separation_point(self, s, amin, smin, area, fluid_props):
        asep = fluid_props['r_sep'] * amin

        # This ensures the separation area is selected at a point past the minimum area
        log_wsep = np.log(1-smoothstep(s, smin, fluid_props['k'])) +  log_gaussian(area, asep, fluid_props['sigma'])
        wsep = np.exp(log_wsep - log_wsep.max())
        ssep = wavg(s, s, wsep)

        # assert s_sep >= s_min

        return ssep, asep

    def dseparation_point(self, s, amin, smin, a, fluid_props):
        # breakpoint()
        asep = fluid_props['r_sep'] * amin
        dasep_damin = fluid_props['r_sep']
        dasep_dsmin = 0.0 
        dasep_da = 0.0

        # This ensures the separation area is selected at a point past the minimum area
        wpostsep = 1-smoothstep(s, smin, fluid_props['k'])
        dwpostsep_dsmin = -dsmoothstep_dx0(s, smin, fluid_props['k'])

        wgaussian = gaussian(a, asep, fluid_props['sigma'])
        dwgaussian_da = dgaussian_dx(a, asep, fluid_props['sigma'])
        dwgaussian_damin = dgaussian_dx0(a, asep, fluid_props['sigma']) * dasep_damin

        # wsep = wpostsep * wgaussian
        log_wsep = np.log(wpostsep) +  log_gaussian(a, asep, fluid_props['sigma'])
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

    def fluid_pressure(self, surface_state, fluid_props):
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
        psup, psub = fluid_props['p_sup'], fluid_props['p_sub']
        rho = fluid_props['rho_air']
        asub = fluid_props['a_sub']
        alpha, k = fluid_props['alpha'], fluid_props['k']
        s = self.s_vertices

        # x = surface_state[0][:-1:2]
        y = surface_state[0][1::2]

        a = 2 * (fluid_props['y_midline'] - y)
        asafe = smoothlb(a, 2*fluid_props['y_gap_min'], fluid_props['beta'])

        # Calculate minimum and separation areas/locations
        wmin = expweight(asafe, alpha)
        amin = wavg(s, asafe, wmin)
        smin = wavg(s, s, wmin)

        ssep, asep = self.separation_point(s, amin, smin, asafe, fluid_props)
        
        # 1D Bernoulli approximation of the flow
        psep = psup
        flow_rate_sqr = 2/rho*(psep - psub)/(asub**-2 - asep**-2)

        pbern = psub + 1/2*rho*flow_rate_sqr*(asub**-2 - asafe**-2)

        sep_multiplier = smoothstep(self.s_vertices, ssep, k=k)

        p = sep_multiplier * pbern
        q = flow_rate_sqr**0.5

        # Find the first point where s passes s_min/s_sep, then we can linearly interpolate
        idx_min = np.argmax(s>smin)
        idx_sep = np.argmax(s>ssep)
        xy_min = surface_state[0].reshape(-1, 2)[idx_min]
        xy_sep = surface_state[0].reshape(-1, 2)[idx_sep]
        # breakpoint()
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

    def flow_sensitivity(self, surface_state, fluid_props):
        """
        Return the sensitivities of pressure and flow rate to the surface state.

        Parameters
        ----------
        surface_state : tuple of (u, v, a) each of (NUM_VERTICES, GEOMETRIC_DIM) np.ndarray
            States of the surface vertices, ordered following the flow (increasing x coordinate).
        fluid_props : BlockVec
            A dictionary of fluid property keyword arguments.
        """
        assert surface_state[0].size%2 == 0
        psup, psub = fluid_props['p_sup'], fluid_props['p_sub']
        rho = fluid_props['rho_air']
        asub = fluid_props['a_sub']
        alpha, k = fluid_props['alpha'], fluid_props['k']
        s = self.s_vertices

        # x = surface_state[0][:-1:2]
        y = surface_state[0][1::2]

        a = 2 * (fluid_props['y_midline'] - y)
        da_dy = -2

        asafe = smoothlb(a, 2*fluid_props['y_gap_min'], fluid_props['beta'])
        dasafe_da = dsmoothlb_df(a, 2*fluid_props['y_gap_min'], fluid_props['beta'])

        wmin = expweight(asafe, alpha)
        dwmin_da = dexpweight_df(asafe, alpha) * dasafe_da

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
        coeff = 2*(psup - psub)/rho
        dcoeff_dpsub = -2/rho

        qsqr = coeff/(asub**-2 - asep**-2)
        dqsqr_dasep = -coeff/(asub**-2 - asep**-2)**2 * (2/asep**3)
        dqsqr_dpsub = dcoeff_dpsub/(asub**-2 - asep**-2)

        # Find Bernoulli pressure
        pbern = psub + 1/2*rho*qsqr*(asub**-2 - asafe**-2)
        dpbern_da = 1/2*rho*qsqr*(2/asafe**3) * dasafe_da
        dpbern_dasep = 1/2*rho*(asub**-2 - asafe**-2) * dqsqr_dasep
        dpbern_dy = np.diag(dpbern_da*da_dy) + dpbern_dasep[:, None]*dasep_dy
        dpbern_dpsub = 1.0 + 1/2*rho*dqsqr_dpsub*(asub**-2 - asafe**-2)

        # Correct Bernoulli pressure by applying a smooth mask after separation
        sepweight = smoothstep(self.s_vertices, ssep, k=k)
        dsepweight_dy = dsmoothstep_dx0(self.s_vertices, ssep, k=k)[:, None] * dssep_dy

        # p = sepweight * pbern
        dp_dy = sepweight[:, None]*dpbern_dy + dsepweight_dy*pbern[:, None]

        dp_dpsub = sepweight*dpbern_dpsub
        dq_dpsub = 0.5*qsqr**-0.5 * dqsqr_dpsub

        dp_du = np.zeros((surface_state[0].size//2, surface_state[0].size))
        dp_du[:, :-1:2] = 0
        dp_du[:, 1::2] = dp_dy

        ## Calculate the flow rate sensitivity
        dq_du = np.zeros(surface_state[0].size)
        dq_du[1::2] = dqsqr_dasep/(2*qsqr**(1/2)) * dasep_dy

        return dq_du, dp_du, dq_dpsub, dp_dpsub

    def flow_sensitivity_solid(self, model, surface_state, fluid_props, adjoint=False):
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
        _dq_du, _dp_du, *_ = self.flow_sensitivity(surface_state, fluid_props)

        dp_du = PETSc.Mat().create(PETSc.COMM_SELF)
        dp_du.setType('aij')

        shape = None
        if not adjoint:
            shape = (self.state1['p'].size, model.solid.vert_to_vdof.size)
        else:
            shape = (model.solid.vert_to_vdof.size, self.state1['p'].size)
        dp_du.setSizes(shape)
        dp_du.setUp()

        pressure_vertices = model.surface_vertices
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

    def apply_dres_dp_adj(self, x):
        b = self.get_properties_vec()
        b.set(0.0)
        return b

    # def get_glottal_width(self, surface_state):
    #     """
    #     Return the non smoothed, glottal width
    #     """
    #     _, y = surface_state[0][:, 0], surface_state[0][:, 1]

    #     area = 2 * (self.properties['y_midline'] - y)
    #     a_min = smooth_minimum(area, self.s_vertices, self.properties['alpha'])
    #     return a_min

# Below are a collection of smoothened functions for selecting the minimum area, separation point,
# and simulating separation

def smoothlb(f, f_lb, beta=100):
    """
    Return the value of `f` subject to a smooth lower bound `f_lb`

    Function is based on a scaled and shifted version of the 'SoftPlus' function. This function
    smoothly blends a constant function when f<f_lb with a linear function when f>f_lb.

    The 'region' of smoothness is roughly characterized by 'df = f-f_lb', where the function is 95%
    a straight line when `beta*df = 3`.

    Parameters
    ----------
    f : array_like
    f_lb : float
        The minimum possible value of `f`
    beta : float
        The level of smoothness of the bounded function. This quantity has units of [cm^-1] if `f`
        has units of [cm]. Larger values of beta increase the sharpness of the bound.
    """
    # Manually return 1 if the exponent is large enough to cause overflow
    exponent = beta*(f-f_lb)
    idx_underflow = exponent <= -50.0
    idx_normal = np.logical_and(exponent > -50.0, exponent <= 50.0)
    idx_overflow = exponent > 50.0

    out = np.zeros(exponent.shape)
    out[idx_underflow] = f_lb
    out[idx_normal] = 1/beta*np.log(1 + np.exp(exponent[idx_normal])) + f_lb
    out[idx_overflow] = f[idx_overflow]
    return out

def dsmoothlb_df(f, f_lb, beta=100):
    """
    Return the sensitivity of `smooth_lower_bound` to `f`

    Parameters
    ----------
    f : array_like
    f_lb : float
        The minimum possible value of `f`
    beta : float
        The level of smoothness of the bounded function. This quantity has units of [cm^-1] if `f`
        has units of [cm]. Larger values of beta increase the sharpness of the bound.
    """
    # Manually return limiting values if the exponents are large enough to cause overflow
    exponent = beta*(f-f_lb)
    # idx_underflow = exponent <= -50.0
    idx_normal = np.logical_and(exponent > -50.0, exponent <= 50.0)
    idx_overflow = exponent > 50.0

    out = np.zeros(exponent.shape)
    # out[idx_underflow] = 0
    out[idx_normal] = np.exp(exponent[idx_normal]) / (1+np.exp(exponent[idx_normal]))
    out[idx_overflow] = 1.0
    return out

def expweight(f, alpha=-1000.0):
    """
    Return exponential weights as exp(alpha*f) 
    """
    # For numerical stability subtract a judicious constant from `alpha*x` to prevent exponents
    # being too large (overflow). This constant factors when you weights in an average
    K_STABILITY = np.max(alpha*f)
    w = np.exp(alpha*f - K_STABILITY)
    return w

def dexpweight_df(f, alpha=-1000.0):
    K_STABILITY = np.max(alpha*f)
    # w = np.exp(alpha*f - K_STABILITY)
    dw_df = alpha*np.exp(alpha*f - K_STABILITY)
    return dw_df

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

def smoothmin(f, s, alpha=-1000):
    """
    Return the smooth approximation to the minimum element of f integrated over s.

    The 'region' of smoothness is roughly characterized by 'df'. If f_min has the maximum weight,
    a value at f_min+df will have 5% of that weight if alpha*df = -3.

    Parameters
    ----------
    f : array_like
        Array of values to compute the minimum of
    s : array_like
        surface coordinates of each value
    alpha : float
        Factor that control the sharpness of the minimum. The function approaches the true minimum
        as `alpha` approachs negative infinity.
    """
    # For numerical stability subtract a judicious constant from `alpha*x` to prevent exponents
    # being too small or too large. This constant factors out due to the division.
    K_STABILITY = np.max(alpha*f)
    w = np.exp(alpha*f - K_STABILITY)

    return trapz(f*w, s) / trapz(w, s)

# dsmooth_minimum_df = autograd.grad(smooth_minimum, 0)

def dsmoothmin_df(f, s, alpha=-1000):
    """
    Return the derivative of the smooth minimum with respect to x.

    Parameters
    ----------
    f : array_like
        Array of values to compute the minimum of
    alpha : float
        Factor that control the sharpness of the minimum. The function approaches the true minimum
        function as `alpha` approachs negative infinity.
    """
    K_STABILITY = np.max(alpha*f)
    w = np.exp(alpha*f - K_STABILITY)
    dw_df = alpha*np.exp(alpha*f - K_STABILITY)

    num = trapz(f*w, s)
    den = trapz(w, s)

    dnum_df = dtrapz_df(f*w, s)*(w + f*dw_df)
    dden_df = dtrapz_df(w, s)*dw_df

    return dnum_df/den - num/den**2 * dden_df

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

def sigmoid(x):
    """
    Return the sigmoid function evaluated at `x`
    """
    exponent = -x
    idx_underflow = exponent <= -50.0
    idx_normal = np.logical_and(exponent > -50.0, exponent <= 50.0)
    idx_overflow = exponent > 50.0

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

def smoothstep(x, x0, k=100):
    """
    Return the mirrored logistic function evaluated at x-x0

    This steps from 1.0 when x << x0 to 0.0 when x >> x0.

    The 'region' of smoothness is roughly characterized by dx. If x = x0 + dx, the cutoff function
    will drop to just 5% if k*dx = 3.
    """
    arg = -k*(x-x0)
    return sigmoid(arg)

def dsmoothstep_dx(x, x0, k=100):
    """
    Return the logistic function evaluated at x-xref
    """
    arg = -k*(x-x0)
    darg_dx = -k
    return dsigmoid_dx(arg) * darg_dx

def dsmoothstep_dx0(x, x0, k=100):
    """
    Return the logistic function evaluated at x-xref
    """
    arg = -k*(x-x0)
    darg_dx0 = k
    return dsigmoid_dx(arg) * darg_dx0

def log_gaussian(x, x0, sigma=1.0):
    """
    Return the log of the gaussian with mean `x0` and variance `sigma`
    """
    return -((x-x0)/sigma)**2

def gaussian(x, x0, sigma=1.0):
    """
    Return the 'gaussian' with mean `x0` and variance `sigma`
    """
    return np.exp(-((x-x0)/sigma)**2)

def dgaussian_dx(x, x0, sigma=1.0):
    """
    Return the sensitivity of `gaussian` to `x`
    """
    return gaussian(x, x0, sigma) * -2*((x-x0)/sigma) / sigma

def dgaussian_dx0(x, x0, sigma=1.0):
    """
    Return the sensitivity of `gaussian` to `x0`
    """
    return gaussian(x, x0, sigma) * -2*((x-x0)/sigma) / -sigma
