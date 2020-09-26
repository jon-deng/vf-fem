"""
Functionality related to fluids
"""

import numpy as np
# import autograd
import autograd.numpy as np

import dolfin as dfn
from petsc4py import PETSc

from .parameters.properties import FluidProperties
from .constants import PASCAL_TO_CGS, SI_DENSITY_TO_CGS
from .linalg import BlockVec

## 1D Bernoulli approximation codes
SEPARATION_FACTOR = 1.0

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
        self.properties = FluidProperties(self)

        # the 'mesh' (also the x coordinates in the reference configuration)
        self.x_vertices = x_vertices

        # the surface y coordinates of the solid
        self.y_surface = y_surface

        # form type quantities associated with the mesh
        # displacement and velocity along the surface at state 0 and 1
        self.u0surf = np.zeros(2*x_vertices.size)
        self.u1surf = np.zeros(2*x_vertices.size)

        self.v0surf = np.zeros(2*x_vertices.size)
        self.v1surf = np.zeros(2*x_vertices.size)

        self.p1 = np.zeros(x_vertices.shape)
        self.p0 = np.zeros(x_vertices.shape)

        self.q0 = np.zeros((1,))
        self.q1 = np.zeros((1,))

        # Calculate surface coordinates which are needed to compute surface integrals
        dx = self.x_vertices[1:] - self.x_vertices[:-1]
        dy = self.y_surface[1:] - self.y_surface[:-1]
        ds = (dx**2+dy**2)**0.5
        self.s_vertices = np.concatenate(([0.0], np.cumsum(ds)))

    def set_ini_state(self, qp0):
        """
        Set the initial fluid state
        """
        self.q0[:] = qp0[0]
        self.p0[:] = qp0[1]

    def set_fin_state(self, qp1):
        """
        Set the final fluid state
        """
        self.q1[:] = qp1[0]
        self.p1[:] = qp1[1]

    def set_ini_surf_state(self, uv0):
        """
        Set the initial surface displacement and velocity
        """
        self.u0surf[:] = uv0[0]
        self.v0surf[:] = uv0[1]

    def set_fin_surf_state(self, uv1):
        """
        Set the final surface displacement and velocity
        """
        self.u1surf[:] = uv1[0]
        self.v1surf[:] = uv1[1]

    def set_time_step(self, dt):
        """
        Set the time step
        """
        # This is a quasi-steady fluid so time doesn't matter. I put this here for consistency if
        # you were to use a unsteady fluid.
        pass

    def set_properties(self, props):
        """
        Set the fluid properties
        """
        for key in props:
            if self.properties[key].shape == ():
                self.properties[key][()] = props[key]
            else:
                self.properties[key][:] = props[key]

    def get_properties(self):
        """
        Return the fluid properties
        """
        return self.properties.copy()

    def get_ini_surf_state(self):
        """
        Return the initial surface displacement and velocity
        """
        xy = self.u0surf.copy()
        dxy_dt = self.v0surf
        xy[:-1:2] = self.u0surf[:-1:2] + self.x_vertices
        xy[1::2] = self.u0surf[1::2] + self.y_surface

        surf_state = (xy, dxy_dt)
        return surf_state

    def get_fin_surf_state(self):
        """
        Return the final surface displacement and velocity
        """
        xy = self.u1surf.copy()
        dxy_dt = self.v1surf
        xy[:-1:2] = self.u1surf[:-1:2] + self.x_vertices
        xy[1::2] = self.u1surf[1::2] + self.y_surface

        surf_state = (xy, dxy_dt)
        return surf_state

    def get_state(self):
        """
        Return the state
        """
        vecs = (np.zeros((1,)), np.zeros(self.x_vertices.size))
        return BlockVec(vecs, ('q', 'p'))

    def get_state_vecs(self):
        """
        Return empty flow speed and pressure state vectors
        """
        return np.zeros((1,)), np.zeros(self.x_vertices.size)

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

        self.set_ini_surf_state(uvsurf0)
        self.set_fin_surf_state(uvsurf1)

        self.set_properties(fluid_props)

    ## Methods that subclasses must implement
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
        'p_sub': ('const', ()),
        'p_sup': ('const', ()),
        'a_sub': ('const', ()),
        'a_sup': ('const', ()),
        'rho': ('const', ()),
        'y_midline': ('const', ()),
        'alpha': ('const', ()),
        'k': ('const', ()),
        'sigma': ('const', ()),
        'beta': ('const', ()),
        'y_gap_min': ('const', ())}

    PROPERTY_DEFAULTS = {
        'p_sub': 800 * PASCAL_TO_CGS,
        'p_sup': 0 * PASCAL_TO_CGS,
        'a_sub': 100000,
        'a_sup': 0.6,
        'rho': 1.1225 * SI_DENSITY_TO_CGS,
        'y_midline': 0.61,
        'alpha': -3/0.002,
        'k': 3/0.002,
        'sigma': 2.5*0.002,
        'beta': 3/.002,
        'y_gap_min': 0.001}

    def solve_qp1(self):
        """
        Return the final flow state
        """
        return self.fluid_pressure(self.get_fin_surf_config(), self.properties)

    def solve_qp0(self):
        """
        Return the initial flow state
        """
        return self.fluid_pressure(self.get_ini_surf_config(), self.properties)

    def solve_dqp1_du1(self, adjoint=False):
        """
        Return the final flow state
        """
        return self.flow_sensitivity(self.get_fin_surf_config(), self.properties)

    def solve_dqp0_du0(self, adjoint=False):
        """
        Return the final flow state
        """
        return self.flow_sensitivity(self.get_ini_surf_config(), self.properties)

    # TODO: Refactor to use the DOF map from solid to fluid rather than `ForwardModel`
    def solve_dqp1_du1_solid(self, model, adjoint=False):
        """
        Return the final flow state
        """
        return self.flow_sensitivity_solid(model, self.get_fin_surf_config(), self.properties,
                                           adjoint)

    def solve_dqp0_du0_solid(self, model, adjoint=False):
        """
        Return the final flow state
        """
        return self.flow_sensitivity_solid(model, self.get_ini_surf_config(), self.properties,
                                           adjoint)

    ## internal methods
    def fluid_pressure(self, surface_state, fluid_props):
        """
        Computes the pressure loading at a series of surface nodes according to Pelorson (1994)

        TODO: I think it would make more sense to treat this as a generic Bernoulli pressure
        calculator. It could be refactored to not use an self variable, instead it would pass a
        reference surface mesh, and current surface mesh (x, y coordinates of vertices in increasing
        streamwise direction).

        Parameters
        ----------
        surface_state : tuple of (u, v, a) each of (NUM_VERTICES, GEOMETRIC_DIM) np.ndarray
            States of the surface vertices, ordered following the flow (increasing x coordinate).
        fluid_props : properties.FluidProperties
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
        y_midline = fluid_props['y_midline']
        p_sup, p_sub = fluid_props['p_sup'], fluid_props['p_sub']
        rho = fluid_props['rho']
        a_sub = fluid_props['a_sub']
        alpha, k, sigma = fluid_props['alpha'], fluid_props['k'], fluid_props['sigma']

        x, y = surface_state[0].reshape(-1, 2)[:, 0], surface_state[0].reshape(-1, 2)[:, 1]

        area = 2 * (y_midline - y)
        area_safe = smooth_lower_bound(area, 2*fluid_props['y_gap_min'], fluid_props['beta'])
        dt_area = -2 * surface_state[1].reshape(-1, 2)[:, 1]

        # Calculate minimum and separation areas/locations
        a_min = smooth_minimum(area_safe, self.s_vertices, alpha)
        dt_a_min = np.sum(dsmooth_minimum_df(area_safe, self.s_vertices, alpha) * dt_area)
        a_sep = SEPARATION_FACTOR * a_min
        dt_a_sep = SEPARATION_FACTOR * dt_a_min

        # 1D Bernoulli approximation of the flow
        p_sep = p_sup
        flow_rate_sqr = 2/rho*(p_sep - p_sub)/(a_sub**-2 - a_sep**-2)
        dt_flow_rate_sqr = 2/rho*(p_sep - p_sub)*-1*(a_sub**-2 - a_sep**-2)**-2 \
                           * (2*a_sep**-3 * dt_a_sep)

        p_bernoulli = p_sub + 1/2*rho*flow_rate_sqr*(a_sub**-2 - area_safe**-2)

        # Calculate x_sep for plotting/information purposes
        x_sep = smooth_selection(x, area_safe, a_sep, self.s_vertices, sigma)
        s_sep = smooth_selection(self.s_vertices, area_safe, a_sep, self.s_vertices, sigma=sigma)
        sep_multiplier = smooth_cutoff(self.s_vertices, s_sep, k=k)

        p = sep_multiplier * p_bernoulli

        flow_rate = flow_rate_sqr**0.5
        dt_flow_rate = 0.5*flow_rate_sqr**(-0.5) * dt_flow_rate_sqr

        idx_min = 0
        idx_sep = 0
        xy_min = surface_state[0][idx_min]
        xy_sep = surface_state[0][idx_sep]
        info = {'flow_rate': flow_rate,
                'dt_flow_rate': dt_flow_rate,
                'idx_min': idx_min,
                'idx_sep': idx_sep,
                'xy_min': xy_min,
                'xy_sep': xy_sep,
                'x_sep': x_sep,
                'a_min': a_min,
                'a_sep': a_sep,
                'area': area_safe,
                'pressure': p}
        return BlockVec((np.array([flow_rate]), p), ('q', 'p')), info

    def flow_sensitivity(self, surface_state, fluid_props):
        """
        Return the sensitivities of pressure and flow rate to the surface state.

        Parameters
        ----------
        surface_state : tuple of (u, v, a) each of (NUM_VERTICES, GEOMETRIC_DIM) np.ndarray
            States of the surface vertices, ordered following the flow (increasing x coordinate).
        fluid_props : properties.FluidProperties
            A dictionary of fluid property keyword arguments.
        """
        assert surface_state[0].size%2 == 0

        y_midline = fluid_props['y_midline']
        p_sup, p_sub = fluid_props['p_sup'], fluid_props['p_sub']
        rho = fluid_props['rho']
        a_sub = fluid_props['a_sub']
        alpha, k, sigma = fluid_props['alpha'], fluid_props['k'], fluid_props['sigma']

        x, y = surface_state[0].reshape(-1, 2)[:, 0], surface_state[0].reshape(-1, 2)[:, 1]

        area = 2 * (y_midline - y)
        darea_dy = -2 # darea_dx = 0

        area_safe = smooth_lower_bound(area, 2*fluid_props['y_gap_min'], fluid_props['beta'])
        darea_safe_darea = dsmooth_lower_bound_df(area, 2*fluid_props['y_gap_min'], fluid_props['beta'])

        a_min = smooth_minimum(area_safe, self.s_vertices, alpha)
        da_min_darea = dsmooth_minimum_df(area_safe, self.s_vertices, alpha) * darea_safe_darea

        a_sep = SEPARATION_FACTOR * a_min
        da_sep_da_min = SEPARATION_FACTOR
        da_sep_dy = da_sep_da_min * da_min_darea * darea_dy

        # Calculate the flow rate using Bernoulli equation
        p_sep = p_sup
        coeff = 2*(p_sep - p_sub)/rho
        flow_rate_sqr = coeff/(a_sub**-2 - a_sep**-2)
        dflow_rate_sqr_da_sep = -coeff/(a_sub**-2 - a_sep**-2)**2 * (2/a_sep**3)

        # Find Bernoulli pressure
        p_bernoulli = p_sub + 1/2*rho*flow_rate_sqr*(a_sub**-2 - area_safe**-2)
        dp_bernoulli_darea = 1/2*rho*flow_rate_sqr*(2/area_safe**3) * darea_safe_darea
        dp_bernoulli_da_sep = 1/2*rho*(a_sub**-2 - area_safe**-2) * dflow_rate_sqr_da_sep
        dp_bernoulli_dy = np.diag(dp_bernoulli_darea*darea_dy) + dp_bernoulli_da_sep[:, None]*da_sep_dy

        # Correct Bernoulli pressure by applying a smooth mask after separation
        # x_sep = smooth_selection(x, area, a_sep, self.s_vertices, sigma)
        # dx_sep_dx = dsmooth_selection_dx(x, area, a_sep, self.s_vertices, sigma)
        # dx_sep_dy = dsmooth_selection_dy(x, area, a_sep, self.s_vertices, sigma)*darea_dy \
        #             + dsmooth_selection_dy0(x, area, a_sep, self.s_vertices, sigma)*da_sep_dy

        s_sep = smooth_selection(self.s_vertices, area_safe, a_sep, self.s_vertices, sigma=sigma)
        ds_sep_dy = dsmooth_selection_dy(self.s_vertices, area_safe, a_sep, self.s_vertices,
                                         sigma) * darea_safe_darea * darea_dy \
                    + dsmooth_selection_dy0(self.s_vertices, area_safe, a_sep, self.s_vertices,
                                            sigma) * da_sep_dy

        sep_multiplier = smooth_cutoff(self.s_vertices, s_sep, k=k)
        dsep_multiplier_dy = dsmooth_cutoff_dx0(self.s_vertices, s_sep, k=k)[:, None] * ds_sep_dy

        # p = sep_multiplier * p_bernoulli
        dp_dy = sep_multiplier[:, None]*dp_bernoulli_dy + dsep_multiplier_dy*p_bernoulli[:, None]

        dp_du = np.zeros((surface_state[0].size//2, surface_state[0].size))
        dp_du[:, :-1:2] = 0
        dp_du[:, 1::2] = dp_dy

        ## Calculate the flow rate sensitivity
        dflow_rate_du = np.zeros(surface_state[0].size)
        dflow_rate_du[1::2] = dflow_rate_sqr_da_sep/(2*flow_rate_sqr**(1/2)) \
                              * da_sep_da_min * da_min_darea * darea_dy

        return dflow_rate_du, dp_du

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
        _dq_du, _dp_du = self.flow_sensitivity(surface_state, fluid_props)

        dp_du = PETSc.Mat().create(PETSc.COMM_SELF)
        dp_du.setType('aij')

        shape = None
        if not adjoint:
            shape = (self.p1.size, model.solid.vert_to_vdof.size)
        else:
            shape = (model.solid.vert_to_vdof.size, self.p1.size)
        dp_du.setSizes(shape)
        dp_du.setUp()

        pressure_vertices = model.surface_vertices
        solid_dofs, fluid_dofs = model.get_fsi_vector_dofs()
        # ()
        rows, cols = None, None
        if not adjoint:
            rows = np.arange(self.p1.size, dtype=np.int32)
            cols = solid_dofs
        else:
            rows = solid_dofs
            cols = np.arange(self.p1.size, dtype=np.int32)

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

    def get_glottal_width(self, surface_state):
        """
        Return the non smoothed, glottal width
        """
        _, y = surface_state[0][:, 0], surface_state[0][:, 1]

        area = 2 * (self.properties['y_midline'] - y)
        a_min = smooth_minimum(area, self.s_vertices, self.properties['alpha'])
        return a_min

    def get_ini_surf_config(self):
        """
        Return the initial surface configuration

        This is the surface state, but the displacements are converted to current configuration
        positions
        """
        xy = self.u0surf.copy()
        dxy_dt = self.v0surf
        xy[:-1:2] = self.u0surf[:-1:2] + self.x_vertices
        xy[1::2] = self.u0surf[1::2] + self.y_surface

        surf_state = (xy, dxy_dt)
        return surf_state

    def get_fin_surf_config(self):
        """
        Return the final surface configuration

        This is the surface state, but the displacements are converted to current configuration
        positions
        """
        xy = self.u1surf.copy()
        dxy_dt = self.v1surf
        xy[:-1:2] = self.u1surf[:-1:2] + self.x_vertices
        xy[1::2] = self.u1surf[1::2] + self.y_surface

        surf_state = (xy, dxy_dt)
        return surf_state

# Below are a collection of smoothened functions for selecting the minimum area, separation point,
# and simulating separation

def smooth_lower_bound(f, f_lb, beta=100):
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

def dsmooth_lower_bound_df(f, f_lb, beta=100):
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

def smooth_minimum(f, s, alpha=-1000):
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

def dsmooth_minimum_df(f, s, alpha=-1000):
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

def smooth_cutoff(x, x0, k=100):
    """
    Return the mirrored logistic function evaluated at x-x0

    The 'region' of smoothness is roughly characterized by dx. If x = x0 + dx, the cutoff function
    will drop to just 5% if k*dx = 3.
    """
    arg = -k*(x-x0)
    return sigmoid(arg)

def dsmooth_cutoff_dx(x, x0, k=100):
    """
    Return the logistic function evaluated at x-xref
    """
    arg = -k*(x-x0)
    darg_dx = -k
    return dsigmoid_dx(arg) * darg_dx

def dsmooth_cutoff_dx0(x, x0, k=100):
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
    return np.log(1/(sigma*(2*np.pi)**0.5)) + -0.5*((x-x0)/sigma)**2

def log_dgaussian_dx(x, x0, sigma=1.0):
    """
    Return the sensitivity of `log_gaussian` to `x`
    """
    return log_gaussian(x, x0, sigma) * -(x-x0)/sigma**2

def log_dgaussian_dx0(x, x0, sigma=1.0):
    """
    Return the sensitivity of `log_gaussian` to `x0`
    """
    return log_gaussian(x, x0, sigma) * -(x-x0)/sigma**2

def gaussian(x, x0, sigma=1.0):
    """
    Return the gaussian with mean `x0` and variance `sigma`
    """
    return 1/(sigma*(2*np.pi)**0.5) * np.exp(-0.5*((x-x0)/sigma)**2)

def dgaussian_dx(x, x0, sigma=1.0):
    """
    Return the sensitivity of `gaussian` to `x`
    """
    return gaussian(x, x0, sigma) * -(x-x0)/sigma**2

def dgaussian_dx0(x, x0, sigma=1.0):
    """
    Return the sensitivity of `gaussian` to `x0`
    """
    return gaussian(x, x0, sigma) * (x-x0)/sigma**2

def smooth_selection(x, y, y0, s, sigma=1.0):
    """
    Return the `x` value from an `(x, y)` pair where `y` equals `y0`.

    Weights are computed according to a gaussian distribution. The 'region' of smoothness is roughly
    characterized by 'sigma'. If y = y0 +- 2.5*sigma, it's weight will be about 5% of the weight
    corresponding to y = y0.

    Parameters
    ----------
    x, y : array_like
        Paired array of values
    sigma : float
        Standard deviation of the selection criteria
    """
    # assert x.size == y.size
    # Use the log density for a numerically stable computation. Subtracting off a constant from the
    # exponentiation doesn't change anything in the final ratio of weights
    log_w = log_gaussian(y, y0, sigma)
    w = np.exp(log_w - np.max(log_w))

    return trapz(x*w, s) / trapz(w, s)

# dsmooth_selection_dx = autograd.grad(smooth_selection, 0)
# dsmooth_selection_dy = autograd.grad(smooth_selection, 1)
# dsmooth_selection_dy0 = autograd.grad(smooth_selection, 2)
def dsmooth_selection_dx(x, y, y0, s, sigma=1.0):
    """
    Return the derivative of `gaussian_selection` w.r.t `x`.

    Weights are computed according to a gaussian distribution.

    Parameters
    ----------
    x, y : array_like
        Paired array of values
    sigma : float
        Standard deviation of the selection criteria
    """
    # assert x.size == y.size
    log_w = log_gaussian(y, y0, sigma)
    w = np.exp(log_w - np.max(log_w))

    # The returned value would be
    # return np.sum(x*weights) / np.sum(weights)
    # so the derivative is given by
    return dtrapz_df(x*w, s) / trapz(w, s) * w

def dsmooth_selection_dy(x, y, y0, s, sigma=1.0):
    """
    Return the derivative of `gaussian_selection` w.r.t `y`.

    Weights are computed according to a gaussian distribution.

    Parameters
    ----------
    x, y : array_like
        Paired array of values
    sigma : float
        Standard deviation of the selection criteria
    """
    # assert x.size == y.size
    # log_w = log_gaussian(y, y0, sigma)
    w = gaussian(y, y0, sigma)
    dw_dy = dgaussian_dx(y, y0, sigma)

    num = trapz(x*w, s)
    dnum_dy = dtrapz_df(x*w, s) * x*dw_dy

    den = trapz(w, s)
    dden_dy = dtrapz_df(w, s) * dw_dy

    # out = num/den
    dout_dy = dnum_dy/den - num/den**2*dden_dy

    return dout_dy

def dsmooth_selection_dy0(x, y, y0, s, sigma=1.0):
    """
    Return the `x` value from an `(x, y)` pair where `y` equals `y0`.

    Weights are computed according to a gaussian distribution.

    Parameters
    ----------
    x, y : array_like
        Paired array of values
    sigma : float
        Standard deviation of the selection criteria
    """
    # assert x.size == y.size
    # log_w = log_gaussian(y, y0, sigma)
    w = gaussian(y, y0, sigma)
    dw_dy0 = dgaussian_dx0(y, y0, sigma)

    num = trapz(x*w, s)
    dnum_dy0 = np.dot(dtrapz_df(x*w, s), x*dw_dy0)

    den = trapz(w, s)
    dden_dy0 = np.dot(dtrapz_df(w, s), dw_dy0)

    # out = num/den
    dout_dy0 = dnum_dy0/den - num/den**2*dden_dy0

    return dout_dy0
