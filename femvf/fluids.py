"""
Functionality related to fluids
"""

import math
import numpy as np
# import autograd
import autograd.numpy as np

import dolfin as dfn
from petsc4py import PETSc

from .parameters.properties import FluidProperties
from .constants import PASCAL_TO_CGS, SI_DENSITY_TO_CGS

## 1D Euler model
def fluid_pressure_vasu(uva, x0, xr, fluid_props):
    """
    Return fluid surface pressures according to a 1D flow model.

    The flow model is given by "" vasudevan etc.

    Parameters
    ----------
    x : tuple of (u, v, a) each of (#surface vertices, geometric dimension) np.ndarray
        States of the surface vertices, ordered following the flow (increasing x coordinate).
    fluid_props : dict
        A dictionary of fluid properties.
    xr : np.ndarray
        The nodal locations of the fixed reference grid over which the problem is solved.

    Returns
    -------
    q, p : np.ndarray
        An array of flow rate and pressure vectors for each each vertex
    dqdx, dpdx : np.ndarray
        Arrays of sensititivies of nodal flow rates and pressures to the surface displacements and
        velocities
    xy_min, xy_sep: (2,) np.ndarray
        The coordinates of the vertices at minimum and separation areas
    """
    u, v, a = uva

    area = 2*u[..., 1]
    darea_dt = 2*v[..., 1]

def res_fluid(n, p_bcs, qp0, qp1, xy_ref, uva0, uva1, fluid_props, dt):
    """
    Return the momentum and continuity equation residuals applied at node `n`.

    Momentum and continuity residuals are returned based on the equation between state 0 and 1
    over a given time step and a guess of the final fluid and solid states.

    Parameters
    ----------
    qp0, qp1 : tuple(array_like, array_like)
        The fluid flow rate and pressure states in a tuple
    uva : tuple(array_like, array_like, array_like)
        The fluid-structure interace displacement, velocity, and acceleration states in a tuple
    xy_ref, uva0, uva1 : tuple(array_like, array_like)
        xy coordinates/displacements of the fluid-structure interace at the reference configuration,
        and current and future timesteps
    fluid_props : femvf.parameters.properties.FluidProperties
        The fluid properties
    dt : float
        The timestep between the initial and final states of the iteration.

    Returns
    -------
    tuple(float, float)
        The continuity and momentum equation residuals respectively.
    """
    ## Set the finite differencing stencil according to the node
    fdiff = None
    NUM_NODE = xy_ref.size

    if n == NUM_NODE-1:
        fdiff = bd
    elif n == 0:
        fdiff = fd
    else:
        fdiff = cd

    ## Precompute some variables needed in computing residuals for continuity and momentum
    # The reference grid for ALE is evenly spaced between the first and last FSI interface
    # x-coordinates
    dx = (xy_ref[-1, 0] - xy_ref[0, 0]) / NUM_NODE
    u1, v1, _ = uva1

    rho = fluid_props['rho']

    # Flow rate and pressure
    q0, _ = qp0
    q1, p1 = qp1

    area = 2*u1[..., 1]
    darea_dt = 2*v1[..., 1]

    # The x-coordinates of the moving mesh are the initial surface node values plus the
    # x-displacement
    # This is needed to calculate deformation gradient for use in ALE
    # x0 = xr + u0[..., 0]
    x1 = xy_ref[..., 0] + u1[..., 0]

    def_grad = 1/fdiff(x1, n, dx)
    darea_dx = fdiff(area, n, dx)
    dq_dx = fdiff(q1, n, dx)
    dp_dx = fdiff(p1, n, dx)
    dqarea_dx = fdiff(area*q1, n, dx)

    ## Calculate the momentum and continuity residuals
    res_continuity = darea_dt[n] - darea_dx*def_grad*v1[n, 0] + dqarea_dx*def_grad

    dq_dt = (q1[n]-q0[n])/dt
    sep_criteria = q0[n]*fluid_props['rho']*(-q0[n]*dq_dx - dq_dt)
    xx = separation_factor(0.25, 0.1, sep_criteria)
    tau = (1-xx)*rho*q1[n]*dq_dx*def_grad
    res_momentum = rho*q1[n]*dq_dx*def_grad + rho*(dq_dt-dq_dx*def_grad*v1[n, 0]) \
                   + dp_dx*def_grad - tau

    return res_continuity, res_momentum

def res_fluid_quasistatic(n, p_bcs, qp0, xy_ref, uva0, fluid_props):
    """
    Return the momentum and continuity equation residuals applied at node `n`.

    Momentum and continuity residuals are returned based on the equation between state 0 and 1
    over a given time step and a guess of the final fluid and solid states.

    Parameters
    ----------
    n : int
        The node number to compute the residual at
    p_bcs : tuple(float, float)
        The pressure boundary conditions at the inlet and outlet
    qp0 : tuple(array_like[:], array_like[:])
        The fluid flow rate and pressure states in a tuple
    uva0 : tuple(array_like[:, 2], array_like[:, 2], array_like[:, 2])
        The fluid-structure interace displacement, velocity, and acceleration states in a tuple
    xy_ref : array_like[:, 2]
        The fluid-structure interace coordinates in the reference configuration
    fluid_props : femvf.parameters.properties.FluidProperties
        The fluid properties

    Returns
    -------
    tuple(float, float)
        The continuity and momentum equation residuals respectively.
    """
    ## Set the finite differencing stencil according to the node
    fdiff = None
    NUM_NODE = xy_ref.shape[0]

    if n == NUM_NODE-1:
        fdiff = bd
    elif n == 0:
        fdiff = fd
    else:
        fdiff = cd

    ## Precompute some variables needed in computing residuals for continuity and momentum
    # The reference grid for ALE is evenly spaced between the first and last FSI interface
    # x-coordinates
    dx = (xy_ref[-1, 0] - xy_ref[0, 0]) / NUM_NODE
    u0, v0, _ = uva0

    rho = fluid_props['rho']

    # Flow rate and pressure
    q0, p0 = qp0
    p0[0], p0[-1] = p_bcs[0], p_bcs[1]

    # The x-coordinates of the moving mesh are the initial surface node values plus the
    # x-displacement
    xy0 = xy_ref[..., :] + u0[..., :]
    area = 2*(fluid_props['y_midline'] - xy0[..., 1])

    def_grad = 1/fdiff(xy0[..., 0], n, dx)
    # darea_dx = fdiff(area, n, dx)
    dq_dx = fdiff(q0, n, dx)
    dp_dx = fdiff(p0, n, dx)
    dqarea_dx = fdiff(area*q0, n, dx)

    ## Calculate the momentum and continuity residuals
    res_continuity = dqarea_dx*def_grad

    # The separation criteria is based on q dp/dx but most suggest using the inviscid approximation,
    # which is dp/dx = -rho*q*dq/dx
    sep_criteria = q0[n]*fluid_props['rho'] * (-q0[n]*dq_dx)
    xx = separation_factor(0.0, 0.1, sep_criteria)
    tau = (1-xx)*rho*q0[n]*dq_dx*def_grad
    res_momentum = rho*q0[n]*dq_dx*def_grad + dp_dx*def_grad - tau

    info = {'tau': tau, 'separation_factor': xx}

    return res_continuity, res_momentum, info

def separation_factor(sep_factor_min, alpha_max, alpha):
    r"""
    Return the separation factor (a fancy x)

    parameters
    ----------
    alpha : float
        A separation criteria given by :math:`\alpha=q\frac{dp}{dx}` or approximated by the inviscid
        approximation using :math:`\alpha \approx q*(-\rho q \frac{dq}{dx}-rho\frac{dq}{dt})`
    """
    sep_factor = None
    if alpha < 0:
        sep_factor = 1
    elif alpha < alpha_max:
        sep_factor = sep_factor_min/2*(1-math.cos(math.pi*alpha/alpha_max))
    else:
        sep_factor = sep_factor_min

    return sep_factor

def cd(f, n, dx):
    """
    Return the central difference approximation of f at n.
    """
    return (f[n+1]-f[n-1]) / (2*dx)

def cd_df(f, n, dx):
    """
    Return the derivative of the central difference approximation of f at n, to the vector of f.
    """
    idxs = [n-1, n+1]
    vals = [-1/(2*dx), 1/(2*dx)]

def fd(f, n, dx):
    """
    Return the central difference approximation of f at n.
    """
    return (f[n+1]-f[n]) / dx

def fd_df(f, n, dx):
    """
    Return the derivative of the central difference approximation of f at n, to the vector of f.
    """
    idxs = [n, n+1]
    vals = [-1/dx, 1/dx]

def bd(f, n, dx):
    """
    Return the central difference approximation of f at n.
    """
    return (f[n]-f[n-1]) / dx

def bd_df(f, n, dx):
    """
    Return the derivative of the central difference approximation of f at n, to the vector of f.
    """
    idxs = [n-1, n]
    vals = [-1/dx, 1/dx]

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

    def set_ini_state(self, q0, p0):
        pass

    def set_fin_state(self, q1, p1):
        pass

    def set_ini_surf_state(self, u0, v0):
        self.u0surf[:] = u0
        self.v0surf[:] = v0

    def set_fin_surf_state(self, u1, v1):
        self.u1surf[:] = u1
        self.v1surf[:] = v1

    def set_time_step(self, dt):
        # This is a quasi-steady fluid so time doesn't matter. I put this here for consistency if
        # you were to use a unsteady fluid.
        pass

    def set_properties(self, props):
        for key in props:
            if self.properties[key].shape == ():
                self.properties[key][()] = props[key]
            else:
                self.properties[key][:] = props[key]

    def get_properties(self):
        return self.properties.copy()

    def get_ini_surf_state(self):
        xy = self.u0surf.copy()
        dxy_dt = self.v0surf
        xy[:-1:2] = self.u0surf[:-1:2] + self.x_vertices
        xy[1::2] = self.u0surf[1::2] + self.y_surface

        surf_state = (xy, dxy_dt)
        return surf_state

    def get_fin_surf_state(self):
        xy = self.u1surf.copy()
        dxy_dt = self.v1surf
        xy[:-1:2] = self.u1surf[:-1:2] + self.x_vertices
        xy[1::2] = self.u1surf[1::2] + self.y_surface

        surf_state = (xy, dxy_dt)
        return surf_state

    def get_state_vecs(self):
        return np.zeros((1,)), np.zeros(self.x_vertices.size)

    def get_surf_vector(self):
        return np.zeros(self.x_vertices.shape[0]*2)

    def get_surf_scalar(self):
        return np.zeros(self.x_vertices.shape[0])

    def set_iter_params(self, qp0, uvsurf0, qp1, uvsurf1, fluid_props):
        self.set_ini_state(*qp0)
        self.set_fin_state(*qp1)

        self.set_ini_surf_state(*uvsurf0)
        self.set_fin_surf_state(*uvsurf1)

        self.set_properties(fluid_props)

    def get_fp1(self):
        return self.p1 - self.fluid_pressure((self.u1surf, self.v1surf))

    def get_fp0(self):
        return self.p0 - self.fluid_pressure((self.u0surf, self.v0surf))

    def get_dfp1_dp1(self):
        return 1.0

    def get_dfp0_dp0(self):
        return 1.0

    def get_dfp1_du1(self, u_order='fluid'):
        return -1*self.flow_sensitivity

    # Methods that subclasses must implement
    def solve_qp1(self):
        raise NotImplementedError("Fluid models have to implement this")

    def solve_qp0(self):
        raise NotImplementedError("Fluid models have to implement this")

    def solve_dqp1_du1(self, adjoint=False):
        raise NotImplementedError("Fluid models have to implement this")

    def solve_dqp0_du0(self, adjoint=False):
        raise NotImplementedError("Fluid models have to implement this")

    def solve_dqp1_du1_solid(self, adjoint=False):
        raise NotImplementedError("Fluid models have to implement this")

    def solve_dqp0_du0_solid(self, adjoint=False):
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
        'sigma': ('const', ())}

    PROPERTY_DEFAULTS = {
        'p_sub': 800 * PASCAL_TO_CGS,
        'p_sup': 0 * PASCAL_TO_CGS,
        'a_sub': 100000,
        'a_sup': 0.6,
        'rho': 1.1225 * SI_DENSITY_TO_CGS,
        'y_midline': 0.61,
        'alpha': -3000,
        'k': 50,
        'sigma': 0.002}

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
        dt_area = -2 * (y)

        # Calculate minimum and separation areas/locations
        a_min = smooth_minimum(area, self.s_vertices, alpha)
        dt_a_min = np.sum(dsmooth_minimum_df(area, self.s_vertices, alpha) * dt_area)
        a_sep = SEPARATION_FACTOR * a_min
        dt_a_sep = SEPARATION_FACTOR * dt_a_min

        # 1D Bernoulli approximation of the flow
        p_sep = p_sup
        flow_rate_sqr = 2/rho*(p_sep - p_sub)/(a_sub**-2 - a_sep**-2)
        dt_flow_rate_sqr = 2/rho*(p_sep - p_sub)*-1*(a_sub**-2 - a_sep**-2)**-2 * (2*a_sep**-3 * dt_a_sep)

        p_bernoulli = p_sub + 1/2*rho*flow_rate_sqr*(a_sub**-2 - area**-2)

        # Calculate x_sep for plotting/information purposes
        x_sep = smooth_selection(x, area, a_sep, self.s_vertices, sigma)
        s_sep = smooth_selection(self.s_vertices, area, a_sep, self.s_vertices, sigma=sigma)
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
                'area': area,
                'pressure': p}
        return flow_rate, p, info

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

        a_min = smooth_minimum(area, self.s_vertices, alpha)
        da_min_darea = dsmooth_minimum_df(area, self.s_vertices, alpha)

        a_sep = SEPARATION_FACTOR * a_min
        da_sep_da_min = SEPARATION_FACTOR
        da_sep_dy = da_sep_da_min * da_min_darea * darea_dy

        # Calculate the flow rate using Bernoulli equation
        p_sep = p_sup
        coeff = 2*(p_sep - p_sub)/rho
        flow_rate_sqr = coeff/(a_sub**-2 - a_sep**-2)
        dflow_rate_sqr_da_sep = -coeff/(a_sub**-2 - a_sep**-2)**2 * (2/a_sep**3)

        # Find Bernoulli pressure
        p_bernoulli = p_sub + 1/2*rho*flow_rate_sqr*(a_sub**-2 - area**-2)
        dp_bernoulli_darea = 1/2*rho*flow_rate_sqr*(2/area**3)
        dp_bernoulli_da_sep = 1/2*rho*(a_sub**-2 - area**-2) * dflow_rate_sqr_da_sep
        dp_bernoulli_dy = np.diag(dp_bernoulli_darea*darea_dy) + dp_bernoulli_da_sep[:, None]*da_sep_dy

        # Correct Bernoulli pressure by applying a smooth mask after separation
        # x_sep = smooth_selection(x, area, a_sep, self.s_vertices, sigma)
        # dx_sep_dx = dsmooth_selection_dx(x, area, a_sep, self.s_vertices, sigma)
        # dx_sep_dy = dsmooth_selection_dy(x, area, a_sep, self.s_vertices, sigma)*darea_dy \
        #             + dsmooth_selection_dy0(x, area, a_sep, self.s_vertices, sigma)*da_sep_dy

        s_sep = smooth_selection(self.s_vertices, area, a_sep, self.s_vertices, sigma=sigma)
        ds_sep_dy = dsmooth_selection_dy(self.s_vertices, area, a_sep, self.s_vertices, sigma)*darea_dy \
                    + dsmooth_selection_dy0(self.s_vertices, area, a_sep, self.s_vertices, sigma)*da_sep_dy

        sep_multiplier = smooth_cutoff(self.s_vertices, s_sep, k=k)
        dsep_multiplier_dy = dsmooth_cutoff_dx0(self.s_vertices, s_sep, k=k)[:, None] * ds_sep_dy

        # p = sep_multiplier * p_bernoulli
        dp_dy = sep_multiplier[:, None]*dp_bernoulli_dy + dsep_multiplier_dy*p_bernoulli[:, None]

        dp_du = np.zeros((surface_state[0].size//2, surface_state[0].size))
        dp_du[:, :-1:2] = 0
        dp_du[:, 1::2] = dp_dy

        ## Calculate the flow rate sensitivity
        dflow_rate_du = np.zeros(surface_state[0].size)
        dflow_rate_du[1::2] = dflow_rate_sqr_da_sep/(2*flow_rate_sqr**(1/2)) * da_sep_da_min * da_min_darea * darea_dy

        return dflow_rate_du, dp_du

    def flow_sensitivity_solid(self, model, surface_state, fluid_props, adjoint=False):
        """
        Returns sparse matrices/vectors for the sensitivity of pressure and flow rate to displacement.

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
        x, y = surface_state[0][:, 0], surface_state[0][:, 1]

        area = 2 * (self.properties['y_midline'] - y)
        a_min = smooth_minimum(area, self.s_vertices, self.properties['alpha'])
        return a_min

    def get_ini_surf_config(self):
        xy = self.u0surf.copy()
        dxy_dt = self.v0surf
        xy[:-1:2] = self.u0surf[:-1:2] + self.x_vertices
        xy[1::2] = self.u0surf[1::2] + self.y_surface

        surf_state = (xy, dxy_dt)
        return surf_state

    def get_fin_surf_config(self):
        xy = self.u1surf.copy()
        dxy_dt = self.v1surf
        xy[:-1:2] = self.u1surf[:-1:2] + self.x_vertices
        xy[1::2] = self.u1surf[1::2] + self.y_surface

        surf_state = (xy, dxy_dt)
        return surf_state

# Below are a collection of smoothened functions for selecting the minimum area, separation point,
# and simulating separation

def smooth_minimum(f, s, alpha=-1000):
    """
    Return the smooth approximation to the minimum element of f integrated over s.

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

def d2smooth_minimum_df2(f, s, alpha=-1000):
    # TODO: Need to to implement this
    K_STABILITY = np.max(alpha*f)
    w = np.exp(alpha*f - K_STABILITY)
    dw_df = alpha*np.exp(alpha*f - K_STABILITY)
    d2w_df2 = alpha**2*np.exp(alpha*f - K_STABILITY)

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

    return np.sum( (s[1:]-s[:-1])*(f[1:]+f[:-1])/2 )

def dtrapz_df(f, s):
    out = np.zeros(f.size)
    out[:-1] += (s[1:]-s[:-1]) / 2
    out[1:] += (s[1:]-s[:-1]) / 2
    return out

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid_dx(x):
    """
    This is a diagonal matrix
    """
    sig = sigmoid(x)
    return sig * (1-sig)

def smooth_cutoff(x, x0, k=100):
    """
    Return the mirrored logistic function evaluated at x-x0
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
    return np.log(1/(sigma*(2*np.pi)**0.5)) + -0.5*((x-x0)/sigma)**2

def log_dgaussian_dx(x, x0, sigma=1.0):
    return log_gaussian(x, x0, sigma) * -(x-x0)/sigma**2

def log_dgaussian_dx0(x, x0, sigma=1.0):
    return log_gaussian(x, x0, sigma) * -(x-x0)/sigma**2

def gaussian(x, x0, sigma=1.0):
    return 1/(sigma*(2*np.pi)**0.5) * np.exp(-0.5*((x-x0)/sigma)**2)

def dgaussian_dx(x, x0, sigma=1.0):
    return gaussian(x, x0, sigma) * -(x-x0)/sigma**2

def dgaussian_dx0(x, x0, sigma=1.0):
    return gaussian(x, x0, sigma) * (x-x0)/sigma**2

def smooth_selection(x, y, y0, s, sigma=1.0):
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
