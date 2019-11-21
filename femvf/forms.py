"""
Module for definitions of weak forms.

Units are in CGS
"""
from os import path

import numpy as np

import dolfin as dfn
import ufl
from petsc4py import PETSc as pc

from . import fluids
from . import constants as const
from .properties import FluidProperties, SolidProperties

from .operators import LinCombOfMats

# dfn.parameters['form_compiler']['optimize'] = True
# dfn.parameters['form_compiler']['cpp_optimize'] = True

def newmark_v(u, u0, v0, a0, dt, gamma=1/2, beta=1/4):
    """
    Returns the Newmark method velocity update.

    Parameters
    ----------
    u : ufl.Argument or ufl.Coefficient
        A trial function, to solve for a displacement, or a coefficient function
    u0, v0, a0 : ufl.Coefficient
        The initial states
    gamma, beta : float
        Newmark time integration parameters

    Returns
    -------
    v1
    """
    return gamma/beta/dt*(u-u0) - (gamma/beta-1.0)*v0 - dt*(gamma/2.0/beta-1.0)*a0

def newmark_v_du1(dt, gamma=1/2, beta=1/4):
    """See `newmark_v`"""
    return gamma/beta/dt

def newmark_v_du0(dt, gamma=1/2, beta=1/4):
    """See `newmark_v`"""
    return -gamma/beta/dt

def newmark_v_dv0(dt, gamma=1/2, beta=1/4):
    """See `newmark_v`"""
    return - (gamma/beta-1.0)

def newmark_v_da0(dt, gamma=1/2, beta=1/4):
    """See `newmark_v`"""
    return - dt*(gamma/2.0/beta-1.0)

def newmark_a(u, u0, v0, a0, dt, gamma=1/2, beta=1/4):
    """
    Returns the Newmark method acceleration update.

    Parameters
    ----------
    u : ufl.Argument
    u0, v0, a0 : ufl.Argument
        Initial states
    gamma, beta : float
        Newmark time integration parameters

    Returns
    -------
    a1
    """
    return 1/beta/dt**2*(u-u0-dt*v0) - (1/2/beta-1)*a0

def newmark_a_du1(dt, gamma=1/2, beta=1/4):
    """See `newmark_a`"""
    return 1.0/beta/dt**2

def newmark_a_du0(dt, gamma=1/2, beta=1/4):
    """See `newmark_a`"""
    return -1.0/beta/dt**2

def newmark_a_dv0(dt, gamma=1/2, beta=1/4):
    """See `newmark_a`"""
    return -1.0/beta/dt

def newmark_a_da0(dt, gamma=1/2, beta=1/4):
    """See `newmark_a`"""
    return -(1/2/beta-1)

def linear_elasticity(u, emod, nu):
    """
    Returns the cauchy stress for a small-strain displacement field

    Parameters
    ----------
    u : dfn.TrialFunction, ufl.Argument
        Trial displacement field
    emod : dfn.Function, ufl.Coefficient
        Elastic modulus
    nu : float
        Poisson's ratio
    """
    lame_lambda = emod*nu/(1+nu)/(1-2*nu)
    lame_mu = emod/2/(1+nu)

    return 2*lame_mu*strain(u) + lame_lambda*ufl.tr(strain(u))*ufl.Identity(u.geometric_dimension())

def strain(u):
    """
    Returns the strain tensor for a displacement field.

    Parameters
    ----------
    u : dfn.TrialFunction, ufl.Argument
        Trial displacement field
    """
    return 1/2 * (ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

def biform_k(a, b, emod, nu):
    """
    Return stiffness bilinear form

    Integrates linear_elasticity(a) with the strain(b)
    """
    return ufl.inner(linear_elasticity(a, emod, nu), strain(b))*ufl.dx

def biform_m(a, b, rho):
    """
    Return the mass bilinear form

    Integrates a with b
    """
    return rho*ufl.dot(a, b) * ufl.dx

def _sort_surface_vertices(surface_coordinates):
    """
    Returns a list sorting the vertices in increasing streamwise direction.

    Assumes the inferior-superior direction is oriented along the positive x axis.

    Parameters
    ----------
    surface_coordinates : (..., 2) array_like
        An array of surface coordinates, with x and y locations stored in the last dimension.
    """
    # Determine the very first coordinate
    idx_sort = [np.argmin(surface_coordinates[..., 0])]

    while len(idx_sort) < surface_coordinates.shape[0]:
        # Calculate array of distances to every other coordinate
        vector_distances = surface_coordinates - surface_coordinates[idx_sort[-1]]
        distances = np.sum(vector_distances**2, axis=-1)**0.5
        distances[idx_sort] = np.nan

        idx_sort.append(np.nanargmin(distances))

    return np.array(idx_sort)

def _linear_elastic_forms(mesh, facet_function, facet_labels, cell_function, cell_labels,
    solid_props=None, fluid_props=None):
        """
        Return a dictionary of variational forms and other parameters.

        This is specific to the forward model and only exists to separate out this long ass section of
        code.
        """

        if solid_props is None:
            solid_props = SolidProperties()
        if fluid_props is None:
            fluid_props = FluidProperties()

        scalar_fspace = dfn.FunctionSpace(mesh, 'CG', 1)
        vector_fspace = dfn.VectorFunctionSpace(mesh, 'CG', 1)

        vector_trial = dfn.TrialFunction(vector_fspace)
        vector_test = dfn.TestFunction(vector_fspace)

        scalar_trial = dfn.TrialFunction(scalar_fspace)
        scalar_test = dfn.TestFunction(scalar_fspace)

        # Newmark update parameters
        gamma = dfn.Constant(1/2)
        beta = dfn.Constant(1/4)

        # Solid material properties
        y_collision = dfn.Constant(solid_props['y_collision'])
        k_collision = dfn.Constant(solid_props['k_collision'])
        rho = dfn.Constant(solid_props['density'])
        nu = dfn.Constant(solid_props['poissons_ratio'])
        rayleigh_m = dfn.Constant(solid_props['rayleigh_m'])
        rayleigh_k = dfn.Constant(solid_props['rayleigh_k'])
        emod = dfn.Function(scalar_fspace)

        emod.vector()[:] = solid_props['elastic_modulus']

        # Initial and final states
        # u: displacement, v: velocity, a: acceleration
        u0 = dfn.Function(vector_fspace)
        v0 = dfn.Function(vector_fspace)
        a0 = dfn.Function(vector_fspace)

        u1 = dfn.Function(vector_fspace)

        # Time step
        dt = dfn.Constant(1e-4)

        # Surface pressures
        pressure = dfn.Function(scalar_fspace)

        # Symbolic calculations to get the variational form for a linear-elastic solid
        trial_v = newmark_v(vector_trial, u0, v0, a0, dt, gamma, beta)
        trial_a = newmark_a(vector_trial, u0, v0, a0, dt, gamma, beta)

        inertia = biform_m(trial_a, vector_test, rho)

        # stress = linear_elasticity(vector_trial, emod, nu)

        stiffness = biform_k(vector_trial, vector_test, emod, nu)

        damping = rayleigh_m * biform_m(trial_v, vector_test, rho) \
                  + rayleigh_k * biform_k(trial_v, vector_test, emod, nu)

        # Compute the pressure loading Neumann boundary condition on the reference configuration
        # using Nanson's formula. This is because the 'total lagrangian' formulation is used.
        deformation_gradient = ufl.grad(u0) + ufl.Identity(2)
        deformation_cofactor = ufl.det(deformation_gradient) * ufl.inv(deformation_gradient).T

        ds = dfn.Measure('ds', domain=mesh, subdomain_data=facet_function)
        fluid_force = -pressure*deformation_cofactor*dfn.FacetNormal(mesh)

        traction = ufl.dot(fluid_force, vector_test)*ds(facet_labels['pressure'])

        # Use the penalty method to account for collision
        collision_normal = dfn.Constant([0, 1])
        x_reference = dfn.Function(vector_fspace)

        vert_to_vdof = dfn.vertex_to_dof_map(vector_fspace)
        x_reference.vector()[vert_to_vdof.reshape(-1)] = mesh.coordinates().reshape(-1)

        gap = ufl.dot(x_reference+u1, collision_normal) - y_collision
        positive_gap = (gap + abs(gap)) / 2


        penalty = ufl.dot(k_collision*positive_gap**2*-1*collision_normal, vector_test) \
                * ds(facet_labels['pressure'])
        f1_linear = ufl.action(inertia + stiffness + damping, u1)
        f1_nonlin = -traction - penalty
        f1 = f1_linear + f1_nonlin

        df1_du1_linear = ufl.derivative(f1_linear, u1, vector_trial)
        df1_du1_nonlin = ufl.derivative(f1_nonlin, u1, vector_trial)
        df1_du1 = df1_du1_linear + df1_du1_nonlin

        ## Boundary conditions
        # Specify DirichletBC at the VF base
        bc_base = dfn.DirichletBC(vector_fspace, dfn.Constant([0, 0]),
                                  facet_function, facet_labels['fixed'])

        bc_base_adj = dfn.DirichletBC(vector_fspace, dfn.Constant([0, 0]),
                                      facet_function, facet_labels['fixed'])

        ## Adjoint forms
        # Note: For an externally calculated pressure, you have to correct the derivative based on
        # the sensitivity of pressure loading in `f1` to either `u0` and/or `u1` depending on if
        # it's strongly coupled.
        df1_du0_adj_linear = dfn.adjoint(ufl.derivative(f1_linear, u0, vector_trial))
        df1_du0_adj_nonlin = dfn.adjoint(ufl.derivative(f1_nonlin, u0, vector_trial))
        df1_du0_adj = df1_du0_adj_linear + df1_du0_adj_nonlin

        df1_dv0_adj_linear = dfn.adjoint(ufl.derivative(f1_linear, v0, vector_trial))
        df1_dv0_adj_nonlin = 0
        df1_dv0_adj = df1_dv0_adj_linear + df1_dv0_adj_nonlin

        df1_da0_adj_linear = dfn.adjoint(ufl.derivative(f1_linear, a0, vector_trial))
        df1_da0_adj_nonlin = 0
        df1_da0_adj = df1_da0_adj_linear + df1_da0_adj_nonlin

        df1_dp_adj = dfn.adjoint(ufl.derivative(f1, pressure, scalar_trial))

        df1_du1_adj_linear = dfn.adjoint(df1_du1_linear)
        df1_du1_adj_nonlin = dfn.adjoint(df1_du1_nonlin)
        df1_du1_adj = df1_du1_adj_linear + df1_du1_adj_nonlin

        df1_demod = ufl.derivative(f1, emod, scalar_trial)
        df1_dpressure_adj = dfn.adjoint(ufl.derivative(f1, pressure, scalar_trial))

        # Work done by pressure from u0 to u1
        # fluid_work = ufl.dot(fluid_force, u1-u0) * ds(facet_labels['pressure'])
        # dfluid_work_du0 = ufl.derivative(fluid_work, u0, vector_test)
        # dfluid_work_du1 = ufl.derivative(fluid_work, u1, vector_test)
        # dfluid_work_dp = ufl.derivative(fluid_work, pressure, scalar_test)

        forms = {
            'bcs.base': bc_base,
            # 'bcs.base'
            'fspace.vector': vector_fspace,
            'fspace.scalar': scalar_fspace,
            'test.vector': vector_test,
            'test.scalar': scalar_test,
            'trial.vector': vector_trial,
            'trial.scalar': scalar_trial,
            'arg.u1': u1,
            'coeff.dt': dt,
            'coeff.gamma': gamma,
            'coeff.beta': beta,
            'coeff.u0': u0,
            'coeff.v0': v0,
            'coeff.a0': a0,
            'coeff.pressure': pressure,
            'coeff.rho': rho,
            'coeff.rayleigh_m': rayleigh_m,
            'coeff.rayleigh_k': rayleigh_k,
            'coeff.y_collision': y_collision,
            'coeff.emod': emod,
            'coeff.nu': nu,
            'coeff.k_collision': k_collision,
            'lin.f1': f1,
            'bilin.df1_du1': df1_du1,
            'bilin.df1_du1_adj': df1_du1_adj,
            'bilin.df1_du0_adj': df1_du0_adj,
            'bilin.df1_dv0_adj': df1_dv0_adj,
            'bilin.df1_da0_adj': df1_da0_adj,
            'bilin.df1_dpressure_adj': df1_dpressure_adj,
            'bilin.df1_demod': df1_demod}

        return forms

class ForwardModel:
    """
    Stores all the things related to the vocal fold forward model solved thru fenics.

    TODO: Instantiation is kind of messy and ugly. Prettify/clean it up.
    TODO: Class contains alot of extra, non-essential stuff. Think about what are the essential
        things that are included, how to compartmentalize the things etc.
    TODO: think about how to include extraneous forms for computing some functional etc...

    Parameters
    ----------
    mesh_path : str
        Path to a fenics mesh xml file. An xml file containing facet and cell functions are also
        loaded in the directory.
    facet_labels, cell_labels : dict of {str: int}
        A dictionary of named markers for facets and cells. `facet_labels` needs atleast two
        entries {'pressure': ...} and {'fixed': ...} denoting the facets with applied pressure
        and fixed conditions respectively.

    Attributes
    ----------
    mesh : dfn.Mesh
    facet_function : dfn.MeshFunction
    cell_function : dfn.MeshFunction

    fluid_props : .properties.FluidProperties
    solid_props : .properties.SolidProperties

    surface_vertices : array_like
        A list of vertex numbers on the pressure surface. They are ordered in increasing streamwise
        direction.

    vector_function_space : dfn.VectorFunctionSpace
    scalar_function_space : dfn.FunctionSpace
    vertex_to_vdof : array_like
    vertex_to_sdof : array_like

    form coefficients and states

    main forms for solving
    """

    def __init__(self, mesh_path, facet_labels, cell_labels):
        ## Setting the mesh
        base_path, ext = path.splitext(mesh_path)
        facet_function_path = base_path +  '_facet_region.xml'
        cell_function_path = base_path + '_physical_region.xml'

        if ext == '':
            mesh_path = mesh_path + '.xml'

        self.mesh = dfn.Mesh(mesh_path)
        self.facet_function = dfn.MeshFunction('size_t', self.mesh, facet_function_path)
        self.cell_function = dfn.MeshFunction('size_t', self.mesh, cell_function_path)

        # Create a vertex marker from the boundary marker
        edge_to_vertex = np.array([[vertex.index() for vertex in dfn.vertices(edge)]
                                   for edge in dfn.edges(self.mesh)])
        pressure_edges = self.facet_function.where_equal(facet_labels['pressure'])
        fixed_edges = self.facet_function.where_equal(facet_labels['fixed'])

        pressure_vertices = np.unique(edge_to_vertex[pressure_edges].reshape(-1))
        fixed_vertices = np.unique(edge_to_vertex[fixed_edges].reshape(-1))

        vertex_marker = dfn.MeshFunction('size_t', self.mesh, 0)
        vertex_marker.set_all(0)
        vertex_marker.array()[pressure_vertices] = facet_labels['pressure']
        vertex_marker.array()[fixed_vertices] = facet_labels['fixed']

        surface_vertices = np.array(vertex_marker.where_equal(facet_labels['pressure']))
        surface_coordinates = self.mesh.coordinates()[surface_vertices]

        # Sort the pressure surface vertices from inferior to superior
        idx_sort = _sort_surface_vertices(surface_coordinates)
        self.surface_vertices = surface_vertices[idx_sort]
        self.surface_coordinates = surface_coordinates[idx_sort]

        ## Set properties
        # Default property values are used
        self.fluid_props = FluidProperties()
        self.solid_props = SolidProperties()

        ## Variational Forms
        self._forms = _linear_elastic_forms(self.mesh,
            self.facet_function, facet_labels, self.cell_function, cell_labels)

        # Add some commonly used things as parameters
        self.vector_function_space = self.forms['fspace.vector']
        self.scalar_function_space = self.forms['fspace.scalar']
        self.vert_to_vdof = dfn.vertex_to_dof_map(self.forms['fspace.vector']).reshape(-1, 2)
        self.vert_to_sdof = dfn.vertex_to_dof_map(self.forms['fspace.scalar'])

        self.gamma = self.forms['coeff.gamma']
        self.beta = self.forms['coeff.beta']
        self.emod = self.forms['coeff.emod']
        self.y_collision = self.forms['coeff.y_collision']
        self.dt = self.forms['coeff.dt']
        self.u0 = self.forms['coeff.u0']
        self.v0 = self.forms['coeff.v0']
        self.a0 = self.forms['coeff.a0']
        self.u1 = self.forms['arg.u1']

        self.f1 = self.forms['lin.f1']
        self.df1_du1 = self.forms['bilin.df1_du1']

        self.bc_base_adj = self.forms['bcs.base']
        self.bc_base = self.forms['bcs.base']

        self.scalar_trial = self.forms['trial.scalar']
        self.vector_trial = self.forms['trial.vector']

        # self.assem_cache = {}
        # self.reset_cache()
        # self.reset_adj_cache()

        self.df1_du1_mat = dfn.assemble(self.forms['bilin.df1_du1'])
        self.df1_du1_adj_mat = dfn.assemble(self.forms['bilin.df1_du1_adj'])
        self.df1_du0_adj_mat = dfn.assemble(self.forms['bilin.df1_du0_adj'])
        self.df1_dv0_adj_mat = dfn.assemble(self.forms['bilin.df1_dv0_adj'])
        self.df1_da0_adj_mat = dfn.assemble(self.forms['bilin.df1_da0_adj'])

    @property
    def forms(self):
        return self._forms

    # def reset_cache(self):
    #     """
    #     Sets cached stiffness and mass matrices.
    #     """
    #     out = {
    #         'M': dfn.assemble(biform_m(self.vector_trial, self.vector_test, self.rho)),
    #         'K': dfn.assemble(biform_k(self.vector_trial, self.vector_test, self.emod, self.nu))}
    #     self.assem_cache.update(out)

    # def reset_adj_cache(self):
    #     """
    #     Sets cached adjoint stiffness and mass matrices.
    #     """
    #     out = {
    #         'M.adj': dfn.assemble(
    #             dfn.adjoint(biform_m(self.vector_trial, self.vector_test, self.rho))),
    #         'K.adj': dfn.assemble(
    #             dfn.adjoint(biform_k(self.vector_trial, self.vector_test, self.emod, self.nu)))}
    #     self.assem_cache.update(out)

    # Core solver functions
    def get_ref_config(self):
        """
        Returns the current configuration of the body.

        Coordinates of the body are ordered according to vertices.

        Returns
        -------
        array_like
            An array of mesh coordinate point ordered with increasing vertices.
        """
        return self.mesh.coordinates()

    def get_cur_config(self):
        """
        Returns the current configuration of the body.

        Coordinates of the body are ordered according to vertices.

        Returns
        -------
        array_like
            An array of mesh coordinate point ordered with increasing vertices.
        """
        displacement = self.u0.vector()[self.vert_to_vdof]
        return self.mesh.coordinates() + displacement

    def get_surface_state(self):
        """
        Returns the state (u, v, a) of surface vertices of the model.

        The displacement, u, returned is the actual position rather than the displacement relative
        to the reference configuration. Also, states are ordered according to
        `self.surface_vertices`.

        Returns
        -------
        tuple of array_like
            A tuple of arrays of surface positions, velocities and accelerations.
        """
        surface_dofs = self.vert_to_vdof[self.surface_vertices].reshape(-1)

        u = self.u0.vector()[surface_dofs].reshape(-1, 2)
        v = self.v0.vector()[surface_dofs].reshape(-1, 2)
        a = self.a0.vector()[surface_dofs].reshape(-1, 2)

        x_surface = (self.surface_coordinates + u, v, a)

        return x_surface

    def get_pressure(self):
        """
        Updates pressure coefficient using a bernoulli flow model.

        Parameters
        ----------
        fluid_props : dict
            A dictionary of fluid properties for the 1D bernoulli fluids model
        """
        # Update the pressure loading based on the deformed surface
        x_surface = self.get_surface_state()

        # Check that the surface doesn't cross over the midline
        assert np.max(x_surface[0][..., 1]) < self.fluid_props['y_midline']

        pressure, fluid_info = fluids.get_pressure_form(self, x_surface, self.fluid_props)

        self.forms['coeff.pressure'].assign(pressure)

        return fluid_info

    def get_flow_sensitivity(self):
        """
        Updates pressure sensitivity using a bernoulli flow model.

        Parameters
        ----------
        fluid_props : dict
            A dictionary of fluid properties

        Returns
        -------
        dp_du0 : np.ndarray
            Sensitivity of surface pressures w.r.t. the initial displacement.
        dq_du0 : np.ndarray
            Sensitivity of the flow rate w.r.t. the initial displacement.
        """
        # Calculate sensitivities of fluid quantities based on the deformed surface
        x_surface = self.get_surface_state()
        dp_du0, dq_du0 = fluids.get_flow_sensitivity(self, x_surface, self.fluid_props)

        return dp_du0, dq_du0

    # @profile
    def assem_f1(self, u1=None):
        """
        Return the residual vector

        Parameters
        ----------
        u1 : dfn.cpp.la.Vector
        """
        return dfn.assemble(self.forms['lin.f1'])
        # M = self.assem_cache['M']
        # K = self.assem_cache['K']

        # dt = self.dt.values()[0]
        # gamma, beta = self.gamma.values()[0], self.beta.values()[0]

        # rm = self.rayleigh_m.values()[0]
        # rk = self.rayleigh_k.values()[0]

        # u0 = self.u0.vector()
        # v0 = self.v0.vector()
        # a0 = self.a0.vector()

        # if u1 is None:
        #     u1 = self.u1.vector()

        # v1 = newmark_v(u1, u0, v0, a0, dt, gamma, beta)
        # a1 = newmark_a(u1, u0, v0, a0, dt, gamma, beta)

        # return M*(a1 + rm*v1) + K*(u1 + rk*v1) + dfn.assemble(self.f1_nonlin)

    # TODO: Adding matrices has an overhead due to different sparsity patterns of the component
    # matrices. You may be able to speed up addition by having an output matrix with sparsity that
    # is a superset of all the component matrix sparsities. I believe, fenics does not directly
    # support this type of add, so you need to use petsc directly.
    def assem_df1_du1(self):
        """
        Return the residual vector jacobian

        Parameters
        ----------
        u1 : dfn.cpp.la.Vector
        """
        return dfn.assemble(self.forms['df1_du1'])

    def assem_df1_du1_adj(self):
        return dfn.assemble(self.forms['bilin.df1_du1_adj'])

    def assem_df1_du0_adj(self):
        return dfn.assemble(self.forms['bilin.df1_du0_adj'])

    def assem_df1_dv0_adj(self):
        return dfn.assemble(self.forms['bilin.df1_dv0_adj'])

    def assem_df1_da0_adj(self):
        return dfn.assemble(self.forms['bilin.df1_da0_adj'])

    # Convenience functions
    def get_glottal_width(self):
        """
        Return glottal width
        """
        x_surface = self.get_surface_state()

        return self.fluid_props['y_midline'] - np.max(x_surface[0][..., 1])

    def get_collision_gap(self):
        """
        Return the smallest distance to the collision plane
        """
        u_surface = self.get_surface_state()[0]

        return self.y_collision.values()[0] - np.max(u_surface[..., 1])

    def get_ymax(self):
        """
        Return the maximum y-coordinate of the reference configuration
        """
        x_surface = self.get_surface_state()

        return np.max(x_surface[0][..., 1])

    def get_collision_verts(self):
        """
        Return vertex numbers of nodes in collision.
        """
        # import ipdb; ipdb.set_trace()
        u_surface = self.get_surface_state()[0]
        verts = self.surface_vertices[u_surface[..., 1] > self.y_collision.values()[0]]
        return verts

    # Parameter value setting functions
    def set_ini_state(self, u0, v0, a0):
        """
        Sets the state variables u, v, and a at the start of the step.

        Parameters
        ----------
        u0, v0, a0 : array_like
        """
        self.forms['coeff.u0'].vector()[:] = u0
        self.forms['coeff.v0'].vector()[:] = v0
        self.forms['coeff.a0'].vector()[:] = a0

    def set_fin_state(self, u1):
        """
        Sets the displacement at the end of the time step.

        Named `set_final_state` so that it pairs up with `set_initial_state`,  even though there is
        only an input `u`, instead of `u`, `v` and `a`.

        This could be an initial guess in the case of non-linear governing equations, or a solved
        state so that the non-linear form can be linearized for the given state.

        Parameters
        ----------
        u1 : array_like
        """
        self.forms['arg.u1'].vector()[:] = u1

    def set_time_step(self, dt):
        """
        Sets the time step.
        """
        self.forms['coeff.dt'].assign(dt)

    def set_solid_props(self, solid_props):
        """
        Sets solid properties given a dictionary of solid properties.

        Parameters
        ----------
        solid_props : properties.SolidProperties
        """
        labels = SolidProperties.TYPES.keys()
        forms = self.forms
        coefficients = [forms['coeff.emod'], forms['coeff.nu'], forms['coeff.rho'],
                        forms['coeff.rayleigh_m'], forms['coeff.rayleigh_k'],
                        forms['coeff.y_collision']]

        for coefficient, label in zip(coefficients, labels):
            if label in solid_props:
                if label == 'elastic_modulus':
                    coefficient.vector()[:] = solid_props[label]
                else:
                    coefficient.assign(solid_props[label])

    def set_fluid_props(self, fluid_props):
        """
        Sets fluid properties given a dictionary of fluid properties.

        This just sets the pressure vector given the fluid boundary conditions.

        Parameters
        ----------
        fluid_props : properties.FluidProperties
        """
        self.fluid_props = fluid_props

    def set_params(self, x0, solid_props=None, fluid_props=None):
        """
        Set all parameters needed to integrate the model.

        Parameters
        ----------
        x0 : array_like
        dt : float
        fluid_props : dict
        solid_props : dict
        """
        self.set_ini_state(*x0)

        if fluid_props is not None:
            self.set_fluid_props(fluid_props)

        if solid_props is not None:
            self.set_solid_props(solid_props)

        fluid_info = self.get_pressure()

        return fluid_info

    def set_params_fromfile(self, statefile, n, update_props=False):
        """
        Set all parameters needed to integrate the model from a recorded value.

        Iteration `n` is the implicit relation
        :math: f^{n}(u_n, u_{n-1}, p)
        that gives the displacement at index `n`, given the state at `n-1` and all additional
        parameters.

        Parameters
        ----------
        statefile : statefileutils.StateFile
        n : int
            Index of iteration to set
        """
        # Get data from the state file
        solid_props, fluid_props = None, None
        if update_props:
            fluid_props = statefile.get_fluid_props(n)
            solid_props = statefile.get_solid_props()

        x0 = statefile.get_state(n)

        # Assign the values to the model
        fluid_info = self.set_params(x0, solid_props=solid_props, fluid_props=fluid_props)

        return fluid_info

    def set_iter_params(self, x0, dt, u1=None, solid_props=None, fluid_props=None):
        """
        Set all parameters needed to integrate the model and an initial guess.

        Parameters
        ----------
        x0 : array_like
        dt : float
        fluid_props : dict
        solid_props : dict
        u1 : array_like, optional
        """
        self.set_params(x0, solid_props=solid_props, fluid_props=fluid_props)

        self.set_time_step(dt)

        if u1 is not None:
            self.set_fin_state(u1)

        fluid_info = self.get_pressure()

        return fluid_info

    def set_iter_params_fromfile(self, statefile, n, set_final_state=True, update_props=False):
        """
        Set all parameters needed to integrate the model and an initial guess, based on a recorded
        iteration.

        Iteration `n` is the implicit relation
        :math: f^{n}(u_n, u_{n-1}, p)
        that gives the displacement at index `n`, given the state at `n-1` and all additional
        parameters.

        Parameters
        ----------
        statefile : statefileutils.StateFile
        n : int
            Index of iteration to set
        """
        # Get data from the state file
        solid_props, fluid_props = None, None
        if update_props:
            fluid_props = statefile.get_fluid_props(0)
            solid_props = statefile.get_solid_props()

        x0 = statefile.get_state(n-1)
        u1 = None
        if set_final_state:
            u1 = statefile.get_u(n)

        dt = statefile.get_time(n) - statefile.get_time(n-1)

        # Assign the values to the model
        fluid_info = self.set_iter_params(x0, dt, solid_props=solid_props, fluid_props=fluid_props,
                                          u1=u1)

        return fluid_info, fluid_props

if __name__ == '__main__':
    mesh_path = '../geometry2.xml'
    model = ForwardModel(mesh_path, {'pressure': 1, 'fixed': 3}, {})