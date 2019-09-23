"""
Module for definitions of weak forms.

Units are in CGS
"""
import sys
import os
from os import path

import numpy as np

import dolfin as dfn
import ufl

from . import transforms
from . import fluids
from . import constants as const

# dfn.parameters['form_compiler']['optimize'] = True
# dfn.parameters['form_compiler']['cpp_optimize'] = True

def newmark_v(u, u0, v0, a0, dt, gamma=1/2, beta=1/4):
    """
    Returns the Newmark method displacement update.

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


class ForwardModel:
    """
    Stores all the things related to the vocal fold forward model solved thru fenics.

    TODO: Instantiation is kind of messy and ugly. Prettify/clean it up.
    TODO: Class contains alot of extra, non-essential stuff. Think about what are the essential
        things that are included, how to compartmentalize the things etc.
    TODO: think about how to include extraneous forms for computing some functional etc...

    Parameters
    ----------
    mesh : str
        Path to a fenics mesh xml file. A xml file containing facet and cell markers are also loaded
        in the directory.
    facet_marker_id, cell_marker_id : dict of {str: int}
        A dictionary of named markers for facets and cells. `facet_marker_id` needs atleast two
        entries {'pressure': ...} and {'fixed': ...} denoting the surfaces with applied pressure
        and fixed conditions respectively.

    Attributes
    ----------
    mesh : dfn.Mesh

    list of key vertices ie fsi interface

    vertex_to_vdof
    vertex_to_sdof

    vector_function_space
    scalar_function_space

    form coefficients and states

    main forms for solving
    """
    def __init__(self, mesh_path, facet_marker_id, cell_marker_id):

        base_path, ext = path.splitext(mesh_path)
        facet_marker_path = base_path +  '_facet_region.xml'
        cell_marker_path = base_path + '_physical_region.xml'

        if ext == '':
            mesh_path = mesh_path + '.xml'

        ## Fluid properties
        self.fluid_properties = const.DEFAULT_FLUID_PROPERTIES

        ## Mesh parameters and attributes
        self.mesh = dfn.Mesh(mesh_path)
        self.facet_marker = dfn.MeshFunction('size_t', self.mesh, facet_marker_path)
        self.body_marker = dfn.MeshFunction('size_t', self.mesh, cell_marker_path)

        # Create a vertex marker from the boundary marker
        edge_to_vertex = np.array([[vertex.index() for vertex in dfn.vertices(edge)]
                                   for edge in dfn.edges(self.mesh)])
        pressure_edges = self.facet_marker.where_equal(facet_marker_id['pressure'])
        fixed_edges = self.facet_marker.where_equal(facet_marker_id['fixed'])

        pressure_vertices = np.unique(edge_to_vertex[pressure_edges].reshape(-1))
        fixed_vertices = np.unique(edge_to_vertex[fixed_edges].reshape(-1))

        vertex_marker = dfn.MeshFunction('size_t', self.mesh, 0)
        vertex_marker.set_all(0)
        vertex_marker.array()[pressure_vertices] = facet_marker_id['pressure']
        vertex_marker.array()[fixed_vertices] = facet_marker_id['fixed']

        surface_vertices = np.array(vertex_marker.where_equal(facet_marker_id['pressure']))
        surface_coordinates = self.mesh.coordinates()[surface_vertices]

        # Sort the pressure surface vertices from inferior to superior
        # TODO: This only works if surface coordinates have strictly increasing x-coordinate from
        # inferior to superior. Meshes that don't might have some weird results.
        idx_sort = _sort_surface_vertices(surface_coordinates)
        self.surface_vertices = surface_vertices[idx_sort]
        self.surface_coordinates = surface_coordinates[idx_sort]

        ## Set Variational Forms
        # Newmark update parameters
        self.gamma = dfn.Constant(1/2)
        self.beta = dfn.Constant(1/4)

        self.y_collision = dfn.Constant(const.DEFAULT_SOLID_PROPERTIES['y_collision'])
        # self.y_midline = const.DEFAULT_FLUID_PROPERTIES['y_midline']

        self.scalar_function_space = dfn.FunctionSpace(self.mesh, 'CG', 1)
        self.vector_function_space = dfn.VectorFunctionSpace(self.mesh, 'CG', 1)

        self.vert_to_vdof = dfn.vertex_to_dof_map(self.vector_function_space).reshape(-1, 2)
        self.vert_to_sdof = dfn.vertex_to_dof_map(self.scalar_function_space)

        self.vector_trial = dfn.TrialFunction(self.vector_function_space)
        self.vector_test = dfn.TestFunction(self.vector_function_space)

        self.scalar_trial = dfn.TrialFunction(self.scalar_function_space)
        self.scalar_test = dfn.TestFunction(self.scalar_function_space)

        # Solid material properties
        self.rho = dfn.Constant(1)
        self.nu = dfn.Constant(0.48)
        self.rayleigh_m = dfn.Constant(1e-4)
        self.rayleigh_k = dfn.Constant(1e-4)
        self.emod = dfn.Function(self.scalar_function_space)

        self.emod.vector()[:] = 11.8e3 * const.PASCAL_TO_CGS

        # Initial conditions
        self.u0 = dfn.Function(self.vector_function_space)
        self.v0 = dfn.Function(self.vector_function_space)
        self.a0 = dfn.Function(self.vector_function_space)

        # Next time step displacement
        self.u1 = dfn.Function(self.vector_function_space)

        # Time step
        self.dt = dfn.Constant(1e-4)

        # Pressure forcing
        self.pressure = dfn.Function(self.scalar_function_space)

        # Define the variational forms
        trial_v = newmark_v(self.vector_trial, self.u0, self.v0, self.a0, self.dt, self.gamma, self.beta)
        trial_a = newmark_a(self.vector_trial, self.u0, self.v0, self.a0, self.dt, self.gamma, self.beta)

        inertia = self.rho*ufl.dot(trial_a, self.vector_test)*ufl.dx

        stress = linear_elasticity(self.vector_trial, self.emod, self.nu)
        stiffness = ufl.inner(stress, strain(self.vector_test))*ufl.dx

        damping = self.rayleigh_m * self.rho*ufl.dot(trial_v, self.vector_test)*ufl.dx \
                  + self.rayleigh_k * ufl.inner(linear_elasticity(trial_v, self.emod, self.nu),
                                                strain(self.vector_test))*ufl.dx

        # Compute the pressure loading Neumann boundary condition thru Nanson's formula
        deformation_gradient = ufl.grad(self.u0) + ufl.Identity(2)
        deformation_cofactor = ufl.det(deformation_gradient) * ufl.inv(deformation_gradient).T

        ds = dfn.Measure('ds', domain=self.mesh, subdomain_data=self.facet_marker)
        fluid_force = -self.pressure*deformation_cofactor*dfn.FacetNormal(self.mesh)

        traction = ufl.dot(fluid_force, self.vector_test)*ds(facet_marker_id['pressure'])

        fu = inertia + stiffness + damping - traction

        # Non linear equations during collision. Add a penalty to account for this
        collision_normal = dfn.Constant([0, 1])
        x_reference = dfn.Function(self.vector_function_space)
        x_reference.vector()[self.vert_to_vdof.reshape(-1)] = self.mesh.coordinates().reshape(-1)

        gap = ufl.dot(x_reference+self.u1, collision_normal) - (self.y_collision)
        positive_gap = (gap + abs(gap)) / 2

        self.k_collision = dfn.Constant(1e11)
        penalty = ufl.dot(self.k_collision*positive_gap**2*-1*collision_normal, self.vector_test) \
                  * ds(facet_marker_id['pressure'])

        self.fu_nonlin = ufl.action(fu, self.u1) - penalty
        self.jac_fu_nonlin = ufl.derivative(self.fu_nonlin, self.u1, self.vector_trial)

        ## Boundary conditions
        # Specify DirichletBC at the VF base
        self.bc_base = dfn.DirichletBC(self.vector_function_space, dfn.Constant([0, 0]),
                                       self.facet_marker, facet_marker_id['fixed'])

        self.bc_base_adjoint = dfn.DirichletBC(self.vector_function_space, dfn.Constant([0, 0]),
                                               self.facet_marker, facet_marker_id['fixed'])

        # Define some additional forms for diagnostics
        # force_form = ufl.inner(linear_elasticity(self.u0, self.emod, self.nu), strain(test))*ufl.dx - traction

        ## Forms needed for adjoint computation
        # Note: For an externally calculated pressure, you have to correct the derivative based on the
        # sensitivity of the pressure loading in f1 to either u0 or u1 (or both depending if it's strongly
        # coupled).
        self.f1 = self.fu_nonlin
        self.df1_du0_adjoint = dfn.adjoint(ufl.derivative(self.f1, self.u0, self.vector_trial))
        self.df1_da0_adjoint = dfn.adjoint(ufl.derivative(self.f1, self.a0, self.vector_trial))
        self.df1_dv0_adjoint = dfn.adjoint(ufl.derivative(self.f1, self.v0, self.vector_trial))
        self.df1_dp_adjoint = dfn.adjoint(ufl.derivative(self.f1, self.pressure, self.scalar_trial))

        self.df1_du1_adjoint = dfn.adjoint(ufl.derivative(self.f1, self.u1, self.vector_trial))

        # Work done by pressure from u0 to u1
        self.fluid_work = ufl.dot(fluid_force, self.u1-self.u0) * ds(facet_marker_id['pressure'])
        self.dfluid_work_du0 = ufl.derivative(self.fluid_work, self.u0, self.vector_test)
        self.dfluid_work_du1 = ufl.derivative(self.fluid_work, self.u1, self.vector_test)
        self.dfluid_work_dp = ufl.derivative(self.fluid_work, self.pressure, self.scalar_test)

    def get_reference_configuration(self):
        """
        Returns the current configuration of the body.

        Coordinates of the body are ordered according to vertices.

        Returns
        -------
        array_like
            An array of mesh coordinate point ordered with increasing vertices.
        """

        return self.mesh.coordinates()

    def get_current_configuration(self):
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


    def set_initial_state(self, u0, v0, a0):
        """
        Sets the state variables u, v, and a at the start of the step.

        Parameters
        ----------
        u0, v0, a0 : dfn.Function
        """
        self.u0.assign(u0)
        self.v0.assign(v0)
        self.a0.assign(a0)

    def set_final_displacement(self, u1):
        """
        Sets the displacement at the end of the time step.

        This could be an initial guess in the case of non-linear governing equations, or a solved
        state so that the non-linear form can be linearized for the given state.

        Parameters
        ----------
        u1 : dfn.Function
        """
        self.u1.assign(u1)

    def set_time_step(self, dt):
        """
        Sets the time step.
        """
        self.dt.assign(dt)

    def set_solid_properties(self, solid_props):
        """
        Sets solid properties given a dictionary of solid properties.

        Parameters
        ----------
        solid_properties : dict
        """
        labels = const.DEFAULT_SOLID_PROPERTIES
        coefficients = [self.emod, self.nu, self.rho, self.rayleigh_m, self.rayleigh_k,
                        self.y_collision]

        for coefficient, label in zip(coefficients, labels):
            if label in solid_props:
                if label == 'elastic_modulus' and not isinstance(solid_props[label], dfn.Function):
                    coefficient.vector()[:] = solid_props[label]
                else:
                    coefficient.assign(solid_props[label])

    def set_fluid_properties(self, fluid_props):
        """
        Sets fluid properties given a dictionary of fluid properties.

        This just sets the pressure vector given the fluid boundary conditions.

        Parameters
        ----------
        fluid_props : dict
        """
        self.fluid_properties = fluid_props

        return self.set_pressure()

    def set_pressure(self):
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
        # print(np.max(x_surface[0][..., 1]))
        assert np.max(x_surface[0][..., 1]) < self.fluid_properties['y_midline']

        pressure, fluid_info = fluids.get_pressure_form(self, x_surface, self.fluid_properties)

        self.pressure.assign(pressure)
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
        dp_du0, dq_du0 = fluids.get_flow_sensitivity(self, x_surface, self.fluid_properties)

        return dp_du0, dq_du0

    def set_iteration(self, x0, dt, fluid_props, solid_props, u1=None):
        """
        Sets all initial coefficient values for a step, based on a recorded solution.

        Parameters
        ----------
        statefile : statefileutils.StateFile

        n : int
            Index of iteration to set
        """
        self.set_time_step(dt)
        self.set_initial_state(*x0)
        self.set_fluid_properties(fluid_props)
        self.set_solid_properties(solid_props)

        if u1 is not None:
            self.set_final_displacement(u1)

        fluid_info = self.set_pressure()

        return fluid_info

    def set_iteration_fromfile(self, statefile, n, set_final_state=True):
        """
        Sets all coefficient values based on a recorded iteration.

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
        fluid_props = statefile.get_fluid_properties(n-1)
        solid_props = statefile.get_solid_properties()
        x0 = statefile.get_state(n-1, function_space=self.vector_function_space)
        u1 = None
        if set_final_state:
            u1 = statefile.get_u(n, function_space=self.vector_function_space)

        dt = statefile.get_time(n) - statefile.get_time(n-1)

        # Assign the values to the model
        fluid_info = self.set_iteration(x0, dt, fluid_props, solid_props, u1=u1)

        return fluid_info, fluid_props
