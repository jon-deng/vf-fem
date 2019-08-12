"""
Module for definitions of weak forms.

Units are in CGS
"""
import sys
import os

import numpy as np

import dolfin as dfn
import ufl

from . import transforms
from . import fluids
from . import constants as const
from . import default_mesh

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

class ForwardModel:
    """
    Stores all the things related to the vocal fold forward model solved thru fenics.

    TODO: intialization is kinf of messy and ugly. prettify/clean
    TODO: Class contains alot of extra, non-essential stuff. Think about what are the essential
        things that are included.
    TODO: think about how to include extraneous forms for computing some functional etc...

    Parameters
    ----------
    mesh : str
        Path to a fenics mesh xml file.
    facet_marker, cell_marker : str
        Paths to fenics facet / cell xml marker files.
    id_facet_marker, id_cell_marker : dict of {str: int}
        A dictionary of named markers for facets and cells. `facet_marker` needs atleast two
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
    def __init__(self, mesh_path, facet_marker, cell_marker, id_facet_marker, id_cell_marker):

        ## Mesh parameters and attributes
        self.mesh = dfn.Mesh(mesh_path)

        _facet_marker = dfn.MeshFunction('size_t', self.mesh, facet_marker)
        body_marker = dfn.MeshFunction('size_t', self.mesh, cell_marker)

        # Create a vertex marker from the boundary marker
        edge_to_vertex = np.array([[vertex.index() for vertex in dfn.vertices(edge)]
                                   for edge in dfn.edges(self.mesh)])
        pressure_edges = _facet_marker.where_equal(id_facet_marker['pressure'])
        fixed_edges = _facet_marker.where_equal(id_facet_marker['fixed'])

        pressure_vertices = np.unique(edge_to_vertex[pressure_edges].reshape(-1))
        fixed_vertices = np.unique(edge_to_vertex[fixed_edges].reshape(-1))

        vertex_marker = dfn.MeshFunction('size_t', self.mesh, 0)
        vertex_marker.set_all(0)
        vertex_marker.array()[pressure_vertices] = id_facet_marker['pressure']
        vertex_marker.array()[fixed_vertices] = id_facet_marker['fixed']

        self.surface_vertices = np.array(vertex_marker.where_equal(id_facet_marker['pressure']))
        self.surface_coordinates = self.mesh.coordinates()[self.surface_vertices]

        # Sort the pressure surface vertices from inferior to superior
        idx_sort = np.argsort(self.surface_coordinates[..., 0])
        self.surface_vertices = self.surface_vertices[idx_sort]
        # surface_coordinates = surface_coordinates[idx_sort]

        ## Set Variational Forms
        # Newmark update parameters
        self.gamma = dfn.Constant(1/2)
        self.beta = dfn.Constant(1/4)

        self.collision_eps = 0.001
        self.y_midline = const.DEFAULT_FLUID_PROPERTIES['y_midline']

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

        raleigh_m = dfn.Constant(1e-4)
        raleigh_k = dfn.Constant(1e-4)

        damping = raleigh_m * self.rho*ufl.dot(trial_v, self.vector_test)*ufl.dx \
                  + raleigh_k * ufl.inner(linear_elasticity(trial_v, self.emod, self.nu),
                                          strain(self.vector_test))*ufl.dx

        # Compute the pressure loading neumann boundary condition thru Nanson's formula
        deformation_gradient = ufl.grad(self.u0) + ufl.Identity(2)
        deformation_cofactor = ufl.det(deformation_gradient) * ufl.inv(deformation_gradient).T

        ds = dfn.Measure('ds', domain=self.mesh, subdomain_data=_facet_marker)
        fluid_force = -self.pressure*deformation_cofactor*dfn.FacetNormal(self.mesh)

        traction = ufl.dot(fluid_force, self.vector_test)*ds(id_facet_marker['pressure'])

        fu = inertia + stiffness + damping - traction

        # Non linear equations during collision. Add a penalty to account for this
        collision_normal = dfn.Constant([0, 1])
        x_reference = dfn.Function(self.vector_function_space)
        x_reference.vector()[self.vert_to_vdof.reshape(-1)] = self.mesh.coordinates().reshape(-1)

        gap = ufl.dot(x_reference+self.u1, collision_normal) - (self.y_midline - self.collision_eps)
        positive_gap = (gap + abs(gap)) / 2

        k_collision = dfn.Constant(1e11)
        penalty = ufl.dot(k_collision*positive_gap**2*-1*collision_normal, self.vector_test) * ds(id_facet_marker['pressure'])

        self.fu_nonlin = ufl.action(fu, self.u1) - penalty
        self.jac_fu_nonlin = ufl.derivative(self.fu_nonlin, self.u1, self.vector_trial)

        ## Boundary conditions
        # Specify DirichletBC at the VF base
        self.bc_base = dfn.DirichletBC(self.vector_function_space, dfn.Constant([0, 0]), _facet_marker,
                                       id_facet_marker['fixed'])

        self.bc_base_adjoint = dfn.DirichletBC(self.vector_function_space, dfn.Constant([0, 0]), _facet_marker,
                                               id_facet_marker['fixed'])

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
        self.fluid_work = ufl.dot(fluid_force, self.u1-self.u0) * ds(id_facet_marker['pressure'])
        self.dfluid_work_du0 = ufl.derivative(self.fluid_work, self.u0, self.vector_test)
        self.dfluid_work_du1 = ufl.derivative(self.fluid_work, self.u1, self.vector_test)
        self.dfluid_work_dp = ufl.derivative(self.fluid_work, self.pressure, self.scalar_test)

    def set_pressure(self, fluid_props):
        """
        Updates pressure coefficient using a bernoulli flow model.

        Parameters
        ----------
        fluid_props : dict
            A dictionary of fluid properties for the 1d bernoulli fluids model
        """

        # Calculated the deformed pressure surface of the body
        surface_slice = self.vert_to_vdof[self.surface_vertices].reshape(-1)
        _u = self.u0.vector()[surface_slice].reshape(-1, 2)
        _v = self.v0.vector()[surface_slice].reshape(-1, 2)
        _a = self.a0.vector()[surface_slice].reshape(-1, 2)
        x_surface = (self.surface_coordinates + _u, _v, _a)

        # Update the pressure loading based on the deformed surface
        pressure, info = fluids.set_pressure_form(self, x_surface, fluid_props)

        self.pressure.assign(pressure)
        return info

    def set_flow_sensitivity(self, fluid_props):
        """
        Updates pressure sensitivity using a bernoulli flow model.

        Parameters
        ----------
        fluid_props : dict
            A dictionary of fluid properties
        """
        # Calculated the deformed pressure surface of the body
        surface_slice = self.vert_to_vdof[self.surface_vertices].reshape(-1)
        _u = self.u0.vector()[surface_slice].reshape(-1, 2)
        _v = self.v0.vector()[surface_slice].reshape(-1, 2)
        _a = self.a0.vector()[surface_slice].reshape(-1, 2)
        x_surface = (self.surface_coordinates + _u, _v, _a)

        # Calculate sensitivities of fluid quantities based on the deformed surface
        dp_du0, dq_du0 = fluids.set_flow_sensitivity(self, x_surface, fluid_props)

        return dp_du0, dq_du0
