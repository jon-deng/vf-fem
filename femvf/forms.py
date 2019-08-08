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

####################################################################################################
## Mesh definition

# loading the mesh
# mesh, boundary_marker, domainid_pressure, domainid_fixed = default_mesh.givememesh()

mesh_dir = os.path.expanduser('~/GraduateSchool/Projects/FEMVFOptimization/meshes/')
mesh_base_filename = 'geometry2'
path_to_mesh = os.path.join(mesh_dir, mesh_base_filename + '.xml')
path_to_mesh_function = os.path.join(mesh_dir, mesh_base_filename + '_facet_region.xml')

mesh = dfn.Mesh(path_to_mesh)
boundary_marker = dfn.MeshFunction('size_t', mesh, path_to_mesh_function)

domainid_pressure = 1
domainid_fixed = 3

# Create a vertex marker from the boundary marker
edge_to_vertex = np.array([[vertex.index() for vertex in dfn.vertices(edge)]
                            for edge in dfn.edges(mesh)])
pressure_edges = boundary_marker.where_equal(domainid_pressure)
fixed_edges = boundary_marker.where_equal(domainid_fixed)

pressure_vertices = np.unique(edge_to_vertex[pressure_edges].reshape(-1))
fixed_vertices = np.unique(edge_to_vertex[fixed_edges].reshape(-1))

vertex_marker = dfn.MeshFunction('size_t', mesh, 0)
vertex_marker.set_all(0)
vertex_marker.array()[pressure_vertices] = domainid_pressure
vertex_marker.array()[fixed_vertices] = domainid_fixed

### End mesh stuff
surface_vertices = np.array(vertex_marker.where_equal(domainid_pressure))
surface_coordinates = mesh.coordinates()[surface_vertices]

# Sort the pressure surface vertices from inferior to superior
idx_sort = np.argsort(surface_coordinates[..., 0])
surface_vertices = surface_vertices[idx_sort]
surface_coordinates = surface_coordinates[idx_sort]

####################################################################################################
## Geometric parameters

thickness_bottom = np.amax(mesh.coordinates()[:, 0])

####################################################################################################
## Governing equation variational forms

collision_eps = 0.001
y_midline = const.DEFAULT_FLUID_PROPERTIES['y_midline']

scalar_function_space = dfn.FunctionSpace(mesh, 'CG', 1)
vector_function_space = dfn.VectorFunctionSpace(mesh, 'CG', 1)
trial_u = dfn.TrialFunction(vector_function_space)
test = dfn.TestFunction(vector_function_space)

scalar_trial = dfn.TrialFunction(scalar_function_space)
scalar_test = dfn.TestFunction(scalar_function_space)

vert_to_vdof = dfn.vertex_to_dof_map(vector_function_space).reshape(-1, 2)
vert_to_sdof = dfn.vertex_to_dof_map(scalar_function_space)

# Newmark updates
gamma = dfn.Constant(1/2)
beta = dfn.Constant(1/4)
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

# Solid material properties
rho = dfn.Constant(1)
nu = dfn.Constant(0.48)
emod = dfn.Function(scalar_function_space)
emod.vector()[:] = 11.8e3 * const.PASCAL_TO_CGS

# Stress and strain functions
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

# Initial conditions
u0 = dfn.Function(vector_function_space)
v0 = dfn.Function(vector_function_space)
a0 = dfn.Function(vector_function_space)

# Next time step displacement
u1 = dfn.Function(vector_function_space)

# Time step
dt = dfn.Constant(1e-4)

# Pressure forcing
pressure = dfn.Function(scalar_function_space)

# Define the variational forms
trial_v = newmark_v(trial_u, u0, v0, a0, dt, gamma=gamma, beta=beta)
trial_a = newmark_a(trial_u, u0, v0, a0, dt, gamma=gamma, beta=beta)

inertia = rho*ufl.dot(trial_a, test)*ufl.dx
stiffness = ufl.inner(linear_elasticity(trial_u, emod, nu), strain(test))*ufl.dx

raleigh_m = dfn.Constant(1e-4)
raleigh_k = dfn.Constant(1e-4)

damping = raleigh_m * rho*ufl.dot(trial_v, test)*ufl.dx \
          + raleigh_k * ufl.inner(linear_elasticity(trial_v, emod, nu), strain(test))*ufl.dx

# Compute the pressure loading neumann boundary condition thru Nanson's formula
deformation_gradient = ufl.grad(u0) + ufl.Identity(2)
deformation_cofactor = ufl.det(deformation_gradient) * ufl.inv(deformation_gradient).T

ds = dfn.Measure('ds', domain=mesh, subdomain_data=boundary_marker)
fluid_force = -pressure*deformation_cofactor*dfn.FacetNormal(mesh)

traction = ufl.dot(fluid_force, test)*ds(domainid_pressure)

fu = inertia + stiffness + damping - traction

lhs = dfn.lhs(fu)
rhs = dfn.rhs(fu)

# Non linear equations during collision. Add a penalty to account for this
collision_normal = dfn.Constant([0, 1])
x_reference = dfn.Function(vector_function_space)
x_reference.vector()[vert_to_vdof.reshape(-1)] = mesh.coordinates().reshape(-1)
# gap = (x_reference.sub(1) + u1.sub(1))# - (y_midline-collision_eps)
# gap = u1.sub(1)
gap = ufl.dot(x_reference+u1, collision_normal) - (y_midline - collision_eps)
positive_gap = (gap + abs(gap)) / 2

k_collision = dfn.Constant(1e11)
penalty = ufl.dot(k_collision*positive_gap**2*-1*collision_normal, test) * ds(domainid_pressure)

fu_nonlin = ufl.action(fu, u1) - penalty
jac_fu_nonlin = ufl.derivative(fu_nonlin, u1, trial_u)

####################################################################################################
# Boundary conditions

# Specify DirichletBC at the VF base
bc_base = dfn.DirichletBC(vector_function_space, dfn.Constant([0, 0]), boundary_marker,
                          domainid_fixed)

bc_base_adjoint = dfn.DirichletBC(vector_function_space, dfn.Constant([0, 0]), boundary_marker,
                                  domainid_fixed)

# Define some additional forms for diagnostics
force_form = ufl.inner(linear_elasticity(u0, emod, nu), strain(test))*ufl.dx - traction

####################################################################################################
## Forms needed for adjoint computation
# Note: For an externally calculated pressure, you have to correct the derivative based on the
# sensitivity of the pressure loading in f1 to either u0 or u1 (or both depending if it's strongly
# coupled).
f1 = fu_nonlin
df1_du0_adjoint = dfn.adjoint(ufl.derivative(f1, u0, trial_u))
df1_da0_adjoint = dfn.adjoint(ufl.derivative(f1, a0, trial_u))
df1_dv0_adjoint = dfn.adjoint(ufl.derivative(f1, v0, trial_u))
df1_dp_adjoint = dfn.adjoint(ufl.derivative(f1, pressure, scalar_trial))

df1_du1_adjoint = dfn.adjoint(ufl.derivative(f1, u1, trial_u))

# Work done by pressure from u0 to u1
fluid_work = ufl.dot(fluid_force, u1-u0) * ds(domainid_pressure)

### Fluid update functions
def set_pressure(fluid_props):
    """
    Updates pressure coefficient using a bernoulli flow model.

    Parameters
    ----------
    fluid_props : dict
        A dictionary of fluid properties for the 1d bernoulli fluids model
    """
    _u = u0.vector()[vert_to_vdof[surface_vertices].reshape(-1)].reshape(-1, 2)
    _v = v0.vector()[vert_to_vdof[surface_vertices].reshape(-1)].reshape(-1, 2)
    _a = a0.vector()[vert_to_vdof[surface_vertices].reshape(-1)].reshape(-1, 2)
    x_surface = (surface_coordinates + _u, _v, _a)
    info = fluids.set_pressure_form(pressure, x_surface, surface_vertices, vert_to_sdof,
                                    fluid_props)
    return info

def set_flow_sensitivity(fluid_props):
    """
    Updates pressure sensitivity using a bernoulli flow model.

    Parameters
    ----------
    fluid_props : dict
        A dictionary of fluid properties
    """
    delta_xy = u0.vector()[vert_to_vdof[surface_vertices].reshape(-1)].reshape(-1, 2)
    xy = surface_coordinates + delta_xy
    dp_du0, dq_du0 = fluids.set_flow_sensitivity(xy, surface_vertices, vert_to_sdof, vert_to_vdof,
                                                 fluid_props)

    return dp_du0, dq_du0
