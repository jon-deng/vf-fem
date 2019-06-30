"""
Module for definitions of weak forms.

Units are in CGS
"""

import numpy as np

import dolfin as dfn
import ufl

import transforms
import fluids
import constants as const

# dfn.parameters['form_compiler']['optimize'] = True
# dfn.parameters['form_compiler']['cpp_optimize'] = True

## Mesh generation
mesh = dfn.RectangleMesh(dfn.Point(-0.5, -0.5), dfn.Point(0.5, 0.5), 5, 15)

# Mark the mesh boundaries with dfn.MeshFunction
# Use the MeshFunction class to mark portions of the mesh with integer values.
# mark left, top, and right lines of the rectangle as one and the bottom as another.
boundary_marker = dfn.MeshFunction('size_t', mesh, 1)
vertex_marker = dfn.MeshFunction('size_t', mesh, 0)

# You need to subclass 'SubDomain' to 'mark' portions of the mesh in 'boundary_marker'
class OmegaFixed(dfn.SubDomain):
    """Marks the fixed boundary"""
    def inside(self, x, on_boundary):
        """Marks the appropriate nodes"""
        return np.abs(x[1] - -0.5) <= dfn.DOLFIN_EPS
omega_fixed = OmegaFixed()

class OmegaPressure(dfn.SubDomain):
    """Marks the pressure boundary"""
    def inside(self, x, on_boundary):
        """Marks the appropriate nodes"""
        return (np.abs(x[0] - -0.5) <= dfn.DOLFIN_EPS
                or np.abs(x[1] - 0.5) <= dfn.DOLFIN_EPS
                or np.abs(x[0] - 0.5) <= dfn.DOLFIN_EPS)
omega_pressure = OmegaPressure()

# Set a function to mark collision nodes
collision_eps = 0.005
y_midline = const.DEFAULT_FLUID_PROPERTIES['y_midline']
class OmegaContact(dfn.SubDomain):
    """Marks the contact boundary"""
    def inside(self, x, on_boundary):
        """Marks the appropriate nodes"""
        return on_boundary and x[1] >= y_midline-collision_eps
omega_contact = OmegaContact()

domainid_pressure = 1
domainid_fixed = 2
domainid_contact = 3

omega_pressure.mark(boundary_marker, domainid_pressure)
omega_pressure.mark(vertex_marker, domainid_pressure)
surface_vertices = np.array(vertex_marker.where_equal(domainid_pressure))

omega_fixed.mark(boundary_marker, domainid_fixed)
omega_fixed.mark(vertex_marker, domainid_fixed)
fixed_vertices = np.array(vertex_marker.where_equal(domainid_fixed))

## Deform the mesh to a VF-like shape
depth = 0.6
thickness_bottom = 1.0
thickness_top = 1/4 * thickness_bottom
x_inf_edge = 1/2 * thickness_bottom
x_sup_edge = x_inf_edge + thickness_top

x1 = [0, 0]
x2 = [thickness_bottom, 0]
x3 = [x_sup_edge, depth]
x4 = [x_inf_edge, depth-0.002]
x4 = [x_inf_edge, depth]

mesh.coordinates()[...] = transforms.bilinear(mesh.coordinates(), x1, x2, x3, x4)

# Sort the pressure surface vertices from inferior to superior
surface_coordinates = mesh.coordinates()[surface_vertices]

idx_sort = np.argsort(surface_coordinates[..., 0])
surface_vertices = surface_vertices[idx_sort]
surface_coordinates = surface_coordinates[idx_sort]

####################################################################################################
## Governing equation variational forms
scalar_function_space = dfn.FunctionSpace(mesh, 'CG', 1)
vector_function_space = dfn.VectorFunctionSpace(mesh, 'CG', 1)
trial = dfn.TrialFunction(vector_function_space)
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
trial_v = newmark_v(trial, u0, v0, a0, dt, gamma=gamma, beta=beta)
trial_a = newmark_a(trial, u0, v0, a0, dt, gamma=gamma, beta=beta)

inertia = rho*ufl.dot(trial_a, test)*ufl.dx
stiffness = ufl.inner(linear_elasticity(trial, emod, nu), strain(test))*ufl.dx

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
gap = ufl.dot(x_reference+u1, collision_normal) - (y_midline-collision_eps)
positive_gap = (gap + abs(gap)) / 2

k_collision = dfn.Constant(1e10)
penalty = ufl.dot(k_collision*positive_gap**2*-1*collision_normal, test) * ds(domainid_pressure)

fu_nonlin = ufl.action(fu, u1) - penalty
jac_fu_nonlin = ufl.derivative(fu_nonlin, u1, trial)

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
# sensitivity of the pressure loading in f1 to either u0 or u1 (or both).
f1 = fu_nonlin
df1_du0_adjoint = dfn.adjoint(ufl.derivative(f1, u0, trial))
df1_da0_adjoint = dfn.adjoint(ufl.derivative(f1, a0, trial))
df1_dv0_adjoint = dfn.adjoint(ufl.derivative(f1, v0, trial))
df1_dp_adjoint = dfn.adjoint(ufl.derivative(f1, pressure, scalar_trial))

df1_du1_adjoint = dfn.adjoint(ufl.derivative(f1, u1, trial))

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
    delta_xy = u0.vector()[vert_to_vdof[surface_vertices].reshape(-1)].reshape(-1, 2)
    xy_surface = surface_coordinates + delta_xy
    info = fluids.set_pressure_form(pressure, xy_surface, surface_vertices, vert_to_sdof,
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
