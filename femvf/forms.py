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
from .properties import FluidProperties, LinearElasticRayleigh

# from .operators import LinCombOfMats

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
    Returns the Cauchy stress for a small-strain displacement field

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
        
def linear_elastic_rayleigh(mesh, facet_function, facet_labels, cell_function, cell_labels):
    """
    Return a dictionary of variational forms and other parameters.
    """
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
    y_collision = dfn.Constant(1.0)
    k_collision = dfn.Constant(1.0)
    rho = dfn.Constant(1.0)
    nu = dfn.Constant(1.0)
    rayleigh_m = dfn.Constant(1.0)
    rayleigh_k = dfn.Constant(1.0)
    emod = dfn.Function(scalar_fspace)

    emod.vector()[:] = 1.0

    # Initial and final states
    # u: displacement, v: velocity, a: acceleration
    dt = dfn.Constant(1e-4)

    u0 = dfn.Function(vector_fspace)
    v0 = dfn.Function(vector_fspace)
    a0 = dfn.Function(vector_fspace)

    u1 = dfn.Function(vector_fspace)
    v1 = newmark_v(vector_trial, u0, v0, a0, dt, gamma, beta)
    a1 = newmark_a(vector_trial, u0, v0, a0, dt, gamma, beta)

    # Surface pressures
    pressure = dfn.Function(scalar_fspace)

    # Symbolic calculations to get the variational form for a linear-elastic solid

    inertia = biform_m(a1, vector_test, rho)

    stiffness = biform_k(vector_trial, vector_test, emod, nu)

    ra_damping = rayleigh_m * biform_m(v1, vector_test, rho) \
                 + rayleigh_k * biform_k(v1, vector_test, emod, nu)

    # Compute the pressure loading Neumann boundary condition on the reference configuration
    # using Nanson's formula. This is because the 'total lagrangian' formulation is used.
    deformation_gradient = ufl.grad(u0) + ufl.Identity(2)
    deformation_cofactor = ufl.det(deformation_gradient) * ufl.inv(deformation_gradient).T

    ds = dfn.Measure('ds', domain=mesh, subdomain_data=facet_function)
    fluid_force = -pressure*deformation_cofactor*dfn.FacetNormal(mesh)

    traction = ufl.dot(fluid_force, vector_test)*ds(facet_labels['pressure'])

    # Use the penalty method to account for collision
    collision_normal = dfn.Constant([0.0, 1.0])
    x_reference = dfn.Function(vector_fspace)

    vert_to_vdof = dfn.vertex_to_dof_map(vector_fspace)
    x_reference.vector()[vert_to_vdof.reshape(-1)] = mesh.coordinates().reshape(-1)

    gap = ufl.dot(x_reference+u1, collision_normal) - y_collision
    positive_gap = (gap + abs(gap)) / 2

    # Uncomment/comment the below lines to choose between exponential or quadratic penalty springs
    penalty = ufl.dot(k_collision*positive_gap**2*-1*collision_normal, vector_test) \
              * ds(facet_labels['pressure'])

    f1_linear = ufl.action(inertia + stiffness + ra_damping, u1)
    f1_nonlin = -traction - penalty
    f1 = f1_linear + f1_nonlin

    df1_du1_linear = ufl.derivative(f1_linear, u1, vector_trial)
    df1_du1_nonlin = ufl.derivative(f1_nonlin, u1, vector_trial)
    df1_du1 = df1_du1_linear + df1_du1_nonlin

    ## Boundary conditions
    # Specify DirichletBC at the VF base
    bc_base = dfn.DirichletBC(vector_fspace, dfn.Constant([0.0, 0.0]),
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

    df1_du1_adj_linear = dfn.adjoint(df1_du1_linear)
    df1_du1_adj_nonlin = dfn.adjoint(df1_du1_nonlin)
    df1_du1_adj = df1_du1_adj_linear + df1_du1_adj_nonlin

    df1_demod = ufl.derivative(f1, emod, scalar_trial)
    df1_dpressure_adj = dfn.adjoint(ufl.derivative(f1, pressure, scalar_trial))

    forms = {
        'bcs.base': bc_base,

        'fspace.vector': vector_fspace,
        'fspace.scalar': scalar_fspace,

        'test.vector': vector_test,
        'test.scalar': scalar_test,
        'trial.vector': vector_trial,
        'trial.scalar': scalar_trial,

        'coeff.arg.u1': u1,

        'coeff.time.dt': dt,
        'coeff.time.gamma': gamma,
        'coeff.time.beta': beta,
        
        'coeff.state.u0': u0,
        'coeff.state.v0': v0,
        'coeff.state.a0': a0,

        'coeff.fsi.pressure': pressure,

        'coeff.prop.rho': rho,
        'coeff.prop.nu': nu,
        'coeff.prop.emod': emod,
        'coeff.prop.rayleigh_m': rayleigh_m,
        'coeff.prop.rayleigh_k': rayleigh_k,
        'coeff.prop.y_collision': y_collision,
        'coeff.prop.k_collision': k_collision,

        'form.un.f1': f1,

        'form.bi.df1_du1': df1_du1,
        'form.bi.df1_du1_adj': df1_du1_adj,
        'form.bi.df1_du0_adj': df1_du0_adj,
        'form.bi.df1_dv0_adj': df1_dv0_adj,
        'form.bi.df1_da0_adj': df1_da0_adj,
        'form.bi.df1_dpressure_adj': df1_dpressure_adj,
        'form.bi.df1_demod': df1_demod}
    return forms

def kelvin_voigt(mesh, facet_function, facet_labels, cell_function, cell_labels):
    """
    Return a dictionary of variational forms for a kelvin-voigt model and
    """
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
    y_collision = dfn.Constant(1.0)
    k_collision = dfn.Constant(1.0)
    rho = dfn.Constant(1.0)
    nu = dfn.Constant(1.0)
    rayleigh_m = dfn.Constant(1.0)
    rayleigh_k = dfn.Constant(1.0)
    emod = dfn.Function(scalar_fspace)

    emod.vector()[:] = 1.0

    # Initial and final states
    # u: displacement, v: velocity, a: acceleration
    dt = dfn.Constant(1e-4)

    u0 = dfn.Function(vector_fspace)
    v0 = dfn.Function(vector_fspace)
    a0 = dfn.Function(vector_fspace)

    u1 = dfn.Function(vector_fspace)
    v1 = newmark_v(vector_trial, u0, v0, a0, dt, gamma, beta)
    a1 = newmark_a(vector_trial, u0, v0, a0, dt, gamma, beta)

    # Surface pressures
    pressure = dfn.Function(scalar_fspace)

    # Symbolic calculations to get the variational form for a linear-elastic solid

    inertia = biform_m(a1, vector_test, rho)

    stiffness = biform_k(vector_trial, vector_test, emod, nu)

    # Kelvin-Voigt type damping term
    kv_eta = dfn.Function(scalar_fspace)
    kv_damping = ufl.inner(kv_eta*strain(v1), strain(vector_test)) * ufl.dx

    # Compute the pressure loading Neumann boundary condition on the reference configuration
    # using Nanson's formula. This is because the 'total lagrangian' formulation is used.
    deformation_gradient = ufl.grad(u0) + ufl.Identity(2)
    deformation_cofactor = ufl.det(deformation_gradient) * ufl.inv(deformation_gradient).T

    ds = dfn.Measure('ds', domain=mesh, subdomain_data=facet_function)
    fluid_force = -pressure*deformation_cofactor*dfn.FacetNormal(mesh)

    traction = ufl.dot(fluid_force, vector_test)*ds(facet_labels['pressure'])

    # Use the penalty method to account for collision
    collision_normal = dfn.Constant([0.0, 1.0])
    x_reference = dfn.Function(vector_fspace)

    vert_to_vdof = dfn.vertex_to_dof_map(vector_fspace)
    x_reference.vector()[vert_to_vdof.reshape(-1)] = mesh.coordinates().reshape(-1)

    gap = ufl.dot(x_reference+u1, collision_normal) - y_collision
    positive_gap = (gap + abs(gap)) / 2

    # Uncomment/comment the below lines to choose between exponential or quadratic penalty springs
    penalty = ufl.dot(k_collision*positive_gap**2*-1*collision_normal, vector_test) \
              * ds(facet_labels['pressure'])

    f1_linear = ufl.action(inertia + stiffness + kv_damping, u1)
    f1_nonlin = -traction - penalty
    f1 = f1_linear + f1_nonlin

    df1_du1_linear = ufl.derivative(f1_linear, u1, vector_trial)
    df1_du1_nonlin = ufl.derivative(f1_nonlin, u1, vector_trial)
    df1_du1 = df1_du1_linear + df1_du1_nonlin

    ## Boundary conditions
    # Specify DirichletBC at the VF base
    bc_base = dfn.DirichletBC(vector_fspace, dfn.Constant([0.0, 0.0]),
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

    df1_du1_adj_linear = dfn.adjoint(df1_du1_linear)
    df1_du1_adj_nonlin = dfn.adjoint(df1_du1_nonlin)
    df1_du1_adj = df1_du1_adj_linear + df1_du1_adj_nonlin

    df1_demod = ufl.derivative(f1, emod, scalar_trial)
    df1_dpressure_adj = dfn.adjoint(ufl.derivative(f1, pressure, scalar_trial))

    forms = {
        'bcs.base': bc_base,

        'fspace.vector': vector_fspace,
        'fspace.scalar': scalar_fspace,

        'test.vector': vector_test,
        'test.scalar': scalar_test,
        'trial.vector': vector_trial,
        'trial.scalar': scalar_trial,

        'coeff.arg.u1': u1,
        'coeff.time.dt': dt,
        'coeff.time.gamma': gamma,
        'coeff.time.beta': beta,

        'coeff.state.u0': u0,
        'coeff.state.v0': v0,
        'coeff.state.a0': a0,

        'coeff.fsi.pressure': pressure,

        'coeff.prop.rho': rho,
        'coeff.prop.eta': kv_eta,
        'coeff.prop.emod': emod,
        'coeff.prop.nu': nu,
        'coeff.prop.y_collision': y_collision,
        'coeff.prop.k_collision': k_collision,

        'form.un.f1': f1,
        'form.bi.df1_du1': df1_du1,
        'form.bi.df1_du1_adj': df1_du1_adj,
        'form.bi.df1_du0_adj': df1_du0_adj,
        'form.bi.df1_dv0_adj': df1_dv0_adj,
        'form.bi.df1_da0_adj': df1_da0_adj,
        'form.bi.df1_dpressure_adj': df1_dpressure_adj,
        'form.bi.df1_demod': df1_demod}
    return forms
