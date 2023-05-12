"""
Continuum operations implemented in UFL
"""

import ufl
import dolfin as dfn

def stress_isotropic(strain, emod, nu):
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
    return 2*lame_mu*strain + lame_lambda*ufl.tr(strain)*ufl.Identity(strain.ufl_shape[0])

def def_grad(u):
    """
    Returns the deformation gradient

    Parameters
    ----------
    u : dfn.TrialFunction, ufl.Argument
        Trial displacement field
    """
    spp = ufl.grad(u)
    if u.geometric_dimension() == 2:
        return ufl.as_tensor(
            [[spp[0, 0], spp[0, 1], 0],
            [spp[1, 0], spp[1, 1], 0],
            [        0,         0, 0]]
        ) + ufl.Identity(3)
    else:
        return spp + ufl.Identity(3)

def def_cauchy_green(u):
    """
    Returns the right cauchy-green deformation tensor

    Parameters
    ----------
    u : dfn.TrialFunction, ufl.Argument
        Trial displacement field
    """
    def_grad = def_grad(u)
    return def_grad.T*def_grad

def strain_green_lagrange(u):
    """
    Returns the strain tensor

    Parameters
    ----------
    u : dfn.TrialFunction, ufl.Argument
        Trial displacement field
    """
    C = def_cauchy_green(u)
    return 1/2*(C - ufl.Identity(3))

def strain_inf(u):
    """
    Returns the strain tensor

    Parameters
    ----------
    u : dfn.TrialFunction, ufl.Argument
        Trial displacement field
    """
    spp = 1/2 * (ufl.grad(u) + ufl.grad(u).T)
    if u.geometric_dimension() == 2:
        return ufl.as_tensor(
            [[spp[0, 0], spp[0, 1], 0],
            [spp[1, 0], spp[1, 1], 0],
            [        0,         0, 0]]
        )
    else:
        return spp

def strain_lin_green_lagrange(u, du):
    """
    Returns the linearized Green-Lagrange strain tensor

    Parameters
    ----------
    u : dfn.TrialFunction, ufl.Argument
        Displacement to linearize about
    du : dfn.TrialFunction, ufl.Argument
        Trial displacement field
    """
    E = strain_green_lagrange(u)
    return ufl.derivative(E, u, du)

def strain_lin2_green_lagrange(u0, u):
    """
    Returns the double linearized Green-Lagrange strain tensor

    Parameters
    ----------
    u0 : dfn.TrialFunction, ufl.Argument
        Displacement to linearize about
    u : dfn.TrialFunction, ufl.Argument
        Trial displacement field
    """
    spp = 1/2*(ufl.grad(u).T*ufl.grad(u0) + ufl.grad(u0).T*ufl.grad(u))
    if u0.geometric_dimension() == 2:
        return ufl.as_tensor(
            [[spp[0, 0], spp[0, 1], 0],
            [spp[1, 0], spp[1, 1], 0],
            [        0,         0, 0]]
        )
    else:
        return spp

def pressure_contact_penalty(xref, u, k, ycoll, n=dfn.Constant([0.0, 1.0])):
    """
    Return the contact pressure expression according to the penalty method

    Parameters
    ----------
    xref : dfn.Function
        Reference configuration coordinates
    u : dfn.Function
        Displacement
    k : float or dfn.Constant
        Penalty contact spring value
    d : float or dfn.Constant
        y location of the contact plane
    n : dfn.Constant
        Contact plane normal, facing away from the vocal folds
    """
    gap = ufl.dot(xref+u, n) - ycoll
    positive_gap = (gap + abs(gap)) / 2

    # Uncomment/comment the below lines to choose between exponential or quadratic penalty springs
    return -k*positive_gap**3

def pressure_contact_quad_penalty(gap, kcoll):
    positive_gap = (gap + abs(gap)) / 2
    return kcoll*positive_gap**2

def traction_pressure(p, u, n):
    """

    Parameters
    ----------
    p : Pressure load
    u : displacement
    n : facet outer normal
    """
    deformation_gradient = ufl.grad(u) + ufl.Identity(2)
    deformation_cofactor = ufl.det(deformation_gradient) * ufl.inv(deformation_gradient).T

    return -p*deformation_cofactor*n

def pullback_area_normal(u, n):
    """

    Parameters
    ----------
    p : Pressure load
    u : displacement
    n : facet outer normal
    """
    deformation_gradient = ufl.grad(u) + ufl.Identity(2)
    deformation_cofactor = ufl.det(deformation_gradient) * ufl.inv(deformation_gradient).T

    return deformation_cofactor*n
