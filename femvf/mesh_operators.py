"""
Operators on meshed
"""

import dolfin as dfn
import ufl
import numpy as np
from petsc4py import PETSc

def grad_fem_p1(scalar_trial, scalar_test):
    """
    Return the gradient norm operator

    This return the product :math: A = L^T L :math:, such that a smoothness penalty term can be
    constructed with :math: f^T L^T L f = f^T A f :math:

    Parameters
    ----------
    trial : dfn.TrialFunction
    test : dfn.TestFunction

    Returns
    -------
    PETSc.Matrix
        Operator
    """

    op_form = ufl.dot(ufl.grad(scalar_trial), ufl.grad(scalar_test)) * dfn.dx
    return dfn.assemble(op_form)
