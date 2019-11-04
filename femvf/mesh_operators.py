"""
Operators on meshed
"""

import dolfin as dfn
from petsc4py import PETSc

def grad_lsq(mesh, vert_to_dof):
    """
    Return the gradient operator for a scalar function in dof order

    Parameters
    ----------
    mesh : dfn.mesh
    vert_to_dof : array_like
        Mapping from vertices to degrees of freedom

    Returns
    -------
    PETSc.Matrix
        Operator
    """

    # Assemble the matrix row-by-row
    for vertex in mesh.vertices():
        for edge in mesh.edges(vertex):

