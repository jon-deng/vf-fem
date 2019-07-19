"""
Linear Algebra routines for dolfin
"""

import dolfin as dfn

import petsc4py
petsc4py.init()
from petsc4py import PETSc

def transpose(mat):
    """
    Tranposes a dolfin matrix.

    Parameters
    ----------
    mat : dfn.Matrix
        The matrix

    Returns
    -------
    dfn.Matrix
        The transpose of the matrix
    """
    mat_petsc = dfn.as_backend_type(mat).mat()
    mat_petsc.transpose()

    return mat
    #return dfn.Matrix(dfn.PETScMatrix(mat_petsc))

# def transpose(mat):
#     """
#     Tranposes a dolfin matrix.
#
#     Parameters
#     ----------
#     mat : dfn.Matrix
#         The matrix
#
#     Returns
#     -------
#     dfn.Matrix
#         The transpose of the matrix
#     """
#     mat_petsc_t = PETSc.Mat()
#     mat_petsc_t.create(PETSc.COMM_SELF)
#     mat_petsc_t.setSizes([mat.size(1), mat.size(0)])
#     mat_petsc_t.setUp()
#     mat_petsc_t.setType('aij')
#
#     mat_petsc = dfn.as_backend_type(mat).mat()
#     mat_petsc.transpose(mat_petsc_t)
#
#     mat_petsc_t.assemble()
#     return dfn.PETScMatrix(mat_petsc_t)

# def petsc_to_generic_fenics():
#     """
#     Converts a petsc4py matrix/vector objects to fenics equivalents.
#     """
#     if isinstance(PETSc.Matrix):
#     elif isinstance(PETSc.Vec):
#     else:
#         raise ValueError("You dun goofed")
