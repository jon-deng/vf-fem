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
    for vertex in dfn.vertices(mesh):
        a_op = np.zeros((mesh.num_ed, 2))

        edges = [edge for edge in dfn.edges(vertex)]

        vertex_neighbors = [v for v in dfn.vertices(edge) if v.index() != vertex.index()
                            for edge in edges]

