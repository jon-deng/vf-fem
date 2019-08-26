"""
Generates a crappy trapeziodal mesh just for playing around.
"""

import numpy as np

import dolfin as dfn

from . import transforms
from . import constants as const

def givemethemesh():
    """
    Returns the mesh and a bunch of other stuff.

    Parameters
    ----------

    Returns
    -------
    dfn.Mesh
        The mesh
    dfn.MeshFunction, dfn.MeshFunction
        Mesh functions marking facets and vertices respectively
    int, int
        Integers marking pressure and fixed surfaces respectively
    """
    ## Mesh generation
    mesh = dfn.RectangleMesh(dfn.Point(-0.5, -0.5), dfn.Point(0.5, 0.5), 10, 25)

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

    # # Set a function to mark collision nodes
    # class OmegaContact(dfn.SubDomain):
    #     """Marks the contact boundary"""
    #     def inside(self, x, on_boundary):
    #         """Marks the appropriate nodes"""
    #         return on_boundary and x[1] >= y_midline-collision_eps
    # omega_contact = OmegaContact()

    domainid_pressure = 1
    domainid_fixed = 2

    omega_pressure.mark(boundary_marker, domainid_pressure)
    omega_pressure.mark(vertex_marker, domainid_pressure)

    omega_fixed.mark(boundary_marker, domainid_fixed)
    omega_fixed.mark(vertex_marker, domainid_fixed)

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

    return mesh, boundary_marker, domainid_pressure, domainid_fixed
