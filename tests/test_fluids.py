"""
Tests fluids.py module
"""

import sys
import dolfin as dfn

import autograd
import autograd.numpy as np

import petsc4py
petsc4py.init()

sys.path.append('../')
from femvf import fluids
from femvf import transforms
from femvf import constants

## Mesh generation
mesh = dfn.RectangleMesh(dfn.Point(-0.5, -0.5), dfn.Point(0.5, 0.5), 10, 30)

# Mark the mesh boundaries with dfn.MeshFunction
vertex_marker = dfn.MeshFunction('size_t', mesh, 0)

class OmegaPressure(dfn.SubDomain):
    """Marks the pressure boundary"""
    def inside(self, x, on_boundary):
        """Marks the appropriate nodes"""
        return (np.abs(x[0] - -0.5) <= dfn.DOLFIN_EPS
                or np.abs(x[1] - 0.5) <= dfn.DOLFIN_EPS
                or np.abs(x[0] - 0.5) <= dfn.DOLFIN_EPS)
domainid = 1
OmegaPressure().mark(vertex_marker, domainid)

## Deform the mesh to a crappy vf-like shape
depth = 0.6
thickness_bottom = 0.6
thickness_top = 1/4 * thickness_bottom
x_inferior_edge = 0.5 * thickness_bottom
x_superior_edge = x_inferior_edge + thickness_top
x1 = [0, 0]
x2 = [thickness_bottom, 0]
x3 = [x_superior_edge, 0.55]
x4 = [x_inferior_edge, 0.5]

mesh.coordinates()[...] = transforms.bilinear(mesh.coordinates(), x1, x2, x3, x4)

surface_vertices = np.array(vertex_marker.where_equal(domainid))
surface_coordinates = mesh.coordinates()[surface_vertices]
idx_sort = np.argsort(surface_coordinates[..., 0])
surface_vertices = surface_vertices[idx_sort]
surface_coordinates = surface_coordinates[idx_sort]

def test_pressure_sensitivity(fluid_props):
    # Calculate pressure sensitivity with finite differences
    print("Pressure jacobian via FD")
    dp_du_fd = np.zeros((surface_coordinates.shape[0], surface_coordinates.shape[0]*2))
    dy = np.zeros(surface_coordinates.shape)
    DY = 0.0001
    for ii in range(surface_coordinates.shape[0]):
        dy[...] = 0
        dy[ii, 1] += DY
        pressure, *_ = fluids.fluid_pressure(surface_coordinates+dy, fluid_props)

        dp_du_fd[:, 2*ii+1] = pressure

    pressure, *_ = fluids.fluid_pressure(surface_coordinates, fluid_props)
    dp_du_fd[:, 1::2] -= pressure[..., None]

    dp_du_fd /= DY

    # Calculate pressure sensitivity with auto-differentiation
    print("\nPressure jacobian via AD")
    def fluid_pressure(x):
        return fluids.fluid_pressure(x.reshape(-1, 2), fluid_props)[0]

    dp_du_ad = autograd.jacobian(fluid_pressure, 0)(surface_coordinates.reshape(-1))

    # Calculate pressure sensitivity with the analytical derivation
    print("\nPressure jacobian via AN")
    dp_du_an = fluids.flow_sensitivity(surface_coordinates, fluid_props)[0]

    print("\nComparison")
    print(dp_du_an[:, 1::2].diagonal())
    print(dp_du_ad[:, 1::2].diagonal())

    close = np.isclose(dp_du_an, dp_du_ad)
    print(close)
    print(np.sum(~close))

    #assert np.allclose(dp_du_an, dp_du_ad)

    #ipdb.set_trace()

if __name__ == '__main__':
    fluid_props = constants.DEFAULT_FLUID_PROPERTIES
    test_pressure_sensitivity(fluid_props)
