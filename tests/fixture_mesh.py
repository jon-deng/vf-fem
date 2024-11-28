"""

"""

import pytest
from numpy.typing import NDArray

import numpy as np
import dolfin as dfn
import gmsh

gmsh.initialize()

# from femvf.vis.xdmfutils import write_xdmf, export_mesh_values

class FenicsMeshFixtures:

    NXS = [2]
    @pytest.fixture(params=NXS)
    def nx(self, request):
        return request.param

    NYS = [2]
    @pytest.fixture(params=NYS)
    def ny(self, request):
        return request.param

    NZS = [None, 3]
    @pytest.fixture(params=NZS)
    def nz(self, request):
        return request.param

    @pytest.fixture()
    def mesh(self, nx: int, ny: int, nz: int):
        if nz is None:
            return dfn.UnitSquareMesh(nx, ny)
        else:
            return dfn.UnitCubeMesh(nx, ny, nz)

    @pytest.fixture()
    def mesh_name(self, mesh: dfn.mesh):
        return mesh.__class__.__name__

    @pytest.fixture()
    def extrude_zs(self, nz: int):
        return np.linspace(0, 1, nz)

    @pytest.fixture()
    def mesh_dim(self, mesh):
        return mesh.topology().dim()

    @pytest.fixture()
    def codim2_function_tuple(self, mesh: dfn.Mesh, mesh_dim: int):
        mf = dfn.MeshFunction('size_t', mesh, mesh_dim-2, 0)

        # Mark the top right corner as 'separation'
        class Separation(dfn.SubDomain):

            def inside(self, x, on_boundary):
                is_top = x[1] > 1 - dfn.DOLFIN_EPS
                is_right = x[0] > 1 - dfn.DOLFIN_EPS
                return (is_top and is_right) and on_boundary
        subdomain = Separation()
        subdomain.mark(mf, 1)
        return mf, {'separation': 1}

    @pytest.fixture()
    def facet_function_tuple(self, mesh: dfn.Mesh, mesh_dim: int):
        mf = dfn.MeshFunction('size_t', mesh, mesh_dim-1, 0)

        # Mark the bottom and front/back faces of the unit cube as dirichlet
        class Fixed(dfn.SubDomain):

            def inside(self, x, on_boundary):
                is_bottom = x[1] < dfn.DOLFIN_EPS
                # Only check for front/back surfaces in 3D
                if len(x) > 2:
                    is_front = x[2] > 1-dfn.DOLFIN_EPS
                    is_back = x[2] < dfn.DOLFIN_EPS
                else:
                    is_front = False
                    is_back = False
                return (is_bottom or is_front or is_back) and on_boundary

        fixed = Fixed()
        fixed.mark(mf, 1)
        return mf, {'fixed': 1, 'traction': 0}

    @pytest.fixture()
    def cell_function_tuple(self, mesh: dfn.Mesh, mesh_dim: int):
        mf = dfn.MeshFunction('size_t', mesh, mesh_dim, 0)

        # Mark the bottom and front/back faces of the unit cube as dirichlet
        class TopHalf(dfn.SubDomain):

            def inside(self, x, on_boundary):
                is_tophalf = x[1] > 0.5 + dfn.DOLFIN_EPS
                return is_tophalf

        top_half = TopHalf()
        top_half.mark(mf, 1)
        return mf, {'top': 1, 'bottom': 0}

    @pytest.fixture()
    def mesh_functions(self, mesh_dim, codim2_function_tuple, facet_function_tuple, cell_function_tuple):
        codim2_func, _ = codim2_function_tuple
        facet_func, _ = facet_function_tuple
        cell_func, _ = cell_function_tuple
        return (mesh_dim-3)*(None,) + (codim2_func, facet_func, cell_func)

    @pytest.fixture()
    def mesh_subdomains(self, mesh_dim, codim2_function_tuple, facet_function_tuple, cell_function_tuple):
        _, codim2_fields = codim2_function_tuple
        _, facet_fields = facet_function_tuple
        _, cell_fields = cell_function_tuple
        return (mesh_dim-3)*({},) + (codim2_fields, facet_fields, cell_fields)


class GMSHFixtures:

    MESH_NAMES = ['unit_square', 'unit_cube']
    # MESH_NAMES = ['unit_cube']
    # MESH_NAMES = ['unit_square']

    @pytest.fixture(params=MESH_NAMES)
    def mesh_name(self, request):
        return request.param

    def init_unit_square_mesh(self):

        model = gmsh.model.add("unit_square")

        ## Create the geometry
        # Add vertices
        gmsh.model.geo.addPoint(0, 0, 0, tag=1)
        gmsh.model.geo.addPoint(1, 0, 0, tag=2)
        gmsh.model.geo.addPoint(1, 1, 0, tag=3)
        gmsh.model.geo.addPoint(0, 1, 0, tag=4)

        # Add edges
        gmsh.model.geo.addLine(1, 2, tag=1)
        gmsh.model.geo.addLine(2, 3, tag=2)
        gmsh.model.geo.addLine(3, 4, tag=3)
        gmsh.model.geo.addLine(4, 1, tag=4)

        # Add edge loop
        gmsh.model.geo.addCurveLoop([1, 2, 3, 4], tag=1)

        # Creat the unit square surface
        gmsh.model.geo.addPlaneSurface([1], tag=1)

        ## Mark Physical entities

        # Mark the top left and top right points as "inferior" and "superior"
        gmsh.model.geo.addPhysicalGroup(0, [3], name="superior")
        gmsh.model.geo.addPhysicalGroup(0, [4], name="inferior")

        # Mark the bottom, right, top, and left surfaces
        # TODO: Handle "overlapping" physical groups in load_gmsh
        # gmsh.model.geo.addPhysicalGroup(1, [1], name="bottom")
        # gmsh.model.geo.addPhysicalGroup(1, [2], name="right")
        # gmsh.model.geo.addPhysicalGroup(1, [3], name="top")
        # gmsh.model.geo.addPhysicalGroup(1, [3], name="traction")
        # gmsh.model.geo.addPhysicalGroup(1, [4], name="left")

        gmsh.model.geo.addPhysicalGroup(1, [1], name="dirichlet")
        gmsh.model.geo.addPhysicalGroup(1, [2, 3, 4], name="traction")

        # Mark the plane surface
        gmsh.model.geo.addPhysicalGroup(2, [1], name="volume")

        gmsh.model.geo.synchronize()

        gmsh.option.setNumber("Mesh.MeshSizeMin", 1)

        gmsh.model.mesh.generate(2)

    def init_unit_cube_mesh(self, n_extrude: int, z_extrude: float):
        # TODO: Need to use the extruded 2D mesh for this!

        model = gmsh.model.add("unit_cube")

        ## Create the geometry
        # Add vertices
        gmsh.model.geo.addPoint(0, 0, 0, tag=1)
        gmsh.model.geo.addPoint(1, 0, 0, tag=2)
        gmsh.model.geo.addPoint(1, 1, 0, tag=3)
        gmsh.model.geo.addPoint(0, 1, 0, tag=4)

        # Add edges
        gmsh.model.geo.addLine(1, 2, tag=1)
        gmsh.model.geo.addLine(2, 3, tag=2)
        gmsh.model.geo.addLine(3, 4, tag=3)
        gmsh.model.geo.addLine(4, 1, tag=4)

        # Add edge loop (back, front, then sides)
        # sides start from the bottom-left counter-clockwise about z
        gmsh.model.geo.addCurveLoop([1, 2, 3, 4], tag=1)

        # Add the square faces (back, front, then sides counter-clockwise about z starting from bottom)
        gmsh.model.geo.addPlaneSurface([1], tag=1)

        # gmsh.model.geo.synchronize()

        extrude_vector = (0, 0, z_extrude)
        gmsh.model.geo.extrude(
            [(2, 1)], *extrude_vector, numElements=[n_extrude]
        )
        gmsh.model.geo.synchronize()

        ## Mark Physical entities

        # Mark the volume
        gmsh.model.geo.addPhysicalGroup(3, [1], name="volume")

        # Mark vertices, edges, etc.

        # TODO: Figure out how to determine geometry tags?
        # The tags below are hard-coded from examining tag numbers in the GMSH gui
        # (e.g. the surfaces are extruded from edges with known numbers)
        # extruded_surface = line_tag *

        gmsh.model.geo.addPhysicalGroup(2, [13], name="dirichlet")
        gmsh.model.geo.addPhysicalGroup(2, [17, 21, 25], name="traction")
        gmsh.model.geo.addPhysicalGroup(2, [1], name="back")
        gmsh.model.geo.addPhysicalGroup(2, [26], name="front")

        # Mark the top left and top right edges as "inferior" and "superior"
        gmsh.model.geo.addPhysicalGroup(1, [16], name="superior")
        gmsh.model.geo.addPhysicalGroup(1, [20], name="inferior")

        # Mark the origin point
        # gmsh.model.geo.addPhysicalGroup(0, [1], name="origin")

        gmsh.option.setNumber("Mesh.MeshSizeMin", 1)
        gmsh.model.geo.synchronize()

        gmsh.model.mesh.generate(3)

    @pytest.fixture()
    def extrude_zs(self, mesh_name: str):
        z_extrude = 1
        n_extrude = 2
        if mesh_name == 'unit_square':
            return None
        elif mesh_name == 'unit_cube':
            return np.linspace(0, z_extrude, n_extrude+1)

    @pytest.fixture(params=[(1, 1.0)])
    def extrude_info(self, request):
        return request.param

    @pytest.fixture()
    def mesh_path(self, mesh_name: str, extrude_info: tuple[int, float]):
        n_extrude, z_extrude = extrude_info

        mesh_path = f"{mesh_name}.msh"
        if mesh_name == 'unit_square':
            self.init_unit_square_mesh()
        elif mesh_name == 'unit_cube':
            self.init_unit_cube_mesh(n_extrude, z_extrude)

        gmsh.write(mesh_path)

        return mesh_path