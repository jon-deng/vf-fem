"""

"""

import pytest

import dolfin as dfn
import gmsh

gmsh.initialize()

# from femvf.vis.xdmfutils import write_xdmf, export_mesh_values

class FenicsMeshFixtures:

    @pytest.fixture()
    def mesh_name(self):
        return "UnitSquare"

    @pytest.fixture()
    def mesh(self):
        # TODO: Implement test for 3D case too!
        return dfn.UnitSquareMesh(10, 10)

    @pytest.fixture()
    def mesh_dim(self, mesh):
        return mesh.topology().dim()

    @pytest.fixture()
    def vertex_function_tuple(self, mesh: dfn.Mesh):
        mf = dfn.MeshFunction('size_t', mesh, 0, 0)

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
    def mesh_functions(self, mesh_dim, vertex_function_tuple, facet_function_tuple, cell_function_tuple):
        vertex_func, _ = vertex_function_tuple
        facet_func, _ = facet_function_tuple
        cell_func, _ = cell_function_tuple
        return (vertex_func,) + (mesh_dim-3)*(None,) + (facet_func, cell_func)

    @pytest.fixture()
    def mesh_subdomains(self, mesh_dim, vertex_function_tuple, facet_function_tuple, cell_function_tuple):
        _, vertex_fields = vertex_function_tuple
        _, facet_fields = facet_function_tuple
        _, cell_fields = cell_function_tuple
        return (vertex_fields,) + (mesh_dim-3)*({},) + (facet_fields, cell_fields)


class GMSHFixtures:

    @pytest.fixture()
    def mesh_path(self):

        mesh_path = "unit_square.msh"

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
        gmsh.model.geo.addPhysicalGroup(1, [1], name="bottom")
        gmsh.model.geo.addPhysicalGroup(1, [2], name="right")
        gmsh.model.geo.addPhysicalGroup(1, [3], name="top")
        gmsh.model.geo.addPhysicalGroup(1, [4], name="left")

        gmsh.model.geo.addPhysicalGroup(1, [1], name="dirichlet")
        gmsh.model.geo.addPhysicalGroup(1, [2, 3, 4], name="neumann")

        # Mark the plane surface
        gmsh.model.geo.addPhysicalGroup(2, [1], name="volume")

        gmsh.model.geo.synchronize()

        gmsh.model.mesh.generate(2)

        gmsh.write(mesh_path)

        return mesh_path