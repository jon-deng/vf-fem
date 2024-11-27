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

    MESHES = [dfn.UnitSquareMesh(5, 5), dfn.UnitCubeMesh(5, 5, 5)]
    MESHES = [dfn.UnitSquareMesh(5, 5)]
    MESHES = [dfn.UnitCubeMesh(2, 2, 2)]

    @pytest.fixture(params=MESHES)
    def mesh(self, request):
        return request.param

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
    MESH_NAMES = ['unit_cube']
    MESH_NAMES = ['unit_square']

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

    def init_unit_cube_mesh(self):
        # TODO: Need to use the extruded 2D mesh for this!

        model = gmsh.model.add("unit_cube")

        ## Create the geometry
        # Add vertices
        gmsh.model.geo.addPoint(0, 0, 0, tag=1)
        gmsh.model.geo.addPoint(1, 0, 0, tag=2)
        gmsh.model.geo.addPoint(1, 1, 0, tag=3)
        gmsh.model.geo.addPoint(0, 1, 0, tag=4)

        gmsh.model.geo.addPoint(0, 0, 1, tag=5)
        gmsh.model.geo.addPoint(1, 0, 1, tag=6)
        gmsh.model.geo.addPoint(1, 1, 1, tag=7)
        gmsh.model.geo.addPoint(0, 1, 1, tag=8)

        # Add edges
        # back face
        gmsh.model.geo.addLine(1, 2, tag=1)
        gmsh.model.geo.addLine(2, 3, tag=2)
        gmsh.model.geo.addLine(3, 4, tag=3)
        gmsh.model.geo.addLine(4, 1, tag=4)

        # front face
        gmsh.model.geo.addLine(5, 6, tag=5)
        gmsh.model.geo.addLine(6, 7, tag=6)
        gmsh.model.geo.addLine(7, 8, tag=7)
        gmsh.model.geo.addLine(8, 5, tag=8)

        # back-to-front vertex connectors
        # from the bottom-left counter-clockwise about z
        gmsh.model.geo.addLine(1, 5, tag=9)
        gmsh.model.geo.addLine(2, 6, tag=10)
        gmsh.model.geo.addLine(3, 7, tag=11)
        gmsh.model.geo.addLine(4, 8, tag=12)

        # Add edge loop (back, front, then sides)
        # sides start from the bottom-left counter-clockwise about z
        gmsh.model.geo.addCurveLoop([1, 2, 3, 4], tag=1)
        gmsh.model.geo.addCurveLoop([5, 6, 7, 8], tag=2)

        gmsh.model.geo.addCurveLoop([9, 1, 10, 5], tag=3, reorient=True)
        gmsh.model.geo.addCurveLoop([10, 2, 11, 6], tag=4, reorient=True)
        gmsh.model.geo.addCurveLoop([11, 3, 12, 7], tag=5, reorient=True)
        gmsh.model.geo.addCurveLoop([12, 4, 9, 8], tag=6, reorient=True)

        # Add the square faces (back, front, then sides counter-clockwise about z starting from bottom)
        gmsh.model.geo.addPlaneSurface([1], tag=1)
        gmsh.model.geo.addPlaneSurface([2], tag=2)
        gmsh.model.geo.addPlaneSurface([3], tag=3)
        gmsh.model.geo.addPlaneSurface([4], tag=4)
        gmsh.model.geo.addPlaneSurface([5], tag=5)
        gmsh.model.geo.addPlaneSurface([6], tag=6)

        # Add the closed shell
        gmsh.model.geo.addSurfaceLoop([1, 2, 3, 4, 5, 6], tag=1)

        # Add the cubic volume
        gmsh.model.geo.addVolume([1], tag=1)

        ## Mark Physical entities

        # Mark the top left and top right edges as "inferior" and "superior"
        gmsh.model.geo.addPhysicalGroup(1, [11], name="superior")
        gmsh.model.geo.addPhysicalGroup(1, [12], name="inferior")

        # Mark the bottom, right, top, and left surfaces
        # TODO: Handle "overlapping" physical groups in load_gmsh
        # gmsh.model.geo.addPhysicalGroup(1, [1], name="bottom")
        # gmsh.model.geo.addPhysicalGroup(1, [2], name="right")
        # gmsh.model.geo.addPhysicalGroup(1, [3], name="top")
        # gmsh.model.geo.addPhysicalGroup(1, [3], name="traction")
        # gmsh.model.geo.addPhysicalGroup(1, [4], name="left")

        gmsh.model.geo.addPhysicalGroup(2, [3], name="dirichlet")
        gmsh.model.geo.addPhysicalGroup(2, [4, 5, 6], name="traction")

        # Mark the plane surface
        gmsh.model.geo.addPhysicalGroup(3, [1], name="volume")

        gmsh.model.geo.synchronize()

        gmsh.option.setNumber("Mesh.MeshSizeMin", 1)

        gmsh.model.mesh.generate(3)

    @pytest.fixture()
    def mesh_path(self, mesh_name):

        mesh_path = f"{mesh_name}.msh"
        if mesh_name == 'unit_square':
            self.init_unit_square_mesh()
        elif mesh_name == 'unit_cube':
            self.init_unit_cube_mesh()

        gmsh.write(mesh_path)

        return mesh_path