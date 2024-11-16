"""
Test `meshutils`
"""

import pytest

from femvf import meshutils as mut

class GMSHFixtures:

    @pytest.fixture()
    def mesh_path(self):
        return '../meshes/square_msh4.msh'


class TestMeshIO(GMSHFixtures):

    # def test_meshio(self, mesh_path):

    def test_load_fenics_gmsh(self, mesh_path):
        mesh, mfs, mfs_values = mut.load_fenics_gmsh(mesh_path)

        print(mesh)
        print(mfs)
        print(mfs_values)
        # breakpoint()
