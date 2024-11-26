"""
Test `meshutils`
"""

import pytest

from femvf import meshutils
from tests.fixture_mesh import GMSHFixtures


class TestMeshIO(GMSHFixtures):

    # def test_meshio(self, mesh_path):

    def test_load_fenics_gmsh(self, mesh_path):
        mesh, mfs, mfs_values = meshutils.load_fenics_gmsh(mesh_path)

        print(mesh)
        print(mfs)
        print(mfs_values)
