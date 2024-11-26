"""
Test `meshutils`
"""

import pytest

from femvf import meshutils
from tests.fixture_mesh import GMSHFixtures


class TestMeshIO(GMSHFixtures):

    # def test_meshio(self, mesh_path):

    def test_load_fenics_gmsh(self, mesh_path):
        mesh, mesh_funcs, mesh_subdomains = meshutils.load_fenics_gmsh(mesh_path)
        assert mesh.topology().dim() == len(mesh_funcs)-1
        assert len(mesh_funcs) == len(mesh_subdomains)
