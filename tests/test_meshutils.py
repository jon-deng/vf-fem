"""
Test `meshutils`
"""

from numpy.typing import NDArray

import pytest

import dolfin as dfn
import numpy as np

from femvf import meshutils

from tests.fixture_mesh import GMSHFixtures, FenicsMeshFixtures


class TestMeshIO(GMSHFixtures):

    # def test_meshio(self, mesh_path):

    def test_load_fenics_gmsh(self, mesh_path):
        mesh, mesh_funcs, mesh_subdomains = meshutils.load_fenics_gmsh(mesh_path)
        assert mesh.topology().dim() == len(mesh_funcs)-1
        assert len(mesh_funcs) == len(mesh_subdomains)

class TestMeshOperations(FenicsMeshFixtures):

    @pytest.fixture()
    def dim(self):
        return 1

    @pytest.fixture()
    def mesh_function(self, mesh_functions: list[dfn.MeshFunction], dim: int):
        return mesh_functions[dim]

    @pytest.fixture()
    def mesh_subdomain_data(self, mesh_subdomains: list[dict[str, int]], dim: int):
        return mesh_subdomains[dim]

    def test_filter_mesh_entities_by_subdomain(
        self,
        mesh: dfn.Mesh,
        mesh_function: dfn.MeshFunction,
        mesh_subdomain_data: dict[str, int]
    ):
        filtering_mesh_values = set([mesh_subdomain_data[key] for key in ['traction']])
        mesh_entities = [ent for ent in dfn.entities(mesh, 1)]

        # TODO: Fix hard-coded ['traction'] names?
        assert meshutils.filter_mesh_entities_by_subdomain(
            mesh_entities, mesh_function, filtering_mesh_values
        )

    def test_filter_mesh_entities_by_plane(
        self, mesh: dfn.Mesh, extrude_zs: NDArray[np.float64]
    ):
        mesh_entities = [ent for ent in dfn.entities(mesh, 1)]

        # origin = np.zeros(3)
        normal = np.array([0, 0, 1])

        for z in extrude_zs:
            assert meshutils.filter_mesh_entities_by_plane(
                mesh_entities, origin=np.array([0, 0, z]), normal=normal
            )
