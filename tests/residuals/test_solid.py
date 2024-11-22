"""
Test solid model form definitions
"""

import pytest

import dolfin as dfn

from femvf.residuals import solid

from tests.fixture_mesh import FenicsMeshFixtures


class TestResidual(FenicsMeshFixtures):
    RESIDUAL_CLASSES = (
        solid.Rayleigh,
        solid.KelvinVoigt,
        solid.SwellingKelvinVoigt,
        solid.SwellingKelvinVoigtWEpithelium
    )

    @pytest.fixture(params=RESIDUAL_CLASSES)
    def ResidualClass(self, request):
        return request.param

    def test_init(
            self,
            ResidualClass: solid.PredefinedSolidResidual,
            mesh: dfn.Mesh,
            mesh_functions: list[dfn.MeshFunction],
            mesh_subdomains
        ):

        dim = mesh.topology().dim()
        dirichlet_bcs = {
            'coeff.state.u1': [(dfn.Constant(dim*[0]), 'facet', 'fixed')],
            # 'coeff.state.u0': [(dfn.Constant(dim*[0]), 'facet', 'fixed')]
        }

        ResidualClass(mesh, mesh_functions, mesh_subdomains, dirichlet_bcs)
