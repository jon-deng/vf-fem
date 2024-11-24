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

    @staticmethod
    def init_residual(ResidualClass, mesh, mesh_functions, mesh_subdomains):

        dim = mesh.topology().dim()
        dirichlet_bcs = {
            'coeff.state.u1': [(dfn.Constant(dim*[0]), 'facet', 'fixed')],
            # 'coeff.state.u0': [(dfn.Constant(dim*[0]), 'facet', 'fixed')]
        }

        return ResidualClass(mesh, mesh_functions, mesh_subdomains, dirichlet_bcs)

    # TODO: Not sure duplicated init + residual is good design?? Seem off

    def test_init(
        self,
        ResidualClass: solid.PredefinedSolidResidual,
        mesh: dfn.Mesh,
        mesh_functions: list[dfn.MeshFunction],
        mesh_subdomains
    ):
        assert self.init_residual(ResidualClass, mesh, mesh_functions, mesh_subdomains)

    @pytest.fixture()
    def residual(
        self,
        ResidualClass: solid.PredefinedSolidResidual,
        mesh: dfn.Mesh,
        mesh_functions: list[dfn.MeshFunction],
        mesh_subdomains
    ):
        return self.init_residual(ResidualClass, mesh, mesh_functions, mesh_subdomains)

    def test_assemble_form(self, residual):
        assert dfn.assemble(residual.form.form)
