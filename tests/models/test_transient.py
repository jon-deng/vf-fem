"""
Test transient models
"""

from numpy.typing import NDArray

import pytest

import numpy as np
import dolfin as dfn

from femvf.residuals import solid as slr, fluid as flr
from femvf.models import transient
from femvf.load import derive_1dfluid_from_2dsolid, derive_1dfluid_from_3dsolid

from tests.fixture_mesh import FenicsMeshFixtures


class TestSolid(FenicsMeshFixtures):

    @pytest.fixture(
        params=[slr.Rayleigh, slr.KelvinVoigt, slr.SwellingKelvinVoigt]
    )
    def SolidResidual(self, request):
        return request.param

    def init_residual(self, SolidResidual, mesh, mesh_functions, mesh_subdomains):
        dim = mesh.topology().dim()
        dirichlet_bcs = {
            'coeff.state.u1': [(dfn.Constant(dim*[0]), 'facet', 'fixed')]
        }
        return SolidResidual(mesh, mesh_functions, mesh_subdomains, dirichlet_bcs)

    def test_init(
            self,
            SolidResidual: slr.PredefinedSolidResidual,
            mesh,
            mesh_functions,
            mesh_subdomains
        ):
        residual = self.init_residual(SolidResidual, mesh, mesh_functions, mesh_subdomains)
        assert transient.FenicsModel(residual)

    @pytest.fixture()
    def residual(
            self,
            SolidResidual: slr.PredefinedSolidResidual,
            mesh,
            mesh_functions,
            mesh_subdomains
        ):
        return self.init_residual(SolidResidual, mesh, mesh_functions, mesh_subdomains)

    # TODO: Think of ways you can test a model is working properly?
    @pytest.fixture()
    def model(self, residual: slr.FenicsResidual):
        model = transient.FenicsModel(residual)
        model.dt = 1
        return model

    def test_assem_res(self, model: transient.FenicsModel):
        assert model.assem_res()

    def test_assem_dres_dstate0(self, model: transient.FenicsModel):
        assert model.assem_dres_dstate0()

    def test_assem_dres_dstate1(self, model: transient.FenicsModel):
        assert model.assem_dres_dstate1()

    def test_assem_dres_dcontrol(self, model: transient.FenicsModel):
        assert model.assem_dres_dcontrol()

    def test_assem_dres_dprops(self, model: transient.FenicsModel):
        assert model.assem_dres_dprops()


class TestFluid:

    @pytest.fixture()
    def mesh(self):
        return np.linspace(0, 1, 11)

    @pytest.fixture(
        params=[flr.BernoulliSmoothMinSep, flr.BernoulliFixedSep, flr.BernoulliAreaRatioSep]
    )
    def FluidResidual(self, request):
        return request.param

    def test_init(
        self,
        FluidResidual: flr.PredefinedFluidResidual,
        mesh: NDArray
    ):
        assert transient.JaxModel(FluidResidual(mesh))

    # TODO: Think of ways you can test a model is working properly?


class TestCoupled(FenicsMeshFixtures):

    @pytest.fixture(
        params=[slr.Rayleigh, slr.KelvinVoigt, slr.SwellingKelvinVoigt]
    )
    def SolidResidual(self, request):
        return request.param

    @pytest.fixture(
        params=[flr.BernoulliSmoothMinSep, flr.BernoulliFixedSep, flr.BernoulliAreaRatioSep]
    )
    def FluidResidual(self, request):
        return request.param

    @pytest.fixture()
    def solid(self, SolidResidual, mesh, mesh_functions, mesh_subdomains):
        dim = mesh.topology().dim()
        dirichlet_bcs = {
            'coeff.state.u1': [(dfn.Constant(dim*[0]), 'facet', 'fixed')]
        }
        residual = SolidResidual(mesh, mesh_functions, mesh_subdomains, dirichlet_bcs)
        return transient.FenicsModel(residual)

    def test_init(
        self, solid, FluidResidual
    ):
        fluid_res, solid_pdofs = derive_1dfluid_from_2dsolid(solid, FluidResidual, fsi_facet_labels=['traction'])
        fluid = transient.JaxModel(fluid_res)
        fluid_pdofs = np.arange(solid_pdofs.size)

        assert transient.ExplicitFSIModel(solid, fluid, solid_pdofs, fluid_pdofs)

    # TODO: Think of ways you can test a model is working properly?

