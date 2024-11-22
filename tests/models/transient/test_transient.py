"""
Test transient models
"""

from numpy.typing import NDArray

import pytest

import numpy as np
import dolfin as dfn

from femvf.residuals import solid as slr, fluid as flr
from femvf.models.transient import solid as sld, fluid as fld, coupled as cpd
from femvf.load import derive_1dfluid_from_2dsolid, derive_1dfluid_from_3dsolid

from tests.fixture_mesh import FenicsMeshFixtures


class TestSolid(FenicsMeshFixtures):

    @pytest.fixture(
        params=[slr.Rayleigh, slr.KelvinVoigt, slr.SwellingKelvinVoigt]
    )
    def SolidResidual(self, request):
        return request.param

    def test_init(
            self,
            SolidResidual: slr.PredefinedSolidResidual,
            mesh,
            mesh_functions,
            mesh_subdomains
        ):
        dim = mesh.topology().dim()
        dirichlet_bcs = {
            'coeff.state.u1': [(dfn.Constant(dim*[0]), 'facet', 'fixed')]
        }
        residual = SolidResidual(mesh, mesh_functions, mesh_subdomains, dirichlet_bcs)
        assert sld.Model(residual)

    # TODO: Think of ways you can test a model is working properly?


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
        FluidResidual: flr.PredefinedJaxResidual,
        mesh: NDArray
    ):
        assert fld.Model(FluidResidual(mesh))

    # TODO: Think of ways you can test a model is working properly?


class TestCoupled(FenicsMeshFixtures):

    @pytest.fixture(
        params=[slr.Rayleigh, slr.KelvinVoigt, slr.SwellingKelvinVoigt]
    )
    def SolidModel(self, request):
        return request.param

    @pytest.fixture(
        params=[flr.BernoulliSmoothMinSep, flr.BernoulliFixedSep, flr.BernoulliAreaRatioSep]
    )
    def FluidModel(self, request):
        return request.param

    @pytest.fixture()
    def solid(self, SolidModel, mesh, vertex_function_tuple, facet_function_tuple, cell_function_tuple):
        mf_tuples = [vertex_function_tuple, facet_function_tuple, cell_function_tuple]
        mfs = [mf_tuple[0] for mf_tuple in mf_tuples]
        mfs_values = [mf_tuple[1] for mf_tuple in mf_tuples]
        return SolidModel(mesh, mfs, mfs_values, fixed_facet_labels=['fixed'], fsi_facet_labels=['traction'])

    def test_init(
        self, solid, FluidModel
    ):
        fluid, solid_pdofs = derive_1dfluid_from_2dsolid(solid, FluidModel, fsi_facet_labels=['traction'])
        fluid_pdofs = np.arange(solid_pdofs.size)
        solid

        assert cpd.ExplicitFSIModel(solid, fluid, solid_pdofs, fluid_pdofs)

    # TODO: Think of ways you can test a model is working properly?

