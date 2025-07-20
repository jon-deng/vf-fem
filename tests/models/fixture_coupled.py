"""
Coupled solid/fluid residual fixtures
"""

from numpy.typing import NDArray

import pytest

import numpy as np
import dolfin as dfn

from femvf.residuals import solid as slr, fluid as flr
from femvf.models import transient
from femvf.load import derive_1D_interface_from_facet_subdomain

from tests.fixture_mesh import FenicsMeshFixtures


class CoupledResidualFixtures(FenicsMeshFixtures):

    @pytest.fixture(
        params=[slr.Rayleigh, slr.KelvinVoigt, slr.SwellingKelvinVoigt]
    )
    def SolidResidual(self, request):
        return request.param

    @pytest.fixture()
    def solid_res(self, SolidResidual, mesh, mesh_functions, mesh_subdomains):
        dim = mesh.topology().dim()
        dirichlet_bcs = {
            'state/u1': [(dfn.Constant(dim*[0]), 'facet', 'fixed')]
        }
        residual = SolidResidual(mesh, mesh_functions, mesh_subdomains, dirichlet_bcs)
        return residual

    @pytest.fixture(
        params=[flr.BernoulliSmoothMinSep, flr.BernoulliFixedSep, flr.BernoulliAreaRatioSep]
    )
    def FluidResidual(self, request):
        return request.param

    @pytest.fixture()
    def pressure_function_space(self, solid_res: slr.FenicsResidual):
        return solid_res.form['control/p1'].function_space()

    @pytest.fixture()
    def facet_function(self, mesh_functions: list[dfn.MeshFunction]):
        dim = len(mesh_functions)
        return mesh_functions[dim-2]

    @pytest.fixture()
    def facet_subdomain_data(self, mesh_subdomains: list[dict[str, int]]):
        dim = len(mesh_subdomains)
        return mesh_subdomains[dim-2]

    @pytest.fixture()
    def fluid_1Dinterface_info(
        self,
        mesh: dfn.Mesh,
        pressure_function_space: dfn.FunctionSpace,
        facet_function: dfn.MeshFunction,
        facet_subdomain_data: dict[str, int],
        extrude_zs: NDArray[np.float64]
    ):
        fsi_subdomain_names = ['pressure']
        facet_values = set(facet_subdomain_data[name] for name in fsi_subdomain_names)
        s, dofs_fsi_solid, dofs_fsi_fluid = derive_1D_interface_from_facet_subdomain(
            mesh, pressure_function_space, facet_function, facet_values, extrude_zs
        )
        return s, dofs_fsi_solid, dofs_fsi_fluid

    @pytest.fixture()
    def fluid_res(
        self, FluidResidual: flr.PredefinedFluidResidual, fluid_1Dinterface_info
    ):
        s = fluid_1Dinterface_info[0]
        return FluidResidual(s)

    @pytest.fixture()
    def solid_pdofs(self, fluid_1Dinterface_info):
        return fluid_1Dinterface_info[1]

    @pytest.fixture()
    def fluid_pdofs(self, fluid_1Dinterface_info):
        return fluid_1Dinterface_info[2]