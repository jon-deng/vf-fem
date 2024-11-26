"""
Tests for `femvf.load`
"""

from typing import Any
from numpy.typing import NDArray

import pytest

import dolfin as dfn

from femvf import load
from femvf.residuals import solid as slr

from tests.fixture_mesh import GMSHFixtures


class TestLoad(GMSHFixtures):

    SOLID_RESIDUALS = [slr.Rayleigh, slr.KelvinVoigt]

    @pytest.fixture(params=SOLID_RESIDUALS)
    def SolidResidual(self, request):
        return request.param

    @pytest.fixture()
    def dirichlet_bcs(self):
        return {
            'coeff.state.u1': [(dfn.Constant([0, 0]), 'facet', 'dirichlet')]
        }

    def test_load_fenics_model(self, mesh_path, dirichlet_bcs):
        assert load.load_fenics_model(mesh_path, slr.Rayleigh, dirichlet_bcs=dirichlet_bcs)