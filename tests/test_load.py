"""
Tests for `femvf.load`
"""

from typing import Any
from numpy.typing import NDArray

import pytest

import numpy as np
import dolfin as dfn

from femvf import load
from femvf.residuals import solid as slr
from femvf.residuals import fluid as flr

from tests.fixture_mesh import GMSHFixtures


class TestLoad(GMSHFixtures):

    MODEL_TYPES = ['transient', 'dynamical', 'linearized_dynamical']
    MODEL_TYPES = ['dynamical']
    @pytest.fixture(params=MODEL_TYPES)
    def model_type(self, request):
        return request.param

    SOLID_RESIDUALS = [slr.Rayleigh, slr.KelvinVoigt]
    SOLID_RESIDUALS = [slr.Rayleigh]

    @pytest.fixture(params=SOLID_RESIDUALS)
    def SolidResidual(self, request):
        return request.param

    @pytest.fixture()
    def dirichlet_bcs(self):
        return {
            'coeff.state.u1': [(dfn.Constant([0, 0]), 'facet', 'dirichlet')]
        }

    def test_load_fenics_model(self, mesh_path, SolidResidual, model_type, dirichlet_bcs):
        assert load.load_fenics_model(
            mesh_path, SolidResidual, model_type=model_type, dirichlet_bcs=dirichlet_bcs
        )

    # FLUID_RESIDUALS = [flr.BernoulliAreaRatioSep, flr.BernoulliFixedSep]
    FLUID_RESIDUALS = [flr.BernoulliAreaRatioSep]

    @pytest.fixture(params=FLUID_RESIDUALS)
    def FluidResidual(self, request):
        return request.param

    def test_load_jax_model(self, FluidResidual, model_type):
        mesh = np.linspace(0, 1, 10)
        assert load.load_jax_model(mesh, FluidResidual, model_type=model_type)

    @pytest.fixture()
    def solid_kwargs(self, dirichlet_bcs):
        return {
            'dirichlet_bcs': dirichlet_bcs
        }

    @pytest.fixture()
    def fluid_kwargs(self):
        return {}

    def test_load_transient_fsi_model(
        self, mesh_path, SolidResidual, FluidResidual, solid_kwargs, fluid_kwargs, model_type
    ):
        assert load.load_fsi_model(
            mesh_path, SolidResidual, FluidResidual, solid_kwargs, fluid_kwargs, model_type=model_type
        )