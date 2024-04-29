"""
Tests fluid.py module
"""

from typing import Mapping
from numpy.typing import NDArray

import pytest

import numpy as np

import femvf.models.equations.fluid as eqfluid

JaxResidual = eqfluid.JaxResidual


class MeshFixtures:
    @pytest.fixture(
        params=[
            np.linspace(0, 1, 11),
            np.ones((2, 1)) * np.linspace(0, 1, 11),
        ]
    )
    def mesh(self, request):
        return request.param


class ModelParamFixtures:
    @pytest.fixture()
    def model(self, mesh) -> eqfluid.JaxResidual:
        return eqfluid.BernoulliAreaRatioSep(mesh)

    @pytest.fixture()
    def state(self, model: JaxResidual):
        state, control, prop = model.res_args
        state['p'][:] = 100
        return state

    @pytest.fixture()
    def control(self, model: JaxResidual, mesh: NDArray):
        state, control, prop = model.res_args

        SHAPE_FLUID = mesh.shape[:-1]
        N_POINT = mesh.shape[-1]
        values = {
            'psub': 100.0,
            'psup': 0.0,
            'qsub': 1.0,
        }
        for key, value in values.items():
            if key in control:
                control[key][:] = value

        # Use a flat area with multiple minimum area points
        area_1d = np.ones(N_POINT)
        area_1d[5:8] = 0.1
        control['area'][:] = (np.ones(SHAPE_FLUID + (1,)) * area_1d).reshape(-1)
        return control

    @pytest.fixture()
    def prop(self, model: JaxResidual):
        state, control, prop = model.res_args

        values = {
            'rho_air': 1.0,
            'area_lb': 0.0,
        }
        for key, value in values.items():
            if key in prop:
                prop[key][:] = value
        return prop


class TestBernoulliAreaRatioSep(MeshFixtures, ModelParamFixtures):

    @pytest.fixture()
    def model(self, mesh) -> eqfluid.JaxResidual:
        return eqfluid.BernoulliAreaRatioSep(mesh)

    def test_res(self, model: JaxResidual, state, control, prop):
        residual_vector = model.res(state, control, prop)
        print(residual_vector)


class TestBernoulliFlowFixedSep(MeshFixtures, ModelParamFixtures):

    @pytest.fixture()
    def model(self, mesh) -> eqfluid.JaxResidual:
        return eqfluid.BernoulliFlowFixedSep(mesh)

    def test_res(self, model: JaxResidual, state, control, prop):
        residual_vector = model.res(state, control, prop)
        print(residual_vector)


class TestBernoulliSmoothMinSep(MeshFixtures, ModelParamFixtures):

    @pytest.fixture()
    def model(self, mesh) -> eqfluid.JaxResidual:
        return eqfluid.BernoulliSmoothMinSep(mesh)

    def test_res(self, model: JaxResidual, state, control, prop):
        residual_vector = model.res(state, control, prop)
        print(residual_vector)


class TestBernoulliFixedSep(MeshFixtures, ModelParamFixtures):

    @pytest.fixture()
    def model(self, mesh) -> eqfluid.JaxResidual:
        return eqfluid.BernoulliFixedSep(mesh)

    def test_res(self, model: JaxResidual, state, control, prop):
        residual_vector = model.res(state, control, prop)
        print(residual_vector)
