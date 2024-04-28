"""
Tests fluid.py module
"""

from typing import Mapping
from numpy.typing import NDArray

import pytest

import numpy as np

import femvf.models.equations.fluid as eqfluid

JaxResidual = eqfluid.JaxResidual

class TestBernoulliSmoothMinSep:
    @pytest.fixture(
        params=[
            np.linspace(0, 1, 11),
            np.ones((2, 1)) * np.linspace(0, 1, 11),
        ]
    )
    def mesh(self, request):
        return request.param

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
        control['psub'][:] = 100
        control['psup'][:] = 0

        # Use a flat area with multiple minimum area points
        area_1d = np.ones(N_POINT)
        area_1d[5:8] = 0.1
        control['area'][:] = (np.ones(SHAPE_FLUID+(1,))*area_1d).reshape(-1)
        return control

    @pytest.fixture()
    def prop(self, model: JaxResidual):
        state, control, prop = model.res_args
        prop['rho_air'][:] = 1.00
        prop['area_lb'][:] = 0
        return prop

    def test_res(self, model: JaxResidual, state, control, prop):
        residual_vector = model.res(state, control, prop)
        print(residual_vector)
