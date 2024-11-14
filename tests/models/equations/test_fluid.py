"""
Test fluid model form definitions
"""

import pytest

from numpy.typing import NDArray

import numpy as np
from jax import numpy as jnp

from femvf.models.equations import fluid as fld


class TestJaxResidual:

    @pytest.fixture()
    def res_descr(self):

        def res(state: dict[str, NDArray], coefficients: dict[str, NDArray]):
            return state['f'] - coefficients['source']

        res_args = ({'f': np.zeros((10,))}, {'source': np.ones((10,))})

        return res, res_args

    def test_JaxResidual(self, res_descr):
        res, res_args = res_descr
        assert fld.JaxResidual(res, res_args)
