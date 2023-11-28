"""
Test `femvf.parameters.parameterization`
"""

from typing import Union

import pytest

import numpy as np
import dolfin as dfn

from femvf.models.transient.base import BaseTransientModel
from femvf.models.dynamical.base import BaseDynamicalModel
from femvf.models.transient import solid as tsld, fluid as tfld
from femvf.load import load_transient_fsi_model
from femvf import meshutils
from femvf.parameters import parameterization

from blockarray import (blockvec as bv, linalg as blinalg)

from taylor import taylor_convergence

dfn.set_log_level(50)

class TestParameterization:

    @pytest.fixture()
    def model(self) -> Union[BaseDynamicalModel, BaseTransientModel]:
        """
        Return the model to test
        """
        mesh_path = '../meshes/M5_BC--GA3--DZ0.00.msh'
        model = load_transient_fsi_model(
            mesh_path,
            None,
            SolidType=tsld.KelvinVoigtWShape,
            FluidType=tfld.BernoulliAreaRatioSep
        )
        return model

    @pytest.fixture(
        params=[
            # parameterization.Identity,
            parameterization.TractionShape,
            parameterization.ConstantSubset
        ]
    )
    def transform(self, model, request):
        """
        Return the parameterization to test
        """
        Param = request.param
        kwargs = {}
        if issubclass(Param, parameterization.ConstantSubset):
            kwargs = {
                'const_vals': {'umesh': 0},
                'scale': {'emod': 1e-3}
            }
        elif issubclass(Param, parameterization.TractionShape):
            kwargs = {
                'const_vals': {'emod': 1e4, 'nu': 0.3},
            }
        return Param(model, **kwargs)

    @pytest.fixture()
    def x(self, transform):
        """
        Return the linearization point for the parameterization
        """
        ret_x = transform.x.copy()
        ret_x[:] = 5
        return ret_x

    @pytest.fixture()
    def dx(self, transform):
        """
        Return the perturbation direction for the parameterization
        """
        ret_dx = transform.x.copy()
        ret_dx[:] = 1e-2
        return ret_dx

    @pytest.fixture()
    def hy(self, transform):
        """
        Return a dual vector for the model properties
        """
        ret_hy = transform.y.copy()
        ret_hy[:] = 1e-2
        return ret_hy

    def test_apply(self, transform, x):
        """
        Test `transform.apply`
        """
        x.print_summary()
        y = transform.apply(x)
        y.print_summary()

    def test_apply_jvp(self, transform, x, dx):
        """
        Test `transform.apply_jvp`
        """
        def f(x):
            return transform.apply(x).copy()

        def jac(x, dx):
            return transform.apply_jvp(x, dx)

        taylor_convergence(x, dx, f, jac, norm=blinalg.norm)

    def test_apply_vjp(self, transform, x, dx, hy):
        """
        Test `transform.apply_vjp`
        """
        dy = transform.apply_jvp(x, dx)
        g_from_primal = bv.dot(hy, dy)

        hx = transform.apply_vjp(x, hy)
        g_from_dual = bv.dot(dx, hx)

        print(
            "(primal, dual) functional values: "
            f"({g_from_primal}, {g_from_dual})"
        )
        assert np.isclose(g_from_primal, g_from_dual)

# if __name__ == '__main__':