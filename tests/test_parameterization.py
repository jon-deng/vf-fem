"""
Test `femvf.parameters.parameterization`
"""

from typing import Union

import pytest

import numpy as np

from femvf.models.transient.base import BaseTransientModel
from femvf.models.dynamical.base import BaseDynamicalModel
from femvf.models.transient import solid as tsld, fluid as tfld
from femvf.load import load_transient_fsi_model
from femvf import meshutils
from femvf.parameters import parameterization

from blockarray import (blockvec as bv, linalg as blinalg)

from .taylor import taylor_convergence

class TestParameterization:

    @pytest.fixture
    def model(self) -> Union[BaseDynamicalModel, BaseTransientModel]:
        """
        Return the model to test
        """
        mesh_path = '../meshes/M5-3layers.xml'
        model = load_transient_fsi_model(
            mesh_path,
            None,
            SolidType=tsld.KelvinVoigt,
            FluidType=tfld.BernoulliAreaRatioSep
        )
        return model

    def params(self, model):
        """
        Return the parameterization to test
        """
        return parameterization.LayerModuli(model, model.props)

    def x(self, params):
        """
        Return the linearization point for the parameterization
        """
        ret_x = params.x.copy()
        ret_x[:] = 1
        return ret_x

    def dx(self, params):
        """
        Return the perturbation direction for the parameterization
        """
        ret_dx = params.x.copy()
        ret_dx[:] = 1e-2
        return ret_dx

    def hy(self, params):
        """
        Return a dual vector for the model properties
        """
        ret_hy = params.y.copy()
        ret_hy[:] = 1e-2
        return ret_hy

    def test_apply(self, params, x):
        """
        Test `params.apply`
        """
        params.apply(x)

    def test_apply_jvp(self, params, x, dx):
        """
        Test `params.apply_jvp`
        """
        def f(x):
            return params.apply(x)

        def jac(x, dx):
            return params.apply_jvp(x, dx)

        taylor_convergence(x, dx, f, jac, norm=blinalg.norm)

    def test_apply_vjp(self, params, x, dx, hy):
        """
        Test `params.apply_vjp`
        """
        dy = params.apply_jvp(x, dx)
        g_from_primal = bv.dot(hy, dy)

        hx = params.apply_vjp(x, hy)
        g_from_dual = bv.dot(dx, hx)

        print(
            "(primal, dual) functional values: "
            f"({g_from_primal}, {g_from_dual})"
        )
        assert np.isclose(g_from_primal, g_from_dual)

    def test_layer_moduli(self, setup_model):
        model = setup_model
        cell_label_to_dofs = meshutils.process_celllabel_to_dofs_from_forms(
            model.solid.forms,
            model.solid.forms['coeff.prop.emod'].function_space()
        )

        layer_moduli = parameterization.LayerModuli(model, model.props)

        x = layer_moduli.x.copy()
        x['cover'][0] = 1.0
        x['body'][0] = 2.0
        x.print_summary()

        y = layer_moduli.apply(x)
        assert all(np.all(x[label] == y['emod'][cell_label_to_dofs[label]]) for label in x.labels[0])

    def test_identity(self, setup_model):
        model = setup_model
        identity = parameterization.Identity(model, model.props)

        x = identity.x.copy()
        x['emod'][:] = 1.0
        x['rho'][:] = 2.0

        y = identity.apply(x)
        assert all(np.all(x[label] == y[label]) for label in x.labels[0])

# if __name__ == '__main__':