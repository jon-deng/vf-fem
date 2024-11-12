"""
Test `femvf.parameters.transform`
"""

from typing import Union

from functools import reduce

import pytest

import numpy as np
import dolfin as dfn

from femvf.models.transient.base import BaseTransientModel
from femvf.models.dynamical.base import BaseDynamicalModel
from femvf.models.transient import solid as tsld, fluid as tfld
from femvf.load import load_transient_fsi_model
from femvf.parameters import transform as tform

from blockarray import blockvec as bv, linalg as blinalg

from femvf.misc.taylor import taylor_convergence

dfn.set_log_level(50)


def init_default_transform(Transform: tform.Transform, model, x=None, y=None):
    """
    Return a basic transform
    """
    # `TransformFromModel` and `JaxTransformFromModel` instances
    if issubclass(Transform, (tform.TransformFromModel, tform.JaxTransformFromModel)):
        transform_args = (model,)

        if issubclass(Transform, tform.TractionShape):
            kwargs = {'lame_lambda': 101.0, 'lame_mu': 2.0}
        elif issubclass(Transform, tform.LayerModuli):
            kwargs = {}
    # `JaxTransformFromX`:
    elif issubclass(Transform, tform.JaxTransformFromX):
        transform_args = (model.prop,)
        if issubclass(Transform, tform.Identity):
            kwargs = {}
        elif issubclass(Transform, tform.ConstantSubset):
            kwargs = {'const_vals': {'emod': 1e4, 'nu': 0.3}}
        elif issubclass(Transform, tform.Scale):
            kwargs = {'scale': {'emod': 1e4, 'nu': 0.3}}
    # `JaxTransformFromY`:
    elif issubclass(Transform, tform.JaxTransformFromY):
        transform_args = (model.prop,)

    return Transform(*transform_args, **kwargs)


class TestTransform:

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
            FluidType=tfld.BernoulliAreaRatioSep,
        )
        return model

    @pytest.fixture(
        params=[
            tform.Identity,
            tform.TractionShape,
            tform.ConstantSubset,
            tform.Scale,
            (tform.TractionShape, tform.Scale),
            (tform.TractionShape, tform.ConstantSubset),
            (tform.TractionShape, tform.Scale, tform.ConstantSubset),
        ]
    )
    def transform(self, model, request):
        """
        Return the transform to test
        """
        Transform = request.param

        # This handles different initialization calls for

        if isinstance(Transform, tuple):
            _transform = reduce(
                tform.TransformComposition,
                [init_default_transform(x, model) for x in Transform],
            )
        else:
            _transform = init_default_transform(Transform, model)

        return _transform

    @pytest.fixture()
    def x(self, transform):
        """
        Return the linearization point for the transform
        """
        ret_x = transform.x.copy()
        ret_x[:] = 5
        return ret_x

    @pytest.fixture()
    def dx(self, transform):
        """
        Return the perturbation direction for the transform
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
        # x.print_summary()
        y = transform.apply(x)
        # y.print_summary()

    def test_apply_jvp(self, transform, x, dx):
        """
        Test `transform.apply_jvp`
        """

        def f(x):
            return transform.apply(x).copy()

        def jac(x, dx):
            return transform.apply_jvp(x, dx).copy()

        taylor_convergence(x, dx, f, jac, norm=blinalg.norm)

    def test_apply_vjp(self, transform, x, dx, hy):
        """
        Test `transform.apply_vjp`
        """
        dy = transform.apply_jvp(x, dx)
        g_from_primal = bv.dot(hy, dy)

        hx = transform.apply_vjp(x, hy)
        g_from_dual = bv.dot(dx, hx)

        print("(primal, dual) functional values: " f"({g_from_primal}, {g_from_dual})")
        # assert np.isclose(g_from_primal, g_from_dual)


# if __name__ == '__main__':
