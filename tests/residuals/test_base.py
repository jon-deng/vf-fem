"""
Test base residual definitions
"""

import pytest

from numpy.typing import NDArray

import numpy as np
import dolfin as dfn

from femvf.residuals import fluid as fld
from femvf.equations import form
from femvf.residuals import solid as sld

from ..fixture_mesh import FenicsMeshFixtures

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


class UFLFormFixtures(FenicsMeshFixtures):

    @pytest.fixture()
    def measure_dx(self, mesh):
        return dfn.Measure("dx", mesh)

    @pytest.fixture()
    def measure_ds(self, mesh):
        return dfn.Measure("ds", mesh)

    @pytest.fixture()
    def ufl_form_tuple(self, mesh, measure_dx):
        FSPACE_V = dfn.FunctionSpace(mesh, "CG", 1)

        u = dfn.TrialFunction(FSPACE_V)
        v = dfn.TestFunction(FSPACE_V)

        source = dfn.Function(FSPACE_V)
        source.vector()[:] = 1.0

        form = (dfn.inner(dfn.grad(u), dfn.grad(v)) - source) * measure_dx
        coefficients = {"u": u, "source": source}
        return form, coefficients

    @pytest.fixture()
    def ufl_form(self, ufl_form_tuple):
        ufl_form, _ = ufl_form_tuple
        return ufl_form

    @pytest.fixture()
    def ufl_coefficients(self, ufl_form_tuple):
        _, coefficients = ufl_form_tuple
        return coefficients


class FenicsFormFixtures(UFLFormFixtures):

    @pytest.fixture()
    def form(self, ufl_form, ufl_coefficients):
        return form.Form(ufl_form, ufl_coefficients)


class TestFenicsResidual(FenicsFormFixtures):

    @pytest.fixture()
    def form(self, mesh, measure_dx):
        return (
            form.InertialForm({}, measure_dx, mesh)
            + form.IsotropicElasticForm({}, measure_dx, mesh)
        )

    def test_FenicsResidual(
            self,
            form,
            mesh,
            mesh_functions,
            mesh_subdomains
        ):
        assert sld.FenicsResidual(form, mesh, mesh_functions, mesh_subdomains)
