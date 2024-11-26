"""
Test fenics form definitions and other equations
"""

import pytest

import ufl
import dolfin as dfn

from femvf.equations.form import Form
from femvf.models import assemblyutils as aut

from ..fixture_mesh import FenicsMeshFixtures


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

        u = dfn.Function(FSPACE_V)
        v = dfn.TestFunction(FSPACE_V)

        source = dfn.Function(FSPACE_V)
        source.vector()[:] = 1.0

        form = (dfn.inner(dfn.grad(u), dfn.grad(v)) - dfn.inner(source, v)) * measure_dx
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


class TestCachedUFLFormAssembler(UFLFormFixtures):

    @pytest.fixture()
    def form_assembler(self, ufl_form: dfn.Form):
        return aut.CachedUFLFormAssembler(ufl_form)

    def test_assemble(self, form_assembler: aut.CachedUFLFormAssembler):
        assert form_assembler.assemble()


class TestFormAssembler(UFLFormFixtures):

    @pytest.fixture()
    def form_key(self):
        return 'f'

    @pytest.fixture()
    def form_assembler(self, ufl_form: dfn.Form, ufl_coefficients, form_key: str):
        return aut.FormAssembler(Form({form_key: ufl_form}, ufl_coefficients))

    def test_assemble(self, form_assembler: aut.FormAssembler, form_key: str):
        assert form_assembler.assemble(form_key)
        assert form_assembler.assemble(form_key)

    def test_assemble_derivative(self, form_assembler: aut.FormAssembler, form_key: str):
        assert form_assembler.assemble_derivative(form_key, 'u')
        assert form_assembler.assemble_derivative(form_key, 'source')

