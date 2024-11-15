"""
Test fenics form definitions and other equations
"""

import pytest

import dolfin as dfn

from femvf.equations import form


class UFLFormFixtures:

    @pytest.fixture()
    def mesh(self):
        mesh = dfn.UnitSquareMesh(10, 10)
        return mesh

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


class TestFenicsForm(UFLFormFixtures):

    def test_init(self, ufl_form, ufl_coefficients):
        assert form.FenicsForm(ufl_form, ufl_coefficients)


class TestPredefinedVolumeForms(UFLFormFixtures):

    @pytest.fixture(
        params = [
            form.IsotropicElasticForm,
            form.IsotropicIncompressibleElasticSwellingForm,
            form.IsotropicElasticSwellingForm,
            form.IsotropicElasticSwellingPowerLawForm
        ]
    )
    def CellForm(self, request):
        return request.param

    def test_CellForm(self, CellForm, measure_dx, mesh):
        assert CellForm({}, measure_dx, mesh)

    @pytest.fixture(
        params = [
            form.IsotropicMembraneForm,
            form.IsotropicIncompressibleMembraneForm,
            form.SurfacePressureForm,
            form.ManualSurfaceContactTractionForm
        ]
    )
    def FacetForm(self, request):
        return request.param

    def test_FacetForm(self, FacetForm, measure_ds, mesh):
        assert FacetForm({}, measure_ds, mesh)

