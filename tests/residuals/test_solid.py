"""
Test solid model form definitions
"""

import pytest

import dolfin as dfn

from femvf.equations import form
from femvf.residuals import solid as sld


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


class FenicsFormFixtures(UFLFormFixtures):

    @pytest.fixture()
    def form(self, ufl_form, ufl_coefficients):
        return form.FenicsForm(ufl_form, ufl_coefficients)


class TestFenicsResidual(FenicsFormFixtures):

    @pytest.fixture()
    def vertex_function_tuple(self, mesh: dfn.Mesh):
        mf = dfn.MeshFunction('size_t', mesh, 0, 0)
        return mf, {}

    @pytest.fixture()
    def facet_function_tuple(self, mesh: dfn.Mesh):
        num_dim = mesh.topology().dim()
        mf = dfn.MeshFunction('size_t', mesh, num_dim-1, 0)

        # Mark the bottom and front/back faces of the unit cube as dirichlet
        class Fixed(dfn.SubDomain):

            def inside(self, x, on_boundary):
                is_bottom = x[1] < dfn.DOLFIN_EPS
                # Only check for front/back surfaces in 3D
                if len(x) > 2:
                    is_front = x[2] > 1-dfn.DOLFIN_EPS
                    is_back = x[2] < dfn.DOLFIN_EPS
                else:
                    is_front = False
                    is_back = False
                return (is_bottom or is_front or is_back) and on_boundary

        fixed = Fixed()
        fixed.mark(mf, 1)
        return mf, {'fixed': 1, 'traction': 0}

    @pytest.fixture()
    def cell_function_tuple(self, mesh: dfn.Mesh):
        num_dim = mesh.topology().dim()
        mf = dfn.MeshFunction('size_t', mesh, num_dim, 0)

        # Mark the bottom and front/back faces of the unit cube as dirichlet
        class TopHalf(dfn.SubDomain):

            def inside(self, x, on_boundary):
                is_tophalf = x[1] > 0.5 + dfn.DOLFIN_EPS
                return is_tophalf

        top_half = TopHalf()
        top_half.mark(mf, 1)
        return mf, {'top': 1, 'bottom': 0}

    @pytest.fixture()
    def form(self, mesh, measure_dx):

        return form.InertialForm({}, measure_dx, mesh) + form.IsotropicElasticForm({}, measure_dx, mesh)

    def test_FenicsResidual(
            self,
            form,
            mesh,
            vertex_function_tuple,
            facet_function_tuple,
            cell_function_tuple
        ):
        mf_vertex, vertex_values = vertex_function_tuple
        mf_facet, facet_values = facet_function_tuple
        mf_cell, cell_values = cell_function_tuple

        mfs = (mf_vertex, mf_facet, mf_cell)
        mfs_values = (vertex_values, facet_values, cell_values)
        assert sld.FenicsResidual(form, mesh, mfs, mfs_values, ['traction'], ['fixed'])
