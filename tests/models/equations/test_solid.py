"""
Test solid model form definitions
"""

import pytest

import dolfin as dfn

from femvf.models.equations import solid as sld

class TestFenicsForm:

    @pytest.fixture()
    def mesh(self):
        mesh = dfn.UnitSquareMesh(10, 10)
        return mesh

    @pytest.fixture()
    def form_tuple(self, mesh):
        FSPACE_V = dfn.FunctionSpace(mesh, "CG", 1)

        u = dfn.TrialFunction(FSPACE_V)
        v = dfn.TestFunction(FSPACE_V)

        source = dfn.Function(FSPACE_V)
        source.vector()[:] = 1.0

        dx = dfn.Measure("dx", mesh)
        form = (dfn.inner(dfn.grad(u), dfn.grad(v)) - source) * dx
        coefficients = {"u": u, "source": source}
        return form, coefficients

    def test_init(self, form_tuple):
        form, coefficients = form_tuple

        assert sld.FenicsForm(form, coefficients)

