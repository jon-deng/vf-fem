"""
Test fluid model form definitions
"""

import pytest

import numpy as np

from femvf.residuals import fluid as fld


class MeshFixture:

    @pytest.fixture()
    def mesh(self):
        s = np.linspace(0, 1, 11)
        return s


class TestBernoulliFixedSep(MeshFixture):

    @pytest.fixture()
    def area(self, mesh):
        # Simulate a triangular constriction for Bernoulli
        area = np.abs(mesh-0.5)
        min_area = 0.1
        area[area < min_area] = min_area
        return area

    def test_BernoulliFixedSep(self, mesh):
        assert fld.BernoulliFixedSep(mesh, idx_sep=mesh.shape[-1]//2)

    @pytest.fixture()
    def residual(self, mesh):
        return fld.BernoulliFixedSep(mesh, idx_sep=mesh.shape[-1]//2)

    def test_res(self, mesh, residual, area):
        state, control, prop = residual.res_args

        state['q'] = np.zeros(state['q'].shape)
        state['p'] = np.zeros(state['p'].shape)

        control['area'] = area
        control['psub'] = 100*np.ones(mesh.shape[:-1])
        control['psup'] = 0*np.ones(mesh.shape[:-1])
        print(control)
        res = residual.res(state, control, prop)
        print(res)

        # NOTE: the 'q' and 'p' residuals should reflect the bernoulli flow/pressure
        # for the given area function


class TestBernoulliSmoothMinSep(MeshFixture):

    def test_BernoulliSmoothMinSep(self, mesh):
        assert fld.BernoulliSmoothMinSep(mesh)

    # TODO: Add similar tests to `TestBernoulliFixedSep` here!


class TestBernoulliAreaRatioSep(MeshFixture):

    def test_BernoulliAreaRatioSep(self, mesh):
        assert fld.BernoulliAreaRatioSep(mesh)

