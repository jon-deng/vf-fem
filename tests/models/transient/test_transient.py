"""
Test transient models
"""

import pytest

import numpy as np
import dolfin as dfn

from femvf.models.transient import solid as sld, fluid as fld, coupled as cpd


class FenicsMeshFixtures:

    @pytest.fixture()
    def mesh(self):
        # TODO: Implement test for 3D case too!
        return dfn.UnitSquareMesh(10, 10)

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


class TestSolid(FenicsMeshFixtures):

    @pytest.fixture(
        params=[sld.Rayleigh, sld.KelvinVoigt, sld.SwellingKelvinVoigt]
    )
    def SolidModel(self, request):
        return request.param

    def test_init(
            self,
            SolidModel: sld.PredefinedModel,
            mesh,
            vertex_function_tuple,
            facet_function_tuple,
            cell_function_tuple
        ):
        mf_tuples = [vertex_function_tuple, facet_function_tuple, cell_function_tuple]
        mfs = [mf_tuple[0] for mf_tuple in mf_tuples]
        mfs_values = [mf_tuple[1] for mf_tuple in mf_tuples]
        assert SolidModel(mesh, mfs, mfs_values, fixed_facet_labels=['fixed'], fsi_facet_labels=['traction'])

    # TODO: Think of ways you can test a model is working properly?


class TestFluid:

    @pytest.fixture()
    def mesh(self):
        return np.linspace(0, 1, 11)

    @pytest.fixture(
        params=[fld.BernoulliSmoothMinSep, fld.BernoulliFixedSep, fld.BernoulliAreaRatioSep]
    )
    def FluidModel(self, request):
        return request.param

    def test_init(
            self,
            FluidModel: sld.PredefinedModel,
            mesh
        ):
        assert FluidModel(mesh)

    # TODO: Think of ways you can test a model is working properly?