"""
Tests fluids.py module
"""

import sys
import unittest

import dolfin as dfn

import pandas as pd
import matplotlib.pyplot as plt
import autograd
import autograd.numpy as np
# import numpy as np

import petsc4py
petsc4py.init()

sys.path.append('../')
from femvf.model import load_fsi_model
from femvf import fluids, transforms, meshutils
from femvf.parameters.properties import FluidProperties
from femvf.constants import PASCAL_TO_CGS

class CommonSetup(unittest.TestCase):
    # def setUp(self):
    #     """
    #     This setup produces a interface mesh that is a box with sloped sides.
    #     The inlet portion slopes up at a shallow angle for the inlet contraction, then a shallower
    #     angle for the medial portion of the glottis portion and finally a steep angle for the
    #     diffuser portion.
    #     """
    #     ## Mesh generation
    #     mesh = dfn.RectangleMesh(dfn.Point(-0.5, -0.5), dfn.Point(0.5, 0.5), 10, 30)

    #     # Mark the mesh boundaries with dfn.MeshFunction
    #     vertex_marker = dfn.MeshFunction('size_t', mesh, 0)

    #     class OmegaPressure(dfn.SubDomain):
    #         """Marks the pressure boundary"""
    #         def inside(self, x, on_boundary):
    #             """Marks the appropriate nodes"""
    #             return (np.abs(x[0] - -0.5) <= dfn.DOLFIN_EPS
    #                     or np.abs(x[1] - 0.5) <= dfn.DOLFIN_EPS
    #                     or np.abs(x[0] - 0.5) <= dfn.DOLFIN_EPS)
    #     domainid = 1
    #     OmegaPressure().mark(vertex_marker, domainid)

    #     ## Deform the mesh to a crappy vf-like shape
    #     depth = 0.6
    #     thickness_bottom = 0.6
    #     thickness_top = 1/4 * thickness_bottom
    #     x_inferior_edge = 0.5 * thickness_bottom
    #     x_superior_edge = x_inferior_edge + thickness_top
    #     x1 = [0, 0]
    #     x2 = [thickness_bottom, 0]
    #     x3 = [x_superior_edge, 0.55]
    #     x4 = [x_inferior_edge, 0.5]

    #     mesh.coordinates()[...] = transforms.bilinear(mesh.coordinates(), x1, x2, x3, x4)

    #     surface_vertices = np.array(vertex_marker.where_equal(domainid))
    #     surface_coordinates = mesh.coordinates()[surface_vertices]
    #     idx_sort = forms.sort_vertices_by_nearest_neighbours(surface_coordinates)
    #     surface_vertices = surface_vertices[idx_sort]

    #     self.surface_coordinates = surface_coordinates[idx_sort]
    #     self.fluid_properties = properties.FluidProperties()

    #     self.area = 2*(self.fluid_properties['y_midline'] - self.surface_coordinates[..., 1])
    #     self.p_sub = 800.0*PASCAL_TO_CGS
    #     self.p_sup = 0*PASCAL_TO_CGS

    def setUp(self):
        """
        This setup produces a smooth concave interface mesh
        """
        mesh_path = '../meshes/M5-3layers.xml'

        self.model = load_fsi_model(mesh_path, None, Fluid=fluids.Bernoulli)
        self.surface_coordinates = self.model.surface_coordinates

        self.fluid = self.model.fluid
        self.fluid_properties = self.fluid.get_properties()

        self.area = 2*(self.fluid_properties['y_midline'] - self.surface_coordinates[..., 1])
        self.p_sub = 800.0*PASCAL_TO_CGS
        self.p_sup = 0*PASCAL_TO_CGS

class TestBernoulli(CommonSetup):
    # @unittest.skip("Validated")
    def test_fluid_pressure(self):
        """
        Tests if bernoulli fluid pressures are calculated correctly
        """
        fluid = self.fluid
        fluid.set_properties(self.fluid_properties)

        xy_surf, fluid_props = self.surface_coordinates, self.fluid_properties

        surf_state = (xy_surf, np.zeros(xy_surf.shape), np.zeros(xy_surf.shape))
        q_test, p_test, info = fluid.fluid_pressure(surf_state)

        area = 2*(fluid_props['y_midline'] - xy_surf[..., 1])
        p_verify = fluid_props['p_sub'] + 1/2*fluid_props['rho']*info['flow_rate']**2*(1/fluid_props['a_sub']**2 - 1/area**2)

        # Plot the pressures computed from Bernoulli
        fig, ax = plt.subplots(1, 1)
        ax.plot(xy_surf[:, 0], p_test/10)
        ax.plot(xy_surf[:, 0], p_verify/10)
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("Pressure [Pa]")

        ax_surf = ax.twinx()
        ax_surf.plot(xy_surf[:, 0], xy_surf[:, 1], ls='-.', c='k')
        ax_surf.set_ylabel("y [cm]")

        plt.show()

    # @unittest.skip("...")
    def test_flow_sensitivity_fd(self):
        """
        Test if derivatives of the flow quantities/residuals are correct using finite differences
        """
        np.random.seed(0)
        fluid = self.fluid
        fluid.set_properties(self.fluid_properties)

        surface_coordinates = self.surface_coordinates

        # Set a displacement change direction
        hs = np.concatenate([[0], 2**np.arange(-3, 3, dtype=np.float)])
        du = np.zeros(fluid.u1surf.size)
        du[:] = np.random.rand(du.size)*1e-5

        # Calculate pressure sensitivity using the `Fluid` function
        x0 = (surface_coordinates, 0, 0)
        _, dp_du = fluid.flow_sensitivity(x0)

        # Calculated the predicted change in pressure
        dp = dp_du@du

        # Calculate perturbed pressures for a range of step sizes
        ps = []
        for h in hs:
            x1 = (surface_coordinates + h*du.reshape(-1, 2), 0, 0)
            ps.append(fluid.fluid_pressure(x1)[1])
        ps = np.array(ps)

        dp_true = ps[1]-ps[0]

        taylor_remainder_2 = np.abs(ps[1:, :] - ps[0, :] - hs[1:][:, None]*dp)
        order_2 = np.log(taylor_remainder_2[1:]/taylor_remainder_2[:-1]) / np.log(2)

        error = dp - dp_true
        breakpoint()

        self.assertTrue((error))

    def test_flow_sensitivity_solid(self):
        """
        Test if derivatives of the flow quantities/residuals are correct using finite differences
        """
        np.random.seed(0)
        fluid = self.fluid
        fluid.set_properties(self.fluid_properties)

        surface_coordinates = self.surface_coordinates

        # Set a displacement change direction
        du_fluid = np.zeros(fluid.u1surf.size)
        du_fluid[:] = np.random.rand(du_fluid.size)*1e-2

        du_solid = dfn.Function(self.model.solid.vector_fspace).vector().vec()
        vdof_solid, vdof_fluid = self.model.get_fsi_vector_dofs()
        du_solid[vdof_solid] = du_fluid[vdof_fluid]

        # Calculate pressure sensitivity using the `Fluid` function
        x0 = (surface_coordinates, 0, 0)
        _, dp_du_fluid = fluid.flow_sensitivity(x0)
        _, dp_du_solid = fluid.get_flow_sensitivity_solid(self.model, x0)
        _, dp_du_solid_adj = fluid.get_flow_sensitivity_solid(self.model, x0, adjoint=True)

        # Calculated the predicted changes in pressure
        dp_fluid = dp_du_fluid@du_fluid
        dp_solid = dp_du_solid*du_solid

        dp_solid_adj = dp_du_solid_adj.getVecRight()
        breakpoint()
        dp_du_solid_adj.multTranspose(du_solid, dp_solid_adj)

        error = dp_solid[:] - dp_fluid
        error_adj = dp_solid_adj[:] - dp_fluid

        self.assertTrue(error)

    # @unittest.skip("...")
    def test_flow_sensitivity(self):
        fluid = self.fluid
        fluid.set_properties(self.fluid_properties)

        surface_coordinates = self.surface_coordinates

        ## Calculate pressure sensitivity using the `Fluid` class function
        x = (surface_coordinates, np.zeros(surface_coordinates.shape), np.zeros(surface_coordinates.shape))
        _, dp_du_an = fluid.flow_sensitivity(x)

        ## Calculate pressure sensitivity with finite differences
        dp_du_fd = np.zeros((surface_coordinates.shape[0], surface_coordinates.shape[0]*2))
        dy = np.zeros(surface_coordinates.shape)
        DY = 1e-7
        for ii in range(surface_coordinates.shape[0]):
            dy[...] = 0
            dy[ii, 1] += DY
            x = (surface_coordinates+dy, np.zeros(dy.shape), np.zeros(dy.shape))
            _, pressure, *_ = fluid.fluid_pressure(x)

            dp_du_fd[:, 2*ii+1] = np.array(pressure)

        x = (surface_coordinates, np.zeros(dy.shape), np.zeros(dy.shape))
        q, pressure, *_ = fluid.fluid_pressure(x)
        dp_du_fd[:, 1::2] -= pressure[..., None]

        dp_du_fd /= DY

        ## Check if calculated pressure sensitivity compares well with finite differences
        close = np.isclose(dp_du_fd, dp_du_an, rtol=1e-3, atol=1e-4)
        self.assertTrue(np.all(close))

        err = np.abs(dp_du_fd - dp_du_an)
        rel_err = err/np.maximum(np.abs(dp_du_fd), 1.0)
        breakpoint()

        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        mappable = ax.matshow(rel_err)
        fig.colorbar(mappable, ax=ax)
        ax.set_ylabel("pressure node")
        ax.set_xlabel("(x, y) node")
        ax.set_title("")
        plt.show()

        ## Calculate pressure sensitivity with auto-differentiation
        def fluid_pressure(x):
            x_ = x.reshape(-1, 2)
            q, p, _ = fluid.fluid_pressure((x_, np.zeros(x_.shape), np.zeros(x_.shape)))
            return p

        dp_du_ad = autograd.jacobian(fluid_pressure, 0)(surface_coordinates.reshape(-1))

        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        err = np.abs(dp_du_ad - dp_du_an)
        rel_err = err/np.maximum(np.abs(dp_du_ad), 1.0)

        mappable = ax.matshow(err)
        fig.colorbar(mappable, ax=ax)
        ax.set_ylabel("pressure node")
        ax.set_xlabel("(x, y) node")
        plt.show()

        self.assertTrue(np.all(close))

    def test_get_flow_sensitivity_solid(self):
        surface_coordinates = self.surface_coordinates
        x = (surface_coordinates, np.zeros(surface_coordinates.shape), np.zeros(surface_coordinates.shape))

        _, dp_du = self.fluid.get_flow_sensitivity_solid(self.model, x)

        _, dp_du_t = self.fluid.get_flow_sensitivity_solid(self.model, x, adjoint=True)

class TestSmoothApproximations(unittest.TestCase):
    def test_gaussian(self):
        dgaussian_dx_ad = autograd.grad(fluids.gaussian, 0)
        dgaussian_dx0_ad = autograd.grad(fluids.gaussian, 1)
        x, x0, sigma = 0.1, 0.0, 0.1

        self.assertAlmostEqual(dgaussian_dx_ad(x, x0, sigma), fluids.dgaussian_dx(x, x0, sigma))

        self.assertAlmostEqual(dgaussian_dx0_ad(x, x0, sigma), fluids.dgaussian_dx0(x, x0, sigma))

    def test_sigmoid(self):
        dsigmoid_dx_ad = autograd.grad(fluids.sigmoid, 0)
        x = 0.1

        self.assertAlmostEqual(dsigmoid_dx_ad(x), fluids.dsigmoid_dx(x))

    def test_smooth_cutoff(self):
        dsmooth_cutoff_dx_ad = autograd.grad(fluids.smooth_cutoff, 0)
        dsmooth_cutoff_dx0_ad = autograd.grad(fluids.smooth_cutoff, 1)
        x, k = 0.1, 100.0

        self.assertAlmostEqual(dsmooth_cutoff_dx_ad(x, k), fluids.dsmooth_cutoff_dx(x, k))
        self.assertAlmostEqual(dsmooth_cutoff_dx0_ad(x, k), fluids.dsmooth_cutoff_dx0(x, k))

    def test_smooth_selection(self):
        dsmooth_selection_dx_ad = autograd.grad(fluids.smooth_selection, 0)
        dsmooth_selection_dy_ad = autograd.grad(fluids.smooth_selection, 1)
        dsmooth_selection_dy0_ad = autograd.grad(fluids.smooth_selection, 2)
        x, y, y0, sigma = 1.0, 2.0, 2.1, 0.1

        x, y, y0, sigma = np.array([1.0, 2.0]), np.array([2.1, 2.2]), 2.1, 0.1
        s = np.array([0.0, 1.0])

        a = dsmooth_selection_dx_ad(x, y, y0, s, sigma)
        b = fluids.dsmooth_selection_dx(x, y, y0, s, sigma)
        self.assertTrue(np.all(np.isclose(a, b)))

        a = dsmooth_selection_dy_ad(x, y, y0, s, sigma)
        b = fluids.dsmooth_selection_dy(x, y, y0, s, sigma)
        self.assertTrue(np.all(np.isclose(a, b)))

        a = dsmooth_selection_dy0_ad(x, y, y0, s, sigma)
        b = fluids.dsmooth_selection_dy0(x, y, y0, s, sigma)
        self.assertTrue(np.all(np.isclose(a, b)))

if __name__ == '__main__':
    # unittest.main()

    test = TestBernoulli()
    test.setUp()
    # test.test_get_flow_sensitivity_solid()
    # test.test_flow_sensitivity_fd()
    test.test_flow_sensitivity_solid()
