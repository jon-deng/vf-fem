"""
Tests fluids.py module
"""

import sys
import unittest

import dolfin as dfn

# import pandas as pd
import matplotlib.pyplot as plt
import autograd
import autograd.numpy as np
# import numpy as np

import petsc4py
petsc4py.init()

sys.path.append('../')
from femvf.model import load_fsi_model
from femvf import fluids
from femvf.constants import PASCAL_TO_CGS

class CommonSetup(unittest.TestCase):
    """
    Class containing fluid model setup code that is used in the other test classes
    """

    def setUp(self):
        """
        This setup produces a smooth concave interface mesh
        """
        mesh_path = '../meshes/M5-3layers.xml'

        self.model = load_fsi_model(mesh_path, None, Fluid=fluids.Bernoulli)
        self.surface_coordinates = np.stack([self.model.fluid.x_vertices,
                                             self.model.fluid.y_surface], axis=-1)

        self.fluid = self.model.fluid
        self.fluid_properties = self.fluid.get_properties()

        self.fluid_properties['y_midline'][()] = self.surface_coordinates[..., 1].max()+1e-3
        self.fluid_properties['y_gap_min'][()] = 1e-3
        self.fluid_properties['beta'][()] = 10

        self.area = 2*(self.fluid_properties['y_midline'] - self.surface_coordinates[..., 1])
        self.p_sub = 800.0*PASCAL_TO_CGS
        self.p_sup = 0*PASCAL_TO_CGS

class TestBernoulli(CommonSetup):

    ## Tests for internal methods
    def test_fluid_pressure(self):
        """
        Tests if bernoulli fluid pressures are calculated correctly by qualitatively comparing
        with manually calculated Bernoulli pressures
        """
        fluid = self.fluid
        fluid.set_properties(self.fluid_properties)

        xy_surf, fluid_props = self.surface_coordinates, self.fluid_properties

        surf_state = (xy_surf, np.zeros(xy_surf.shape), np.zeros(xy_surf.shape))
        q_test, p_test, info = fluid.fluid_pressure(surf_state, fluid_props)

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

    def test_flow_sensitivity(self):
        """
        Test if derivatives of the flow quantities/residuals are correct using finite differences
        """
        np.random.seed(0)
        fluid = self.fluid
        fluid.set_properties(self.fluid_properties)

        surface_coordinates = self.surface_coordinates

        ## Set a surface state step direction and step sizes
        hs = np.concatenate([[0], 2**np.arange(-6, 30, dtype=np.float)])
        du = np.zeros(fluid.u1surf.size)
        du[:] = np.random.rand(du.size)*1e-5

        ## Calculate p/q sensitivity and convergence order using FD
        # Calculate perturbed flow states
        ps, qs = [], []
        for h in hs:
            x1 = (surface_coordinates + h*du.reshape(-1, 2), np.zeros(surface_coordinates.shape))
            q, p, _ = fluid.fluid_pressure(x1, self.fluid_properties)
            ps.append(p)
            qs.append(q)
        ps = np.array(ps)
        qs = np.array(qs)

        dp_true = ps[1]-ps[0]
        dq_true = qs[1]-qs[0]

        ## Calculate pressure sensitivity using the flow_sensitivity function (the one being tested)
        x0 = (surface_coordinates, 0)
        dq_du, dp_du = fluid.flow_sensitivity(x0, self.fluid_properties)

        ## Calculate the predicted change in pressure and compare the two quantities
        dp = dp_du@du
        dq = dq_du.dot(du)

        taylor_remainder_2 = np.abs(ps[1:, :] - ps[0, :] - hs[1:][:, None]*dp)
        p_order_2 = np.log(taylor_remainder_2[1:]/taylor_remainder_2[:-1]) / np.log(2)
        print("2nd order Taylor (p)", p_order_2)

        taylor_remainder_2 = np.abs(qs[1:] - qs[0] - hs[1:]*dq)
        q_order_2 = np.log(taylor_remainder_2[1:]/taylor_remainder_2[:-1]) / np.log(2)
        print("2nd order Taylor (q)", q_order_2)

        error = dp - dp_true
        breakpoint()
        print(p_order_2[:, 25])
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
        _, dp_du_fluid = fluid.flow_sensitivity(x0, self.fluid_properties)
        _, dp_du_solid = fluid.flow_sensitivity_solid(self.model, x0, self.fluid_properties)
        _, dp_du_solid_adj = fluid.flow_sensitivity_solid(self.model, x0, self.fluid_properties,
                                                          adjoint=True)

        # Calculated the predicted changes in pressure
        dp_fluid = dp_du_fluid@du_fluid
        dp_solid = dp_du_solid*du_solid

        dp_solid_adj = dp_du_solid_adj.getVecRight()
        breakpoint()
        dp_du_solid_adj.multTranspose(du_solid, dp_solid_adj)

        error = dp_solid[:] - dp_fluid
        error_adj = dp_solid_adj[:] - dp_fluid

        self.assertTrue(error)

    ## Tests for user methods
    # TODO : have to think about how to implement these tests
    def test_solve_qp0(self):
        fluid = self.fluid

        u0 = fluid.get_surf_vector()
        u0[:] = 0.1

        fluid.set_ini_surf_state(u0, 0.0)
        qp0 = fluid.solve_qp0()

    def test_solve_qp1(self):
        fluid = self.fluid

        u1 = fluid.get_surf_vector()
        u1[:] = 0.1

        fluid.set_fin_surf_state(u1, 0.0)
        qp1 = fluid.solve_qp1()

    def test_solve_dqp1_du1(self, adjoint=False):
        pass

    def test_solid_dqp0_du0(self, adjoint=False):
        pass

    def test_solid_dqp0_du0_solid(self, adjoint=False):
        pass

    def test_solid_dqp1_du1_solid(self, adjoint=False):
        pass

class TestQuasiSteady1DFluid(CommonSetup):
    """
    This class tests methods common to all QuasiSteady1DFluid models (excludes methods implemented
    by subclasses)
    """
    def test_get_ini_surf_config(self):
        fluid = self.fluid
        x_surf = self.surface_coordinates

        ## Set test data for the ini surface displacement
        u0, v0 = fluid.get_surf_vector(), fluid.get_surf_vector()
        u0[:], v0[:] = np.random.rand(*u0.shape), np.random.rand(*v0.shape)

        ## Calculate the initial surface state manually
        surf_config_ref = (x_surf.reshape(-1) + u0, v0)

        ## Calculate the initial surface state through the Bernoulli fluid model
        fluid.set_ini_surf_state(u0, v0)
        surf_config = fluid.get_ini_surf_state()

        self.assertTrue(np.all(surf_config_ref[0] == surf_config[0]))

    def test_get_fin_surf_state(self):
        fluid = self.fluid
        x_surf = self.surface_coordinates

        ## Set test data for the final surface displacement
        u1, v1 = fluid.get_surf_vector(), fluid.get_surf_vector()
        u1[:], v1[:] = np.random.rand(*u1.shape), np.random.rand(*v1.shape)

        ## Calculate the final surface state manually
        surf_config_ref = (x_surf.reshape(-1) + u1, v1)

        ## Calculate the final surface state through the Bernoulli fluid model
        fluid.set_fin_surf_state(u1, v1)
        surf_config = fluid.get_fin_surf_state()

        self.assertTrue(np.all(surf_config_ref[0] == surf_config[0]))

class TestSmoothApproximations(CommonSetup):
    def setUp(self):
        super().setUp()

        beta = 100.0
        alpha, k, sigma = -500, 75, 0.005
        self.fluid_properties['beta'][()] = beta
        self.fluid_properties['alpha'][()] = alpha
        self.fluid_properties['k'][()] = k
        self.fluid_properties['sigma'][()] = sigma

    def test_dsmooth_lower_bound_df(self):
        a0 = np.array([0.1])
        a_lb = 0.001

        da = 1e-6

        df_da = fluids.dsmooth_lower_bound_df(a0, a_lb, beta=self.fluid_properties['beta'])

        f0 = fluids.smooth_lower_bound(a0, a_lb, beta=self.fluid_properties['beta'])
        f1 = fluids.smooth_lower_bound(a0+da, a_lb, beta=self.fluid_properties['beta'])
        df_da_fd = (f1-f0)/da
        breakpoint()

    def test_smooth_minimum_weights(self):
        """
        Plots the values of the smoothing factors
        """
        fluid = self.fluid
        fluid.set_properties(self.fluid_properties)

        xy_surf, fluid_props = self.surface_coordinates, self.fluid_properties
        y = xy_surf.reshape(-1, 2)[:, 1]

        area = 2 * (fluid_props['y_midline'] - y)

        # print(fluid_props.)
        print([fluid_props[key] for key in ('alpha', 'k', 'sigma')])
        # breakpoint()
        K_STABILITY = np.max(fluid_props['alpha']*area)
        w_smooth_min = np.exp(fluid_props['alpha']*area - K_STABILITY)

        fig, ax = plt.subplots(1, 1)
        ax.plot(fluid.x_vertices, y)
        ax = ax.twinx()
        ax.plot(fluid.x_vertices, w_smooth_min, marker='o', ls='none')
        plt.show()

    def test_smooth_selection_weights(self):
        """
        Plots the values of the smoothing factors
        """
        fluid = self.fluid
        fluid.set_properties(self.fluid_properties)

        xy_surf, fluid_props = self.surface_coordinates, self.fluid_properties
        y = xy_surf.reshape(-1, 2)[:, 1]

        area = 2 * (fluid_props['y_midline'] - y)

        # print(fluid_props.)
        print([fluid_props[key] for key in ('alpha', 'k', 'sigma')])
        log_w = fluids.log_gaussian(area, np.min(area), fluid_props['sigma'])
        w = np.exp(log_w - np.max(log_w))

        fig, ax = plt.subplots(1, 1)
        ax.plot(fluid.x_vertices, y)

        ax = ax.twinx()
        ax.plot(fluid.x_vertices, w, marker='o', ls='none')
        plt.show()

    def test_smooth_cutoff(self):
        """
        Plots the values of the smoothing factors
        """
        fluid = self.fluid
        fluid.set_properties(self.fluid_properties)

        xy_surf, fluid_props = self.surface_coordinates, self.fluid_properties
        y = xy_surf.reshape(-1, 2)[:, 1]

        area = 2 * (fluid_props['y_midline'] - y)

        # print(fluid_props.)
        print([fluid_props[key] for key in ('alpha', 'k', 'sigma')])
        w = fluids.smooth_cutoff(fluid.s_vertices, np.mean(fluid.s_vertices), fluid_props['k'])

        fig, ax = plt.subplots(1, 1)
        ax.plot(fluid.x_vertices, y)

        ax = ax.twinx()
        ax.plot(fluid.x_vertices, w, marker='o', ls='none')
        plt.show()

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

    def test_dsmooth_cutoff(self):
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
    # test.test_fluid_pressure()
    test.test_flow_sensitivity()
    # test.test_get_ini_surf_config()
    # test.test_get_fin_surf_config()
    # test.test_get_flow_sensitivity_solid()
    # test.test_flow_sensitivity_fd()
    # test.test_flow_sensitivity_solid()

    # test = TestSmoothApproximations()
    # test.setUp()
    # test.test_dsmooth_lower_bound_df()
    # test.test_smooth_minimum_weights()
    # test.test_smooth_selection_weights()
    # test.test_smooth_cutoff()
