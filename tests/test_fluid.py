"""
Tests fluid.py module
"""

import unittest

import dolfin as dfn

# import pandas as pd
import matplotlib.pyplot as plt
# import autograd
# import autograd.numpy as np
import numpy as np

import petsc4py
petsc4py.init()

from femvf.models import load_fsi_model, fluid
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

        # self.model = load_fsi_model(mesh_path, None, Fluid=fluid.Bernoulli)
        # self.surface_coordinates = np.stack([self.model.fluid.x_vertices,
        #                                      self.model.fluid.y_surface], axis=-1).reshape(-1)
        # self.fluid = self.model.fluid

        x = np.linspace(-0.5, 0.5, 50)
        y = 0.5-x**2
        # print(y)
        self.fluid = fluid.Bernoulli(x, y)
        self.surface_coordinates = np.stack([x, y], axis=-1).reshape(-1)
        
        self.p_sub = 800.0*PASCAL_TO_CGS
        self.p_sup = 100*PASCAL_TO_CGS
        # self.p_sub = 100.0*PASCAL_TO_CGS
        # self.p_sup = 800.0*PASCAL_TO_CGS

        self.fluid_properties = self.fluid.get_properties_vec()
        self.fluid_properties['y_midline'][()] = y.max()+1e-3
        self.fluid_properties['ygap_lb'][()] = 0.0
        self.fluid_properties['r_sep'][()] = 1.0

        # self.fluid_properties['zeta_amin'][()] = 0
        self.fluid_properties['zeta_lb'][()] = 0
        # self.fluid_properties['zeta_sep'][()] = 0
        # self.fluid_properties['zeta_ainv'][()] = 0

        self.area = 2*(self.fluid_properties['y_midline'] - y)

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

        surf_state = (xy_surf.reshape(-1), np.zeros(xy_surf.shape).reshape(-1))
        breakpoint()
        qp_test, info = fluid.fluid_pressure(*surf_state, self.p_sub, self.p_sup, fluid_props)
        q_test, p_test = qp_test['q'], qp_test['p']

        area = 2*(fluid_props['y_midline'] - xy_surf[1::2])
        p_verify = None
        if self.p_sub >= self.p_sup:
            p_verify = self.p_sub + 1/2*fluid_props['rho_air']*q_test**2*(1/fluid_props['a_sub']**2 - 1/area**2)
        else:
            p_verify = self.p_sup + 1/2*fluid_props['rho_air']*q_test**2*(1/fluid_props['a_sup']**2 - 1/area**2)

        # Plot the pressures computed from Bernoulli
        fig, axs = plt.subplots(2, 1, sharex=True)
        ax = axs[0]
        ax.plot(xy_surf[:-1:2], p_test/10, label="Bernoulli Model")
        ax.plot(xy_surf[:-1:2], p_verify/10, label="Manual Bernoulli")
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("Pressure [Pa]")
        # ax.set_ylim(-self.p_sub/10/5, self.p_sub/10*1.1)
        ax.set_ylim(0)

        ax = axs[1]
        ax.plot(xy_surf[:-1:2], xy_surf[1::2], ls='-.', c='k')
        ax.set_ylabel("y [cm]")
        
        axs[0].legend()
        fig.savefig('test_fluid_Bernoulli.png')
        plt.show()
        
    def test_flow_sensitivity(self):
        """
        Test if derivatives of the flow quantities/residuals are correct using finite differences
        """
        np.random.seed(0)
        fluid = self.fluid
        fluid.set_properties(self.fluid_properties)

        psub0, psup0 = 8000, 0
        usurf0 = self.surface_coordinates
        vsurf0 = np.zeros(usurf0.shape)

        ## Set a surface state step direction and step sizes
        hs = np.concatenate([[0], 2**np.arange(-5, 5, dtype=np.float)])
        du = np.zeros(fluid.control['usurf'].size)
        du[:] = np.linspace(0, 1.0e-5, du.size)
        # du[:] = np.random.rand(du.size)*1e-5
        # du[:] = 0.0

        dpsub = 100.0
        dpsup = 0.0

        ## Calculate p/q sensitivity and convergence order using FD
        # Calculate perturbed flow states
        ps, qs = [], []
        for h in hs:
            usurf = usurf0 + h*du
            vsurf = vsurf0
            psub = psub0 + h*dpsub
            psup = psup0 + h*dpsup
            
            qp, _ = fluid.fluid_pressure(usurf, vsurf, psub, psup, self.fluid_properties)
            qs.append(qp['q'][0])
            ps.append(qp['p'])
        ps = np.array(ps)
        qs = np.array(qs)

        dp_fd = (ps[1]-ps[0])/hs[1]
        dq_fd = (qs[1]-qs[0])/hs[1]

        ## Calculate pressure sensitivity using the flow_sensitivity function (the one being tested)
        dq_du, dp_du, dq_dpsub, dp_dpsub, dq_dpsup, dp_dpsup = fluid.flow_sensitivity(usurf0, vsurf0, psub0, psup0, self.fluid_properties)

        ## Calculate the predicted change in pressure and compare the two quantities
        dq = dq_du.dot(du) + dq_dpsub*dpsub + dq_dpsup*dpsup
        dp = dp_du@du + dp_dpsub*dpsub + dp_dpsup*dpsup

        taylor_remainder_2 = np.abs(ps[1:] - ps[0] - hs[1:, None]*dp)
        p_order_2 = np.log(taylor_remainder_2[1:]/taylor_remainder_2[:-1]) / np.log(2)
        print("2nd order Taylor (p)", p_order_2)

        relerr = np.abs(dp - dp_fd)/np.abs(dp_fd)
        print("Relative error (p)", relerr)

        taylor_remainder_2 = np.abs(qs[1:] - qs[0] - hs[1:]*dq)
        q_order_2 = np.log(taylor_remainder_2[1:]/taylor_remainder_2[:-1]) / np.log(2)
        print("2nd order Taylor (q)", q_order_2)

        relerr = np.abs(dq - dq_fd)/np.abs(dq_fd)
        print("Relative error (q)", relerr)
        
        # print(p_order_2[:, 25])
        # self.assertTrue((error))

    def test_flow_sensitivity_psub(self):
        """
        Test if derivatives of the flow quantities/residuals are correct using finite differences
        """
        np.random.seed(0)
        fluid = self.fluid
        props = self.fluid_properties.copy()
        p_sub0 = self.p_sub

        surface_coordinates = self.surface_coordinates
        x0 = (surface_coordinates, np.zeros(self.surface_coordinates.shape))

        ## Set a surface state step direction and step sizes
        hs = np.concatenate([[0], 2**np.arange(0, 5, dtype=np.float)])
        dpsub = 1.0

        ## Calculate p/q sensitivity and convergence order using FD
        # Calculate perturbed flow states
        ps, qs = [], []
        for h in hs:
            props['p_sub'][()] = p_sub0 + h*dpsub
            qp, _ = fluid.fluid_pressure(x0, props)
            ps.append(qp[1])
            qs.append(qp[0])
        ps = np.array(ps)
        qs = np.array(qs)

        dp_true = ps[1]-ps[0]
        dq_true = qs[1]-qs[0]

        ## Calculate pressure sensitivity using the flow_sensitivity function (the one being tested)
        # x0 = (surface_coordinates, 0)
        *_, dq, dp = fluid.flow_sensitivity(x0, self.fluid_properties)

        ## Calculate the predicted change in pressure and compare the two quantities

        taylor_remainder_2 = np.abs(ps[1:, :] - ps[0, :] - hs[1:][:, None]*dp)
        p_order_2 = np.log(taylor_remainder_2[1:]/taylor_remainder_2[:-1]) / np.log(2)
        print("2nd order Taylor (p)", p_order_2)

        taylor_remainder_2 = np.abs(qs[1:] - qs[0] - hs[1:]*dq)
        q_order_2 = np.log(taylor_remainder_2[1:]/taylor_remainder_2[:-1]) / np.log(2)
        print("2nd order Taylor (q)", q_order_2)

        error = dp - dp_true
        
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
        du_fluid = np.zeros(fluid.control['usurf'].size)
        du_fluid[:] = np.random.rand(du_fluid.size)*1e-2

        du_solid = dfn.Function(self.model.solid.vector_fspace).vector().vec()
        vdof_solid, vdof_fluid = self.model.get_fsi_vector_dofs()
        du_solid[vdof_solid] = du_fluid[vdof_fluid]

        # Calculate pressure sensitivity using the `Fluid` function
        x0 = (surface_coordinates, 0, 0)
        _, dp_du_fluid, *_ = fluid.flow_sensitivity(x0, self.fluid_properties)
        _, dp_du_solid = fluid.flow_sensitivity_solid(self.model, x0, self.fluid_properties)
        _, dp_du_solid_adj = fluid.flow_sensitivity_solid(self.model, x0, self.fluid_properties,
                                                          adjoint=True)

        # Calculated the predicted changes in pressure
        dp_fluid = dp_du_fluid@du_fluid
        dp_solid = dp_du_solid*du_solid

        dp_solid_adj = dp_du_solid_adj.getVecRight()
        
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

        fluid.set_ini_surf_state((u0, 0.0))
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
        fluid.set_ini_surf_state((u0, v0))
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

        self.s = np.linspace(0, 1.0, 50)
        self.f = self.s**2

        self.beta = 100.0
        self.alpha = 75 
        self.k =  -500
        self.sigma = 0.005

    def test_dsmoothlb_df(self):
        a0 = np.array([0.1])
        a_lb = 0.001

        da = 1e-6

        df_da = fluid.dsmoothlb_df(a0, a_lb, alpha=self.fluid_properties['zeta_lb'])

        f0 = fluid.smoothlb(a0, a_lb, alpha=self.fluid_properties['zeta_lb'])
        f1 = fluid.smoothlb(a0+da, a_lb, alpha=self.fluid_properties['zeta_lb'])
        df_da_fd = (f1-f0)/da
    
    def test_smoothmin(self):
        """
        Plots the values of the smoothing factors
        """
        fluid = self.fluid
        fluid.set_properties(self.fluid_properties)

        xy_surf, fluid_props = self.surface_coordinates, self.fluid_properties
        y = xy_surf.reshape(-1, 2)[:, 1]

        area = 2 * (fluid_props['y_midline'] - y)

        # print(fluid_props.)
        print([fluid_props[key] for key in ('zeta_amin', 'zeta_sep', 'zeta_ainv')])
        
        K_STABILITY = np.max(fluid_props['zeta_amin']*area)
        w_smooth_min = np.exp(fluid_props['zeta_amin']*area - K_STABILITY)

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
        print([fluid_props[key] for key in ('zeta_amin', 'zeta_sep', 'zeta_ainv')])
        log_w = fluid.log_gaussian(area, np.min(area), fluid_props['zeta_ainv'])
        w = np.exp(log_w - np.max(log_w))

        fig, ax = plt.subplots(1, 1)
        ax.plot(fluid.x_vertices, y)

        ax = ax.twinx()
        ax.plot(fluid.x_vertices, w, marker='o', ls='none')
        plt.show()

    def test_smoothstep(self):
        """
        Plots the values of the smoothing factors
        """
        xy_surf, fluid_props = self.surface_coordinates, self.fluid_properties
        y = xy_surf.reshape(-1, 2)[:, 1]
        w = fluid.smoothstep(fluid.s_vertices, np.mean(fluid.s_vertices), fluid_props['zeta_sep'])

        fig, ax = plt.subplots(1, 1)
        ax.plot(fluid.x_vertices, y)

        ax = ax.twinx()
        ax.plot(fluid.x_vertices, w, marker='o', ls='none')
        plt.show()

    def test_gaussian(self):
        dgaussian_dx_ad = autograd.grad(fluid.gaussian, 0)
        dgaussian_dx0_ad = autograd.grad(fluid.gaussian, 1)
        x, x0, sigma = 0.1, 0.0, 0.1

        self.assertAlmostEqual(dgaussian_dx_ad(x, x0, sigma), fluid.dgaussian_dx(x, x0, sigma))

        self.assertAlmostEqual(dgaussian_dx0_ad(x, x0, sigma), fluid.dgaussian_dx0(x, x0, sigma))

    def test_sigmoid(self):
        dsigmoid_dx_ad = autograd.grad(fluid.sigmoid, 0)
        x = 0.1

        self.assertAlmostEqual(dsigmoid_dx_ad(x), fluid.dsigmoid_dx(x))

    def test_dsmoothstep(self):
        dsmooth_cutoff_dx_ad = autograd.grad(fluid.smoothstep, 0)
        dsmooth_cutoff_dx0_ad = autograd.grad(fluid.smoothstep, 1)
        x, k = 0.1, 100.0

        self.assertAlmostEqual(dsmooth_cutoff_dx_ad(x, k), fluid.dsmoothstep_dx(x, k))
        self.assertAlmostEqual(dsmooth_cutoff_dx0_ad(x, k), fluid.dsmoothstep_dx0(x, k))

    # def test_wavg(self):

class TestSmoothApproximationsInDiscontinuousLimit(CommonSetup):
    def setUp(self):
        super().setUp()

        self.s = np.linspace(0, 1.0, 50)
        self.f = self.s**2

        # self.zeta = 0.005
        self.alpha = 1e-4
        # self.zeta_lb = 0.005
        # self.zeta_min = 0.005 
        # self.zeta_sig =  0.005
        # self.zeta_gau = 0.005

    def test_min(self):
        f_ = self.f - self.f.min() + 1.0
        w1 = fluid.expweight(f_, alpha=self.alpha)
        w2 = fluid.expweight(f_, alpha=0.0)

        f1min = fluid.wavg(self.s, f_, w1)
        f2min = fluid.wavg(self.s, f_, w2)

        print(relative_error(f1min, f2min))

    def test_smoothlb(self):
        LB = 0.1
        y1 = fluid.smoothlb(self.f, LB, alpha=self.alpha)
        y2 = fluid.smoothlb(self.f, LB, alpha=0.0)

        print(relative_error(y1, y2))

    def test_gaussian(self):
        F0 = 0.1
        y1 = fluid.gaussian(self.f, F0, alpha=self.alpha)
        y2 = fluid.gaussian(self.f, F0, alpha=0.0)

        print(relative_error(y1, y2))

    def test_smoothstep(self):
        S0 = np.mean(self.s)
        # S0 = self.s[5]
        y1 = fluid.smoothstep(self.s, S0, alpha=self.alpha)
        y2 = fluid.smoothstep(self.s, S0, alpha=0.0)

        print(relative_error(y1, y2))

    def test_separation(self):
        pass


def relative_error(f1, f2):
    return np.linalg.norm(f2-f1)/np.linalg.norm(f1)

if __name__ == '__main__':
    # unittest.main()

    # test = TestSmoothApproximationsInDiscontinuousLimit()
    # test.setUp()
    # test.test_min() 
    # test.test_smoothlb()
    # test.test_gaussian()
    # test.test_smoothstep()

    test = TestBernoulli()
    test.setUp()
    test.test_fluid_pressure()
    # test.test_flow_sensitivity()
    # test.test_flow_sensitivity_psub()
    # test.test_flow_sensitivity_solid()

    # test = TestSmoothApproximations()
    # test.setUp()
    # test.test_dsmooth_lower_bound_df()
    # test.test_smooth_minimum_weights()
    # test.test_smooth_selection_weights()
    # test.test_smooth_cutoff()
