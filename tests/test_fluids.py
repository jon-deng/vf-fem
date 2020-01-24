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
from femvf import fluids, transforms, properties, forms
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
        mesh_path = '../meshes/geometry2.xml'
        facet_labels, cell_labels = {'pressure': 1, 'fixed': 3}, {}
        mesh, facet_function, _ = forms.load_mesh(mesh_path, facet_labels, cell_labels)

        surface_edges = facet_function.where_equal(facet_labels['pressure'])
        surface_vertices = forms.vertices_from_edges(surface_edges, mesh)

        # Get surface coordinates and sort along the flow direction
        surface_coordinates = mesh.coordinates()[surface_vertices]
        idx_sort = forms.sort_vertices_by_nearest_neighbours(surface_coordinates)
        surface_vertices = surface_vertices[idx_sort]
        surface_coordinates = surface_coordinates[idx_sort]

        self.surface_coordinates = surface_coordinates
        self.fluid_properties = properties.FluidProperties()

        self.area = 2*(self.fluid_properties['y_midline'] - self.surface_coordinates[..., 1])
        self.p_sub = 800.0*PASCAL_TO_CGS
        self.p_sup = 0*PASCAL_TO_CGS

@unittest.skip("I don't care about the euler fluid law yet.")
class Test1DEuler(CommonSetup):

    def test_res_fluid(self):
        xy_ref = self.surface_coordinates
        p_bcs = (self.p_sub, self.p_sup)

        qp0 = (np.zeros(xy_ref.shape[0]), np.zeros(xy_ref.shape[0]))
        qp1 = (np.zeros(xy_ref.shape[0]), np.zeros(xy_ref.shape[0]))

        uva0 = (np.zeros(xy_ref.shape), np.zeros(xy_ref.shape), np.zeros(xy_ref.shape))
        uva1 = (np.zeros(xy_ref.shape), np.zeros(xy_ref.shape), np.zeros(xy_ref.shape))

        n = 3
        fluid_props = self.fluid_properties
        dt = 0.001

        fluids.res_fluid(n, p_bcs, qp0, qp1, xy_ref, uva0, uva1, fluid_props, dt)

    def test_res_fluid_quasistatic(self):
        xy_ref = self.surface_coordinates
        fluid_props = self.fluid_properties
        fluid_props['p_sub'] = self.p_sub
        fluid_props['p_sup'] = self.p_sup

        # Calculate an initial guess based on the bernoulli fluid law
        x = (xy_ref, np.zeros(xy_ref.shape), np.zeros(xy_ref.shape))
        fluid_props['a_sub'] = self.area[0] # This is because I didn't set the subglottal area conditions for 1d euler version
        p_guess, info = fluids.fluid_pressure(x, fluid_props)
        q_guess = info['flow_rate']/self.area
        # breakpoint()

        # You can approximate the momentum residual using
        # p_guess[2:]-p_guess[:-2] + rho*q_guess[1:-1]*(q_guess[2:]-q_guess[:-2])

        # Set up arguments for 1d quasi-static euler
        p_bcs = (self.p_sub, self.p_sup)
        # qp0 = (np.zeros(xy_ref.shape[0]), np.zeros(xy_ref.shape[0]))
        qp0 = (q_guess, p_guess)
        uva0 = (np.zeros(xy_ref.shape), np.zeros(xy_ref.shape), np.zeros(xy_ref.shape))

        res_continuity, res_momentum, sep = [], [], []
        for n in range(xy_ref.shape[0]):
            res_cont, res_mome, info = fluids.res_fluid_quasistatic(n, p_bcs, qp0, xy_ref, uva0, fluid_props)
            res_continuity.append(res_cont)
            res_momentum.append(res_mome)
            sep.append(info['separation_factor'])

        print(f"Continuity residual is {np.array(res_continuity)}")
        print(f"Momentum residual is {np.array(res_momentum)}")
        print(self.area)
        print(np.array(sep))

class TestBernoulli(CommonSetup):
    @unittest.skip("Validated")
    def test_fluid_pressure(self):
        """
        Tests if bernoulli fluid pressures are calculated correctly
        """
        xy_surf, fluid_props = self.surface_coordinates, self.fluid_properties
        surf_state = (xy_surf, np.zeros(xy_surf.shape), np.zeros(xy_surf.shape))
        p_test, info = fluids.fluid_pressure(surf_state, fluid_props)

        area = 2*(fluid_props['y_midline'] - xy_surf[..., 1])
        p_verify = fluid_props['p_sub'] + 1/2*fluid_props['rho']*info['flow_rate']**2*(1/fluid_props['a_sub']**2 - 1/area**2)

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
    def test_flow_sensitivity(self):
        surface_coordinates = self.surface_coordinates
        fluid_props = self.fluid_properties

        ## Calculate pressure sensitivity with finite differences
        dp_du_fd = np.zeros((surface_coordinates.shape[0], surface_coordinates.shape[0]*2))
        dy = np.zeros(surface_coordinates.shape)
        DY = 1e-7
        for ii in range(surface_coordinates.shape[0]):
            dy[...] = 0
            dy[ii, 1] += DY
            x = (surface_coordinates+dy, np.zeros(dy.shape), np.zeros(dy.shape))
            pressure, *_ = fluids.fluid_pressure(x, fluid_props)
            # breakpoint()

            dp_du_fd[:, 2*ii+1] = np.array(pressure)

        x = (surface_coordinates, np.zeros(dy.shape), np.zeros(dy.shape))
        pressure, *_ = fluids.fluid_pressure(x, fluid_props)
        dp_du_fd[:, 1::2] -= pressure[..., None]

        dp_du_fd /= DY

        # Calculate pressure sensitivity with the analytical derivation
        x = (surface_coordinates, np.zeros(dy.shape), np.zeros(dy.shape))
        dp_du_an = fluids.flow_sensitivity(x, fluid_props)[0]

        close = np.isclose(dp_du_fd, dp_du_an, rtol=1e-3, atol=1e-4)

        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        err = np.abs(dp_du_fd - dp_du_an)
        rel_err = np.where(np.abs(dp_du_fd) == 0, 0, err/np.abs(dp_du_fd))
        mappable = ax.matshow(rel_err)
        fig.colorbar(mappable, ax=ax)
        ax.set_ylabel("pressure node")
        ax.set_xlabel("(x, y) node")
        plt.show()

        # print(dp_du_fd[:, 1])
        # print(dp_du_an[:, 1])

        print(dp_du_fd[28, 59])
        print(dp_du_an[28, 59])

        # Calculate pressure sensitivity with auto-differentiation
        def fluid_pressure(x):
            x_ = x.reshape(-1, 2)
            return fluids.fluid_pressure((x_, np.zeros(x_.shape), np.zeros(x_.shape)), fluid_props)[0]

        dp_du_ad = autograd.jacobian(fluid_pressure, 0)(surface_coordinates.reshape(-1))

        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        err = np.abs(dp_du_ad - dp_du_an)
        rel_err = np.where(np.abs(dp_du_ad) == 0, 0, err/np.abs(dp_du_ad))
        mappable = ax.matshow(err)
        fig.colorbar(mappable, ax=ax)
        ax.set_ylabel("pressure node")
        ax.set_xlabel("(x, y) node")
        plt.show()

        breakpoint()
        self.assertTrue(np.all(close))

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

        a = dsmooth_selection_dx_ad(x, y, y0, sigma)
        b = fluids.dsmooth_selection_dx(x, y, y0, sigma)
        self.assertTrue(np.all(np.isclose(a, b)))

        a = dsmooth_selection_dy_ad(x, y, y0, sigma)
        b = fluids.dsmooth_selection_dy(x, y, y0, sigma)
        self.assertTrue(np.all(np.isclose(a, b)))

        a = dsmooth_selection_dy0_ad(x, y, y0, sigma)
        b = fluids.dsmooth_selection_dy0(x, y, y0, sigma)
        self.assertTrue(np.all(np.isclose(a, b)))

if __name__ == '__main__':
    unittest.main()
