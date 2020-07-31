"""
Compare gradient computed via adjoint method with gradient computed via FD.

To verify accuracy of the gradient, use the Taylor remainder convergence test [1, 2].

References
----------
[1] http://www.dolfin-adjoint.org/en/latest/documentation/verification.html
[2] P. E. Farrell, D. A. Ham, S. W. Funke and M. E. Rognes. Automated derivation of the adjoint of
    high-level transient finite element programs.
    https://arxiv.org/pdf/1204.5577.pdf
"""

import os
from time import perf_counter

import unittest

import numpy as np
import matplotlib.pyplot as plt
import dolfin as dfn

from femvf import statefile as sf, linalg
from femvf.forward import implicit_increment
from femvf.adjoint import adjoint
from femvf.solids import Rayleigh, KelvinVoigt
from femvf.fluids import Bernoulli
from femvf.model import load_fsi_model
from femvf.constants import PASCAL_TO_CGS
from femvf.parameters import parameterization
from femvf.functionals import basic

from femvf.utils import line_search, line_search_p

# sys.path.append(path.expanduser('~/lib/vf-optimization'))
# from optvf import functionals as extra_funcs

dfn.set_log_level(30)
np.random.seed(123)

def get_starting_rayleigh_model():
    ## Set the mesh to be used and initialize the forward model
    mesh_dir = '../meshes'
    mesh_base_filename = 'geometry2'
    mesh_base_filename = 'M5-3layers-refined'

    mesh_path = os.path.join(mesh_dir, mesh_base_filename + '.xml')
    model = load_fsi_model(mesh_path, None, Solid=Rayleigh, Fluid=Bernoulli)

    ## Set the fluid/solid parameters
    emod = 2.5e3 * PASCAL_TO_CGS

    k_coll = 1e11
    y_gap = 0.02
    y_coll_offset = 0.01
    alpha, k, sigma = -3000, 50, 0.002

    fluid_props = model.fluid.get_properties()
    fluid_props['y_midline'][()] = np.max(model.solid.mesh.coordinates()[..., 1]) + y_gap
    fluid_props['p_sub'][()] = 1000 * PASCAL_TO_CGS
    fluid_props['alpha'][()] = alpha
    fluid_props['k'][()] = k
    fluid_props['sigma'][()] = sigma

    solid_props = model.solid.get_properties()
    solid_props['emod'][:] = emod
    solid_props['rayleigh_m'][()] = 0.0
    solid_props['rayleigh_k'][()] = 3e-4
    solid_props['k_collision'][()] = k_coll
    solid_props['y_collision'][()] = fluid_props['y_midline'][()] - y_coll_offset

    return model, solid_props, fluid_props

def get_starting_kelvinvoigt_model():
    ## Set the mesh to be used and initialize the forward model
    mesh_dir = '../meshes'
    mesh_base_filename = 'geometry2'
    mesh_base_filename = 'M5-3layers-medial-surface-refinement'

    mesh_path = os.path.join(mesh_dir, mesh_base_filename + '.xml')
    model = load_fsi_model(mesh_path, None, Solid=KelvinVoigt, Fluid=Bernoulli)

    ## Set the fluid/solid parameters
    emod = 6e3 * PASCAL_TO_CGS

    k_coll = 1e13
    y_gap = 0.02
    y_gap = 0.1
    y_coll_offset = 0.01

    y_gap = 0.01
    y_coll_offset = 0.0025
    alpha, k, sigma = -3000, 50, 0.002

    fluid_props = model.fluid.get_properties()
    fluid_props['y_midline'][()] = np.max(model.solid.mesh.coordinates()[..., 1]) + y_gap
    fluid_props['p_sub'][()] = 800 * PASCAL_TO_CGS
    fluid_props['alpha'][()] = alpha
    fluid_props['k'][()] = k
    fluid_props['sigma'][()] = sigma
    fluid_props['y_gap_min'][()] = y_coll_offset
    # fluid_props['y_gap_min'][()] = -10000

    solid_props = model.solid.get_properties()
    solid_props['emod'][:] = emod
    solid_props['eta'][()] = 3.0
    solid_props['k_collision'][()] = k_coll
    # solid_props['y_collision'][()] = fluid_props['y_midline'][()] - y_coll_offset
    solid_props['y_collision'][()] = fluid_props['y_midline'] - y_coll_offset

    return model, solid_props, fluid_props

class TaylorTestUtils(unittest.TestCase):

    def plot_taylor_convergence(self, grad_on_step_dir, h, g):
        # Plot the adjoint gradient and finite difference approximations of the gradient
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        ax.plot(h[1:], (g[1:] - g[0])/h[1:],
                color='C0', marker='o', label='FD Approximations')
        ax.axhline(grad_on_step_dir, color='C1', label="Adjoint gradient")
        ax.ticklabel_format(axis='y', style='sci')
        ax.set_xlabel("Step size")
        ax.set_ylabel(r"$\frac{df}{dh}$")
        ax.legend()
        return fig, ax

    def plot_grad_uva(self, model, grad_uva):
        """
        Plot the (u, v, a) initial state gradients
        """
        tri = model.get_triangulation()
        fig, axs = plt.subplots(3, 2, constrained_layout=True)
        for ii, (grad, label) in enumerate(zip(grad_uva, ['u0', 'v0', 'a0'])):
            for jj in range(2):
                grad_comp = grad[:].reshape(-1, 2)[:, jj]
                mappable = axs[ii, jj].tripcolor(tri, grad_comp[model.solid.vert_to_sdof])

                axs[ii, jj].set_title(f"$\\nabla_{{{label}}} f$")

                fig.colorbar(mappable, ax=axs[ii, jj])
        return fig, axs

class TestBasicGradient(TaylorTestUtils):
    COUPLING = 'explicit'
    OVERWRITE_FORWARD_SIMULATIONS = False
    # FUNCTIONAL = basic.FinalDisplacementNorm
    # FUNCTIONAL = basic.ElasticEnergyDifference
    # FUNCTIONAL = basic.PeriodicError
    # FUNCTIONAL = basic.PeriodicEnergyError
    # FUNCTIONAL = basic.SubglottalWork
    # FUNCTIONAL = basic.TransferWorkbyDisplacementIncrement
    # FUNCTIONAL = basic.TransferWorkbyVelocity
    # FUNCTIONAL = basic.FinalFlowRateNorm
    # FUNCTIONAL = basic.TransferEfficiency
    FUNCTIONAL = basic.AcousticPower
    # FUNCTIONAL = basic.AcousticEfficiency

    def setUp(self):
        """
        Set the model, parameters, and functional to test with

        This should set the starting point of the line search; all the parameters needed to solve
        a single simulation are given by this set up.
        """
        self.CASE_NAME = 'singleperiod'
        self.model, self.solid_props, self.fluid_props = get_starting_kelvinvoigt_model()

        t_start, t_final = 0, 0.01
        times_meas = np.linspace(t_start, t_final, 128)
        self.times = times_meas

        self.uva0 = (0, 0, 0)

        self.functional = self.FUNCTIONAL(self.model)
        self.functional.constants['n_start'] = 50
        # self.hs = np.concatenate(([0], 2.0**np.arange(-8, 1)), axis=0)

    def get_taylor_order(self, save_path, hs,
                         duva=(0, 0, 0), dsolid_props=None, dfluid_props=None, dtimes=None):
        """
        Runs a line search of simulations along a specified direction
        """
        if self.OVERWRITE_FORWARD_SIMULATIONS or not os.path.exists(save_path):
            if os.path.exists(save_path):
                os.remove(save_path)
            line_search(hs, self.model, self.uva0, self.solid_props, self.fluid_props, self.times,
                        duva=duva, dsolid_props=dsolid_props, dfluid_props=dfluid_props, dtimes=dtimes,
                        filepath=save_path, coupling=self.COUPLING)
        else:
            print("Using existing files")

        gs, remainders, orders, grads = grad_and_taylor_order(
            save_path, self.functional, hs, self.model, duva=duva, dsolid_props=dsolid_props,
            dfluid_props=dfluid_props, dtimes=dtimes, coupling=self.COUPLING)

        remainder_1, remainder_2 = remainders
        order_1, order_2 = orders
        (grad_uva, grad_solid, grad_fluid, grad_times), grad_step, grad_step_fd = grads

        # self.plot_taylor_convergence(grad_step, hs, gs)
        # self.plot_grad_uva(self.model, grad_uva)
        # plt.show()

        print('1st order Taylor', order_1)
        print('2nd order Taylor', order_2)

        return (order_1, order_2)

    def test_emod(self):
        save_path = f'out/linesearch_emod_{self.COUPLING}.h5'
        hs = np.concatenate(([0], 2.0**(np.arange(2, 9)-11)), axis=0)
        step_size = 0.5e0 * PASCAL_TO_CGS

        dsolid = self.solid_props.copy()
        dsolid.vector[:] = 0
        dsolid['emod'][:] = 1.0*step_size*10

        order_1, order_2 = self.get_taylor_order(save_path, hs, dsolid_props=dsolid)
        # self.assertTrue(np.all(np.isclose(order_1, 1.0)))
        # self.assertTrue(np.all(np.isclose(order_2, 2.0)))

    def test_u0(self):
        save_path = f'out/linesearch_u0_{self.COUPLING}.h5'
        hs = np.concatenate(([0], 2.0**(np.arange(2, 9)-16)), axis=0)

        ## Pick a step direction
        # Increment `u` linearly as a function of x and y in the y direction
        # step_dir = np.zeros(xy.size)
        # step_dir[:-1:2] = -(y-y.min()) / (y.max()-y.min())
        # step_dir[1::2] = -(y-y.min()) / (y.max()-y.min())

        # Increment `u` only along the pressure surface
        surface_dofs = self.model.solid.vert_to_vdof.reshape(-1, 2)[self.model.surface_vertices]
        xy_surface = self.model.solid.mesh.coordinates()[self.model.surface_vertices, :]
        x_surf, y_surf = xy_surface[:, 0], xy_surface[:, 1]
        step_dir = dfn.Function(self.model.solid.vector_fspace).vector()

        x_frac = (x_surf-x_surf.min())/(x_surf.max()-x_surf.min())
        step_dir[np.array(surface_dofs[:, 0])] = 1*(1.0-x_frac) + 0.25*x_frac
        step_dir[np.array(surface_dofs[:, 1])] = -1*(1.0-x_frac) + 0.25*x_frac
        # step_dir[np.array(surface_dofs[:, 0])] = (y_surf/y_surf.max())**2
        # step_dir[np.array(surface_dofs[:, 1])] = (y_surf/y_surf.max())**2

        # Increment `u` only in the interior of the body
        # step_dir[:] = 1.0
        # step_dir[surface_dofs[:, 0].flat] = 0.0
        # step_dir[surface_dofs[:, 1].flat] = 0.0

        self.model.solid.bc_base.apply(step_dir)

        duva = (step_dir*0.01, 0.0, 0.0)

        order_1, order_2 = self.get_taylor_order(save_path, hs, duva=duva)
        # self.assertTrue(np.all(np.isclose(order_1, 1.0)))
        # self.assertTrue(np.all(np.isclose(order_2, 2.0)))

    def test_v0(self):
        save_path = f'out/linesearch_v0_{self.COUPLING}.h5'
        hs = np.concatenate(([0], 2.0**(np.arange(2, 9)-9)), axis=0)

        xy = self.model.get_ref_config()[dfn.dof_to_vertex_map(self.model.solid.scalar_fspace)]
        y = xy[:, 1]

        # Set the step direction as a linear (de)increase in x and y displacement in the y direction
        # step_dir = np.zeros(xy.size)
        step_dir = dfn.Function(self.model.solid.vector_fspace).vector()
        _step_dir = step_dir[:].copy()
        _step_dir[:-1:2] = -(y-y.min()) / (y.max()-y.min())
        _step_dir[1::2] = -(y-y.min()) / (y.max()-y.min())
        step_dir[:] = _step_dir

        self.model.solid.bc_base.apply(step_dir)
        duva = (0.0, step_dir*0.1, 0.0)

        order_1, order_2 = self.get_taylor_order(save_path, hs, duva=duva)
        # self.assertTrue(np.all(np.isclose(order_1, 1.0)))
        # self.assertTrue(np.all(np.isclose(order_2, 2.0)))

    def test_a0(self):
        save_path = f'out/linesearch_a0_{self.COUPLING}.h5'
        hs = np.concatenate(([0], 2.0**(np.arange(2, 9))), axis=0)

        xy = self.model.get_ref_config()[dfn.dof_to_vertex_map(self.model.solid.scalar_fspace)]
        y = xy[:, 1]

        # Set the step direction as a linear (de)increase in x and y displacement in the y direction
        step_dir = dfn.Function(self.model.solid.vector_fspace).vector()
        _step_dir = step_dir[:].copy()
        _step_dir[:-1:2] = -(y-y.min()) / (y.max()-y.min())
        _step_dir[1::2] = -(y-y.min()) / (y.max()-y.min())
        step_dir[:] = _step_dir

        self.model.solid.bc_base.apply(step_dir)
        duva = (0.0, 0.0, step_dir)

        order_1, order_2 = self.get_taylor_order(save_path, hs, duva=duva)
        # self.assertTrue(np.all(np.isclose(order_1, 1.0)))
        # self.assertTrue(np.all(np.isclose(order_2, 2.0)))

    def test_times(self):
        save_path = f'out/linesearch_dt_{self.COUPLING}.h5'
        hs = np.concatenate(([0], 2.0**(np.arange(2, 9))), axis=0)

        dtimes = np.zeros(self.times.size)
        dtimes[1:] = np.arange(1, dtimes.size)*1e-9

        order_1, order_2 = self.get_taylor_order(save_path, hs, dtimes=dtimes)
        # self.assertTrue(np.all(np.isclose(order_1, 1.0)))
        # self.assertTrue(np.all(np.isclose(order_2, 2.0)))

class TestBasicGradientSingleStep(TestBasicGradient):
    COUPLING = 'explicit'
    OVERWRITE_FORWARD_SIMULATIONS = True
    FUNCTIONAL = basic.FinalDisplacementNorm
    # FUNCTIONAL = basic.ElasticEnergyDifference
    # FUNCTIONAL = basic.PeriodicError
    # FUNCTIONAL = basic.PeriodicEnergyError
    # FUNCTIONAL = basic.TransferEfficiency
    # FUNCTIONAL = basic.SubglottalWork
    # FUNCTIONAL = basic.FinalFlowRateNorm

    def setUp(self):
        """
        Set the model and baseline simulation parameters

        This should set the starting point of the line search; all the parameters needed to solve
        a single simulation are given by this set up.
        """
        self.CASE_NAME = 'singlestep'

        ## parameter set
        self.model, self.solid_props, self.fluid_props = get_starting_kelvinvoigt_model()

        t_start, t_final = 0, 0.001
        times_meas = np.linspace(t_start, t_final, 2)
        self.times = times_meas

        self.uva0 = (0, 0, 0)

        # Step sizes and scale factor
        self.hs = np.concatenate(([0], 2.0**np.arange(-8, 1)), axis=0)

    def test_emod(self):
        save_path = f'out/linesearch_emod_{self.COUPLING}_{self.CASE_NAME}.h5'
        hs = np.concatenate(([0], 2.0**(np.arange(2, 9)-9)), axis=0)
        step_size = 0.5e0 * PASCAL_TO_CGS

        dsolid = self.solid_props.copy()
        dsolid.vector[:] = 0
        dsolid['emod'][:] = 1.0*step_size/5

        order_1, order_2 = self.get_taylor_order(save_path, hs, dsolid_props=dsolid)
        # self.assertTrue(np.all(np.isclose(order_1, 1.0)))
        # self.assertTrue(np.all(np.isclose(order_2, 2.0)))

    def test_u0(self):
        save_path = f'out/linesearch_u0_{self.COUPLING}_{self.CASE_NAME}.h5'
        hs = np.concatenate(([0], 2.0**(np.arange(2, 9)-16)), axis=0)

        ## Pick a step direction
        # Increment `u` linearly as a function of x and y in the y direction
        # step_dir = np.zeros(xy.size)
        # step_dir[:-1:2] = -(y-y.min()) / (y.max()-y.min())
        # step_dir[1::2] = -(y-y.min()) / (y.max()-y.min())

        # Increment `u` only along the pressure surface
        surface_dofs = self.model.solid.vert_to_vdof.reshape(-1, 2)[self.model.surface_vertices]
        xy_surface = self.model.solid.mesh.coordinates()[self.model.surface_vertices, :]
        x_surf, y_surf = xy_surface[:, 0], xy_surface[:, 1]
        step_dir = dfn.Function(self.model.solid.vector_fspace).vector()

        x_frac = (x_surf-x_surf.min())/(x_surf.max()-x_surf.min())
        step_dir[np.array(surface_dofs[:, 0])] = 1*(1.0-x_frac) + 0.25*x_frac
        step_dir[np.array(surface_dofs[:, 1])] = -1*(1.0-x_frac) + 0.25*x_frac
        # step_dir[np.array(surface_dofs[:, 0])] = (y_surf/y_surf.max())**2
        # step_dir[np.array(surface_dofs[:, 1])] = (y_surf/y_surf.max())**2

        # Increment `u` only in the interior of the body
        # step_dir[:] = 1.0
        # step_dir[surface_dofs[:, 0].flat] = 0.0
        # step_dir[surface_dofs[:, 1].flat] = 0.0

        self.model.solid.bc_base.apply(step_dir)

        duva = (step_dir*0.00005, 0.0, 0.0)

        order_1, order_2 = self.get_taylor_order(save_path, hs, duva=duva)
        # self.assertTrue(np.all(np.isclose(order_1, 1.0)))
        # self.assertTrue(np.all(np.isclose(order_2, 2.0)))

    def test_v0(self):
        save_path = f'out/linesearch_v0_{self.COUPLING}_{self.CASE_NAME}.h5'
        hs = np.concatenate(([0], 2.0**(np.arange(2, 9)-9)), axis=0)

        xy = self.model.get_ref_config()[dfn.dof_to_vertex_map(self.model.solid.scalar_fspace)]
        y = xy[:, 1]

        # Set the step direction as a linear (de)increase in x and y displacement in the y direction
        # step_dir = np.zeros(xy.size)
        step_dir = dfn.Function(self.model.solid.vector_fspace).vector()
        _step_dir = step_dir[:].copy()
        _step_dir[:-1:2] = -(y-y.min()) / (y.max()-y.min())
        _step_dir[1::2] = -(y-y.min()) / (y.max()-y.min())
        step_dir[:] = _step_dir

        self.model.solid.bc_base.apply(step_dir)
        duva = (0.0, step_dir*1e-5, 0.0)

        order_1, order_2 = self.get_taylor_order(save_path, hs, duva=duva)
        # self.assertTrue(np.all(np.isclose(order_1, 1.0)))
        # self.assertTrue(np.all(np.isclose(order_2, 2.0)))

    def test_a0(self):
        save_path = f'out/linesearch_a0_{self.COUPLING}_{self.CASE_NAME}.h5'
        hs = np.concatenate(([0], 2.0**(np.arange(2, 9))), axis=0)

        xy = self.model.get_ref_config()[dfn.dof_to_vertex_map(self.model.solid.scalar_fspace)]
        y = xy[:, 1]

        # Set the step direction as a linear (de)increase in x and y displacement in the y direction
        step_dir = dfn.Function(self.model.solid.vector_fspace).vector()
        _step_dir = step_dir[:].copy()
        _step_dir[:-1:2] = -(y-y.min()) / (y.max()-y.min())
        _step_dir[1::2] = -(y-y.min()) / (y.max()-y.min())
        step_dir[:] = _step_dir

        self.model.solid.bc_base.apply(step_dir)
        duva = (0.0, 0.0, step_dir*1e-4)

        order_1, order_2 = self.get_taylor_order(save_path, hs, duva=duva)
        # self.assertTrue(np.all(np.isclose(order_1, 1.0)))
        # self.assertTrue(np.all(np.isclose(order_2, 2.0)))

    def test_times(self):
        save_path = f'out/linesearch_dt_{self.COUPLING}.h5'
        hs = np.concatenate(([0], 2.0**(np.arange(2, 9))), axis=0)

        dtimes = np.zeros(self.times.size)
        dtimes[1:] = np.arange(1, dtimes.size)*1e-11

        order_1, order_2 = self.get_taylor_order(save_path, hs, dtimes=dtimes)

class TestPeriodicKelvinVoigtGradient(TaylorTestUtils):
    COUPLING = 'explicit'
    OVERWRITE_FORWARD_SIMULATIONS = False
    FUNCTIONAL = basic.FinalDisplacementNorm

    def setUp(self):
        """
        Set the model and baseline simulation parameters

        This should set the starting point of the line search; all the parameters needed to solve
        a single simulation are given by this set up.
        """
        self.model, solid_props, fluid_props = get_starting_kelvinvoigt_model()

        xy = self.model.get_ref_config().flat[self.model.solid.vdof_to_vert].reshape(-1, 2)
        y = xy[:, 1]

        t_start, t_final = 0.0, 0.01

        constants = {
            'default_solid_props': solid_props,
            'default_fluid_props': fluid_props,
            'NUM_STATES_PER_PERIOD': 128}
        self.p = parameterization.PeriodicKelvinVoigt(self.model, constants)
        self.p['period'][()] = t_final-t_start

    def get_taylor_order(self, save_path, hs, dp):
        """
        Runs a line search of simulations along a specified direction
        """
        if self.OVERWRITE_FORWARD_SIMULATIONS or not os.path.exists(save_path):
            if os.path.exists(save_path):
                os.remove(save_path)
            line_search_p(hs, self.model, self.p, dp, coupling=self.COUPLING,
                          filepath=save_path)
        else:
            print("Using existing files")

        functional = self.FUNCTIONAL(self.model)
        gs, remainders, orders, grads = grad_and_taylor_order_p(
            save_path, functional, hs, self.model, self.p, dp, coupling=self.COUPLING)

        remainder_1, remainder_2 = remainders
        order_1, order_2 = orders
        (grad_uva, grad_solid, grad_fluid, grad_times), grad_step, grad_step_fd = grads

        self.plot_taylor_convergence(grad_step, hs, gs)
        self.plot_grad_uva(self.model, grad_uva)
        plt.show()

        print(order_1)
        print(order_2)

        return (order_1, order_2)

    def test_u0(self):
        save_path = f'out/linesearch_periodickelvinvoigt_u0_{self.COUPLING}.h5'

        hs = np.concatenate(([0], 2.0**(np.arange(-6, 0))), axis=0)

        dp = self.p.copy()
        dp.vector[:] = 0
        du0 = dfn.Function(self.model.solid.vector_fspace).vector()
        du0[:] = 1e-5
        self.model.solid.bc_base.apply(du0)

        dp['u0'].reshape(-1)[:] = du0

        order_1, order_2 = self.get_taylor_order(save_path, hs, dp)
        # self.assertTrue(np.all(order_1 == 1.0))
        # self.assertTrue(np.all(order_2 == 2.0))

    def test_v0(self):
        save_path = f'out/linesearch_periodickelvinvoigt_v0_{self.COUPLING}.h5'

        hs = np.concatenate(([0], 2.0**(np.arange(-6, 0))), axis=0)

        dp = self.p.copy()
        dp.vector[:] = 0
        dv0 = dfn.Function(self.model.solid.vector_fspace).vector()
        dv0[:] = 1e-1
        self.model.solid.bc_base.apply(dv0)

        dp['v0'].reshape(-1)[:] = dv0

        order_1, order_2 = self.get_taylor_order(save_path, hs, dp)
        # self.assertTrue(np.all(order_1 == 1.0))
        # self.assertTrue(np.all(order_2 == 2.0))

    def test_period(self):
        save_path = f'out/linesearch_periodickelvinvoigt_period_{self.COUPLING}.h5'

        hs = np.concatenate(([0], 2.0**(np.arange(-6, 0))), axis=0)

        dp = self.p.copy()
        dp.vector[:] = 0
        dp['period'][()] = 1.0e-4

        order_1, order_2 = self.get_taylor_order(save_path, hs, dp)
        # self.assertTrue(np.all(order_1 == 1.0))
        # self.assertTrue(np.all(order_2 == 2.0))

def grad_and_taylor_order(filepath, functional, hs, model,
                          duva=(0, 0, 0), dsolid_props=None, dfluid_props=None, dtimes=None,
                          coupling='explicit'):
    """
    """
    ## Calculate the functional value at each point along the line search
    total_runtime = 0
    functionals = list()
    for n, h in enumerate(hs):
        with sf.StateFile(model, filepath, group=f'{n}', mode='r') as f:
            runtime_start = perf_counter()
            val = functional(f)
            functionals.append(val)
            runtime_end = perf_counter()
            total_runtime += runtime_end-runtime_start
            print(f"f = {val:.2e}")
    print(f"Runtime {total_runtime:.2f} seconds")

    functionals = np.array(functionals)

    ## Calculate the gradient via the adjoint equations at the 0th point
    print("\nComputing Gradient via adjoint calculation")

    grad_uva, grad_solid, grad_fluid, grad_times = None, None, None, None
    runtime_start = perf_counter()
    with sf.StateFile(model, filepath, group='0', mode='r') as f:
        _, grad_uva, grad_solid, grad_fluid, grad_times = adjoint(model, f, functional, coupling)
    runtime_end = perf_counter()
    print(f"Runtime {runtime_end-runtime_start:.2f} seconds")

    ## Project the gradient along the step direction
    directions = [duva, dsolid_props, dfluid_props, dtimes]
    gradients = [grad_uva, grad_solid, grad_fluid, grad_times]
    grad_step = 0

    for ii in range(3):
        # Make sure the direction is a vector since sometimes it's a float = 0, representing no
        # displacement
        _direction = np.zeros(grad_uva[ii].size())
        _direction[:] = duva[ii]
        grad_step += np.dot(grad_uva[ii], _direction)

    if dfluid_props is not None:
        grad_step += np.dot(grad_fluid.vector, dfluid_props.vector)

    if dsolid_props is not None:
        grad_step += np.dot(grad_solid.vector, dsolid_props.vector)

    if dtimes is not None:
        grad_step += np.dot(grad_times, dtimes)

    grad_step_fd = (functionals[1:] - functionals[0])/hs[1:]

    ## Compare the adjoint and finite difference projected gradients
    remainder_1 = np.abs(functionals[1:] - functionals[0])
    remainder_2 = np.abs(functionals[1:] - functionals[0] - hs[1:]*grad_step)

    order_1 = np.log(remainder_1[1:]/remainder_1[:-1]) / np.log(2)
    order_2 = np.log(remainder_2[1:]/remainder_2[:-1]) / np.log(2)

    return functionals, (remainder_1, remainder_2), (order_1, order_2), \
           ((grad_uva, grad_solid, grad_fluid, grad_times), grad_step, grad_step_fd)

def grad_and_taylor_order_p(filepath, functional, hs, model, p, dp, coupling='explicit'):
    """
    """
    ## Calculate the functional value at each point along the line search
    total_runtime = 0
    functionals = list()
    for n, h in enumerate(hs):
        with sf.StateFile(model, filepath, group=f'{n}', mode='r') as f:
            runtime_start = perf_counter()
            val = functional(f)
            functionals.append(val)
            runtime_end = perf_counter()
            total_runtime += runtime_end-runtime_start
            print(f"f = {val:.2e}")
    print(f"Runtime {total_runtime:.2f} seconds")

    functionals = np.array(functionals)

    ## Calculate the gradient via the adjoint equations at the 0th point
    print("\nComputing Gradient via adjoint calculation")

    grad = None
    runtime_start = perf_counter()
    with sf.StateFile(model, filepath, group='0', mode='r') as f:
        _, grad_uva, grad_solid, grad_fluid, grad_times = adjoint(model, f, functional, coupling=coupling)
    runtime_end = perf_counter()
    print(f"Runtime {runtime_end-runtime_start:.2f} seconds")

    ## Project the gradient along the step direction
    grad_p = p.dconvert(grad_uva, grad_solid, grad_fluid, grad_times)
    grad_step = np.dot(grad_p.vector, dp.vector)
    grad_step_fd = (functionals[1:] - functionals[0])/hs[1:]

    ## Compare the adjoint and finite difference projected gradients
    remainder_1 = np.abs(functionals[1:] - functionals[0])
    remainder_2 = np.abs(functionals[1:] - functionals[0] - hs[1:]*grad_step)

    order_1 = np.log(remainder_1[1:]/remainder_1[:-1]) / np.log(2)
    order_2 = np.log(remainder_2[1:]/remainder_2[:-1]) / np.log(2)

    return functionals, (remainder_1, remainder_2), (order_1, order_2), \
           ((grad_uva, grad_solid, grad_fluid, grad_times), grad_step, grad_step_fd)

class TestResidualJacobian(unittest.TestCase):
    """
    Tests if jacobians of residuals are correct
    """
    def test_df1_dstate1_implicit(self):
        np.random.seed(0)

        # Set up the parameters and objects needed for the iteration
        model, sp, fp = get_starting_kelvinvoigt_model()
        model.set_solid_props(sp)
        model.set_fluid_props(fp)
        uva0 = tuple([dfn.Function(model.solid.vector_fspace).vector() for i in range(3)])
        model.set_ini_solid_state(*uva0)
        q0, p0, _ = model.get_pressure()
        qp0 = (q0, p0)
        dt = 1e-3

        # Set state change directions du1, dp1
        du1 = dfn.Function(model.solid.vector_fspace).vector()
        # _du1 = np.array(du1[:])
        # _du1[:20] = 1e-5
        # du1[:] = _du1
        du1[:] = np.random.rand(du1.size())*1e-5
        model.solid.bc_base.apply(du1)

        dp1 = np.random.rand(qp0[1].size)
        dp1 = np.ones(qp0[1].size)*1e-4

        # Find the state at 1 so you can linearize about this point
        uva1, qp1, _ = implicit_increment(model, uva0, qp0, dt)

        # Set up the block matrix dFup_dup_adj and dFup_dup
        model.set_iter_params(uva0=uva0, dt=dt, uva1=uva1, qp1=qp1)
        dfu_du_adj = model.assem_df1_du1_adj()
        model.solid.bc_base.apply(dfu_du_adj)
        dfu_du_adj = dfn.as_backend_type(dfu_du_adj).mat()

        dfu_dp_adj = dfn.as_backend_type(dfn.assemble(model.solid.forms['form.bi.df1_dpressure_adj'])).mat()
        solid_dofs, fluid_dofs = model.get_fsi_scalar_dofs()
        dfu_dp_adj = linalg.reorder_mat_rows(dfu_dp_adj, solid_dofs, fluid_dofs, fluid_dofs.size)

        _, dp_du = model.get_flow_sensitivity_solid_ord(adjoint=True)
        dfp_du_adj = 0.0 - dp_du

        blocks = [[dfu_du_adj, dfp_du_adj],
                  [dfu_dp_adj,        1.0]]
        dfup_dup_adj = linalg.form_block_matrix(blocks)

        # Calculate the linear change in residual using the residual jacobian
        dup, dres_jac = dfup_dup_adj.getVecs()
        dup[:du1.size()] = du1
        dup[du1.size():] = dp1
        dfup_dup_adj.multTranspose(dup, dres_jac)

        dres_u, dres_p = du1.copy(), dp1.copy()
        dres_u[:] = dres_jac[:du1.size()]
        dres_p[:] = dres_jac[du1.size():]
        model.solid.bc_base.apply(dres_u)

        # Calculate residuals at up1
        model.set_iter_params(uva0=uva0, uva1=uva1[0], qp1=qp1)
        res_u0 = dfn.assemble(model.solid.f1)
        model.solid.bc_base.apply(res_u0)

        model.set_ini_solid_state(uva1[0], 0, 0)
        res_p0 = qp1[1] - model.get_pressure()[1]

        # Calculate residuals at up1 + dup1
        model.set_iter_params(uva0=uva0, uva1=uva1[0]+du1, qp1=(qp1[0], qp1[1]+dp1))
        res_u1 = dfn.assemble(model.solid.f1)
        model.solid.bc_base.apply(res_u1)

        model.set_ini_solid_state(uva1[0]+du1, 0, 0)
        res_p1 = qp1[1]+dp1 - model.get_pressure()[1]

        # Calculate the actual residual change
        dres_u_true = dfn.as_backend_type(res_u1-res_u0).vec()
        dres_p_true = res_p1-res_p0

        # Compare and
        error_u = dres_u.vec() - dres_u_true
        error_p = dres_p - dres_p_true

if __name__ == '__main__':
    # unittest.main()

    test = TestBasicGradient()
    test.setUp()
    test.test_emod()
    test.test_u0()
    test.test_v0()
    test.test_a0()
    test.test_times()

    # test = TestBasicGradientSingleStep()
    # test.setUp()
    # test.test_emod()
    # test.test_u0()
    # test.test_v0()
    # test.test_a0()
    # test.test_times()

    # test = TestPeriodicKelvinVoigtGradient()
    # test.setUp()
    # test.test_u0()
    # test.test_v0()
    # test.test_period()
