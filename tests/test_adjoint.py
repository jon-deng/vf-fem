"""
This module tests the adjoint method by comparing with gradients computed from finite differences

Specifically, the Taylor remainder convergence test is used [1, 2].

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

from femvf.models import (
    load_fsi_model, load_fsai_model,
    Rayleigh, KelvinVoigt, Bernoulli, WRAnalog)

from femvf.forward import integrate
from femvf.adjoint import adjoint
from femvf.constants import PASCAL_TO_CGS
from femvf.parameters import parameterization
from femvf.functionals import solid as fsolid, fluid as ffluid, acoustic as facous, math as fmath
from femvf import linalg

from femvf.utils import line_search, line_search_p, functionals_on_line_search

# from optvf import functionals as extra_funcs

dfn.set_log_level(30)
np.random.seed(123)

def get_starting_rayleigh_model(coupling='explicit'):
    ## Set the mesh to be used and initialize the forward model
    mesh_dir = '../meshes'
    mesh_base_filename = 'geometry2'
    mesh_base_filename = 'M5-3layers-refined'

    mesh_path = os.path.join(mesh_dir, mesh_base_filename + '.xml')
    model = load_fsi_model(mesh_path, None, Solid=Rayleigh, Fluid=Bernoulli, coupling=coupling)

    ## Set the fluid/solid parameters
    emod = 2.5e3 * PASCAL_TO_CGS

    k_coll = 1e11
    y_gap = 0.02
    y_coll_offset = 0.01
    alpha, k, sigma = -3000, 50, 0.002

    fluid_props = model.fluid.get_properties_vec()
    fluid_props['y_midline'][()] = np.max(model.solid.mesh.coordinates()[..., 1]) + y_gap
    fluid_props['p_sub'][()] = 1000 * PASCAL_TO_CGS
    fluid_props['alpha'][()] = alpha
    fluid_props['k'][()] = k
    fluid_props['sigma'][()] = sigma

    solid_props = model.solid.get_properties_vec()
    solid_props['emod'][:] = emod
    solid_props['rayleigh_m'][()] = 0.0
    solid_props['rayleigh_k'][()] = 3e-4
    solid_props['k_collision'][()] = k_coll
    solid_props['y_collision'][()] = fluid_props['y_midline'][()] - y_coll_offset

    return model, linalg.concatenate(solid_props, fluid_props)

def get_starting_kelvinvoigt_model(coupling='explicit'):
    ## Set the mesh to be used and initialize the forward model
    mesh_dir = '../meshes'
    mesh_base_filename = 'geometry2'
    mesh_base_filename = 'M5-3layers-medial-surface-refinement'

    mesh_path = os.path.join(mesh_dir, mesh_base_filename + '.xml')
    model = load_fsi_model(mesh_path, None, Solid=KelvinVoigt, Fluid=Bernoulli, coupling=coupling)

    ## Set the fluid/solid parameters
    emod = 6e3 * PASCAL_TO_CGS

    k_coll = 1e13
    y_gap = 0.02
    y_gap = 0.1
    y_coll_offset = 0.01

    y_gap = 0.01
    y_coll_offset = 0.0025
    alpha, k, sigma = -3000, 50, 0.002

    fluid_props = model.fluid.get_properties_vec()
    fluid_props['y_midline'][()] = np.max(model.solid.mesh.coordinates()[..., 1]) + y_gap
    fluid_props['alpha'][()] = alpha
    fluid_props['k'][()] = k
    fluid_props['sigma'][()] = sigma
    fluid_props['y_gap_min'][()] = y_coll_offset
    # fluid_props['y_gap_min'][()] = -10000

    solid_props = model.solid.get_properties_vec()
    solid_props['emod'][:] = emod
    solid_props['eta'][()] = 3.0
    solid_props['k_collision'][()] = k_coll
    # solid_props['y_collision'][()] = fluid_props['y_midline'][()] - y_coll_offset
    solid_props['y_collision'][()] = fluid_props['y_midline'] - y_coll_offset

    return model, linalg.concatenate(solid_props, fluid_props)

def get_starting_fsai_model(coupling='explicit'):
    mesh_dir = '../meshes'
    mesh_base_filename = 'geometry2'
    mesh_base_filename = 'M5-3layers-refined'
    mesh_path = os.path.join(mesh_dir, mesh_base_filename + '.xml')

    ## Configure the model and its parameters
    acoustic = WRAnalog(44)
    model = load_fsai_model(mesh_path, None, acoustic, Solid=Rayleigh, Fluid=Bernoulli,
                            coupling='explicit')

    # Set the properties
    y_gap = 0.01
    alpha, k, sigma = -3000, 50, 0.002

    fl_props = model.fluid.get_properties_vec(set_default=True)
    fl_props['y_midline'][()] = np.max(model.solid.mesh.coordinates()[..., 1]) + y_gap
    fl_props['alpha'][()] = alpha
    fl_props['k'][()] = k
    fl_props['sigma'][()] = sigma

    sl_props = model.solid.get_properties_vec(set_default=True)
    xy = model.solid.scalar_fspace.tabulate_dof_coordinates()
    x = xy[:, 0]
    y = xy[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    sl_props['emod'][:] = 1/2*5.0e3*PASCAL_TO_CGS*((x-x_min)/(x_max-x_min) + (y-y_min)/(y_max-y_min)) + 2.5e3*PASCAL_TO_CGS
    sl_props['rayleigh_m'][()] = 0
    sl_props['rayleigh_k'][()] = 4e-3
    sl_props['k_collision'][()] = 1e11
    sl_props['y_collision'][()] = fl_props['y_midline'] - y_gap*1/2

    ac_props = model.acoustic.get_properties_vec(set_default=True)
    ac_props['area'][:] = 4.0
    ac_props['length'][:] = 12.0
    ac_props['soundspeed'][:] = 340*100

    props = linalg.concatenate(sl_props, fl_props, ac_props)
    
    return model, props


def taylor_order(f0, hs, fs, 
                 gstate, gcontrols, gprops, gtimes,
                 dstate, dcontrols, dprops, dtimes):
    
    ## Project the gradient along the step direction
    directions = linalg.concatenate(dstate, dprops, dtimes)
    gradients = linalg.concatenate(gstate, gprops, gtimes)

    grad_step = linalg.dot(directions, gradients)
    grad_step_fd = (fs - f0)/hs
    # breakpoint()

    ## Compare the adjoint and finite difference gradients projected on the step direction
    remainder_1 = np.abs(fs - f0)
    remainder_2 = np.abs((fs - f0) - hs*grad_step)

    order_1 = np.log(remainder_1[1:]/remainder_1[:-1]) / np.log(2)
    order_2 = np.log(remainder_2[1:]/remainder_2[:-1]) / np.log(2)
    return (order_1, order_2), (grad_step, grad_step_fd)

class TaylorTest(unittest.TestCase):

    def compute_baseline(self):
        ## Compute the baseline forward simulation and functional/gradient at the baseline (via adjoint)
        base_path = f"out/{self.CASE_NAME}-0.h5"
        if self.OVERWRITE_LSEARCH or not os.path.isfile(base_path):
            if os.path.isfile(base_path):
                os.remove(base_path)
            integrate(self.model, self.state0, self.controls, self.props, self.times,
                      h5file=base_path)

        print("Computing gradient via adjoint")
        t_start = perf_counter()

        f0, grads = None, None
        with sf.StateFile(self.model, base_path, mode='r') as f:
            f0, *grads = adjoint(self.model, f, self.functional)

        print(f"Duration {perf_counter()-t_start:.4f} s")
        
        return f0, grads

    def get_taylor_order(self, lsearch_fname, hs,
                         dstate=None, dcontrols=None, dprops=None, dtimes=None):
        """
        Runs a line search of simulations along a specified direction
        """
        ## Set a zero search direction if one isn't specified
        if dstate is None:
            dstate = self.model.get_state_vec()
            dstate.set(0.0)

        if dcontrols is None:
            dcontrols = [self.model.get_control_vec()]
            dcontrols[0].set(0.0)

        if dprops is None:
            dprops = self.model.get_properties_vec()
            dprops.set(0.0)

        if dtimes is None:
            dtimes = self.times.copy()
            dtimes.set(0.0)

        ## Conduct a line search along the specified direction
        if self.OVERWRITE_LSEARCH or not os.path.exists(lsearch_fname):
            if os.path.exists(lsearch_fname):
                os.remove(lsearch_fname)
            lsearch_fname = line_search(
                hs, self.model, self.state0, self.controls, self.props, self.times,
                dstate, dcontrols, dprops, dtimes, filepath=lsearch_fname)
        else:
            print("Using existing files")

        fs = functionals_on_line_search(hs, self.functional, self.model, lsearch_fname)
        assert not np.all(fs == self.f0) # Make sure that f changes along the step direction

        ## Compute the taylor convergence order
        gstate, gcontrols, gprops, gtimes = self.grads
        (order_1, order_2), (grad_step, grad_step_fd) = taylor_order(
            self.f0, hs, fs,
            gstate, gcontrols, gprops, gtimes,
            dstate, dcontrols, dprops, dtimes)

        # self.plot_taylor_convergence(grad_step, hs, gs)
        # self.plot_grad_uva(self.model, grad_uva)
        # plt.show()

        print('1st order Taylor', order_1)
        print('2nd order Taylor', order_2)
        
        breakpoint()
        return (order_1, order_2)

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

class TestBasicGradient(TaylorTest):
    COUPLING = 'explicit'
    OVERWRITE_LSEARCH = True
    FUNCTIONAL = fsolid.FinalDisplacementNorm
    # FUNCTIONAL = fsolid.FinalVelocityNorm
    FUNCTIONAL = ffluid.FinalPressureNorm

    def setUp(self):
        """
        Set the model, parameters, functional to test, and the gradient/forward model at the
        baseline parameter set

        This should set the starting point of the line search; all the parameters needed to solve
        a single simulation are given by this set up.
        """
        self.CASE_NAME = 'singleperiod'

        ## Load the model and set baseline parameters (point the model is linearized around)
        # self.model, self.props = get_starting_kelvinvoigt_model(self.COUPLING)
        self.model, self.props = get_starting_fsai_model(self.COUPLING)

        t_start, t_final = 0, 0.01
        times_meas = np.linspace(t_start, t_final, 32)
        self.times = linalg.BlockVec((times_meas,), ('times',))

        self.state0 = self.model.get_state_vec()
        self.state0['v'][:] = 1e-3
        self.model.solid.bc_base.apply(self.state0['v'])

        control = self.model.get_control_vec()
        control['psub'][:] = 800 * PASCAL_TO_CGS
        # control['psup'][:] = 0.0 * PASCAL_TO_CGS
        self.controls = [control]

        ## Specify the functional to test the gradient with
        # self.functional = fmath.add(basic.FinalDisplacementNorm(self.model),
        #                             basic.FinalVelocityNorm(self.model))

        self.functional = self.FUNCTIONAL(self.model)

        func_power_cons = fmath.sub(ffluid.AcousticPower(self.model), fmath.Scalar(self.model, 0.0))
        func_periodic_cons = fmath.mul(fmath.Scalar(self.model, 1.0), fsolid.PeriodicError(self.model))
        func = fmath.add(func_power_cons, func_periodic_cons)
        self.functional = ffluid.AcousticPower(self.model)
        self.functional = facous.AcousticPower(self.model)

        # self.functional = fmath.add(
        #     fmath.mul(1e5+0.00000001, basic.FinalDisplacementNorm(self.model)),
        #                   basic.FinalVelocityNorm(self.model))

        # self.functional = self.FUNCTIONAL(self.model)
        # self.functional.constants['n_start'] = 50

        ## Compute the baseline forward simulation and functional/gradient at the baseline (via adjoint)
        self.f0, self.grads = self.compute_baseline()

    def test_emod(self):
        save_path = f'out/linesearch_emod_{self.COUPLING}.h5'
        hs = 2.0**(np.arange(2, 9)-10)
        step_size = 0.5e0 * PASCAL_TO_CGS

        dprops = self.props.copy()
        dprops.set(0.0)
        dprops['emod'][:] = 1.0*step_size*10

        order_1, order_2 = self.get_taylor_order(save_path, hs, dprops=dprops)
        # self.assertTrue(np.all(np.isclose(order_1, 1.0)))
        # self.assertTrue(np.all(np.isclose(order_2, 2.0)))

    def test_u0(self):
        save_path = f'out/linesearch_u0_{self.COUPLING}.h5'
        hs = 2.0**(np.arange(2, 9)-16)

        ## Pick a step direction
        # Increment `u` linearly as a function of x and y in the y direction
        # step_dir = np.zeros(xy.size)
        # step_dir[:-1:2] = -(y-y.min()) / (y.max()-y.min())
        # step_dir[1::2] = -(y-y.min()) / (y.max()-y.min())

        # Increment `u` only along the pressure surface
        surface_dofs = self.model.solid.vert_to_vdof.reshape(-1, 2)[self.model.fsi_verts]
        xy_surface = self.model.solid.mesh.coordinates()[self.model.fsi_verts, :]
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
        dstate = self.model.get_state_vec()
        dstate['u'][:] = step_dir*0.01

        order_1, order_2 = self.get_taylor_order(save_path, hs, dstate=dstate)
        # self.assertTrue(np.all(np.isclose(order_1, 1.0)))
        # self.assertTrue(np.all(np.isclose(order_2, 2.0)))

    def test_v0(self):
        save_path = f'out/linesearch_v0_{self.COUPLING}.h5'
        hs = 2.0**(np.arange(2, 9)-9)

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
        dstate = self.model.get_state_vec()
        dstate['v'][:] = step_dir*0.1

        order_1, order_2 = self.get_taylor_order(save_path, hs, dstate=dstate)
        # self.assertTrue(np.all(np.isclose(order_1, 1.0)))
        # self.assertTrue(np.all(np.isclose(order_2, 2.0)))

    def test_a0(self):
        save_path = f'out/linesearch_a0_{self.COUPLING}.h5'
        hs = 2.0**(np.arange(2, 9))

        xy = self.model.get_ref_config()[dfn.dof_to_vertex_map(self.model.solid.scalar_fspace)]
        y = xy[:, 1]

        # Set the step direction as a linear (de)increase in x and y displacement in the y direction
        step_dir = dfn.Function(self.model.solid.vector_fspace).vector()
        _step_dir = step_dir[:].copy()
        _step_dir[:-1:2] = -(y-y.min()) / (y.max()-y.min())
        _step_dir[1::2] = -(y-y.min()) / (y.max()-y.min())
        step_dir[:] = _step_dir

        self.model.solid.bc_base.apply(step_dir)
        dstate = self.model.get_state_vec()
        dstate['a'][:] = step_dir

        order_1, order_2 = self.get_taylor_order(save_path, hs, dstate=dstate)
        # self.assertTrue(np.all(np.isclose(order_1, 1.0)))
        # self.assertTrue(np.all(np.isclose(order_2, 2.0)))

    def test_times(self):
        save_path = f'out/linesearch_dt_{self.COUPLING}.h5'
        hs = 2.0**(np.arange(2, 9))

        dtimes = self.times.copy()
        dtimes['times'][1:] = np.arange(1, dtimes['times'].size)*1e-9

        
        order_1, order_2 = self.get_taylor_order(save_path, hs, dtimes=dtimes)
        # self.assertTrue(np.all(np.isclose(order_1, 1.0)))
        # self.assertTrue(np.all(np.isclose(order_2, 2.0)))

class TestBasicGradientSingleStep(TaylorTest):
    COUPLING = 'explicit'
    # COUPLING = 'implicit'
    OVERWRITE_LSEARCH = True
    # FUNCTIONAL = fsolid.FinalVelocityNorm
    # FUNCTIONAL = fsolid.FinalDisplacementNorm
    # FUNCTIONAL = ffluid.FinalPressureNorm
    FUNCTIONAL = ffluid.FinalFlowRateNorm

    def setUp(self):
        """
        Set the model and baseline simulation parameters

        This should set the starting point of the line search; all the parameters needed to solve
        a single simulation are given by this set up.
        """
        self.CASE_NAME = 'singlestep'

        ## parameter set
        # self.model, self.props = get_starting_kelvinvoigt_model(self.COUPLING)
        self.model, self.props = get_starting_fsai_model(self.COUPLING)
        self.model.set_properties(self.props)

        t_start, t_final = 0, 0.001
        times_meas = np.linspace(t_start, t_final, 2)

        # t_start, t_final = 0, 0.002
        # times_meas = np.linspace(t_start, t_final, 3)
        self.times = linalg.BlockVec((times_meas,), ('times',))

        control = self.model.get_control_vec()
        uva0 = self.model.solid.get_state_vec()
        uva0['v'][:] = 0.0
        self.model.solid.bc_base.apply(uva0['v'])
        self.model.set_ini_solid_state(uva0)

        control['psub'][:] = 800 * PASCAL_TO_CGS
        # control['psup'][:] = 0.0 * PASCAL_TO_CGS
        self.controls = [control]
        self.model.set_control(control)

        self.model.set_fin_solid_state(uva0)
        qp0, _ = self.model.fluid.solve_qp1()

        self.state0 = self.model.get_state_vec()
        self.state0[3:5] = qp0
    
        # Step sizes and scale factor
        self.functional = self.FUNCTIONAL(self.model)

        ## Compute the baseline forward simulation and functional/gradient at the baseline (via adjoint)
        self.f0, self.grads = self.compute_baseline()

    def test_emod(self):
        save_path = f'out/linesearch_emod_{self.COUPLING}_{self.CASE_NAME}.h5'
        hs = 2.0**(np.arange(7)-5)
        step_size = 0.5e0 * PASCAL_TO_CGS

        dprops = self.props.copy()
        dprops.set(0.0)
        dprops['emod'][:] = 1.0*step_size/5

        order_1, order_2 = self.get_taylor_order(save_path, hs, dprops=dprops)
        # self.assertTrue(np.all(np.isclose(order_1, 1.0)))
        # self.assertTrue(np.all(np.isclose(order_2, 2.0)))

    def test_u0(self):
        save_path = f'out/linesearch_u0_{self.COUPLING}_{self.CASE_NAME}.h5'
        hs = 2.0**(np.arange(2, 9)-12)

        ## Pick a step direction
        # Increment `u` linearly as a function of x and y in the y direction
        # step_dir = np.zeros(xy.size)
        # step_dir[:-1:2] = -(y-y.min()) / (y.max()-y.min())
        # step_dir[1::2] = -(y-y.min()) / (y.max()-y.min())

        # Increment `u` only along the pressure surface
        surface_dofs = self.model.solid.vert_to_vdof.reshape(-1, 2)[self.model.fsi_verts]
        xy_surface = self.model.solid.mesh.coordinates()[self.model.fsi_verts, :]
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
        dstate = self.model.get_state_vec()

        dstate['u'][:] = step_dir*0.00005
        dstate['v'][:] = 0.0
        dstate['a'][:] = 0.0

        order_1, order_2 = self.get_taylor_order(save_path, hs, dstate=dstate)
        # self.assertTrue(np.all(np.isclose(order_1, 1.0)))
        # self.assertTrue(np.all(np.isclose(order_2, 2.0)))

    def test_v0(self):
        save_path = f'out/linesearch_v0_{self.COUPLING}_{self.CASE_NAME}.h5'
        hs = 2.0**(np.arange(2, 9)-6)

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
        dstate = self.model.get_state_vec()
        dstate['u'][:] = 0
        dstate['v'][:] = step_dir*1e-5
        dstate['a'][:] = 0.0

        order_1, order_2 = self.get_taylor_order(save_path, hs, dstate=dstate)
        # self.assertTrue(np.all(np.isclose(order_1, 1.0)))
        # self.assertTrue(np.all(np.isclose(order_2, 2.0)))

    def test_a0(self):
        save_path = f'out/linesearch_a0_{self.COUPLING}_{self.CASE_NAME}.h5'
        hs = 2.0**(np.arange(2, 9)+5)

        xy = self.model.get_ref_config()[dfn.dof_to_vertex_map(self.model.solid.scalar_fspace)]
        y = xy[:, 1]

        # Set the step direction as a linear (de)increase in x and y displacement in the y direction
        step_dir = dfn.Function(self.model.solid.vector_fspace).vector()
        _step_dir = step_dir[:].copy()
        _step_dir[:-1:2] = -(y-y.min()) / (y.max()-y.min())
        _step_dir[1::2] = -(y-y.min()) / (y.max()-y.min())
        step_dir[:] = _step_dir

        self.model.solid.bc_base.apply(step_dir)
        dstate = self.model.get_state_vec()
        dstate['u'][:] = 0.0
        dstate['v'][:] = 0.0
        dstate['a'][:] = step_dir*1e-5

        order_1, order_2 = self.get_taylor_order(save_path, hs, dstate=dstate)
        # self.assertTrue(np.all(np.isclose(order_1, 1.0)))
        # self.assertTrue(np.all(np.isclose(order_2, 2.0)))

    def test_times(self):
        save_path = f'out/linesearch_dt_{self.COUPLING}.h5'
        hs = 2.0**(np.arange(2, 9)+5)

        dtimes = self.times.copy()
        dtimes['times'][:] = np.arange(1, dtimes['times'].size)*1e-11

        
        order_1, order_2 = self.get_taylor_order(save_path, hs, dtimes=dtimes)

if __name__ == '__main__':
    # unittest.main()

    test = TestBasicGradient()
    test.setUp()
    test.test_emod()
    # test.test_u0()
    # test.test_v0()
    # test.test_a0()
    # test.test_times()

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
