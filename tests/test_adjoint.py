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

import sys
import os
from os import path
from time import perf_counter
import pickle

import unittest

# from math import round
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import dolfin as dfn

sys.path.append('../')
from femvf import meshutils, statefile as sf
from femvf.forward import forward
from femvf.adjoint import adjoint
from femvf.model import ForwardModel
from femvf.solids import Rayleigh, KelvinVoigt
from femvf.fluids import Bernoulli
from femvf.constants import PASCAL_TO_CGS
from femvf.parameters.properties import SolidProperties, FluidProperties
from femvf.functionals import basic as funcs

# sys.path.append(path.expanduser('~/lib/vf-optimization'))
# from optvf import functionals as extra_funcs

dfn.set_log_level(30)
np.random.seed(123)

def get_starting_rayleigh_model():
    ## Set the mesh to be used and initialize the forward model
    mesh_dir = '../meshes'
    mesh_base_filename = 'geometry2'
    facet_labels = {'pressure': 1, 'fixed': 3}

    mesh_base_filename = 'M5-3layers-refined'
    facet_labels = {'pressure': 5, 'fixed': 4}

    mesh_path = os.path.join(mesh_dir, mesh_base_filename + '.xml')
    cell_labels = {}

    mesh, facet_func, cell_func = meshutils.load_fenics_mesh(mesh_path)
    solid = Rayleigh(mesh, facet_func, facet_labels, cell_func, cell_labels)

    xfluid, yfluid = meshutils.streamwise1dmesh_from_edges(mesh, facet_func, facet_labels['pressure'])
    fluid = Bernoulli(xfluid, yfluid)
    model = ForwardModel(solid, fluid)

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
    facet_labels = {'pressure': 1, 'fixed': 3}

    # mesh_base_filename = 'M5-3layers'
    mesh_base_filename = 'M5-3layers-medial-surface-refinement'
    facet_labels = {'pressure': 5, 'fixed': 4}

    mesh_path = os.path.join(mesh_dir, mesh_base_filename + '.xml')
    cell_labels = {}

    mesh, facet_func, cell_func = meshutils.load_fenics_mesh(mesh_path, facet_labels, cell_labels)
    solid = KelvinVoigt(mesh, facet_func, facet_labels, cell_func, cell_labels)

    xfluid, yfluid = meshutils.streamwise1dmesh_from_edges(mesh, facet_func, facet_labels['pressure'])
    fluid = Bernoulli(xfluid, yfluid)
    model = ForwardModel(solid, fluid)

    ## Set the fluid/solid parameters
    emod = 6e3 * PASCAL_TO_CGS

    k_coll = 1e13
    y_gap = 0.02
    y_gap = 0.1
    y_coll_offset = 0.01

    y_gap = 0.01
    y_coll_offset = 0.0025
    alpha, k, sigma = -3000, 50, 0.002

    fluid_props = fluid.get_properties()
    fluid_props['y_midline'][()] = np.max(model.solid.mesh.coordinates()[..., 1]) + y_gap
    fluid_props['p_sub'][()] = 800 * PASCAL_TO_CGS
    fluid_props['alpha'][()] = alpha
    fluid_props['k'][()] = k
    fluid_props['sigma'][()] = sigma

    solid_props = solid.get_properties()
    solid_props['emod'][:] = emod
    solid_props['eta'][()] = 3.0
    solid_props['k_collision'][()] = k_coll
    # solid_props['y_collision'][()] = fluid_props['y_midline'][()] - y_coll_offset
    solid_props['y_collision'][()] = fluid_props['y_midline'] - y_coll_offset

    return model, solid_props, fluid_props

class TaylorTest(unittest.TestCase):
    def test_adjoint(self):
        hs = self.hs
        step_dir = self.step_dir
        model = self.model
        save_path = self.save_path

        run_info = None
        with open(path.splitext(save_path)[0]+".pickle", 'rb') as f:
            run_info = pickle.load(f)

        ## Select the functional you want to test
        # fkwargs = {}
        Functional = funcs.TransferWorkbyDisplacementIncrement
        # Functional = funcs.TransferWorkbyVelocity

        # Functional = funcs.FinalSurfacePower
        # Functional = funcs.FinalSurfaceDisplacementIncrementNorm
        # Functional = funcs.FinalSurfaceDisplacementNorm
        # Functional = funcs.FinalSurfacePressureNorm
        # Functional = funcs.PeriodicError
        # Functional = funcs.FinalDisplacementNorm
        # Functional = funcs.FinalVelocityNorm
        # Functional = funcs.DisplacementNorm
        # Functional = funcs.VelocityNorm
        # Functional = funcs.StrainEnergy

        # fkwargs = {'tukey_alpha': 0.05, 'f0':100, 'df':50}
        # Functional = extra_funcs.AcousticPower
        # Functional = extra_funcs.AcousticEfficiency
        # Functional = extra_funcs.F0WeightedAcousticPower

        functional = Functional(model)
        functional.constants['n_start'] = 1

        ## Solve the model for each point along the 'direction' of parameter changes
        print(f"\nSolving models along parameter search direction")

        total_runtime = 0
        functionals = list()
        for n, h in enumerate(hs):
            with sf.StateFile(model, save_path, group=f'{n}', mode='r') as f:
                runtime_start = perf_counter()
                val = functional(f)
                functionals.append(val)
                runtime_end = perf_counter()
                total_runtime += runtime_end-runtime_start
                print(f"f = {val:.2e}")
        print(f"Runtime {total_runtime:.2f} seconds")

        functionals = np.array(functionals)

        ## Calculate the gradient using via the adjoint equations at the 0th point
        print("\nComputing Gradient via adjoint calculation")

        runtime_start = perf_counter()
        info = None
        gradient_ad = None
        with sf.StateFile(model, save_path, group='0', mode='r') as f:
            _, gradient, _ = adjoint(model, f, functional)

            gradient_ad = gradient[self.parameter]
            # breakpoint()
        runtime_end = perf_counter()

        print(f"Runtime {runtime_end-runtime_start:.2f} seconds")

        ## Compare the adjoint and finite difference projected gradients
        taylor_remainder_1 = np.abs(functionals[1:] - functionals[0])
        taylor_remainder_2 = np.abs(functionals[1:] - functionals[0] - hs[1:]*np.dot(gradient_ad, step_dir))

        order_1 = np.log(taylor_remainder_1[1:]/taylor_remainder_1[:-1]) / np.log(2)
        order_2 = np.log(taylor_remainder_2[1:]/taylor_remainder_2[:-1]) / np.log(2)

        print("\nSteps:", hs[1:])

        print("\n1st order taylor remainders: \n", taylor_remainder_1)
        print("Numerical order: \n", order_1)

        print("\n2nd order taylor remainders: \n", taylor_remainder_2)
        print("Numerical order: \n", order_2)

        print("\n||dg/dp|| = ", np.linalg.norm(gradient_ad, ord=2))
        print("dg/dp * step_dir = ", np.dot(gradient_ad, step_dir))
        print("FD approximation of dg/dp * step_dir = ", (functionals[1:] - functionals[0])/hs[1:])

        fig, ax = self.plot_taylor_test(gradient_ad, step_dir, hs, functionals)
        fig.savefig(f'Convergence_{self.case_postfix}.png')

        # Plot the glottal width vs time of the simulation at zero step size
        fig, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)

        ax = axs[0]
        ax.plot(run_info['time'], run_info['glottal_width'])
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Glottal width [cm]")

        ax = axs[0].twinx()
        ax.plot(run_info['time'], run_info['glottal_width'] <= 0.002, ls='-.')
        ax.set_ylabel("Collision")

        ax = axs[1]
        ax.plot(run_info['time'], run_info['flow_rate'])
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Flow rate [cm^2/s]")

        ax = axs[2]
        # ax.plot(run_info['time'][1:], run_info['idx_min_area'], label="Minimum area location")
        ax.plot(run_info['time'][1:], run_info['idx_separation'], label="Separation location")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Separation location")

        ax = ax.twinx()
        for n in np.sort(np.unique(run_info['idx_separation'])):
            ax.plot(run_info['time'][1:], run_info['pressure'][:, n]/PASCAL_TO_CGS, label=f"Vertex {n:d}", ls='-.')
        ax.legend()
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Pressure [Pa]")

        fig.savefig(f'Kinematics_{self.case_postfix}.png')

        # Visualize the computed gradient
        # plot the u, v, a gradient components
        from matplotlib.tri import Triangulation
        solid = model.solid
        xy = solid.mesh.coordinates()
        tri = Triangulation(xy[:, 0], xy[:, 1], triangles=solid.mesh.cells())
        fig, axs = plt.subplots(3, 2)
        for ii, comp in enumerate(['u0', 'v0', 'a0']):
            for jj in range(2):
                grad_comp = gradient[comp][:].reshape(-1, 2)[:, jj]
                # breakpoint()
                mapble = axs[ii, jj].tripcolor(tri, grad_comp[solid.vert_to_sdof])

                fig.colorbar(mapble, ax=axs[ii, jj])

        plt.show()

    def plot_taylor_test(self, grad, step_dir, h, g):
        # Plot the adjoint gradient and finite difference approximations of the gradient
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        ax.plot(h[1:], (g[1:] - g[0])/h[1:],
                color='C0', marker='o', label='FD Approximations')
        ax.axhline(np.dot(grad, step_dir), color='C1', label="Adjoint gradient")
        ax.ticklabel_format(axis='y', style='sci')
        ax.set_xlabel("Step size")
        ax.set_ylabel("Gradient of functional")
        ax.legend()
        return fig, ax

class TestEmodGradient(TaylorTest):
    OVERWRITE_FORWARD_SIMULATIONS = True

    def setUp(self):
        """
        Runs the forward model over several parameters 'steps' and saves their history.
        """
        save_path = 'out/emodgrad-states.h5'
        model, solid_props, fluid_props = get_starting_kelvinvoigt_model()

        t_start, t_final = 0, 0.01
        times_meas = np.linspace(t_start, t_final, 128)
        timing_props = {'t0': t_start, 'tmeas': times_meas, 'dt_max': times_meas[1]}

        ## Set the step direction / step size / number of steps
        hs = np.concatenate(([0], 2.0**(np.arange(-6, 3)-2)), axis=0)

        step_size = 0.5e0 * PASCAL_TO_CGS
        dsolid = solid_props.copy()
        dsolid.vector[:] = 0
        dsolid['emod'][:] = 1.0*step_size

        print(f"Computing {len(hs)} finite difference points")
        if self.OVERWRITE_FORWARD_SIMULATIONS or not os.path.exists(save_path):
            if os.path.exists(save_path):
                os.remove(save_path)
            line_search(hs, model, (0, 0, 0), solid_props, fluid_props, times_meas,
                        dsolid_props=dsolid, filepath=save_path)
        else:
            print("Using existing files")

        self.hs = hs
        self.step_dir = np.array(dsolid['emod'])
        self.parameter = 'emod'

        self.model = model
        self.solid_props = solid_props
        self.fluid_props = fluid_props
        self.timing_props = timing_props

        self.save_path = save_path

        self.case_postfix = 'emod'

class Testu0Gradient(TaylorTest):
    OVERWRITE_FORWARD_SIMULATIONS = True

    def setUp(self):
        """
        Runs the forward model over several parameter 'steps' and saves their history.
        """
        save_path = 'out/u0grad-states.h5'
        model, solid_props, fluid_props = get_starting_kelvinvoigt_model()

        t_start, t_final = 0, 0.01
        times_meas = np.linspace(t_start, t_final, 128)
        timing_props = {'t0': t_start, 'tmeas': times_meas, 'dt_max': times_meas[1]}

        ## Set the step direction / step sizes
        hs = np.concatenate(([0], 2.0**(np.arange(-6, 3)-20)), axis=0)

        # Get y--coordinates in DOF order
        xy = model.get_ref_config()[dfn.dof_to_vertex_map(model.solid.scalar_fspace)]
        y = xy[:, 1]

        # Set the step direction as a linear (de)increase in x and y displacement in the y direction
        step_dir = np.zeros(xy.size)
        step_dir[:-1:2] = -(y-y.min()) / (y.max()-y.min())
        step_dir[1::2] = -(y-y.min()) / (y.max()-y.min())
        duva = (step_dir, 0.0, 0.0)

        print(f"Computing {len(hs)} finite difference points")

        fig, ax = plt.subplots(1, 1)
        if self.OVERWRITE_FORWARD_SIMULATIONS or not os.path.exists(save_path):
            if os.path.exists(save_path):
                os.remove(save_path)
            line_search(hs, model, (0, 0, 0), solid_props, fluid_props, times_meas,
                        duva=duva, filepath=save_path)
        else:
            print("Using existing files")

        plt.show()

        self.hs = hs
        self.step_dir = step_dir
        self.parameter = 'u0'

        self.model = model
        self.solid_props = solid_props
        self.fluid_props = fluid_props
        self.timing_props = timing_props

        self.save_path = save_path

        self.case_postfix = 'u0'

class Testu0FSIGradient(TaylorTest):
    OVERWRITE_FORWARD_SIMULATIONS = True

    def setUp(self):
        """
        Runs the forward model over several parameter 'steps' and saves their history.
        """
        save_path = 'out/u0grad-states.h5'
        model, solid_props, fluid_props = get_starting_kelvinvoigt_model()

        t_start, t_final = 0, 1/150
        times_meas = np.linspace(t_start, t_final, 128)
        timing_props = {'t0': t_start, 'tmeas': times_meas, 'dt_max': 100000000000000.0}

        ## Set the step direction / step sizes
        hs = np.concatenate(([0], 2.0**(np.arange(-6, -3)-10)), axis=0)

        # Get y--coordinates in DOF order
        xy = model.get_ref_config()
        y_surface = xy[model.surface_vertices, 1]

        # Set the step direction to only move the FSI (pressure nodes) surfaces
        # Set the x/y surface step direction
        surface_dofs = model.solid.vert_to_vdof.reshape(-1, 2)[model.surface_vertices]
        step_dir = dfn.Function(model.solid.vector_fspace).vector()

        # step only in surface
        step_dir[:] = 0.0
        # breakpoint()
        step_dir[np.array(surface_dofs[:, 0])] = (y_surface/y_surface.max())**2
        step_dir[np.array(surface_dofs[:, 1])] = 0.0

        # step only in interior
        # step_dir[:] = 1.0
        # step_dir[surface_dofs[:, 0].flat] = 0.0
        # step_dir[surface_dofs[:, 1].flat] = 0.0

        # zero the BC along the base
        model.solid.bc_base.apply(step_dir)


        duva = (step_dir, 0.0, 0.0)

        # fig, ax = plt.subplots(1, 1)
        # xy = model.get_ref_config()
        # ax.scatter(xy[:, 0], xy[:, 1])
        # ax.plot(xy[model.surface_vertices, 0], xy[model.surface_vertices, 1])
        # ax.scatter(xy[model.surface_vertices[0], 0], xy[model.surface_vertices[0], 1], color='k')
        # plt.show()

        # print(f"Computing {len(hs)} finite difference points")

        if self.OVERWRITE_FORWARD_SIMULATIONS or not os.path.exists(save_path):
            if os.path.exists(save_path):
                os.remove(save_path)
            line_search(hs, model, (0, 0, 0), solid_props, fluid_props, times_meas,
                        duva=duva, filepath=save_path)
        else:
            print("Using existing files")

        self.hs = hs
        self.step_dir = step_dir
        self.parameter = 'u0'

        self.model = model
        self.solid_props = solid_props
        self.fluid_props = fluid_props
        self.timing_props = timing_props

        self.save_path = save_path

        self.case_postfix = 'u0'

class Testv0Gradient(TaylorTest):
    OVERWRITE_FORWARD_SIMULATIONS = True

    def setUp(self):
        """
        Runs the forward model over several parameters 'steps' and saves their history.
        """
        save_path = 'out/v0grad-states.h5'
        model, solid_props, fluid_props = get_starting_kelvinvoigt_model()

        t_start, t_final = 0, 0.01
        times_meas = np.linspace(t_start, t_final, 128)
        timing_props = {'t0': t_start, 'tmeas': times_meas, 'dt_max': times_meas[1]}

        # N = 1
        # t_start, t_final = 0, 0.01/100 * N
        # times_meas = np.linspace(t_start, t_final, N+1)
        # timing_props = {'t0': t_start, 'tmeas': times_meas, 'dt_max': times_meas[1]}

        ## Set the step direction / step size / number of steps
        hs = np.concatenate(([0], 2.0**(np.arange(-6, 3)-10)), axis=0)

        # Get y--coordinates in DOF order
        xy = model.get_ref_config()[dfn.dof_to_vertex_map(model.solid.scalar_fspace)]
        y = xy[:, 1]

        # Set the step direction as a linear (de)increase in x and y displacement in the y direction
        step_dir = np.zeros(xy.size)
        step_dir[:-1:2] = -(y-y.min()) / (y.max()-y.min())
        step_dir[1::2] = -(y-y.min()) / (y.max()-y.min())
        duva = (0.0, step_dir, 0.0)

        print(f"Computing {len(hs)} finite difference points")

        if self.OVERWRITE_FORWARD_SIMULATIONS or not os.path.exists(save_path):
            if os.path.exists(save_path):
                os.remove(save_path)

            line_search(hs, model, (0, 0, 0), solid_props, fluid_props, times_meas,
                        duva=duva, filepath=save_path)
        else:
            print("Using existing files")

        plt.show()

        self.hs = hs
        self.step_dir = step_dir
        self.parameter = 'v0'

        self.model = model
        self.solid_props = solid_props
        self.fluid_props = fluid_props
        self.timing_props = timing_props

        self.save_path = save_path

        self.case_postfix = 'v0'

class Testa0Gradient(TaylorTest):
    OVERWRITE_FORWARD_SIMULATIONS = True

    def setUp(self):
        """
        Runs the forward model over several parameters 'steps' and saves their history.
        """
        save_path = 'out/a0grad-states.h5'
        model, solid_props, fluid_props = get_starting_kelvinvoigt_model()

        t_start, t_final = 0, 0.01
        times_meas = np.linspace(t_start, t_final, 128)
        timing_props = {'t0': t_start, 'tmeas': times_meas, 'dt_max': times_meas[-1]}

        ## Set the step direction / step size / number of steps
        hs = np.concatenate(([0], 2.0**(np.arange(-6, 3)+2)), axis=0)

        # Get y--coordinates in DOF order
        xy = model.get_ref_config()[dfn.dof_to_vertex_map(model.solid.scalar_fspace)]
        y = xy[:, 1]

        # Set the step direction as a linear (de)increase in x and y displacement in the y direction
        step_dir = np.zeros(xy.size)
        step_dir[:-1:2] = -(y-y.min()) / (y.max()-y.min())
        step_dir[1::2] = -(y-y.min()) / (y.max()-y.min())
        duva = (0.0, 0.0, step_dir)

        print(f"Computing {len(hs)} finite difference points")

        if self.OVERWRITE_FORWARD_SIMULATIONS or not os.path.exists(save_path):
            if os.path.exists(save_path):
                os.remove(save_path)

            line_search(hs, model, (0, 0, 0), solid_props, fluid_props, times_meas,
                        duva=duva, filepath=save_path)
        else:
            print("Using existing files")

        plt.show()

        self.hs = hs
        self.step_dir = step_dir
        self.parameter = 'a0'

        self.model = model
        self.solid_props = solid_props
        self.fluid_props = fluid_props
        self.timing_props = timing_props

        self.save_path = save_path

        self.case_postfix = 'a0'

class TestdtGradient(TaylorTest):
    OVERWRITE_FORWARD_SIMULATIONS = True

    def setUp(self):
        """
        Runs the forward model over several parameter 'steps' and saves their history.
        """
        save_path = 'out/dtgrad-states.h5'
        model, solid_props, fluid_props = get_starting_kelvinvoigt_model()

        t_start, t_final = 0.0, 0.01
        times_meas = np.linspace(t_start, t_final, 128)
        timing_props = {'t0': t_start, 'tmeas': times_meas, 'dt_max': times_meas[1]}

        ## Set the step direction / step size / number of steps
        hs = np.concatenate(([0], 2.0**(np.arange(-6, 3)-20)), axis=0)

        # Make the step size increase each time step
        step_dir = np.ones(128-1)
        dtimes = np.zeros(times_meas.size)
        dtimes[1:] = np.arange(1, dtimes.size)*1.0

        print(f"Computing {len(hs)} finite difference points")

        fig, ax = plt.subplots(1, 1)
        if self.OVERWRITE_FORWARD_SIMULATIONS or not os.path.exists(save_path):
            if os.path.exists(save_path):
                os.remove(save_path)

            line_search(hs, model, (0, 0, 0), solid_props, fluid_props, times_meas,
                        dtimes=dtimes, filepath=save_path)
        else:
            print("Using existing files")

        plt.show()

        self.hs = hs
        self.step_dir = step_dir
        self.parameter = 'dt'

        self.model = model
        self.solid_props = solid_props
        self.fluid_props = fluid_props
        self.timing_props = timing_props

        self.save_path = save_path

        self.case_postfix = 'dt'

def line_search(hs, model, uva, solid_props, fluid_props, times,
                duva=(0, 0, 0), dsolid_props=None, dfluid_props=None, dtimes=None,
                filepath='temp.h5'):
    if os.path.exists(filepath):
        os.remove(filepath)

    for n, h in enumerate(hs):
        # Increment all the properties along the search direction
        uva_n = tuple([uva[i] + h*duva[i] for i in range(3)])

        solid_props_n = solid_props.copy()
        if dsolid_props is not None:
            solid_props_n.vector[:] += h*dsolid_props.vector

        fluid_props_n = fluid_props.copy()
        if dfluid_props is not None:
            fluid_props_n.vector[:] += h*dfluid_props.vector

        # TODO: This is definitely a hack because I don't have a consistent interface to run the
        # forward sim
        # it should be more like times_n += h*dtimes
        times_n = np.array(times)
        if dtimes is not None:
            times_n += h*dtimes
        timing_props_n = {'t0': times_n[0], 'tmeas': times_n, 'dt_max': np.inf}

        runtime_start = perf_counter()
        info = forward(model, uva_n, solid_props_n, fluid_props_n, timing_props_n,
                       adaptive_step_prm={'abs_tol': None},
                       h5file=filepath, h5group=f'{n}', show_figure=False)
        runtime_end = perf_counter()

        print(runtime_end-runtime_start)

        # Save the run info to a pickled file
        if h == 0:
            with open(path.splitext(filepath)[0] + ".pickle", 'wb') as f:
                pickle.dump(info, f)

    return filepath

if __name__ == '__main__':
    # unittest.main()

    # test = TestEmodGradient()
    # test.setUp()
    # test.test_adjoint()

    # test = Testu0Gradient()
    # test.setUp()
    # test.test_adjoint()

    # test = Testu0FSIGradient()
    # test.setUp()
    # test.test_adjoint()

    # test = Testv0Gradient()
    # test.setUp()
    # test.test_adjoint()

    test = Testa0Gradient()
    test.setUp()
    test.test_adjoint()

    # test = TestdtGradient()
    # test.setUp()
    # test.test_adjoint()

    # test = Test2ndOrderDifferentiability()
    # test.setUp()
    # test.test_c2smoothness()
    # test.show_solution_info()
    # test.test_adjoint()
