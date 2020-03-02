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
from femvf.forward import forward
from femvf.adjoint import adjoint
from femvf import forms
from femvf.constants import PASCAL_TO_CGS
from femvf.properties import FluidProperties, SolidProperties#, TimingProperties
from femvf import functionals as funcs
from femvf import statefile as sf

sys.path.append(path.expanduser('~/lib/vf-optimization'))
from optvf import functionals as extra_funcs

class TestAdjointGradientCalculation(unittest.TestCase):
    OVERWRITE_FORWARD_SIMULATIONS = False

    def setUp(self):
        """
        Runs the forward model over several parameters 'steps' and saves their history.
        """
        dfn.set_log_level(30)
        np.random.seed(123)

        ##### Set up parameters as you see fit

        ## Set the mesh to be used and initialize the forward model
        mesh_dir = '../meshes'
        mesh_base_filename = 'geometry2'
        mesh_path = os.path.join(mesh_dir, mesh_base_filename + '.xml')

        model = forms.ForwardModel(mesh_path, {'pressure': 1, 'fixed': 3}, {})

        ## Set the solution parameters
        dt_sample = 1e-4
        # dt_max = 1e-5
        dt_max = 5e-5
        t_start = 0
        # t_final = (150)*dt_sample
        t_final = 0.2
        times_meas = np.linspace(t_start, t_final, round((t_final-t_start)/dt_max) + 1)

        ## Set the fluid/solid parameters
        emod = model.emod.vector()[:].copy()
        emod[:] = 2.5e3 * PASCAL_TO_CGS

        k_coll = 1e11
        y_gap = 0.02
        y_coll_offset = 0.01
        # alpha, k, sigma = -1000, 200, 0.0005
        # alpha, k, sigma = -1000, 100, 0.001
        alpha, k, sigma = -3000, 50, 0.002
        # alpha, k, sigma = -1000, 25, 0.004

        ## Set the stepping direction
        hs = np.concatenate(([0], 2.0**(np.arange(-6, 3)-2)), axis=0)
        step_size = 0.5e0 * PASCAL_TO_CGS
        step_dir = np.random.rand(emod.size) * step_size
        step_dir = np.ones(emod.size) * step_size

        case_postfix = f'quartic_{t_final:.5f}-{k_coll:.2e}-{y_gap:.2e}-{y_coll_offset:.2e}_{alpha}_{k}_{sigma}'
        save_path = f'out/FiniteDifferenceStates-{case_postfix}.h5'

        timing_props = {'t0': t_start, 'tmeas': times_meas, 'dt_max': dt_max}

        fluid_props = FluidProperties(model)
        fluid_props['y_midline'] = np.max(model.mesh.coordinates()[..., 1]) + y_gap
        fluid_props['p_sub'] = 1000 * PASCAL_TO_CGS
        fluid_props['alpha'] = alpha
        fluid_props['k'] = k
        fluid_props['sigma'] = sigma

        solid_props = SolidProperties(model)
        solid_props['elastic_modulus'] = emod
        solid_props['rayleigh_m'] = 0
        solid_props['rayleigh_k'] = 3e-4
        solid_props['k_collision'] = k_coll
        solid_props['y_collision'] = fluid_props['y_midline'] - y_coll_offset

        # Compute functionals along step direction
        print(f"Computing {len(hs)} finite difference points")
        if not self.OVERWRITE_FORWARD_SIMULATIONS and os.path.exists(save_path):
            print("Using existing files")
        else:
            if os.path.exists(save_path):
                os.remove(save_path)

            for n, h in enumerate(hs):
                runtime_start = perf_counter()
                # _solid_props = solid_props.copy()
                solid_props['elastic_modulus'] = emod + h*step_dir
                info = forward(model, solid_props, fluid_props, timing_props, abs_tol=None,
                               h5file=save_path, h5group=f'{n}', show_figure=False)
                runtime_end = perf_counter()

                if h == 0:
                    # Save the run info to a pickled file
                    with open(save_path+".pickle", 'wb') as f:
                        pickle.dump(info, f)

                print(f"Runtime {runtime_end-runtime_start:.2f} seconds")

        self.hs = hs
        self.step_dir = step_dir
        self.model = model
        self.fluid_props = fluid_props
        self.solid_props = solid_props
        self.timing_props = timing_props

        self.save_path = save_path
        # self.times_meas = times_meas
        # self.dt_max = dt_max

        self.case_postfix = case_postfix

    def test_adjoint(self):
        hs = self.hs
        step_dir = self.step_dir
        model = self.model
        save_path = self.save_path

        run_info = None
        with open(save_path+".pickle", 'rb') as f:
            run_info = pickle.load(f)

        fkwargs = {'tukey_alpha': 0.05, 'f0':100, 'df':10}
        # Functional = funcs.FinalDisplacementNorm
        # Functional = funcs.FinalVelocityNorm
        # Functional = funcs.DisplacementNorm
        # Functional = funcs.VelocityNorm
        # Functional = funcs.StrainEnergy
        Functional = extra_funcs.AcousticEfficiency
        Functional = extra_funcs.F0WeightedAcousticPower

        # Calculate functional values at each step
        print(f"\nComputing functional for each point")

        total_runtime = 0
        functionals = list()
        with sf.StateFile(model, save_path, group='0', mode='r') as f:
            for n, h in enumerate(hs):
                f.root_group_name = f'{n}'
                runtime_start = perf_counter()
                functionals.append(Functional(model, f, **fkwargs)())
                runtime_end = perf_counter()
                total_runtime += runtime_end-runtime_start
                print(runtime_end-runtime_start)
        print(f"Runtime {total_runtime:.2f} seconds")

        functionals = np.array(functionals)

        ## Adjoint
        print("\nComputing Gradient via adjoint calculation")

        runtime_start = perf_counter()
        info = None
        gradient_ad = None
        with sf.StateFile(model, save_path, group='0', mode='r', driver='core') as f:
            _, gradient_ad, _ = adjoint(model, f, Functional, fkwargs)
        runtime_end = perf_counter()

        print(f"Runtime {runtime_end-runtime_start:.2f} seconds")

        ## Comparing adjoint and finite difference projected gradients
        taylor_remainder_1 = np.abs(functionals[1:] - functionals[0])
        taylor_remainder_2 = np.abs(functionals[1:] - functionals[0] - hs[1:]*np.dot(gradient_ad, step_dir))

        order_1 = np.log(taylor_remainder_1[1:]/taylor_remainder_1[:-1]) / np.log(2)
        order_2 = np.log(taylor_remainder_2[1:]/taylor_remainder_2[:-1]) / np.log(2)

        print("\nSteps:", hs[1:])

        print("\n1st order taylor remainders: \n", taylor_remainder_1)
        print("Numerical order: \n", order_1)

        print("\n2nd order taylor remainders: \n", taylor_remainder_2)
        print("Numerical order: \n", order_2)

        print("\n||dg/dp|| = ", gradient_ad.norm('l2'))
        print("dg/dp * step_dir = ", np.dot(gradient_ad, step_dir))
        print("FD approximation of dg/dp * step_dir = ", (functionals[1:] - functionals[0])/hs[1:])

        # Plot the adjoint gradient and finite difference approximations of the gradient
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        ax.plot(hs[1:], (functionals[1:] - functionals[0])/hs[1:],
                color='C0', marker='o', label='FD Approximations')
        ax.axhline(np.dot(gradient_ad, step_dir), color='C1', label="Adjoint gradient")
        # ax.set_xlim(0, None)
        ax.ticklabel_format(axis='y', style='sci')
        ax.set_xlabel("Step size")
        ax.set_ylabel("Gradient of functional")
        ax.legend()

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

        plt.show()

    def show_solution_info(self):
        model = self.model

        solution_file = 'tmp.h5'
        if path.isfile(solution_file):
            os.remove(solution_file)
        run_info = forward(model, self.solid_props, self.fluid_props, self.timing_props,
                           h5file=solution_file, abs_tol=None)

        surface_area = []
        with sf.StateFile(model, solution_file, group='/', mode='r') as f:
            for n in range(f.get_num_states()):
                surface_dofs = model.vert_to_vdof[model.surface_vertices].reshape(-1)
                u = f.get_state(n)[0]
                xy_deformed_surface = model.surface_coordinates + u[surface_dofs].reshape(-1, 2)

                area = 2*(self.fluid_props['y_midline'] - xy_deformed_surface[:, 1])
                # breakpoint()
                # area[area < 2*0.002] = 2*0.002
                surface_area.append(area)

                ## Test of pressure
                # pressure_verify = fluid_props['p_sub'] + 0.5*fluid_props['rho']*run_info['flow_rate'][n]**2 *
                #         (1/fluid_props['a_sub']**2-1/surface_area[:, n]**2))
        surface_area = np.array(surface_area)

        # Plot the glottal width vs time of the simulation at zero step size
        fig, axs = plt.subplots(4, 1, constrained_layout=True, sharex=True)

        ax = axs[0]
        ax.plot(run_info['time'], run_info['glottal_width'])
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Glottal width [cm]")

        ax = axs[0].twinx()
        ax.plot(run_info['time'], run_info['glottal_width'] == 2*0.002)
        ax.set_ylabel("Collision")

        ax = axs[1]
        ax.plot(run_info['time'], run_info['flow_rate'])
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Flow rate [cm^2/s]")

        ax = axs[2]
        ax.plot(run_info['time'][1:], run_info['idx_min_area'], label="Minimum area location")
        ax.plot(run_info['time'][1:], run_info['idx_separation'], label="Separation location")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Vertex index")

        ax = ax.twinx()
        # for n in np.sort(np.unique(run_info['idx_separation'])):
        for n in [30, 31, 32]:
            ax.plot(run_info['time'][:-1], run_info['pressure'][:, n]/PASCAL_TO_CGS, label=f"Vertex {n:d}")

            # ax.plot(run_info['time'],
            #         (fluid_props['p_sub'] + 0.5*fluid_props['rho']*run_info['flow_rate']**2 *
            #             (1/fluid_props['a_sub']**2-1/surface_area[:, n]**2)) / PASCAL_TO_CGS,
            #         label=f"Manual calc")
        ax.legend()
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Pressure [Pa]")

        ax = axs[3]
        # for n in np.sort(np.unique(run_info['idx_separation'])):
        for n in [30]:
            ax.plot(run_info['time'], surface_area[:, n], label=f"Vertex {n:d}")
        ax.legend()
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Area [cm^2]")

        ax = ax.twinx()
        for n in [30]:
            ax.plot(run_info['time'], -(run_info['flow_rate']/surface_area[:, n])**2, label=f"Vertex {n:d}")
        ax.legend()
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Q/Area [cm^2/s / cm]")

        plt.show()

class Test2ndOrderDifferentiability(unittest.TestCase):

    def setUp(self):
        """
        Stores forward model states along parameter steps
        """
        dfn.set_log_level(30)
        np.random.seed(123)
        # np.seterr(all='raise')

        ##### Set up parameters as you see fit

        ## Set the mesh to be used and initialize the forward model
        mesh_dir = '../meshes'

        mesh_base_filename = 'geometry2'
        mesh_path = os.path.join(mesh_dir, mesh_base_filename + '.xml')

        model = forms.ForwardModel(mesh_path, {'pressure': 1, 'fixed': 3}, {})

        p_sub = 1000

        fkwargs = {}
        Functional = funcs.FinalDisplacementNorm
        # Functional = funcs.DisplacementNorm
        # Functional = funcs.VelocityNorm
        # Functional = funcs.StrainEnergy
        # Functional = extra_funcs.AcousticEfficiency

        ## Set the solution parameters
        dt_sample = 1e-4
        # dt_max = 1e-5
        dt_max = 5e-5
        t_start = 0
        # t_final = (150)*dt_sample
        t_final = 0.1
        times_meas = np.linspace(t_start, t_final, round((t_final-t_start)/dt_max) + 1)

        ## Set the fluid/solid parameters
        emod = model.emod.vector()[:].copy()
        emod[:] = 2.5e3 * PASCAL_TO_CGS

        ## Set the stepping direction
        # hs = np.concatenate(([0], 2.0**(np.arange(-5, 4)-2)), axis=0)
        hs = np.linspace(0, 2**3, 50)
        step_size = 0.5e0 * PASCAL_TO_CGS
        step_dir = np.random.rand(emod.size) * step_size
        step_dir = np.ones(emod.size) * step_size

        rewrite_states = True
        save_path = f'out/C2SmoothnessStates-{t_final:.5f}-psub{p_sub:.1f}.h5'

        # Run the simulations
        timing_props = {'t0': t_start, 'tmeas': times_meas, 'dt_max': dt_max}

        y_gap = 0.005
        fluid_props = FluidProperties(model)
        fluid_props['y_midline'] = np.max(model.mesh.coordinates()[..., 1]) + y_gap
        fluid_props['p_sub'] = p_sub * PASCAL_TO_CGS

        solid_props = SolidProperties(model)
        solid_props['elastic_modulus'] = emod
        solid_props['rayleigh_m'] = 0
        solid_props['rayleigh_k'] = 3e-4
        solid_props['k_collision'] = 1e12
        solid_props['y_collision'] = fluid_props['y_midline'] - 0.002

        # Compute functionals along step direction
        print(f"Computing {len(hs)} finite difference points")
        if not rewrite_states and os.path.exists(save_path):
            print("Using existing files")
        else:
            if os.path.exists(save_path):
                os.remove(save_path)

            for n, h in enumerate(hs):
                runtime_start = perf_counter()
                # _solid_props = solid_props.copy()
                solid_props['elastic_modulus'] = emod + h*step_dir
                info = forward(model, solid_props, fluid_props, timing_props, abs_tol=None,
                               h5file=save_path, h5group=f'{n}')

                grad = None
                with sf.StateFile(model, save_path, group=f'{n}', mode='r') as f:
                    _, grad, _ = adjoint(model, f, Functional, fkwargs)
                runtime_end = perf_counter()

                with h5py.File(save_path, mode='a') as f:
                    f[f'{n}/grad'] = grad

                if h == 0:
                    # Save the run info to a pickled file
                    with open(save_path+".pickle", 'wb') as f:
                        pickle.dump(info, f)

                print(f"Runtime {runtime_end-runtime_start:.2f} seconds")

        self.hs = hs
        self.step_dir = step_dir
        self.model = model
        self.fluid_props = fluid_props
        self.solid_props = solid_props
        self.timing_props = timing_props
        self.save_path = save_path

    def test_c2smoothness(self):
        hs = self.hs
        step_dir = self.step_dir
        model = self.model
        save_path = self.save_path
        fkwargs = self.fkwargs
        Functional = self.Functional

        run_info = None
        with open(save_path+".pickle", 'rb') as f:
            run_info = pickle.load(f)

        # Calculate functional values at each step
        print(f"\nComputing functional for each point")
        runtime_start = perf_counter()

        functionals = list()
        with sf.StateFile(model, save_path, group='0', mode='r') as f:
            for n, h in enumerate(hs):
                f.root_group_name = f'{n}'
                functionals.append(Functional(model, f, **fkwargs)())

        runtime_end = perf_counter()
        print(f"Runtime {runtime_end-runtime_start:.2f} seconds")

        functionals = np.array(functionals)

        ## Adjoint
        # print("\nComputing Gradient via adjoint calculation")

        # runtime_start = perf_counter()
        # info = None
        # gradient_ad = None
        # with sf.StateFile(model, save_path, group='0', mode='r') as f:
        #     _, gradient_ad, _ = adjoint(model, f, Functional, fkwargs)
        # runtime_end = perf_counter()

        # print(f"Runtime {runtime_end-runtime_start:.2f} seconds")

        ## Calculate derivatives of the functional
        d0_functional = functionals
        d1_functional = (functionals[2:] - functionals[:-2])/(hs[1]-hs[0])/2
        d2_functional = (functionals[2:] - 2*functionals[1:-1] + functionals[0:-2])/(hs[1]-hs[0])**2

        # Plot the adjoint gradient and finite difference approximations of the gradient
        fig, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
        ax = axs[0]
        ax.plot(hs, d0_functional, color='C0', marker='o')

        ax = axs[1]
        ax.plot(hs[1:-1], d1_functional, color='C0', marker='o')
        ax.set_ylabel(r"$\frac{df}{dx}$")

        ax = ax.twinx()
        ax.plot(hs[1:-1], d2_functional, color='C1', marker='o')
        ax.set_ylabel(r"$\frac{d^2f}{dx^2}$")
        ax.ticklabel_format(axis='y', style='sci')
        ax.set_xlabel("Step size")

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

        plt.show()

        # print(run_info['idx_min_area'])
        # print(run_info['idx_separation'])
        breakpoint()

if __name__ == '__main__':
    # unittest.main()

    test = TestAdjointGradientCalculation()
    test.setUp()
    test.test_adjoint()

    # test = Test2ndOrderDifferentiability()
    # test.setUp()
    # test.test_c2smoothness()
    # test.show_solution_info()
    # test.test_adjoint()
