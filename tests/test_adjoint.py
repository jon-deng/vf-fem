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
import dolfin as dfn

sys.path.append('../')
from femvf.forward import forward
from femvf.adjoint import adjoint
from femvf import forms
from femvf.constants import PASCAL_TO_CGS
from femvf.properties import FluidProperties, SolidProperties
from femvf import functionals as funcs
from femvf import statefile as sf

sys.path.append(path.expanduser('~/lib/vf-optimization'))
from vfopt import functionals as extra_funcs

class TestAdjointGradientCalculation(unittest.TestCase):

    def setUp(self):
        """
        Stores forward model states along parameter steps
        """
        dfn.set_log_level(30)
        np.random.seed(123)
        np.seterr(all='raise')

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
        t_final = 0.25
        times_meas = np.linspace(t_start, t_final, round((t_final-t_start)/dt_sample) + 1)

        ## Set the fluid/solid parameters
        emod = model.emod.vector()[:].copy()
        emod[:] = 2.5e3 * PASCAL_TO_CGS

        ## Set the stepping direction
        hs = np.concatenate(([0], 2.0**np.arange(-5, 4)), axis=0)
        step_size = 1e0 * PASCAL_TO_CGS
        step_dir = np.random.rand(emod.size) * step_size
        step_dir = np.ones(emod.size) * step_size

        rewrite_states = True
        save_path = f'out/FiniteDifferenceStates-uniformstep-notstiff-medduration-modifiedcollarea.h5'

        # Run the simulations
        y_gap = 0.005
        fluid_props = FluidProperties()
        fluid_props['y_midline'] = np.max(model.mesh.coordinates()[..., 1]) + y_gap
        fluid_props['p_sub'] = 1000 * PASCAL_TO_CGS

        solid_props = SolidProperties()
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
                info = forward(model, 0, times_meas, dt_max, solid_props, fluid_props, abs_tol=None,
                               h5file=save_path, h5group=f'{n}')
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
        self.save_path = save_path
        self.times_meas = times_meas
        self.dt_max = dt_max

    def test_adjoint(self):
        hs = self.hs
        step_dir = self.step_dir
        model = self.model
        save_path = self.save_path

        run_info = None
        with open(save_path+".pickle", 'rb') as f:
            run_info = pickle.load(f)

        fkwargs = {}
        Functional = funcs.FinalDisplacementNorm
        # Functional = funcs.DisplacementNorm
        # Functional = funcs.VelocityNorm
        # Functional = funcs.StrainEnergy
        # Functional = extra_funcs.AcousticEfficiency

        # Calculate functional values at each step
        print(f"\nComputing functional for each point")
        runtime_start = perf_counter()

        functionals = list()
        with sf.StateFile(save_path, group='0', mode='r') as f:
            for n, h in enumerate(hs):
                f.root_group_name = f'{n}'
                functionals.append(Functional(model, f, **fkwargs)())

        runtime_end = perf_counter()
        print(f"Runtime {runtime_end-runtime_start:.2f} seconds")

        functionals = np.array(functionals)

        ## Adjoint
        print("\nComputing Gradient via adjoint calculation")

        runtime_start = perf_counter()
        info = None
        gradient_ad = None
        with sf.StateFile(save_path, group='0', mode='r') as f:
            _, gradient_ad = adjoint(model, f, Functional, fkwargs)
        runtime_end = perf_counter()

        print(f"Runtime {runtime_end-runtime_start:.2f} seconds")

        ## Comparing adjoint and finite difference projected gradients
        taylor_remainder_1 = np.abs(functionals[1:] - functionals[0])
        taylor_remainder_2 = np.abs(functionals[1:] - functionals[0] - hs[1:]*np.dot(gradient_ad, step_dir))

        order_1 = np.log(taylor_remainder_1[1:]/taylor_remainder_1[:-1]) / np.log(2)
        order_2 = np.log(taylor_remainder_2[1:]/taylor_remainder_2[:-1]) / np.log(2)

        # taylor_remainder_2_fd = (functionals[1:] - functionals[0]) / hs[1:]
        # order_2_fd = np.log(taylor_remainder_2_fd[1:]/taylor_remainder_2_fd[:-1]) / np.log(2)
        # print("Numerical order of FD: ", order_2_fd)
        # breakpoint()

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

        # Plot the glottal width vs time of the simulation at zero step size
        fig, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)

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
        for n in np.sort(np.unique(run_info['idx_separation'])):
            ax.plot(run_info['time'][1:], run_info['pressure'][:, n]/PASCAL_TO_CGS, label=f"Vertex {n:d}")
        ax.legend()
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Pressure [Pa]")

        plt.show()

        print(run_info['idx_min_area'])
        print(run_info['idx_separation'])
        breakpoint()

    def show_solution_info(self):
        save_path = self.save_path
        model = self.model
        fluid_props = self.fluid_props

        # run_info = None
        # with open(save_path+".pickle", 'rb') as f:
        #     run_info = pickle.load(f)

        run_info = forward(model, 0, self.times_meas, self.dt_max, self.solid_props, fluid_props, abs_tol=None)

        surface_area = []
        with sf.StateFile(self.save_path, group='/0', mode='r') as f:
            for n in range(f.get_num_states()):
                surface_dofs = model.vert_to_vdof[model.surface_vertices].reshape(-1)
                u = f.get_state(n)[0]
                xy_deformed_surface = model.surface_coordinates + u[surface_dofs].reshape(-1, 2)

                area = self.fluid_props['y_midline'] - xy_deformed_surface[:, 1]
                area[area < 2*0.002] = 2*0.002
                surface_area.append(area)
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
        for n in [30]:
            ax.plot(run_info['time'][1:], run_info['pressure'][:, n]/PASCAL_TO_CGS, label=f"Vertex {n:d}")

            ax.plot(run_info['time'],
                    (fluid_props['p_sub'] + 0.5*fluid_props['rho']*run_info['flow_rate']**2 *
                        (1/fluid_props['a_sub']**2-1/surface_area[:, n]**2)) / PASCAL_TO_CGS,
                    label=f"Manual calc")
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

        plt.show()

if __name__ == '__main__':
    # unittest.main()
    test = TestAdjointGradientCalculation()
    test.setUp()
    test.show_solution_info()
    # test.test_adjoint()
