"""
A basic test to see if forward.forward will actually run
"""

import sys
import os
import unittest
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dolfin as dfn
import h5py

sys.path.append('../')
from femvf.forward import forward
from femvf.model import ForwardModel, load_1dfluidfsi_model
from femvf.parameters.properties import SolidProperties, FluidProperties
from femvf.constants import PASCAL_TO_CGS

from femvf.solids import Rayleigh
from femvf.fluids import Bernoulli

class TestForward(unittest.TestCase):
    def setUp(self):
        """Set the mesh to be used"""

        dfn.set_log_level(30)
        np.random.seed(123)

        mesh_dir = '../meshes'

        mesh_base_filename = 'geometry2'
        self.mesh_path = os.path.join(mesh_dir, mesh_base_filename + '.xml')

    def test_forward(self):
        ## Set the model and various simulation parameters (fluid/solid properties, time step etc.)
        model = load_1dfluidfsi_model(self.mesh_path, Solid=Rayleigh, Fluid=Bernoulli)

        dt = 5e-5
        times_meas = [0, 0.005]

        y_gap = 0.01
        alpha, k, sigma = -3000, 50, 0.002
        # Time varying fluid properties
        # fluid_props = constants.DEFAULT_FLUID_PROPERTIES
        # fluid_props['p_sub'][()] = [1500*PASCAL_TO_CGS, 1500*PASCAL_TO_CGS, 1, 1]
        # fluid_props['p_sub_time'][()] = [0, 3e-3, 3e-3, 0.02]
        # Constant fluid properties
        p_sub = 500

        timing_props = {'t0': 0, 'tmeas': times_meas, 'dt_max': dt}

        fluid_props = model.fluid.get_properties()
        fluid_props['y_midline'][()] = np.max(model.solid.mesh.coordinates()[..., 1]) + y_gap
        fluid_props['p_sub'][()] = p_sub * PASCAL_TO_CGS
        fluid_props['alpha'][()] = alpha
        fluid_props['k'][()] = k
        fluid_props['sigma'][()] = sigma

        solid_props = model.solid.get_properties()
        xy = model.solid.scalar_fspace.tabulate_dof_coordinates()
        x = xy[:, 0]
        y = xy[:, 1]
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        solid_props['emod'][:] = 1/2*5.0e3*PASCAL_TO_CGS*((x-x_min)/(x_max-x_min) + (y-y_min)/(y_max-y_min)) + 2.5e3*PASCAL_TO_CGS
        solid_props['rayleigh_m'][()] = 0
        solid_props['rayleigh_k'][()] = 4e-3
        solid_props['k_collision'][()] = 1e11
        solid_props['y_collision'][()] = fluid_props['y_midline'] - y_gap*1/2

        xy = model.solid.vector_fspace.tabulate_dof_coordinates()
        x = xy[:, 0]
        y = xy[:, 1]
        u0 = dfn.Function(solid.vector_fspace).vector()

        save_path = 'test_forward.h5'
        if os.path.isfile(save_path):
            os.remove(save_path)

        print("Running forward model")
        adaptive_step_prm = {'abs_tol': None}
        runtime_start = perf_counter()
        info = forward(model, (u0, 0, 0), solid_props, fluid_props, timing_props,
                       h5file=save_path, h5group='/', adaptive_step_prm=adaptive_step_prm,
                       show_figure=False)
        runtime_end = perf_counter()
        print(f"Runtime {runtime_end-runtime_start:.2f} seconds")

        ### Write a single vector data set out to see if it work with paraview
        with h5py.File(save_path, mode='a') as f:
            f.create_dataset('u0', data=f['u'][15, :], dtype=np.float64)
            f.create_dataset('v0', data=f['v'][15, :], dtype=np.float64)

        fig, ax = plt.subplots(1, 1)
        ax.plot(info['time'], info['glottal_width'])

        plt.show()

    def plot(self):
        pass

if __name__ == '__main__':
    unittest.main()
