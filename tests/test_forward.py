"""
A basic test to see if forward.forward will actually run
"""

# import sys
import os
import unittest
from time import perf_counter

import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import dolfin as dfn
# import h5py

from femvf.forward import integrate
from femvf.model import load_fsi_model
# from femvf.parameters.properties import SolidProperties, FluidProperties
from femvf.constants import PASCAL_TO_CGS

from femvf.solids import Rayleigh, KelvinVoigt
from femvf.fluids import Bernoulli

class TestForward(unittest.TestCase):
    def setUp(self):
        """
        Set the solid mesh
        """
        dfn.set_log_level(30)
        np.random.seed(123)

        mesh_dir = '../meshes'

        mesh_base_filename = 'M5-3layers'
        self.mesh_path = os.path.join(mesh_dir, mesh_base_filename + '.xml')

    def test_forward(self):
        ## Configure the model and model parameters
        model = load_fsi_model(self.mesh_path, None, Solid=Rayleigh, Fluid=Bernoulli)

        y_gap = 0.01
        alpha, k, sigma = -3000, 50, 0.002
        p_sub = 500

        times = np.linspace(0, 0.01, 100)

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
        u0 = dfn.Function(model.solid.vector_fspace).vector()

        save_path = 'test_forward.h5'
        if os.path.isfile(save_path):
            os.remove(save_path)

        ## Run the simulation
        print("Running forward model")
        runtime_start = perf_counter()
        info = integrate(model, (u0, 0, 0), solid_props, fluid_props, times,
                         h5file=save_path, h5group='/', coupling='implicit')
        runtime_end = perf_counter()
        print(f"Runtime {runtime_end-runtime_start:.2f} seconds")

        ## Plot the resulting glottal width
        fig, ax = plt.subplots(1, 1)
        ax.plot(info['time'], info['glottal_width'])
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Glottal width [cm]")

        plt.show()

if __name__ == '__main__':
    test = TestForward()
    test.setUp()
    test.test_forward()
    # unittest.main()
