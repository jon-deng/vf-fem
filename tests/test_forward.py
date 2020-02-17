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

sys.path.append('../')
from femvf.forward import forward
from femvf import forms
from femvf.properties import SolidProperties, FluidProperties#, TimingProperties
from femvf.constants import PASCAL_TO_CGS
from femvf import functionals

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
        model = forms.ForwardModel(self.mesh_path, {'pressure': 1, 'fixed': 3}, {})

        # dt = 2.5e-6
        dt = 5e-5
        times_meas = [0, 0.1]

        y_gap = 0.01
        alpha, k, sigma = -3000, 50, 0.002
        # Time varying fluid properties
        # fluid_props = constants.DEFAULT_FLUID_PROPERTIES
        # fluid_props['p_sub'] = [1500*PASCAL_TO_CGS, 1500*PASCAL_TO_CGS, 1, 1]
        # fluid_props['p_sub_time'] = [0, 3e-3, 3e-3, 0.02]
        # Constant fluid properties
        p_sub = 800

        timing_props = {'t0': 0, 'tmeas': times_meas, 'dt_max': dt}

        fluid_props = FluidProperties(model)
        fluid_props['y_midline'] = np.max(model.mesh.coordinates()[..., 1]) + y_gap
        fluid_props['p_sub'] = p_sub * PASCAL_TO_CGS
        fluid_props['alpha'] = alpha
        fluid_props['k'] = k
        fluid_props['sigma'] = sigma

        solid_props = SolidProperties(model)
        emod = model.emod.vector()[:].copy()
        emod[:] = 5.0e3 * PASCAL_TO_CGS
        solid_props['elastic_modulus'] = emod
        solid_props['rayleigh_m'] = 0
        solid_props['rayleigh_k'] = 3e-4
        solid_props['k_collision'] = 1e11
        solid_props['y_collision'] = fluid_props['y_midline'] - y_gap*1/2

        save_path = 'out/test_forward.h5'
        if os.path.isfile(save_path):
            os.remove(save_path)

        print("Running forward model")
        runtime_start = perf_counter()
        info = forward(model, solid_props, fluid_props, timing_props,
                       h5file=save_path, h5group='/', abs_tol=None, show_figure=False)
        runtime_end = perf_counter()
        print(f"Runtime {runtime_end-runtime_start:.2f} seconds")

        # breakpoint()

        fig, ax = plt.subplots(1, 1)
        ax.plot(info['time'], info['glottal_width'])

        plt.show()

    def plot(self):
        pass

if __name__ == '__main__':
    unittest.main()
