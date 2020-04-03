"""
This example illustrates how to run the forward model.
"""

import sys
import os
from time import perf_counter

import h5py
import dolfin as dfn
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../')
# from femvf import mesh_operators
from femvf.forward import forward
from femvf.model import ForwardModel
from femvf import constants
from femvf import properties as props


if __name__ == '__main__':
    dfn.set_log_level(30)

    dt = 5e-5
    times_meas = np.linspace(0, 0.1, round(0.1/dt)+1)

    # Solid and Fluid properties
    solid_props = props.LinearElasticRayleigh(model)
    fluid_props = props.FluidProperties(model)
    timing_props = {'t0': 0.0, 'tmeas': times_meas, 'dt_max': dt}

    mesh_dir = os.path.expanduser('~/GraduateSchool/Projects/FEMVFOptimization/meshes/')
    mesh_base_filename = 'geometry2'
    mesh_path = os.path.join(mesh_dir, mesh_base_filename + '.xml')

    model = ForwardModel(mesh_path, {'pressure': 1, 'fixed': 3}, {})
   # breakpoint()

    h5file = 'forward_example.h5'
    if os.path.exists(h5file):
        os.remove(h5file)

    runtime_start = perf_counter()
    forward(model, solid_props, fluid_props, timing_props, h5file=h5file, abs_tol=None)
    runtime_end = perf_counter()

    print("Finished!")
    print(f"Duration: {runtime_end-runtime_start:.2f} s")
