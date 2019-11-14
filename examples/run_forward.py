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
from femvf import forms
from femvf import constants
from femvf import properties as props


if __name__ == '__main__':
    dfn.set_log_level(30)

    # Solid and Fluid properties
    solid_props = props.SolidProperties()
    fluid_props = props.FluidProperties()

    mesh_dir = os.path.expanduser('~/GraduateSchool/Projects/FEMVFOptimization/meshes/')
    mesh_base_filename = 'geometry2'
    mesh_path = os.path.join(mesh_dir, mesh_base_filename + '.xml')

    model = forms.ForwardModel(mesh_path, {'pressure': 1, 'fixed': 3}, {})

    dt = 1e-4
    times_meas = [0, 0.1]

    h5file = 'forward_example.h5'
    if os.path.exists(h5file):
        os.remove(h5file)

    runtime_start = perf_counter()
    forward(model, 0, times_meas, dt, solid_props, fluid_props, h5file=h5file)
    runtime_end = perf_counter()

    print("Finished!")
    print(f"Duration: {runtime_end-runtime_start:.2f} s")
