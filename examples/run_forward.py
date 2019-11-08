"""
Just runs the forward model, yep that's all...
"""

import sys
import os
from time import perf_counter

import h5py
import dolfin as dfn
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../')
from femvf.forward import forward
from femvf.adjoint import adjoint
from femvf import forms
from femvf import constants
from femvf import functionals

if __name__ == '__main__':
    dfn.set_log_level(30)

    # Solid and Fluid properties
    solid_props = constants.DEFAULT_SOLID_PROPERTIES
    fluid_props = constants.DEFAULT_FLUID_PROPERTIES

    mesh_dir = os.path.expanduser('~/GraduateSchool/Projects/FEMVFOptimization/meshes/')

    mesh_base_filename = 'geometry2'
    mesh_path = os.path.join(mesh_dir, mesh_base_filename + '.xml')
    mesh_facet_path = os.path.join(mesh_dir, mesh_base_filename + '_facet_region.xml')
    mesh_cell_path = os.path.join(mesh_dir, mesh_base_filename + '_physical_region.xml')

    model = forms.ForwardModel(mesh_path, mesh_facet_path, mesh_cell_path, {'pressure': 1, 'fixed': 3}, {})

    dt = 1e-4
    times_meas = [0, 0.1]

    coords = model.mesh.coordinates()[...]
    triangulation = tri.Triangulation(coords[:, 0], coords[:, 1], triangles=model.mesh.cells())

    # dt = 1e-4
    # times_meas = [0, 0.1]

    h5file = 'forward-noinclusion.h5'
    if os.path.exists(h5file):
        os.remove(h5file)

    runtime_start = perf_counter()
    forward(model, 0, times_meas, dt, solid_props, fluid_props, h5file=h5file)
    runtime_end = perf_counter()
