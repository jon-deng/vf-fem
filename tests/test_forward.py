"""
A basic test to see if forward.forward will actually run
"""

import sys
import os
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
import dolfin as dfn

sys.path.append('../')
from femvf.forward import forward
from femvf import forms
from femvf.constants import DEFAULT_FLUID_PROPERTIES, DEFAULT_SOLID_PROPERTIES, PASCAL_TO_CGS
from femvf import functionals

dfn.set_log_level(30)
np.random.seed(123)

save_path = 'out/test_forward.h5'
if os.path.isfile(save_path):
    os.remove(save_path)


## Set the mesh to be used and initialize the forward model
mesh_dir = '../meshes'

mesh_base_filename = 'geometry2'
mesh_path = os.path.join(mesh_dir, mesh_base_filename + '.xml')

model = forms.ForwardModel(mesh_path, {'pressure': 1, 'fixed': 3}, {})

## Set the solution parameters
dt = 2.5e-6
times_meas = [0, 0.01]


## Set the fluid/solid parameters
emod = model.emod.vector()[:].copy()
emod[:] = 2e3 * PASCAL_TO_CGS

# Time varying fluid properties
# fluid_props = constants.DEFAULT_FLUID_PROPERTIES
# fluid_props['p_sub'] = [1500 * constants.PASCAL_TO_CGS, 1500 * constants.PASCAL_TO_CGS, 1, 1]
# fluid_props['p_sub_time'] = [0, 3e-3, 3e-3, 0.02]
p_sub = 1000
y_gap = 0.005
solid_props = DEFAULT_SOLID_PROPERTIES.copy()
solid_props['elastic_modulus'] = emod
solid_props['rayleigh_m'] = 0
solid_props['rayleigh_k'] = 3e-4
solid_props['k_collision'] = 1e12
solid_props['y_collision'] = np.max(model.mesh.coordinates()[..., 1]) + y_gap - 0.002

# Constant fluid properties
fluid_props = DEFAULT_FLUID_PROPERTIES.copy()
fluid_props['y_midline'] = np.max(model.mesh.coordinates()[..., 1]) + y_gap
fluid_props['p_sub'] = p_sub * PASCAL_TO_CGS

print("Running forward model")
runtime_start = perf_counter()

info = forward(model, 0, times_meas, dt, solid_props, fluid_props, h5file=save_path, h5group='/')

runtime_end = perf_counter()

print(f"Runtime {runtime_end-runtime_start:.2f} seconds")

plt.plot(info['time'], info['glottal_width'])
plt.show()
