"""
Compare gradient computed via adjoint method with gradient computed via FD.
"""

import sys
import os
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
import dolfin as dfn

sys.path.append('../')
from femvf.forward import forward
from femvf.adjoint import adjoint
from femvf import forms
from femvf.constants import DEFAULT_FLUID_PROPERTIES, DEFAULT_SOLID_PROPERTIES, PASCAL_TO_CGS
from femvf import functionals
from femvf import statefile as sf

sys.path.append(os.path.expanduser('~/GraduateSchool/Projects/FEMVFOptimization/'))
import functionals as extra_functionals

dfn.set_log_level(30)
np.random.seed(123)

save_path = 'out/FiniteDifferenceStates.h5'


## Set the mesh to be used and initialize the forward model
mesh_dir = '../meshes'

mesh_base_filename = 'geometry2'
mesh_path = os.path.join(mesh_dir, mesh_base_filename + '.xml')

model = forms.ForwardModel(mesh_path, {'pressure': 1, 'fixed': 3}, {})


## Set the solution parameters
dt = 1e-4
times_meas = [0, 0.25]


## Set the fluid/solid parameters
emod = model.emod.vector()[:].copy()
emod[:] = 5e3 * PASCAL_TO_CGS
step_size = 1e1 * PASCAL_TO_CGS
num_steps = 8

emod_dir = np.random.rand(emod.size)

# Time varying fluid properties
# fluid_props = constants.DEFAULT_FLUID_PROPERTIES
# fluid_props['p_sub'] = [1500 * constants.PASCAL_TO_CGS, 1500 * constants.PASCAL_TO_CGS, 1, 1]
# fluid_props['p_sub_time'] = [0, 3e-3, 3e-3, 0.02]
p_sub = 1000
y_gap = 0.005
solid_props = DEFAULT_SOLID_PROPERTIES.copy()
solid_props['rayleigh_m'] = 0
solid_props['rayleigh_k'] = 3e-4
solid_props['k_collision'] = 1e12
solid_props['y_collision'] = np.max(model.mesh.coordinates()[..., 1]) + y_gap - 0.002

# Constant fluid properties
fluid_props = DEFAULT_FLUID_PROPERTIES.copy()
fluid_props['y_midline'] = np.max(model.mesh.coordinates()[..., 1]) + y_gap
fluid_props['p_sub'] = p_sub * PASCAL_TO_CGS


## Set a functional
fkwargs = {'tukey_alpha': 0.25}
# Functional for vocal eff
# n_start = 50
# fkwargs = {'n_start': n_start}
Functional = extra_functionals.AcousticEfficiency

# Functional for MFDR
# fkwargs = {}
# Functional = functionals.MFDR

# Functional for weighted sum of squared glottal widths
# fkwargs = {}
# Functional = functionals.WSSGlottalWidth

# Functional for total flow
# n_start = 0
# fkwargs = {'n_start': n_start}
# Functional = functionals.VolumeFlow

# Functional for acoustic efficiency
# Functional = functionals.AcousticEfficiency


## Finite Differences
print("Computing Gradient via Finite Differences")

if os.path.exists(save_path):
    os.remove(save_path)

for n in range(num_steps):
    runtime_start = perf_counter()
    solid_props['elastic_modulus'] = emod + n*step_size*emod_dir
    forward(model, 0, times_meas, dt, solid_props, fluid_props, h5file=save_path, h5group=f'{n}')
    runtime_end = perf_counter()

    print(f"Runtime {runtime_end-runtime_start:.2f} seconds")

# Calculate functional values at each finite difference step
functional_fd = list()
with sf.StateFile('out/FiniteDifferenceStates.h5', group='0', mode='r') as f:
    for n in range(num_steps):
        f.group = f'{n}'
        functional_fd.append(Functional(model, f, **fkwargs)())

functional_fd = np.array(functional_fd)


## Adjoint
print("Computing Gradient via Adjoint State")

runtime_start = perf_counter()
info = None
with sf.StateFile(save_path, group='0') as f:
    _, gradient_ad = adjoint(model, f, Functional, fkwargs)
runtime_end = perf_counter()

print(f"Runtime {runtime_end-runtime_start:.2f} seconds")


## Comparing adjoint and finite difference projected gradients
fig, axs = plt.subplots(1, 2, figsize=(7, 3))

step_points = step_size * np.arange(num_steps)
axs[0].plot(step_points, functional_fd, color='C0', marker='o',
            label="Directly computed functionals")

# Project the gradient in the direction of elastic modulus increase
grad_ad_projected = np.sum(gradient_ad * emod_dir)
functional_ad = functional_fd[0] + grad_ad_projected*step_points
axs[0].plot(step_points, functional_ad, color='C1', marker='o',
            label="Linear prediction from gradient")

grad_fd_projected = (functional_fd[2:]-functional_fd[:-2])/(step_points[2:] - step_points[:-2])
error = np.abs((grad_ad_projected-grad_fd_projected)/grad_ad_projected)*100
axs[1].plot(1 + np.arange(error.size), error)

# Formatting
axs[0].set_xlim(step_points[[0, -1]])
axs[0].set_xlabel("Elastic modulus [Pa]")
axs[0].set_ylabel("Objective function")
axs[0].legend()

for ax in axs:
    ax.grid()

axs[1].set_xlim([0, error.size])
axs[1].set_ylim([0, error.max()])

axs[1].set_xlabel(r"$N_{\Delta h}$")
axs[1].set_ylabel(r"% Error")

plt.tight_layout()
plt.show()

print(f"Gradient norm {np.linalg.norm(gradient_ad):.16e}")
print(f"Linear gradient prediction {grad_ad_projected:.16e}")
print(f"Actual FD values {grad_fd_projected[0]:.16e}")
print(f"% Error {error}")
