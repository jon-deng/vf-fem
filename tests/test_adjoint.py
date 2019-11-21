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
from time import perf_counter

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
from femvf import functionals
from femvf import statefile as sf

sys.path.append(os.path.expanduser('~/lib/vf-optimization'))
from vfopt import functionals as extra_functionals

dfn.set_log_level(30)
# np.random.seed(123)

save_path = 'out/FiniteDifferenceStates.h5'


## Set the mesh to be used and initialize the forward model
mesh_dir = '../meshes'

mesh_base_filename = 'geometry2'
mesh_path = os.path.join(mesh_dir, mesh_base_filename + '.xml')

model = forms.ForwardModel(mesh_path, {'pressure': 1, 'fixed': 3}, {})

## Set the solution parameters
dt_sample = 5e-5
dt_max = 5e-5
t_final = (256-1)*5e-5
# times_meas = np.linspace(0, t_final, round(t_final/dt)+1)
times_meas = np.linspace(0, t_final)

## Set the fluid/solid parameters
emod = model.emod.vector()[:].copy()
emod[:] = 5e3 * PASCAL_TO_CGS

## Set the stepping direction
hs = np.concatenate(([0], 2**np.arange(5)), axis=0)
step_size = 1e-2 * PASCAL_TO_CGS
step_dir = np.random.rand(emod.size) * step_size


y_gap = 0.005
solid_props = SolidProperties()
solid_props['rayleigh_m'] = 0
solid_props['rayleigh_k'] = 3e-4
solid_props['k_collision'] = 1e12
solid_props['y_collision'] = np.max(model.mesh.coordinates()[..., 1]) + y_gap - 0.002

# Constant fluid properties
fluid_props = FluidProperties()
fluid_props['y_midline'] = np.max(model.mesh.coordinates()[..., 1]) + y_gap
fluid_props['p_sub'] = 1000 * PASCAL_TO_CGS

fkwargs = {}
# Functional = functionals.VelocityNorm
Functional = functionals.DisplacementNorm
# Functional = functionals.StrainEnergy
# Functional = extra_functionals.AcousticEfficiency

## Compute functionals along step direction
print("Computing Gradient via Finite Differences")

if os.path.exists(save_path):
    os.remove(save_path)

for n, h in enumerate(hs):
    runtime_start = perf_counter()
    solid_props['elastic_modulus'] = emod + h*step_dir
    forward(model, 0, times_meas, dt_max, solid_props, fluid_props, abs_tol=None,
            h5file=save_path, h5group=f'{n}')
    runtime_end = perf_counter()

    print(f"Runtime {runtime_end-runtime_start:.2f} seconds")

# Calculate functional values at each step
functionals = list()
with sf.StateFile('out/FiniteDifferenceStates.h5', group='0', mode='r') as f:
    for n, h in enumerate(hs):
        f.root_group_name = f'{n}'
        functionals.append(Functional(model, f, **fkwargs)())

functionals = np.array(functionals)

## Adjoint
print("Computing Gradient via Adjoint Method")

runtime_start = perf_counter()
info = None
gradient_ad = None
with sf.StateFile(save_path, group='0', mode='r') as f:
    _, gradient_ad = adjoint(model, f, Functional, fkwargs)
runtime_end = perf_counter()

print(f"Runtime {runtime_end-runtime_start:.2f} seconds")

## Comparing adjoint and finite difference projected gradients
# breakpoint()
taylor_remainder_1 = np.abs(functionals[1:] - functionals[0])
taylor_remainder_2 = np.abs(functionals[1:] - functionals[0] - hs[1:]*np.dot(gradient_ad, step_dir))

order_1 = np.log(taylor_remainder_1[1:]/taylor_remainder_1[:-1]) / np.log(2)
order_2 = np.log(taylor_remainder_2[1:]/taylor_remainder_2[:-1]) / np.log(2)
print(hs[1:])
print("\n1st order taylor remainders: ", taylor_remainder_1)
print("Numerical order: ", order_1)

print("\n2nd order taylor remainders: ", taylor_remainder_2)
print("Numerical order: ", order_2)

print("\n||dg/dp|| = ", gradient_ad.norm('l2'))
print("dg/dp * step_dir = ", np.dot(gradient_ad, step_dir))
print("FD approximation of dg/dp * step_dir = ", (functionals[1:] - functionals[0])/hs[1:])

fig, ax = plt.subplots(1, 1)
plt.plot(hs[1:], (functionals[1:] - functionals[0])/hs[1:], marker='o')
ax.axhline(np.dot(gradient_ad, step_dir))
plt.show()

# breakpoint()

# df = pd.DataFrame(columns=['2nd Order Taylor Remainder'])
# Project the gradient in the direction of elastic modulus increase
# grad_ad_projected = np.sum(np.array(gradient_ad) * step_dir)
# functional_ad = functionals[0] + grad_ad_projected * hs

# grad_fd_projected = (functionals[1:]-functionals[0])/hs[1:]
# error = np.abs((grad_ad_projected-grad_fd_projected)/grad_ad_projected)*100

# fig, axs = plt.subplots(1, 2, figsize=(7, 3))

# axs[0].plot(hs, functional_ad, color='C1', marker='o',
#             label="Linear prediction from gradient")
# axs[0].plot(hs, functionals, color='C0', marker='o',
#             label="Directly computed functionals")
# axs[1].plot(1 + np.arange(error.size), error)

# # Formatting
# axs[0].set_xlim(hs[[0, -1]])
# axs[0].set_xlabel("Elastic modulus [Pa]")
# axs[0].set_ylabel("Objective function")
# axs[0].legend()

# for ax in axs:
#     ax.grid()

# axs[1].set_xlim([0, error.size])
# axs[1].set_ylim([0, error.max()])

# axs[1].set_xlabel(r"$N_{\Delta h}$")
# axs[1].set_ylabel(r"% Error")

# plt.tight_layout()
# plt.show()

# print(f"Gradient norm {np.linalg.norm(gradient_ad):.16e}")
# print(f"Linear gradient prediction {grad_ad_projected:.16e}")
# print(f"Actual FD values {grad_fd_projected[0]:.16e}")
# print(f"% Error {error}")
