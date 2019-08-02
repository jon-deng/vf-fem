"""
Verifies if the adjoint is working.

I'm using CGS : cm-g-s units
"""

import sys
from os.path import join

import h5py
import numpy as np

from matplotlib import pyplot as plt

import dolfin as dfn
#import ufl

sys.path.append('../')
from femvf import forms as frm
from femvf.fluids import fluid_pressure
from femvf import constants as const
from femvf import functionals

# Specify the functional
functional = functionals.totalvocaleff
functional = functionals.mfdr
functional = functionals.wss_glottal_width

# Load data and caculate functional value at each FD step
emod = None
cost_fd = list()
with h5py.File('out/FiniteDifferenceStates.h5', mode='r') as f:
    step_size = f['step_size'][()]
    num_steps = f['num_steps'][()]
    emod = f['elastic_modulus'] + np.arange(num_steps)*step_size
    for ii in range(num_steps):
        cost_fd.append(functional(0, f, h5group=f'{ii}'))

# Load the gradient from the adjoint method
grad_ad = None
with h5py.File('out/Adjoint.h5', mode='r') as f:
    grad_ad = f['gradient'][...]

### Figure generation
fig, axs = plt.subplots(1, 2, figsize=(7, 3))

## Plotting
cost_fd = np.array(cost_fd)
axs[0].plot(emod, cost_fd, color='C0', marker='o', label='Functional from F')

# Project the gradient in the direction of uniform increase in elastic modulus
demod = emod-emod[0]
grad_ad_projected = grad_ad.sum()
cost_ad = cost_fd[0] + grad_ad_projected*demod
axs[0].plot(emod, cost_ad, color='C1', marker='o', label='Linear prediction from gradient')

grad_fd_projected = (cost_fd[1:]-cost_fd[0])/(demod[1:])
error = np.abs((grad_ad_projected-grad_fd_projected)/grad_ad_projected)*100
axs[1].plot(np.arange(error.size)+1, error)

## Formatting
axs[0].set_xlabel("Elastic modulus [Pa]")
axs[0].set_ylabel("Cost funciton")
axs[0].legend()

axs[1].set_ylabel(r"% Error")

for ax in axs:
    ax.grid()

axs[0].set_xlabel(r"$E$")
axs[1].set_xlabel(r"$N_{\Delta h}$")

axs[0].set_xlim(emod[[0, -1]])
axs[1].set_xlim([0, error.size])
axs[1].set_ylim([0, error.max()])

plt.tight_layout()
plt.show()

print(f"Adjoint prediction {grad_ad_projected:.16e}")
print(f"FD prediction {grad_fd_projected[0]:.16e}")
print(f"% Error {error}")
