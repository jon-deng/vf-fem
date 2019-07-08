"""
Verifies if the adjoint is working.

I'm using CGS : cm-g-s units
"""

from os.path import join

import h5py
import numpy as np

from matplotlib import pyplot as plt

import dolfin as dfn
#import ufl

import forms as frm
from fluids import fluid_pressure
import constants as const

from functionals import totalvocaleff

# Loading data
elastic_moduli = None
cost = list()

elastic_moduli = None
with h5py.File('out/elasticity_sweep.h5', mode='r') as f:
    elastic_moduli = f['elastic_moduli'][...]
    for ii in range(elastic_moduli.shape[0]):
        ntime = f[join(f'{ii}', 'u')].shape[0]

        cost.append(totalvocaleff(0, f, h5group=f'{ii}'))

        # cost.append(np.mean(f[join(f'{ii}', 'fluid_work')]))
        # cost.append(np.mean(max_y_displacement))
        # cost.append(np.sum(f[join(f'{ii}', 'cost')]))

        # Additional functionals you might want to look at
        # max_y_displacement = np.max(f[join(f'{ii}', 'u')][:, 1::2], axis=-1)

## Figure generation
fig, axs = plt.subplots(1, 2, figsize=(7, 3))

## Plotting
cost = np.array(cost)
axs[0].plot(elastic_moduli, cost, color='C0', marker='o', label='Forward simulation')

## Formatting
axs[0].set_xlabel("Elastic modulus [Pa]")
axs[0].set_ylabel("Cost funciton")
axs[0].legend()

axs[0].grid()

plt.tight_layout()
plt.show()
