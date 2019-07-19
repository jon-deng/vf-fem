"""
Verifies if the adjoint is working.

I'm using CGS : cm-g-s units
"""

import sys
from os.path import join

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import tri

import dolfin as dfn

sys.path.append('../')
from femvf import forms as frm
from femvf.fluids import fluid_pressure
from femvf import constants as const

from femvf.functionals import totalvocaleff

# Loading data
elastic_moduli = None
cost = list()

elastic_moduli = None
d_elastic_moduli = None
with h5py.File('out/elasticity_sweep.h5', mode='r') as f:
    elastic_moduli = f['elastic_moduli'][...]
    d_elastic_moduli = f['d_elastic_moduli'][...]
    cost = f['objective'][...]
    for ii in range(elastic_moduli.shape[0]):
        ntime = f[join(f'{ii}', 'u')].shape[0]

        # cost.append(totalvocaleff(0, f, h5group=f'{ii}'))

        # cost.append(np.mean(f[join(f'{ii}', 'fluid_work')]))
        # cost.append(np.mean(max_y_displacement))
        # cost.append(np.sum(f[join(f'{ii}', 'cost')]))

        # Additional functionals you might want to look at
        # max_y_displacement = np.max(f[join(f'{ii}', 'u')][:, 1::2], axis=-1)

## Figure generation
fig, axs = plt.subplots(1, 2, figsize=(7, 3))

## Plotting the functional @ different steps
cost = np.array(cost)
axs[0].plot(cost*100, color='C0', marker='o')

axs[0].set_xlabel("Step")
axs[0].set_ylabel("Vocal Efficiency [%]")
# axs[0].legend()

axs[0].grid()

## Plotting the direction of the step
axs[1].set_aspect('equal')

coords = frm.mesh.coordinates()[...]
triangulation = tri.Triangulation(coords[:, 0], coords[:, 1], triangles=frm.mesh.cells())

axs[1].set_xlim(-0.1, frm.thickness_bottom+0.1, auto=False)
axs[1].set_ylim(0.0, frm.depth+0.1)

axs[1].axhline(y=frm.y_midline, ls='-.')
axs[1].axhline(y=frm.y_midline-frm.collision_eps, ls='-.', lw=0.5)

axs[1].set_title('Step Direction')

mappable = axs[1].tripcolor(triangulation, d_elastic_moduli[frm.vert_to_sdof], edgecolors='k',
                            shading='flat')
coords_fixed = frm.mesh.coordinates()[frm.fixed_vertices]
axs[1].plot(coords_fixed[:, 0], coords_fixed[:, 1], color='C1')

fig.colorbar(mappable, ax=axs[1])

plt.tight_layout()
plt.show()
