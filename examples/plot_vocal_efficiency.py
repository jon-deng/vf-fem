"""
Plots the vocal efficiency vs time
"""

import numpy as np
from matplotlib import pyplot as plt

import h5py

sys.path.append('../')
from femvf import forms as frm

vocal_eff = None
time = None
glottal_area = None
u = None
with h5py.File('out/forward_pass_state_hist-0.h5', mode='r') as f:
    vocal_eff = f['vocal_efficiency'][...]
    time = f['time'][...]
    u = f['u'][...]

xy_ref = frm.mesh.coordinates()
xy_cur = xy_ref + u[:, frm.vert_to_vdof.reshape(-1)].reshape(u.shape[0], -1, 2)

# Select only y-coords from the mesh position
amin = frm.y_midline - np.amax(xy_cur[..., 1], axis=-1)

fig, axs = plt.subplots(2, 1, figsize=(6, 4))

axs[0].plot(time, amin)
axs[1].plot(time[:-1], vocal_eff*100)

axs[0].set_xticklabels([])
axs[1].set_xlabel("Time [s]")

axs[0].set_ylabel(r"Glottal area $[\mathrm{cm}^2]$")
axs[1].set_ylabel(r"Vocal efficiency $[\%]$")

for ax in axs:
    ax.set_xlim(time[[0, -1]])

plt.tight_layout()
plt.show()
