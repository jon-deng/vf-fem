"""
Plotting some of the optimization results
"""

import sys
from time import perf_counter

import h5py
# import numpy as np
# import dolfin as dfn

sys.path.append('../')
from femvf import forward
from femvf import constants

emod = None
with h5py.File('out/OptimizationElasticModuli.h5', mode='r') as f:
    emod = f['elastic_modulus'][6]

fluid_props = constants.DEFAULT_FLUID_PROPERTIES

save_path = 'yoyoyo.h5'
runtime_start = perf_counter()
forward.forward([0, 0.05], 1e-4, emod, fluid_props, save_path, show_figure=True)
runtime_end = perf_counter()

print(f"Runtime {runtime_end-runtime_start:.2f} seconds")
