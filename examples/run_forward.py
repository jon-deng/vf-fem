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
from femvf import forms as frm
from femvf import constants
from femvf import functionals

if __name__ == '__main__':
    dfn.set_log_level(30)

    # Solid and Fluid properties
    emod = frm.emod.vector()[:].copy()

    cell_to_vertex = np.array([[vertex.index() for vertex in dfn.vertices(cell)]
                               for cell in dfn.cells(frm.mesh)])
    normal_cells = frm.body_marker.where_equal(2)
    inclusion_cells = frm.body_marker.where_equal(1)

    normal_vertices = np.unique(cell_to_vertex[normal_cells].reshape(-1))
    inclusion_vertices = np.unique(cell_to_vertex[inclusion_cells].reshape(-1))

    emod[frm.vert_to_sdof[normal_vertices]] = 10e3 * constants.PASCAL_TO_CGS
    emod[frm.vert_to_sdof[inclusion_vertices]] = 10e3 * constants.PASCAL_TO_CGS

    solid_props = {'elastic_modulus': emod}
    # import ipdb; ipdb.set_trace()

    # Constant fluid properties
    fluid_props = constants.DEFAULT_FLUID_PROPERTIES

    dt = 1e-4
    times_meas = [0, 0.1]

    h5file = 'forward-noinclusion.h5'
    if os.path.exists(h5file):
        os.remove(h5file)

    runtime_start = perf_counter()
    forward(0, times_meas, dt, solid_props, fluid_props, h5file=h5file)
    runtime_end = perf_counter()
