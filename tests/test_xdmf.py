"""
Playing around with making an xmf file to read results in paraview

Writes out vertex values from the statefile
"""
import sys
import os
from os import path

import h5py
import numpy as np
import dolfin as dfn
# import xml

sys.path.append('../')
from femvf.meshutils import load_fenics_mesh
from femvf import statefile as sf
from femvf.model import ForwardModel
from femvf.constants import PASCAL_TO_CGS

from femvf.solids import Rayleigh
from femvf.fluids import Bernoulli

from femvf.vis.xdmfutils import export_vertex_values, write_xdmf

mesh_dir = '../meshes'

mesh_base_filename = 'geometry2'
mesh_path = path.join(mesh_dir, mesh_base_filename + '.xml')

facet_labels = {'pressure': 1, 'fixed': 3}
cell_labels = {}

## Set the model and various simulation parameters (fluid/solid properties, time step etc.)
mesh, facet_func, cell_func = load_fenics_mesh(mesh_path, facet_labels, cell_labels)
solid = Rayleigh(mesh, facet_func, facet_labels, cell_func, cell_labels)

fluid = Bernoulli()
model = ForwardModel(solid, fluid)

statefile_path = './test_forward.h5'
visfile_path = './test_forward-vis.h5'

export_vertex_values(model, statefile_path, visfile_path)

write_xdmf(model, visfile_path)
