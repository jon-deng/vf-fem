"""
Playing around with making an xmf file to read results in paraview

Writes out vertex values from the statefile
"""
# import os
from os import path

# import h5py
# import numpy as np
# import dolfin as dfn
# import xml

from femvf.model import load_fsi_model
from femvf.solids import Rayleigh
from femvf.fluids import Bernoulli

from femvf.vis.xdmfutils import export_vertex_values, write_xdmf

mesh_dir = '../meshes'

mesh_base_filename = 'geometry2'
mesh_path = path.join(mesh_dir, mesh_base_filename + '.xml')

## Set the model and various simulation parameters (fluid/solid properties, time step etc.)
model = load_fsi_model(mesh_path, None, Solid=Rayleigh, Fluid=Bernoulli)

statefile_path = './test_forward.h5'
visfile_path = './test_forward-vis.h5'

export_vertex_values(model, statefile_path, visfile_path)

write_xdmf(model, visfile_path)
