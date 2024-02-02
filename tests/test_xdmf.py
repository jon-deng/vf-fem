"""
Playing around with making an XDMF file to read results in paraview

Writes out vertex values from the statefile
"""

import pytest

import os
import numpy as np

from femvf import statefile as sf
from femvf.load import load_transient_fsi_model
from femvf.models.transient.solid import Rayleigh
from femvf.models.transient.fluid import BernoulliAreaRatioSep
from femvf.forward import integrate

from femvf.vis.xdmfutils import export_vertex_values, write_xdmf

@pytest.fixture()
def mesh_path():
    mesh_dir = '../meshes'
    mesh_name = 'M5_BC--GA3--DZ0.00'
    return os.path.join(mesh_dir, f'{mesh_name}.msh')

@pytest.fixture()
def model(mesh_path):
    model = load_transient_fsi_model(
        mesh_path, None, SolidType=Rayleigh, FluidType=BernoulliAreaRatioSep
    )

    return model

@pytest.fixture()
def state_controls_prop(model):
    state = model.state0
    state[:] = 0

    prop = model.prop
    prop['emod'][:] = 5e4
    prop['rho'][:] = 1.0
    prop.print_summary()

    control = model.control
    for n in range(len(model.fluids)):
        control[f'fluid{n}.psub'] = 500*10
    control.print_summary()
    # breakpoint()
    return state, [control], prop

@pytest.fixture()
def state_fpath(model, state_controls_prop):
    state_fpath = 'out/test_xdmf.h5'
    with sf.StateFile(model, state_fpath, mode='w') as f:
        state, controls, prop = state_controls_prop
        integrate(
            model, f,
            state, controls, prop,
            times=np.linspace(0, 1e-2, 10)
        )

    return state_fpath

def test_write_xdmf(model, state_fpath):

    visfile_path = './test_xdmf--export.h5'

    with sf.StateFile(model, state_fpath, mode='r') as state_file:
        export_vertex_values(model, state_file, visfile_path)
    write_xdmf(model, visfile_path)
