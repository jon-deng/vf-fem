"""
Playing around with making an XDMF file to read results in paraview

Writes out vertex values from the statefile
"""

import pytest

import os
import numpy as np
import h5py

from femvf import statefile as sf
from femvf.load import load_transient_fsi_model
from femvf.models.transient.solid import Rayleigh
from femvf.models.transient.fluid import BernoulliAreaRatioSep
from femvf.forward import integrate

from femvf.vis.xdmfutils import export_vertex_values, write_xdmf, XDMFArray

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

    visfile_fpath = './test_xdmf--export.h5'
    xdmf_fpath = './test_xdmf--export.xdmf'

    with sf.StateFile(model, state_fpath, mode='r') as state_file:
        export_vertex_values(model, state_file, visfile_fpath)

    with h5py.File(visfile_fpath, mode='r') as f:
        static_dataset_descrs = [
            (f['state/u'], 'vector', 'node')
        ]
        static_idxs = [
            (0, ...)
        ]
        # temporal_dataset_descrs = None
        # temporal_idxs = None
        temporal_dataset_descrs = [
            (f['state/u'], 'vector', 'node'),
            (f['state/v'], 'vector', 'node'),
            (f['state/a'], 'vector', 'node'),
            (f['p'], 'scalar', 'node'),
        ]
        temporal_idxs = len(temporal_dataset_descrs)*[
            (slice(None),)
        ]
        write_xdmf(
            visfile_fpath,
            static_dataset_descrs, static_idxs,
            temporal_dataset_descrs, temporal_idxs,
            xdmf_fpath
        )


class TestXDMFArray:

    @pytest.fixture()
    def shape(self):
        return (5, 100)

    @pytest.fixture()
    def xdmf_array(self, shape):
        return XDMFArray(shape)

    def test_to_xdmf_slice(self, xdmf_array):
        print(xdmf_array.to_xdmf_slice((0,)))
