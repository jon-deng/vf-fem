"""
Playing around with making an XDMF file to read results in paraviewos

Writes out vertex values from the statefile
"""

import pytest

from os import path
import numpy as np
import h5py
import dolfin as dfn

from femvf import statefile as sf
from femvf.load import load_transient_fsi_model
from femvf.models.transient.solid import Rayleigh
from femvf.models.transient.fluid import BernoulliAreaRatioSep
from femvf.forward import integrate

from femvf.vis.xdmfutils import export_mesh_values, write_xdmf, XDMFArray
from femvf.postprocess.base import TimeSeries
from femvf.postprocess import solid as slpost

@pytest.fixture()
def mesh_path():
    mesh_dir = '../meshes'
    mesh_name = 'M5_BC--GA3--DZ0.00'
    return path.join(mesh_dir, f'{mesh_name}.msh')

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
    # prop.print_summary()

    control = model.control
    for n in range(len(model.fluids)):
        control[f'fluid{n}.psub'] = 500*10
    # control.print_summary()
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

    post_path = f'{path.splitext(state_fpath)[0]}--post.h5'
    with (
            h5py.File(post_path, mode='w') as post_file,
            sf.StateFile(model, state_fpath, mode='r') as state_file
        ):
        post_file['time.field.p'] = TimeSeries(slpost.FSIPressure(model))(state_file)

    xdmf_data_path = './test_xdmf--export.h5'
    xdmf_path = 'test_xdmf--export.xdmf'

    with (
            h5py.File(state_fpath, mode='r') as state_file,
            h5py.File(post_path, mode='r') as post_file
        ):
        datasets = [
            state_file['mesh/solid'],
            state_file['time'],
        ]
        formats = [None, None]
        labels = ['mesh/solid', 'time']

        mesh = model.solid.residual.mesh()
        fspace_cg1_vector = (
            model.solid.residual.form['coeff.state.u1'].function_space()
        )
        vector_labels = ['state/u', 'state/v', 'state/a']
        datasets += [state_file[label] for label in vector_labels]
        formats += len(vector_labels)*[fspace_cg1_vector]
        labels += vector_labels

        fspace_cg1_scalar = dfn.FunctionSpace(mesh, 'CG', 1)
        scalar_labels = ['time.field.p']
        datasets += [post_file[label] for label in scalar_labels]
        formats += len(scalar_labels)*[fspace_cg1_scalar]
        labels += scalar_labels

        with h5py.File(xdmf_data_path, mode='w') as f:
            export_mesh_values(datasets, formats, f)

    with h5py.File(xdmf_data_path, mode='r') as f:
        static_dataset_descrs = [
            (f['state/u'], 'vector', 'node')
        ]
        static_idxs = [
            (0, ...)
        ]
        temporal_dataset_descrs = None
        temporal_idxs = None
        temporal_dataset_descrs = [
            (f['state/u'], 'vector', 'node'),
            (f['state/v'], 'vector', 'node'),
            (f['state/a'], 'vector', 'node'),
            (f['time.field.p'], 'scalar', 'node'),
        ]
        temporal_idxs = len(temporal_dataset_descrs)*[
            (slice(None),)
        ]
        write_xdmf(
            f['mesh/solid'],
            static_dataset_descrs, static_idxs,
            f['time'],
            temporal_dataset_descrs, temporal_idxs,
            xdmf_path
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
