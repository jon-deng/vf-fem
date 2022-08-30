"""
Benchmark/profile time integration of the forward model
"""

import argparse
import cProfile
import timeit

import numpy as np

from blockarray.blockvec import BlockVector

from femvf.forward import integrate
from femvf import statefile as sf
from femvf.load import load_transient_fsi_model
from femvf.models.transient import (fluid as tfmd, solid as tsmd)

def setup_model(mesh_path):
    """
    Load the model to integrate
    """
    model = load_transient_fsi_model(
        mesh_path, None,
        SolidType=tsmd.KelvinVoigtWEpithelium,
        FluidType=tfmd.BernoulliAreaRatioSep,
        fsi_facet_labels=['pressure'],
        fixed_facet_labels=['fixed'],
        coupling='explicit'
    )
    return model

def run_forward(model, state, control, props, times):
    """
    Integrate the forward model over time
    """
    with sf.StateFile(model, 'out/test.h5', mode='w') as f:
        return integrate(model, f, state, [control], props, times, write=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', action='store_true', default=False)
    args = parser.parse_args()

    model = setup_model('../meshes/M5_CB_GA0.msh')

    state0 = model.state0.copy()
    state0[:] = 0

    control = model.control.copy()
    control[:] = 0
    control['psub'] = 8e3

    props = model.props.copy()
    ymax = model.solid.forms['mesh.mesh'].coordinates()[:, 1].max()
    props['emod'] = 5e4
    props['rho'] = 1
    props['eta'] = 3
    props['nu'] = 0.45
    props['ycontact'] = ymax+0.05
    props['kcontact'] = 1e8

    _times = 1e-4*np.arange(100)
    times = BlockVector((_times,), (1,))

    statement = 'run_forward(model, state0, control, props, times)'
    if args.profile:
        cProfile.run(
            statement,
            './benchmark_forward.profile'
        )
    else:
        time = timeit.timeit(statement, globals=globals(), number=1)
        print(f"Runtime: {time:.2e} s")


