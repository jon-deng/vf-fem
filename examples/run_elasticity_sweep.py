"""
Runs the model over a range of elastic moduli.

I want to use this to see if parameter changes over collisions result in
smooth functionals.

Plotting the case with
emod = 11.8e3 * constants.PASCAL_TO_CGS
elastic_moduli = np.linspace(emod+12*constants.PASCAL_TO_CGS, emod+13*constants.PASCAL_TO_CGS, 100)

shows a non-smooth behaviour at a peak however, the portions to the left and right are pretty nice!
"""

import os
import sys
from time import perf_counter

import h5py
import numpy as np
import dolfin as dfn

sys.path.append('../')
from femvf import forms as frm
from femvf.forward import forward
from femvf.adjoint import adjoint
from femvf import constants
from femvf import functionals

def gradient(tspan, dt, solid_props, fluid_props):
    """
    Returns the gradient and functional value for a functional.
    """
    with h5py.File('_tmp.h5', mode='w') as f:
        pass
    forward(tspan, dt, solid_props, fluid_props, h5file='_tmp.h5')

    totalfluidwork = None
    totalinputwork = None
    with h5py.File('_tmp.h5', mode='r') as f:
        totalfluidwork = functionals.totalfluidwork(0, f)
        totalinputwork = functionals.totalinputwork(0, f)
    fkwargs = {'cache_totalfluidwork': totalfluidwork, 'cache_totalinputwork': totalinputwork}
    grad = adjoint(solid_props, '_tmp.h5', dg_du_kwargs=fkwargs)
    return totalfluidwork/totalinputwork, grad

if __name__ == '__main__':
    dfn.set_log_level(30)

    save_path = f"out/elasticity_sweep.h5"
    tspan = []
    dt = 1e-4

    fluid_props = constants.DEFAULT_FLUID_PROPERTIES

    # fluid_props['p_sub'] = [1500 * constants.PASCAL_TO_CGS, 1500 * constants.PASCAL_TO_CGS, 1, 1]
    # fluid_props['p_sub_time'] = [0, 3e-3, 3e-3, 0.02]
    # fluid_props['p_sub'] = 800

    emod = frm.emod.vector()[:].copy()
    solid_props = {'elastic_modulus': emod}

    grad = gradient([0, 0.1], dt, solid_props, fluid_props)[1]
    grad = grad/grad.max()
    elastic_moduli = emod + 100*np.arange(50)[..., None]*grad
    d_elastic_moduli = grad

    with h5py.File(save_path, mode='w') as f:
        f.create_dataset('d_elastic_moduli', data=d_elastic_moduli)
        f.create_dataset('elastic_moduli', data=elastic_moduli)
        f.create_dataset('objective', shape=(elastic_moduli.shape[0],))

    for ii, elastic_modulus in enumerate(elastic_moduli):
        solid_props = {'elastic_modulus': elastic_modulus}

        runtime_start = perf_counter()
        forward([0, 0.05], dt, solid_props, fluid_props, save_path, h5group=f'{ii}', show_figure=True)
        runtime_end = perf_counter()

        objective = None
        with h5py.File(save_path, mode='a') as f:
            f['objective'][ii] = functionals.totalvocaleff(0, f, h5group=f'{ii}')

        runtime = runtime_end - runtime_start
        print(f"Runtime: {runtime:.2f} s")
