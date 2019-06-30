"""
Compare gradient computed via adjoint method with gradient computed via FD.
"""

from time import perf_counter

import h5py
import dolfin as dfn

from forward import forward
from adjoint import adjoint
import forms as frm
import constants

if __name__ == '__main__':
    dfn.set_log_level(30)
    ## Running finite differences
    print("Computing FD")
    emod = frm.emod.vector()[:].copy()

    # Constant fluid properties
    # fluid_props = constants.DEFAULT_FLUID_PROPERTIES

    # Time varying fluid properties
    fluid_props = constants.DEFAULT_FLUID_PROPERTIES
    fluid_props['p_sub'] = [1500 * constants.PASCAL_TO_CGS, 1500 * constants.PASCAL_TO_CGS, 1, 1]
    fluid_props['p_sub_time'] = [0, 3e-3, 3e-3, 0.02]

    step_size = 0.1*constants.PASCAL_TO_CGS
    num_steps = 10

    save_path = 'out/FiniteDifferenceStates.h5'
    with h5py.File(save_path, mode='w') as f:
        f.create_dataset('elastic_modulus', data=emod[0])
        f.create_dataset('step_size', data=step_size)
        f.create_dataset('num_steps', data=num_steps)

    for ii in range(num_steps):
        tspan = [0, 0.005]
        solid_props = {'elastic_modulus': emod + ii*step_size}

        runtime_start = perf_counter()
        forward(tspan, solid_props, fluid_props, h5path=save_path, h5group=f'{ii}/',
                show_figure=False)
        runtime_end = perf_counter()

        print(f"Runtime {runtime_end-runtime_start:.2f} seconds")

    ## Running adjoint
    print("Computing Adjoint")

    solid_props = {'elastic_modulus': emod}
    runtime_start = perf_counter()
    gradient = adjoint(solid_props, save_path, h5group='0')
    runtime_end = perf_counter()

    print(f"Runtime {runtime_end-runtime_start:.2f} seconds")

    save_path = 'out/Adjoint.h5'
    with h5py.File(save_path, mode='w') as f:
        f.create_dataset('gradient', data=gradient)
