"""
Compare gradient computed via adjoint method with gradient computed via FD.
"""

import sys
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

    ## Finite Differences
    print("Computing Gradient via Finite Differences")
    emod = frm.emod.vector()[:].copy()
    step_size = 0.01*constants.PASCAL_TO_CGS
    num_steps = 5

    # Constant fluid properties
    fluid_props = constants.DEFAULT_FLUID_PROPERTIES

    # Time varying fluid properties
    # fluid_props = constants.DEFAULT_FLUID_PROPERTIES
    # fluid_props['p_sub'] = [1500 * constants.PASCAL_TO_CGS, 1500 * constants.PASCAL_TO_CGS, 1, 1]
    # fluid_props['p_sub_time'] = [0, 3e-3, 3e-3, 0.02]

    dt = 1e-4
    # times_meas = [0, 0.005+dt/4, 0.01]
    times_meas = [0, 0.005-dt/4]
    times_meas = [0, 0.005]

    times_meas = [0, 0.01]

    save_path = 'out/FiniteDifferenceStates.h5'
    with h5py.File(save_path, mode='w') as f:
        f.create_dataset('elastic_modulus', data=emod[0])
        f.create_dataset('step_size', data=step_size)
        f.create_dataset('num_steps', data=num_steps)

    for ii in range(num_steps):
        solid_props = {'elastic_modulus': emod + ii*step_size}

        runtime_start = perf_counter()
        forward(times_meas, dt, solid_props, fluid_props, h5file=save_path, h5group=f'{ii}/',
                show_figure=False)
        runtime_end = perf_counter()

        print(f"Runtime {runtime_end-runtime_start:.2f} seconds")

    ## Adjoint
    print("Computing Gradient via Adjoint State")

    # Functional for vocal eff
    # totalfluidwork = None
    # totalinputwork = None
    # with h5py.File(save_path, mode='r') as f:
    #     totalfluidwork = functionals.totalfluidwork(f, h5group='0')[0]
    #     totalinputwork = functionals.totalinputwork(f, h5group='0')[0]
    # fkwargs = {'cache_totalfluidwork': totalfluidwork, 'cache_totalinputwork': totalinputwork}
    # dg_du = functionals.dtotalvocaleff_du
    # functional = functionals.totalvocaleff

    # Functional for MFDR
    # idx_mfdr = None
    # with h5py.File(save_path, mode='r') as f:
    #     idx_mfdr = functionals.mfdr(f, h5group='0')[1]['idx_mfdr']
    # fkwargs = {'cache_idx_mfdr': idx_mfdr}
    # dg_du = functionals.dmfdr_du
    # functional = functionals.mfdr

    # Functional for weighted sum of squared glottal widths
    fkwargs = {}
    dg_du = functionals.dwss_glottal_width_du
    functional = functionals.wss_glottal_width

    solid_props = {'elastic_modulus': emod}
    runtime_start = perf_counter()
    gradient = adjoint(save_path, h5group='0', dg_du=dg_du, dg_du_kwargs=fkwargs)
    runtime_end = perf_counter()

    print(f"Runtime {runtime_end-runtime_start:.2f} seconds")

    save_path = 'out/Adjoint.h5'
    with h5py.File(save_path, mode='w') as f:
        f.create_dataset('gradient', data=gradient)

    ## Compare adjoint and finite differences
    # Load data and caculate gradient from FD steps
    emod = None
    cost_fd = list()
    with h5py.File('out/FiniteDifferenceStates.h5', mode='r') as f:
        step_size = f['step_size'][()]
        num_steps = f['num_steps'][()]
        emod = f['elastic_modulus'] + np.arange(num_steps)*step_size
        for ii in range(num_steps):
            cost_fd.append(functional(f, h5group=f'{ii}')[0])

    # Load the gradient from the adjoint method
    grad_ad = None
    with h5py.File('out/Adjoint.h5', mode='r') as f:
        grad_ad = f['gradient'][...]

    ### Figure generation
    fig, axs = plt.subplots(1, 2, figsize=(7, 3))

    ## Plotting
    cost_fd = np.array(cost_fd)
    axs[0].plot(emod, cost_fd, color='C0', marker='o', label="Functional from F")

    # Project the gradient in the direction of uniform increase in elastic modulus
    demod = emod-emod[0]
    grad_ad_projected = grad_ad.sum()
    cost_ad = cost_fd[0] + grad_ad_projected*demod
    axs[0].plot(emod, cost_ad, color='C1', marker='o', label="Linear prediction from gradient")

    grad_fd_projected = (cost_fd[1:]-cost_fd[0])/(demod[1:])
    error = np.abs((grad_ad_projected-grad_fd_projected)/grad_ad_projected)*100
    axs[1].plot(np.arange(error.size)+1, error)

    ## Formatting
    axs[0].set_xlabel("Elastic modulus [Pa]")
    axs[0].set_ylabel("Objective function")
    axs[0].legend()

    axs[1].set_ylabel(r"% Error")

    for ax in axs:
        ax.grid()

    axs[0].set_xlabel(r"$E$")
    axs[1].set_xlabel(r"$N_{\Delta h}$")

    axs[0].set_xlim(emod[[0, -1]])
    axs[1].set_xlim([0, error.size])
    axs[1].set_ylim([0, error.max()])

    plt.tight_layout()
    plt.show()

    print(f"Linear gradient prediction {grad_ad_projected:.16e}")
    print(f"Actual FD values {grad_fd_projected[0]:.16e}")
    print(f"% Error {error}")
