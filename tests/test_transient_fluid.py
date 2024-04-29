"""
Various tests of the dynamical fluid models
"""

import numpy as np
import matplotlib.pyplot as plt

from femvf.models.transient import fluid as tfld


def setup_model():
    """
    Return a dynamical fluid model to test
    """
    # This is the surface coordinate [cm]
    s = np.linspace(0, 1, 70)
    # return tfld.BernoulliSmoothMinSep(s)
    return tfld.BernoulliFixedSep(s, idx_sep=15)


def test_pressure_qualitative(model):
    """
    Generate qualitative plot of predicted Bernoulli pressures
    """

    prop = model.prop
    # prop['zeta_min'] = 1e-4
    # prop['zeta_sep'] = 1e-3
    model.set_prop(prop)

    s = model.s
    ygap = 0.1
    control = model.control
    control['psub'][:] = 800.0 * 10
    control['area'][:] = 1 / 2 * (np.cos(2 * np.pi * s / s.max()) + 1.0 + ygap)
    model.set_control(control)

    # the two below both return the fluid flow rate/pressure
    state1_n = model.state1.copy()
    qp, _ = model.solve_state1(state1_n)
    print(f"Residual has shape {qp.bshape}")

    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(model.s, qp['p'])
    axs[0].set_ylabel("p [Ba]")

    axs[1].plot(model.s, control['area'])
    axs[1].set_ylabel("Area [cm]")
    axs[1].set_xlabel("s [cm]")
    fig.savefig('out/test_transient_fluid.png')

    # print(model.assem_dres_dstate().bshape)
    # print(model.assem_dres_dcontrol().bshape)
    # print(model.assem_dres_dprops().bshape)


if __name__ == '__main__':
    model = setup_model()

    test_pressure_qualitative(model)
