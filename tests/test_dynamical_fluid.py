"""
Various tests of the dynamical fluid models
"""

import numpy as np
import matplotlib.pyplot as plt

from femvf.models.dynamical import fluid as dynfld

def setup_model():
    """
    Return a dynamical fluid model to test
    """
    # This is the surface coordinate [cm]
    s = np.linspace(0, 1, 52)
    return dynfld.BernoulliSmoothMinSep(s)

def test_pressure_qualitative(model):
    """
    Generate qualitative plot of predicted Bernoulli pressures
    """
    model.state.set(0.0)

    props = model.props
    props['zeta_min'] = 1e-4
    props['zeta_sep'] = 1e-3
    model.set_props(props)

    s = model.s
    ygap = 0.1
    control = model.control
    control['psub'][:] = 800.0 * 10
    control['area'][:] = 1/2*(np.cos(2*np.pi*s/s.max()) + 1.0 + ygap)
    model.set_control(control)

    qp = -model.assem_res()

    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(model.s, qp['p'])
    axs[0].set_ylabel("p [Ba]")

    axs[1].plot(model.s, control['area'])
    axs[1].set_ylabel("Area [cm]")
    axs[1].set_xlabel("s [cm]")
    fig.savefig('out/test_dynamicalfluid.png')

if __name__ == '__main__':
    model = setup_model()

    test_pressure_qualitative(model)