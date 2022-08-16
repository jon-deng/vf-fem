"""
Various tests of the dynamical fluid models
"""

import numpy as np
import matplotlib.pyplot as plt

from femvf import load
from femvf.models.dynamical import (
    fluid as dynfld,
    solid as dynsld,
    coupled as dyncpd
)

def setup_model():
    """
    Return a dynamical fluid model to test
    """
    # This is the surface coordinate [cm]
    s = np.linspace(0, 1, 70)
    # return dynfld.BernoulliSmoothMinSep(s)
    # return dynfld.BernoulliFixedSep(s, idx_sep=15)
    mesh_path = '../meshes/M5_CB_GA1.msh'
    model = load.load_dynamical_fsi_model(
        mesh_path,
        None,
        SolidType=dynsld.KelvinVoigt,
        FluidType=dynfld.BernoulliFixedSep,
        fsi_facet_labels=('pressure',),
        fixed_facet_labels=('fixed',),
        separation_vertex_label='separation-inf'
    )
    return model

def test_pressure_qualitative(model):
    """
    Generate qualitative plot of predicted Bernoulli pressures
    """
    ymax_mesh = model.solid.forms['mesh.mesh'].coordinates()[:, 1].max()

    props = model.props
    ygap = 0.1 / 10
    props['ymid'] = ymax_mesh + ygap
    print(ymax_mesh)
    props['rho_air'] = 1e-3
    # props['r_sep'] = 1.5
    # props['zeta_min'] = 1e-4
    # props['zeta_sep'] = 1e-3
    model.set_props(props)
    print(model.props['ymid'][:])

    control = model.control
    control['psub'][:] = 800.0 * 10
    model.set_control(control)

    model.state[:] = 0.0
    # Setting the state should automatically update the fluid area control
    model.set_state(model.state)

    qp = -model.fluid.assem_res()
    print(f"Residual has shape {qp.bshape}")

    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(model.fluid.s, qp['p'])
    axs[0].set_ylabel("p [Ba]")

    axs[1].plot(model.fluid.s, model.fluid.control['area'])
    axs[1].set_ylabel("Area [cm]")
    axs[1].set_xlabel("s [cm]")
    fig.savefig('out/test_dynamical_coupled.png')

    print(model.assem_dres_dstate().bshape)
    print(model.assem_dres_dcontrol().bshape)
    print(model.assem_dres_dprops().bshape)

if __name__ == '__main__':
    model = setup_model()

    test_pressure_qualitative(model)