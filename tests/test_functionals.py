import sys
import os
import os.path as path
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
import dolfin as dfn

sys.path.append('../')
from femvf.forward import forward
from femvf import forms
from femvf import statefile as sf
from femvf.properties import SolidProperties, FluidProperties
from femvf.constants import PASCAL_TO_CGS

from femvf import functionals as basic_functionals

sys.path.append(path.expanduser('~/lib/vf-optimization'))
from vfopt import functionals


def test_functional(Functional, model, f, gkwargs):
    """
    Tests the `Functional` `.du` method using finite differences

    Parameters
    ----------
    Functional : type of Functional
    model : ForwardModel
    f : StateFile
        Reference a state file
    """
    n = 125
    x_0 = None
    func_0 = None
    dfunc_du_an = None

    x_0 = f.get_state(n)
    g = Functional(model, f, **gkwargs)
    func_0 = g()

    dfunc_du_an = g.du(n, f.get_iter_params(n-1), f.get_iter_params(n))
    dfunc_dv_an = g.dv(n, f.get_iter_params(n-1), f.get_iter_params(n))
    dfunc_da_an = g.da(n, f.get_iter_params(n-1), f.get_iter_params(n))

    # calculate the functional along a perturbed direction
    def functional_wrapper(u, i=0, n=0):
        """
        Return the functional with state `u_i` replaced at index `n`.
        """
        # Get the original state and subsitute the value to compute the modified functional
        x_orig = f.get_state(n)
        x_subs = []
        for j, component in enumerate(x_orig):
            if j == i:
                x_subs.append(u)
            else:
                x_subs.append(component)

        f.set_state(n, x_subs)

        out = g()

        # Reset the state to its original value
        f.set_state(n, x_orig)

        return out

    # Use a random search direction
    np.random.seed(123)
    du = np.random.rand(*x_0[0].shape)

    model.set_ini_state(*x_0)
    alpha = 1e-1 * min(x_0[0].max(), model.get_collision_gap())
    # alpha = 4e-4 * min(x_0[0].max(), model.get_collision_gap())

    func_1 = functional_wrapper(x_0[0]+alpha*du, n=n)
    # breakpoint()

    # print(func_1)
    # print(func_0)
    print(f"dg/du_n [:10] = {dfunc_du_an[:10]}")
    print("||dg/du_n|| = " f"{dfunc_du_an.norm('l2')}")
    print((func_1-func_0)/alpha)
    print(np.dot(dfunc_du_an, du))

if __name__ == '__main__':
    dfn.set_log_level(30)

    #### Test setup
    # To setup the test, first generate a set of states by solving the forward model
    mesh_dir = '../meshes'

    mesh_base_filename = 'geometry2'
    mesh_path = path.join(mesh_dir, mesh_base_filename + '.xml')

    model = forms.ForwardModel(mesh_path, {'pressure': 1, 'fixed': 3}, {})

    ## Set the solution parameters
    dt_max = 1e-4
    t0 = 0.0

    t_start = 0.0
    t_final = 0.1
    tmeas = np.linspace(t_start, t_final, round((t_final-t_start)/dt_max)+1)

    solid_props = SolidProperties()
    fluid_props = FluidProperties()

    # Use an elasticity that varies linearly with y coordinate.
    y = model.mesh.coordinates()[..., 1]
    y_max = y.max()
    y_min = y.min()

    emod_at_ymin = 5e3*10
    emod_at_ymax = 5e3*10
    emod = emod_at_ymax*(y-y_min)/(y_max-y_min) + emod_at_ymin*(y-y_max)/(y_min-y_max)

    sdof_to_vert = np.sort(model.vert_to_sdof)
    solid_props['elastic_modulus'] = emod[sdof_to_vert]

    h5file = 'out/test_functionals.h5'
    if path.isfile(h5file):
        print("Forward model states already exist. Using existing file.")
    else:
        print("Running forward model to generate data.")
        forward(model, t0, tmeas, dt_max, solid_props, fluid_props, h5file=h5file,
                abs_tol=None)

    #### Functionals
    np.random.seed(123)
    # Functional = functionals.AcousticEfficiency
    # gkwargs = {'tukey_alpha': 0.1}

    # Functional = functionals.AcousticPower
    # gkwargs = {'tukey_alpha': 0.1}

    # Functional = basic_functionals.SubglottalWork
    # gkwargs = {'tukey_alpha': 0.1}

    # Functional = functionals.AcousticEfficiencyReg
    # gkwargs = {'tukey_alpha': 0.2, 'lambda': 0}

    Functional = basic_functionals.DisplacementNorm
    gkwargs = {}

    with sf.StateFile(h5file, mode='a') as f:
        test_functional(Functional, model, f, gkwargs)
