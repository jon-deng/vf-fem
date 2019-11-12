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
    n = 2
    x_0 = None
    func_0 = None
    dfunc_du_an = None

    x_0 = f.get_state(n)
    g = Functional(model, f, **gkwargs)
    func_0 = g()
    dfunc_du_an = g.du(n)

    # calculate the functional along a perturbed direction
    def functional_wrapper(u, n=0):
        """
        Evaluate the functional with state `u` replaced at index `n`.
        """
        x = (u, *x_0[1:])
        f.set_state(n, x)

        return g()

    # Use a random search direction
    du = np.random.rand(*x_0[0].shape)

    model.set_initial_state(*x_0)
    alpha = 1e-2 * min(x_0[0].max(), model.get_collision_gap())
    func_1 = functional_wrapper(x_0[0]+alpha*du, n=n)

    # Reset the state to it has its original value
    f.set_state(n, x_0)

    print(func_1-func_0)
    print(np.sum(alpha*np.multiply(dfunc_du_an, du)))

if __name__ == '__main__':
    dfn.set_log_level(30)
    #### Test setup
    # To setup the test, first generate a set of state by solving the forward model
    mesh_dir = '../meshes'

    mesh_base_filename = 'geometry2'
    mesh_path = path.join(mesh_dir, mesh_base_filename + '.xml')

    model = forms.ForwardModel(mesh_path, {'pressure': 1, 'fixed': 3}, {})

    ## Set the solution parameters
    dt_max = 1e-4
    t0 = 0.0
    t_final = 0.1
    tmeas = np.linspace(0, t_final, 256)
    dtmax = 1e-4

    fluid_props = FluidProperties()
    solid_props = SolidProperties()

    h5file = 'out/test_functionals.h5'
    if not path.isfile(h5file):
        forward(model, t0, tmeas, dt_max, solid_props, fluid_props, h5file=h5file)

    #### Functionals
    Functional = functionals.AcousticEfficiency
    gkwargs = {'m_start': 0}
    with sf.StateFile(h5file, mode='a') as f:
        test_functional(Functional, model, f, gkwargs)
