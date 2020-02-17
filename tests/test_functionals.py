import sys
import os
import os.path as path
from time import perf_counter
import unittest

import numpy as np
import matplotlib.pyplot as plt
import dolfin as dfn

sys.path.append('../')
from femvf.forward import forward
from femvf import forms
from femvf import statefile as sf
from femvf.properties import SolidProperties, FluidProperties, TimingProperties
from femvf.constants import PASCAL_TO_CGS

from femvf import functionals as basic_functionals

sys.path.append(path.expanduser('~/lib/vf-optimization'))
from vfopt import functionals

class TestFunctionals(unittest.TestCase):
    OVERWRITE_FORWARD_SIMULATIONS = False

    def setUp(self):
        """
        Generates the forward model run data
        """
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

        timing_props = TimingProperties(**{'t0': t0, 'tmeas': tmeas, 'dt_max': dt_max})
        solid_props = SolidProperties(model)
        fluid_props = FluidProperties(model)

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
        if path.isfile(h5file) and not self.OVERWRITE_FORWARD_SIMULATIONS:
            print("Forward model states already exist. Using existing file.")
        else:
            print("Running forward model to generate data.")
            forward(model, solid_props, fluid_props, timing_props, h5file=h5file,
                    abs_tol=None)

        self.h5file = h5file
        self.model = model

    def test_functionals(self):
        model = self.model
        ## Set the functional to test
        # Functional = basic_functionals.DisplacementNorm
        # gkwargs = {}

        Functional = basic_functionals.VelocityNorm
        gkwargs = {}

        # Functional = functionals.AcousticEfficiency
        # gkwargs = {'tukey_alpha': 0.1}

        # Functional = functionals.AcousticPower
        # gkwargs = {'tukey_alpha': 0.1}

        # Functional = basic_functionals.SubglottalWork
        # gkwargs = {'tukey_alpha': 0.1}

        # Functional = basic_functionals.FluidtoSolidWork
        # gkwargs = {'tukey_alpha': 0.1}

        # Functional = functionals.T1RegAcousticEfficiency
        # gkwargs = {'tukey_alpha': 0.2, 'lambda': 0}

        # Functional = basic_functionals.StrainEnergy
        # gkwargs = {}

        # Set the direction and step size to test the gradient of the functional
        np.random.seed(123)
        du = np.random.rand(*self.model.u0.vector()[:].shape)

        alpha = 1e-6

        #############################################
        n = 125
        n = 15
        x_0 = None
        func_0 = None

        dfunc_du_an, dfunc_dv_an, dfunc_da_an = None, None, None
        dfunc_du_fd, dfunc_dv_fd, dfunc_da_fd = None, None, None

        with sf.StateFile(self.h5file, mode='a') as f:
            x_0 = f.get_state(n)
            g = Functional(model, f, **gkwargs)

            model.set_ini_state(*x_0)
            alpha_u = alpha * min(x_0[0].max(), model.get_collision_gap())
            alpha_v = alpha * x_0[1].max()
            alpha_a = alpha * x_0[2].max()

            func_0 = g()

            iter_params0, iter_params1 = f.get_iter_params(n), f.get_iter_params(n+1)
            dfunc_du_an = g.du(n, iter_params0, iter_params1)
            dfunc_dv_an = g.dv(n, iter_params0, iter_params1)
            dfunc_da_an = g.da(n, iter_params0, iter_params1)

            dfunc_du_fd = (functional_wrapper(g, x_0[0]+alpha_u*du, f, i=0, n=n)-func_0) / alpha_u
            dfunc_dv_fd = (functional_wrapper(g, x_0[1]+alpha_v*du, f, i=1, n=n)-func_0) / alpha_v
            dfunc_da_fd = (functional_wrapper(g, x_0[2]+alpha_a*du, f, i=2, n=n)-func_0) / alpha_a

        # print(f"dg/du_n [:10] = {dfunc_du_an[:10]}")
        # print("||dg/du_n|| = " f"{dfunc_du_an.norm('l2')}")
        print(np.dot(dfunc_du_an, du), dfunc_du_fd)
        print(np.dot(dfunc_dv_an, du), dfunc_dv_fd)
        print(np.dot(dfunc_da_an, du), dfunc_da_fd)

# Calculate a functional along a perturbed direction
def functional_wrapper(g, u, f, i=0, n=0):
    """
    Return the functional with state `u_i` replaced at index `n` for file `f`.

    Parameters
    ----------
    g : callable
        The Functional object associated with file `f`
    u : dfn.Vector
        A tuple of (u, v, a) states to replace the original values
    f : statefile.StateFile
    """
    # Get the original state and subsitute the values in u to compute the modified functional
    x_orig = f.get_state(n)
    x_subs = []
    for j, component in enumerate(x_orig):
        if j == i:
            x_subs.append(u)
        else:
            x_subs.append(component)

    # Try to change the state but reset to its original value if something goes wrong
    out = None
    try:
        f.set_state(n, x_subs)
        out = g()
    except:
        raise
    finally:
        f.set_state(n, x_orig)

    return out

if __name__ == '__main__':
    unittest.main()

