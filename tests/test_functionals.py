"""
This modules implements tests for functionals
"""

import os
import os.path as path
# from time import perf_counter
import unittest

from tabulate import tabulate

import numpy as np
# import matplotlib.pyplot as plt
import dolfin as dfn

from femvf import statefile as sf
from femvf.forward import integrate
from femvf.models import load_fsi_model, KelvinVoigt, Rayleigh, Bernoulli
from femvf.constants import PASCAL_TO_CGS

from femvf.functionals import basic

class TestFunctionals(unittest.TestCase):
    OVERWRITE_FORWARD_SIMULATIONS = False

    def setUp(self):
        """
        Solves the forward model about a base parameterization

        This generates a history of model states. Functionals are then tested by finite differences
        around the history of model states.
        """
        dfn.set_log_level(30)

        ## Load the model
        mesh_dir = '../meshes'
        mesh_base_filename = 'geometry2'
        mesh_path = path.join(mesh_dir, mesh_base_filename + '.xml')

        model = load_fsi_model(mesh_path, None, Solid=Rayleigh, Fluid=Bernoulli)

        ## Set time integration / fluid / solid parameters
        dt_max = 1e-4
        t0 = 0.0

        t_start = 0.0
        t_final = 0.2

        # Set a tmeas for FFT
        tmeas = np.linspace(t_start, t_final, 512)
        timing_props = tmeas

        # Set parameters for the simulation
        solid_props = model.solid.get_properties_vec()
        fluid_props = model.fluid.get_properties_vec()

        y_gap = 0.01
        alpha, k, sigma = -3000, 50, 0.002
        p_sub = 800
        fluid_props['y_midline'][()] = np.max(model.solid.mesh.coordinates()[..., 1]) + y_gap
        fluid_props['p_sub'][()] = p_sub * PASCAL_TO_CGS
        fluid_props['alpha'][()] = alpha
        fluid_props['k'][()] = k
        fluid_props['sigma'][()] = sigma

        # Use an elasticity that varies linearly with y coordinate.
        y = model.solid.mesh.coordinates()[..., 1]
        y_max = y.max()
        y_min = y.min()

        emod_at_ymin = 5e3*PASCAL_TO_CGS
        emod_at_ymax = 5e3*PASCAL_TO_CGS
        emod = emod_at_ymax*(y-y_min)/(y_max-y_min) + emod_at_ymin*(y-y_max)/(y_min-y_max)

        vert_to_sdof = np.sort(model.solid.vert_to_sdof)
        solid_props['emod'][:] = emod[vert_to_sdof]
        solid_props['rayleigh_m'][()] = 0
        solid_props['rayleigh_k'][()] = 3e-4
        solid_props['k_collision'][()] = 1e11
        solid_props['y_collision'][()] = fluid_props['y_midline'] - y_gap*1/2

        h5file = 'out/test_functionals.h5'
        if path.isfile(h5file) and not self.OVERWRITE_FORWARD_SIMULATIONS:
            print("Forward model states already exist. Using existing file.")
        else:
            if path.isfile(h5file):
                os.remove(h5file)
            print("Running forward model to generate data.")
            adaptive_step_prm = {'abs_tol': None}
            info = integrate(model, (0, 0, 0), solid_props, fluid_props, timing_props,
                             h5file=h5file, adaptive_step_prm=adaptive_step_prm)

        self.h5file = h5file
        self.model = model

    def test_functionals(self):
        model = self.model
        ## Set the functional to test
        # Functional = basic.DisplacementNorm
        # Functional = basic.FinalDisplacementNorm
        # Functional = basic.VelocityNorm
        # Functional = basic.FinalVelocityNorm
        # Functional = functionals.AcousticEfficiency
        # Functional = functionals.AcousticPower
        # Functional = functionals.F0WeightedAcousticPower
        # Functional = basic.SubglottalWork
        # Functional = basic.FluidtoSolidWork
        Functional = basic.StrainWork

        gkwargs = {}

        # Functional = functionals.T1RegAcousticEfficiency
        # gkwargs = {'tukey_alpha': 0.2, 'lambda': 0}

        # Functional = basic.StrainEnergy
        # gkwargs = {}

        g = Functional(model)
        g.constants.update(gkwargs)

        # Set the direction and step size to test the gradient of the functional
        np.random.seed(123)
        du = np.random.rand(*self.model.solid.u0.vector()[:].shape)

        alpha = 1e-8

        idx_meas = None
        with sf.StateFile(self.model, self.h5file, mode='r') as f:
            idx_meas = f.get_meas_indices()
        n = 125
        n = 15
        n = idx_meas[26]

        x_0 = None
        func_0 = None

        dfunc_du_an, dfunc_dv_an, dfunc_da_an = None, None, None
        dfunc_du_fd, dfunc_dv_fd, dfunc_da_fd = None, None, None

        with sf.StateFile(self.model, self.h5file, mode='a') as f:
            x_0 = f.get_state(n)

            model.set_ini_solid_state((x_0[0], x_0[1], x_0[2]))
            alpha_u = alpha * min(x_0[0].max(), model.get_collision_gap())
            alpha_v = alpha * x_0[1].max()
            alpha_a = alpha * x_0[2].max()

            func_0 = g.eval(f)

            iter_params0, iter_params1 = f.get_iter_params(n), f.get_iter_params(n+1)
            dfunc_du_an, dfunc_dv_an, dfunc_da_an = g.duva(f, n, iter_params0, iter_params1)

            dfunc_du_fd = (functional_wrapper(g, x_0[0]+alpha_u*du, f, i=0, n=n)-func_0) / alpha_u
            dfunc_dv_fd = (functional_wrapper(g, x_0[1]+alpha_v*du, f, i=1, n=n)-func_0) / alpha_v
            dfunc_da_fd = (functional_wrapper(g, x_0[2]+alpha_a*du, f, i=2, n=n)-func_0) / alpha_a

        header = ["", "dg/dalpha analytical", "dg/dalpha finite difference"]
        table = [["dg/du", np.dot(dfunc_du_an, du), dfunc_du_fd],
                 ["dg/dv", np.dot(dfunc_dv_an, du), dfunc_dv_fd],
                 ["dg/da", np.dot(dfunc_da_an, du), dfunc_da_fd]]
        print(tabulate(table, headers=header))

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
    # Get the original state and substitute the values in u to compute the modified functional
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
        out = g.eval(f)
    except:
        raise
    finally:
        f.set_state(n, x_orig)

    return out

if __name__ == '__main__':
    unittest.main()
