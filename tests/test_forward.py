"""
Test to see if `forward.integrate` runs
"""

import os
import unittest
import pytest
from time import perf_counter

import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import dolfin as dfn
# import h5py

import femvf.statefile as sf
from femvf.forward import integrate, integrate_linear
from femvf.constants import PASCAL_TO_CGS

from femvf.models.transient import solid as tsmd, fluid as tfmd, acoustic as amd
from femvf.load import load_transient_fsi_model, load_transient_fsai_model
import femvf.postprocess.solid as solidfunc
from femvf.postprocess.base import TimeSeries
# from femvf import callbacks
# from femvf import linalg

from blockarray import blockvec as vec

MESH_DIR = '../meshes'
MESH_BASENAME = 'M5-3layers'
MESH_PATH = os.path.join(MESH_DIR, MESH_BASENAME + '.xml')

class TestIntegrate:

    @pytest.fixture(
        params=[
            ('KelvinVoigt', tsmd.KelvinVoigt, tfmd.BernoulliSmoothMinSep),
            ('Rayleigh', tsmd.Rayleigh, tfmd.BernoulliSmoothMinSep)
        ]
    )
    def model_specification(self, request):
        return request.param

    @pytest.fixture()
    def model(self, model_specification):
        ## Configure the model and its parameters
        case_name, SolidType, FluidType = model_specification
        return load_transient_fsi_model(
            MESH_PATH, None,
            SolidType=SolidType,
            FluidType=FluidType,
            coupling='explicit'
        )

    @pytest.fixture()
    def case_name(self, model_specification):
        case_name, SolidType, FluidType = model_specification
        return case_name

    @pytest.fixture()
    def ini_state(self, model):
        # Set the initial state
        xy = model.solid.XREF[:].copy().reshape(-1, 2)
        x = xy[:, 0]
        y = xy[:, 1]
        u0 = dfn.Function(model.solid.residual.form['coeff.state.u0'].function_space()).vector()

        # model.fluid.set_prop(fluid_props)
        # qp0, *_ = model.fluid.solve_qp0()

        ini_state = model.state0.copy()
        ini_state[:] = 0.0
        ini_state['u'][:] = u0
        # ini_state['q'][()] = qp0['q']
        # ini_state['p'][:] = qp0['p']
        return ini_state

    @pytest.fixture()
    def controls(self, model):
        control = model.control.copy()
        # Set the control vector
        p_sub = 500.0

        control = model.control
        control['psub'][:] = p_sub * PASCAL_TO_CGS
        control['psup'][:] = 0.0 * PASCAL_TO_CGS

        # control['psub'][:] = 0.0 * PASCAL_TO_CGS
        # control['psup'][:] = p_sub * PASCAL_TO_CGS
        return [control]

    @pytest.fixture()
    def prop(self, model):
        # Set the properties
        y_gap = 0.01
        y_midline = np.max(model.solid.residual.mesh().coordinates()[..., 1]) + y_gap

        prop = model.prop.copy()

        prop['ymid'][0] = y_midline

        xy = model.solid.residual.form['coeff.prop.emod'].function_space().tabulate_dof_coordinates()
        x = xy[:, 0]
        y = xy[:, 1]
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        prop['emod'][:] = 1/2*5.0e3*PASCAL_TO_CGS*((x-x_min)/(x_max-x_min) + (y-y_min)/(y_max-y_min)) + 2.5e3*PASCAL_TO_CGS

        # Only set these properties if applicable to the current model
        default_prop = {
            'eta': 4e-3,
            'rho': 1.0,
            'kcontact': 1e11,
            'ycontact': prop['ymid'][0] - y_gap*1/2,
            'zeta_min' : 1e-8,
            'zeta_sep' : 1e-8
        }
        for key, value in default_prop.items():
            if key in prop:
                prop[key] = value

        return prop

    def test_integrate(self, case_name, model, ini_state, controls, prop):

        times = np.linspace(0, 0.01, 100)

        save_path = f'out/test_forward_{case_name}.h5'
        if os.path.isfile(save_path):
            os.remove(save_path)

        self._integrate(model, ini_state, controls, prop, times, save_path)
        self._plot_glottal_width(model, save_path)

        assert True

    def _integrate(self, model, ini_state, controls, prop, times, save_path):

        print("Running forward model")
        runtime_start = perf_counter()
        with sf.StateFile(model, save_path, mode='w') as f:
            fin_state, info = integrate(model, f, ini_state, controls, prop, times)

        runtime_end = perf_counter()
        print(f"Runtime {runtime_end-runtime_start:.2f} seconds")

    def _plot_glottal_width(self, model, save_path):
        ## Plot the resulting glottal width
        with sf.StateFile(model, save_path, mode='r') as f:
            t, gw = proc_time_and_glottal_width(model, f)
        fig, ax = plt.subplots(1, 1)
        ax.plot(t, gw)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Glottal width [cm]")
        fig.savefig(os.path.splitext(save_path)[0] + '.png')

    # def test_integrate_linear(self):
    #     """
    #     To test that the linearized forward integration is done properly:
    #         - Integrate the forward model twice at x and x+dx for some test change, dx
    #         - Compute the change in the final state using the difference
    #         - Compute the change in the final state using the linearized integration and compare
    #     """
    #     ## Set the linearization point
    #     NTIME = 50
    #     model, ini_state, controls, prop = self.config_fsi_model()
    #     control = controls[0]
    #     times = np.linspace(0, 0.01, NTIME)

    #     ## Specify the test change in model parameters
    #     dini_state = model.state0.copy()
    #     dcontrol = model.control.copy()
    #     dprop = model.prop.copy()
    #     dprop[:] = 0.0
    #     # dtimes = vec.BlockVector([np.linspace(0, 1e-6, NTIME)], ['times'])
    #     dtimes = np.linspace(0.0, 0.0, NTIME)
    #     dtimes[-1] = 1e-10
    #     dini_state[:] = 0.0
    #     for vec in [dini_state[label] for label in ['u', 'v', 'a']]:
    #         model.solid.forms['bc.dirichlet'].apply(vec)

    #     ## Integrate the model at x, and x+dx
    #     def _integrate(model, state, control, prop, times, h5file, overwrite=False):
    #         if not overwrite and os.path.isfile(h5file):
    #             print("File already exists. Continuing with old file.")
    #         else:
    #             with sf.StateFile(model, h5file, mode='w') as f:
    #                 integrate(model, f, state, [control], prop, times)

    #     xs = [ini_state, control, prop, times]
    #     dxs = [dini_state, dcontrol, dprop, dtimes]

    #     h5file1 = 'out/test_forward_integrate_linear-1.h5'
    #     _integrate(model, *xs, h5file1)

    #     h5file2 = 'out/test_forward_integrate_linear-2.h5'
    #     _integrate(model, *[x+dx for x, dx in zip(xs, dxs)], h5file2, overwrite=True)

    #     dfin_state_fd = None
    #     with sf.StateFile(model, h5file1, mode='r') as f1, sf.StateFile(model, h5file2, mode='r') as f2:
    #         dfin_state_fd = f2.get_state(f2.size-1) - f1.get_state(f1.size-1)

    #     ## Integrate the linearized model
    #     dfin_state = None
    #     with sf.StateFile(model, h5file1, mode='r') as f:
    #         dfin_state = integrate_linear(
    #             model, f, dini_state, [dcontrol], dprop, dtimes)

    #     err = dfin_state - dfin_state_fd
    #     self.assertAlmostEqual(err.norm()/dfin_state.norm(), 0.0)

def proc_time_and_glottal_width(model, f):
    t = f.get_times()

    glottal_width_sharp = TimeSeries(solidfunc.MinGlottalWidth(model))
    y = glottal_width_sharp(f)

    return t, np.array(y)


# if __name__ == '__main__':
#     np.seterr(invalid='raise')
#     test = TestIntegrate()
#     test.setUp()
#     # test.test_integrate_variable_controls()
#     # test.test_integrate_fsi_kelvinvoigt()
#     test.test_integrate_fsi_rayleigh()
#     # test.test_integrate_approx3D()
#     # test.test_integrate_fsai()
#     # test.test_integrate_linear()
#     # unittest.main()
