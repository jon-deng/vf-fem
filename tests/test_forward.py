"""
Test to see if `forward.integrate` runs
"""

import os
import pytest
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
import dolfin as dfn

import femvf.statefile as sf
from femvf.forward import integrate, integrate_linear
from femvf.constants import PASCAL_TO_CGS

from femvf.models.transient import solid as tsmd, fluid as tfmd, acoustic as amd
from femvf.load import load_transient_fsi_model, load_transient_fsai_model
import femvf.postprocess.solid as solidfunc
from femvf.postprocess.base import TimeSeries

class TestIntegrate:

    @pytest.fixture(
        params=[
            tsmd.KelvinVoigt, tsmd.Rayleigh
        ]
    )
    def solid_type(self, request):
        return request.param

    @pytest.fixture(
        params=[tfmd.BernoulliSmoothMinSep]
    )
    def fluid_type(self, request):
        return request.param

    @pytest.fixture(
        params=[
            'M5_BC--GA0--DZ0.00',
            'M5_BC--GA0--DZ1.00'
        ]
    )
    def mesh_path(self, request):
        mesh_dir = '../meshes'
        return os.path.join(mesh_dir, request.param + '.msh')

    @pytest.fixture()
    def model(self, mesh_path, solid_type, fluid_type):
        ## Configure the model and its parameters
        SolidType, FluidType = (solid_type, fluid_type)
        if 'DZ0.00' in mesh_path:
            zs = None
        else:
            zs = (0.0, 0.5, 1.0)
            zs = np.linspace(0, 1, 6)
        return load_transient_fsi_model(
            mesh_path, None,
            SolidType=SolidType,
            FluidType=FluidType,
            coupling='explicit',
            zs=zs
        )

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
        control['fluid0.psub'][:] = p_sub * PASCAL_TO_CGS
        control['fluid0.psup'][:] = 0.0 * PASCAL_TO_CGS

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

        # Set default properties
        default_prop = {
            'eta': 4e-3,
            'rho': 1.0,
            'kcontact': 1e11,
            'ycontact': prop['ymid'][0] - y_gap*1/2,
        }

        # Set relevant fluid properties
        for ii in range(len(model.fluids)):
            default_prop.update({
                f'fluid{ii}.zeta_min': 1e-8,
                f'fluid{ii}.zeta_sep': 1e-8,
                f'fluid{ii}.rho_air': 1.0
            })

        # This only sets the properties if they exist
        for key, value in default_prop.items():
            if key in prop:
                prop[key] = value

        return prop

    def test_integrate(self, mesh_path, model, ini_state, controls, prop):

        times = np.linspace(0, 0.01, 100)

        mesh_name = os.path.splitext(os.path.split(mesh_path)[1])[0]
        save_path = f'out/{mesh_name}--{model.solid.__class__.__name__}--{model.fluids[0].__class__.__name__}.h5'
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

def proc_time_and_glottal_width(model, f):
    t = f.get_times()

    glottal_width_sharp = TimeSeries(solidfunc.MinGlottalWidth(model))
    y = glottal_width_sharp(f)

    return t, np.array(y)

