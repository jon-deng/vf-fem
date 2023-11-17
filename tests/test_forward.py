"""
Test to see if `forward.integrate` runs
"""

import os
import pytest
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
import dolfin as dfn

from blockarray import blockvec as bv

import femvf.statefile as sf
from femvf.forward import integrate, integrate_linear
from femvf.constants import PASCAL_TO_CGS
from femvf.models.transient import (
    solid as tsmd, fluid as tfmd, acoustic as amd, coupled as cmd
)
from femvf.load import load_transient_fsi_model, load_transient_fsai_model
import femvf.postprocess.solid as solidfunc
from femvf.postprocess.base import TimeSeries
from femvf.vis.xdmfutils import export_vertex_values, write_xdmf

class TestIntegrate:

    @pytest.fixture(
        params=[
            tsmd.KelvinVoigt, tsmd.Rayleigh
        ]
    )
    def solid_type(self, request):
        """Return the solid class"""
        return request.param

    @pytest.fixture(
        params=[tfmd.BernoulliSmoothMinSep]
    )
    def fluid_type(self, request):
        """Return the fluid class"""
        return request.param

    @pytest.fixture(
        params=[
            'M5_BC--GA0--DZ0.00',
            'M5_BC--GA0--DZ1.00'
        ]
    )
    def mesh_path(self, request):
        """Return the mesh path"""
        mesh_dir = '../meshes'
        return os.path.join(mesh_dir, request.param + '.msh')

    @pytest.fixture()
    def model(self, mesh_path, solid_type, fluid_type):
        """Return the model"""
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
        """Return the initial state"""
        xy = model.solid.XREF[:].copy().reshape(-1, 2)
        x = xy[:, 0]
        y = xy[:, 1]
        u0 = dfn.Function(model.solid.residual.form['coeff.state.u0'].function_space()).vector()

        # model.fluid.set_prop(fluid_props)
        # qp0, *_ = model.fluids[0].solve_qp0()

        ini_state = model.state0.copy()
        ini_state[:] = 0.0
        ini_state['u'][:] = u0
        # ini_state['q'][()] = qp0['q']
        # ini_state['p'][:] = qp0['p']
        return ini_state

    @pytest.fixture()
    def controls(self, model):
        """Return the control vector"""
        control = model.control.copy()
        p_sub = 500.0

        control = model.control
        for ii in range(len(model.fluids)):
            control[f'fluid{ii}.psub'][:] = p_sub * PASCAL_TO_CGS
            control[f'fluid{ii}.psup'][:] = 0.0 * PASCAL_TO_CGS
        return [control]

    @pytest.fixture()
    def prop(self, model):
        """Return the properties"""
        y_gap = 0.01
        y_midline = np.max(model.solid.residual.mesh().coordinates()[..., 1]) + y_gap

        prop = model.prop.copy()

        prop['ymid'][0] = y_midline
        prop['ncontact'][1] = 1.0

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
            'nu': 0.45,
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

    @pytest.fixture()
    def times(self):
        return np.linspace(0, 0.01, 100)

    def test_integrate(
            self,
            mesh_path: str,
            model: cmd.BaseTransientFSIModel,
            ini_state: bv.BlockVector,
            controls: bv.BlockVector,
            prop: bv.BlockVector,
            times: np.typing.NDArray
        ):

        psub = controls[0]['fluid0.psub'][0]
        mesh_name = os.path.splitext(os.path.split(mesh_path)[1])[0]
        save_path = f'out/{self.__class__.__name__}--{mesh_name}--{model.solid.__class__.__name__}--{model.fluids[0].__class__.__name__}--psub{psub/10:.1f}.h5'
        if os.path.isfile(save_path):
            os.remove(save_path)

        # prop.print_summary()
        self._integrate(model, ini_state, controls, prop, times, save_path)
        self._plot_glottal_width(model, save_path)
        self._export_paraview(model, save_path)

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

    def _export_paraview(self, model, save_path):
        vertex_data_path = os.path.splitext(save_path)[0] + '--vertex.h5'
        export_vertex_values(model, save_path, vertex_data_path)

        xdmf_name = os.path.split(os.path.splitext(save_path)[0])[-1] + '--vertex.xdmf'
        write_xdmf(model, vertex_data_path, xdmf_name)

class TestLiEtal2020(TestIntegrate):
    """
    Test the forward model with conditions given in (Li et. al., 2020)
    """

    @pytest.fixture(
        params=[
            tsmd.Rayleigh,
        ]
    )
    def solid_type(self, request):
        """Return the solid class"""
        return request.param

    @pytest.fixture(
        params=[tfmd.BernoulliAreaRatioSep]
    )
    def fluid_type(self, request):
        """Return the fluid class"""
        return request.param

    @pytest.fixture(
        # NOTE: (Li2020) specifies a 20 mm VF length, but they don't give
        # details on the geometry shape
        # (I also checked the referenced thesis but couldn't find details
        # on the geometry either)
        params=[
            'M5_BC--GA0--DZ0.00',
            'M5_BC--GA0--DZ2.00'
        ]
    )
    def mesh_path(self, request):
        """Return the mesh path"""
        mesh_dir = '../meshes'
        return os.path.join(mesh_dir, request.param + '.msh')

    @pytest.fixture()
    def model(self, mesh_path, solid_type, fluid_type):
        """Return the model"""
        ## Configure the model and its parameters
        SolidType, FluidType = (solid_type, fluid_type)
        if 'DZ0.00' in mesh_path:
            zs = None
        else:
            # zs = (0.0, 0.5, 2.0)
            zs = np.linspace(0.0, 2.0, 6)
        return load_transient_fsi_model(
            mesh_path, None,
            SolidType=SolidType,
            FluidType=FluidType,
            coupling='explicit',
            zs=zs
        )

    @pytest.fixture(
        # NOTE: (Li2020) uses three different subglottal pressures
        # of 750 Pa, 1000 Pa, and 1250 Pa
        params=[750, 1000, 1250]
    )
    def controls(self, model, request):
        """Return the control vector"""
        control = model.control.copy()
        p_sub = request.param

        control = model.control
        for ii in range(len(model.fluids)):
            control[f'fluid{ii}.psub'][:] = p_sub * PASCAL_TO_CGS
            control[f'fluid{ii}.psup'][:] = 0.0 * PASCAL_TO_CGS
        return [control]

    @pytest.fixture()
    def prop(self, model):
        """Return the properties"""
        # NOTE: (Li2020, Section 2.2) says the initial glottal gap is 0.4 mm
        y_gap = 0.04
        y_max = np.max(model.solid.residual.mesh().coordinates()[..., 1])
        y_midline = y_max + y_gap
        # NOTE: (Li2020, Section 2.2) says that the VF is allowed to have a
        # small gap of 0.2 mm during vibration
        y_contact = y_midline - 0.02

        prop = model.prop.copy()
        prop['ymid'][0] = y_midline

        prop['ycontact'] = y_contact
        prop['ncontact'][1] = 1.0
        prop['kcontact'] = 1e11

        prop['emod'][:] = 15.0e3*PASCAL_TO_CGS
        prop['rho'] = 1.040
        prop['rayleigh_m'] = 0.05
        prop['rayleigh_k'] = 0.0
        prop['nu'] = 0.475

        for ii in range(len(model.fluids)):
            # NOTE: (Li2020, Table 1) Model B1 corresponds to a separation
            # area ratio of 1.0 (i.e. separation happens at the minimum area)
            prop[f'fluid{ii}.r_sep'] = 1.0
            prop[f'fluid{ii}.area_lb'] = 2*(y_midline-y_contact)

        return prop

    @pytest.fixture()
    def times(self):
        tfin = 0.1
        return np.linspace(0, tfin, round(100/0.01*tfin) + 1)

class TestBounceFromDeformation(TestIntegrate):
    """
    Test the forward model with bouncing back from a deformed state
    """

    @pytest.fixture(
        params=[
            tsmd.Rayleigh,
        ]
    )
    def solid_type(self, request):
        """Return the solid class"""
        return request.param

    @pytest.fixture(
        params=[tfmd.BernoulliAreaRatioSep]
    )
    def fluid_type(self, request):
        """Return the fluid class"""
        return request.param

    @pytest.fixture(
        # NOTE: (Li2020) specifies a 20 mm VF length, but they don't give
        # details on the geometry shape
        # (I also checked the referenced thesis but couldn't find details
        # on the geometry either)
        params=[
            # 'M5_BC--GA0--DZ0.00',
            'M5_BC--GA0--DZ2.00'
        ]
    )
    def mesh_path(self, request):
        """Return the mesh path"""
        mesh_dir = '../meshes'
        return os.path.join(mesh_dir, request.param + '.msh')

    @pytest.fixture()
    def model(self, mesh_path, solid_type, fluid_type):
        """Return the model"""
        ## Configure the model and its parameters
        SolidType, FluidType = (solid_type, fluid_type)
        if 'DZ0.00' in mesh_path:
            zs = None
        else:
            # zs = (0.0, 0.5, 2.0)
            zs = np.linspace(0.0, 2.0, 6)
        return load_transient_fsi_model(
            mesh_path, None,
            SolidType=SolidType,
            FluidType=FluidType,
            coupling='explicit',
            zs=zs
        )

    @pytest.fixture()
    def ini_state(self, model):
        """Return the initial state"""
        NDIM = model.solid.residual.mesh().topology().dim()
        # This sets x/y/z deformation
        u0 = dfn.Function(model.solid.residual.form['coeff.state.u0'].function_space()).vector()
        u0 = np.array(u0)
        xy = model.solid.XREF[:].copy().reshape(-1, NDIM)
        if NDIM > 2:
            # Make the deformation respect BCs; deformation increases
            # linearly with y and is parabolic in z with no deformation
            # at the z margins of the VF
            z = xy[:, 2]
            y = xy[:, 1]
            z_max = np.max(z)
            u0[0:-2:NDIM] = 0.2*(y*(z)*(z_max-z)/z_max)
        else:
            y = xy[:, 1]
            u0[0:-2:NDIM] = 0.2*y

        # model.fluid.set_prop(fluid_props)
        # qp0, *_ = model.fluids[0].solve_qp0()

        ini_state = model.state0.copy()
        ini_state[:] = 0.0
        ini_state['u'][:] = u0
        # ini_state['q'][()] = qp0['q']
        # ini_state['p'][:] = qp0['p']
        return ini_state

    @pytest.fixture()
    def controls(self, model):
        """Return the control vector"""
        control = model.control.copy()
        for ii in range(len(model.fluids)):
            control[f'fluid{ii}.psub'][:] = 0.0 * PASCAL_TO_CGS
            control[f'fluid{ii}.psup'][:] = 0.0 * PASCAL_TO_CGS
        return [control]

    @pytest.fixture()
    def prop(self, model):
        """Return the properties"""
        y_gap = 999.0
        y_max = np.max(model.solid.residual.mesh().coordinates()[..., 1])
        y_midline = y_max + y_gap
        y_contact = y_midline

        prop = model.prop.copy()
        prop['ymid'][0] = y_midline

        prop['ycontact'] = y_contact
        prop['ncontact'][1] = 1.0
        prop['kcontact'] = 1e11

        prop['emod'][:] = 15.0e3*PASCAL_TO_CGS
        prop['rho'] = 1.040
        prop['rayleigh_m'] = 0.05
        prop['rayleigh_k'] = 0.0
        prop['nu'] = 0.475

        for ii in range(len(model.fluids)):
            # NOTE: (Li2020, Table 1) Model B1 corresponds to a separation
            # area ratio of 1.0 (i.e. separation happens at the minimum area)
            prop[f'fluid{ii}.r_sep'] = 1.0
            prop[f'fluid{ii}.area_lb'] = 2*(y_midline-y_contact)

        return prop

    @pytest.fixture()
    def times(self):
        tfin = 0.1
        return np.linspace(0, tfin, round(100/0.01*tfin) + 1)

def proc_time_and_glottal_width(model, f):
    t = f.get_times()

    glottal_width_sharp = TimeSeries(solidfunc.MinGlottalWidth(model))
    y = glottal_width_sharp(f)

    return t, np.array(y)

