"""
Test to see if `forward.integrate` runs
"""

import os
import pytest
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
import dolfin as dfn
import pandas as pd

from blockarray import blockvec as bv

import femvf.statefile as sf
from femvf.residuals import solid as slr
from femvf.residuals import fluid as flr
from femvf.forward import integrate, integrate_linear
from femvf.constants import PASCAL_TO_CGS
from femvf.models import transient
from femvf.load import load_fsi_model, derive_1dfluid_from_2dsolid
import femvf.postprocess.solid as solidfunc
from femvf.postprocess.base import TimeSeries
# from femvf.vis.xdmfutils import write_xdmf, export_mesh_values

from vfsig import modal as modalsig

from tests.fixture_mesh import FenicsMeshFixtures


class ModelFixtures(FenicsMeshFixtures):

    RESIDUAL_CLASSES = (
        slr.Rayleigh,
        slr.KelvinVoigt,
        slr.SwellingKelvinVoigt,
        slr.SwellingKelvinVoigtWEpithelium
    )

    @pytest.fixture(params=RESIDUAL_CLASSES)
    def SolidResidual(self, request):
        return request.param

    @staticmethod
    def init_residual(ResidualClass, mesh, mesh_functions, mesh_subdomains):
        dim = mesh.topology().dim()
        dirichlet_bcs = {
            'coeff.state.u1': [(dfn.Constant(dim*[0]), 'facet', 'fixed')],
            # 'coeff.state.u0': [(dfn.Constant(dim*[0]), 'facet', 'fixed')]
        }
        return ResidualClass(mesh, mesh_functions, mesh_subdomains, dirichlet_bcs)

    @pytest.fixture()
    def solid_residual(
        self,
        SolidResidual: slr.PredefinedSolidResidual,
        mesh: dfn.Mesh,
        mesh_functions: list[dfn.MeshFunction],
        mesh_subdomains
    ):
        return self.init_residual(SolidResidual, mesh, mesh_functions, mesh_subdomains)

    @pytest.fixture(
        params=[flr.BernoulliSmoothMinSep, flr.BernoulliFixedSep, flr.BernoulliAreaRatioSep]
    )
    def FluidResidual(self, request):
        return request.param

    @pytest.fixture()
    def solid(self, solid_residual):
        return transient.FenicsModel(solid_residual)

    @pytest.fixture()
    def model(self, solid, FluidResidual):
        res_fluid, solid_pdofs = derive_1dfluid_from_2dsolid(solid.residual, FluidResidual, fsi_facet_labels=['traction'])
        fluid_pdofs = np.arange(solid_pdofs.size)
        fluid = transient.JaxModel(res_fluid)
        return transient.ExplicitFSIModel(solid, fluid, solid_pdofs, fluid_pdofs)


class TestIntegrate(ModelFixtures):

    @pytest.fixture()
    def ini_state(self, model):
        """Return the initial state"""
        xy = model.solid.XREF[:].copy().reshape(-1, 2)
        x = xy[:, 0]
        y = xy[:, 1]
        u0 = dfn.Function(
            model.solid.residual.form['coeff.state.u0'].function_space()
        ).vector()

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
        p_sub = 800.0

        control = model.control
        control[f'psub'][:] = p_sub * PASCAL_TO_CGS
        control[f'psup'][:] = 0.0 * PASCAL_TO_CGS
        return [control]

    @pytest.fixture()
    def prop(self, model):
        """Return the properties"""
        residual = model.solid.residual
        y_gap = 0.05
        y_midline = np.max(residual.mesh().coordinates()[..., 1]) + y_gap

        prop = model.prop.copy()

        prop['ymid'][0] = y_midline
        prop['ncontact'][1] = 1.0

        # xy = (
        #     residual.form['coeff.prop.emod'].function_space()
        #     .tabulate_dof_coordinates()
        # )
        # x = xy[:, 0]
        # y = xy[:, 1]
        # x_min, x_max = x.min(), x.max()
        # y_min, y_max = y.min(), y.max()
        # prop['emod'][:] = (
        #     1/2*5.0e3*PASCAL_TO_CGS*((x-x_min)/(x_max-x_min)
        #     + (y-y_min)/(y_max-y_min)) + 2.5e3*PASCAL_TO_CGS
        # )
        prop['emod'][:] = 10e3 * PASCAL_TO_CGS

        # Set default properties
        default_prop = {
            'eta': 4e-3,
            'rho': 1.0,
            'nu': 0.45,
            'kcontact': 1e11,
            'ycontact': prop['ymid'][0] - y_gap * 1 / 2,
        }

        # Set relevant fluid properties
        default_prop.update(
            {
                f'zeta_min': 1e-8,
                f'zeta_sep': 1e-8,
                f'rho_air': 1.0,
                f'r_sep': 1.0,
            }
        )

        # This only sets the properties if they exist
        for key, value in default_prop.items():
            if key in prop:
                prop[key] = value

        return prop

    @pytest.fixture()
    def times(self):
        times = 2e-5 * np.arange(2**8)
        return times

    def test_integrate(
        self,
        mesh_name: str,
        model: transient.BaseTransientFSIModel,
        ini_state: bv.BlockVector,
        controls: bv.BlockVector,
        prop: bv.BlockVector,
        times: np.typing.NDArray,
    ):
        """
        Test forward time integration of the model
        """

        psub = controls[0]['psub'][0]
        save_path = (
            f'{self.__class__.__name__}--{mesh_name}'
            f'--{model.solid.residual.__class__.__name__}'
            f'--{model.fluid.residual.__class__.__name__}--psub{psub/10:.1f}.h5'
        )
        self.integrate(model, ini_state, controls, prop, times, save_path)

        self.plot_glottal_width(model, save_path)
        self.export_paraview(model, save_path)

        self.export_stats(model, save_path)

        assert True

    def integrate(self, model, ini_state, controls, prop, times, save_path):
        """
        Run the transient simulation and save results
        """
        print("Running forward model")
        runtime_start = perf_counter()
        with sf.StateFile(model, save_path, mode='w') as f:
            fin_state, info = integrate(model, f, ini_state, controls, prop, times)

        runtime_end = perf_counter()
        print(f"Runtime {runtime_end-runtime_start:.2f} seconds")

    def plot_glottal_width(self, model, save_path):
        """
        Plot the glottal area/width waveform from results
        """
        ## Plot the resulting glottal width
        with sf.StateFile(model, save_path, mode='r') as f:
            t = f.get_times()
            gw = TimeSeries(solidfunc.MeanGlottalWidth(model))(f)

        fig, ax = plt.subplots(1, 1)
        ax.plot(t, gw)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Glottal width [cm]")
        fig.savefig(os.path.splitext(save_path)[0] + '.png')

    def export_paraview(self, model, save_path):
        """
        Export paraview visualization data from the results
        """
        # TODO: `write_xdmf` and `export_mesh_values` have been updated
        # vertex_data_path = os.path.splitext(save_path)[0] + '--vertex.h5'
        # export_mesh_values(model, save_path, vertex_data_path)

        # xdmf_name = os.path.split(os.path.splitext(save_path)[0])[-1] + '--vertex.xdmf'
        # write_xdmf(model, vertex_data_path, xdmf_name)
        pass

    def export_stats(self, model, save_path):
        with sf.StateFile(model, save_path, mode='r') as f:
            t = f.get_times()
            gw = TimeSeries(solidfunc.MeanGlottalWidth(model))(f)

        # Truncate parts of the glottal width signal
        idx_truncate_start = 0
        idx_truncate_end = len(t)

        t = t[idx_truncate_start:idx_truncate_end]
        gw = gw[idx_truncate_start:idx_truncate_end]

        dt = t[1] - t[0]
        fo, *_ = modalsig.fundamental_mode_from_rfft(gw - np.mean(gw), dt)
        amplitude = np.max(gw) - np.min(gw)

        column_labels = ['fo', 'amplitude']
        df = pd.DataFrame(index=[0], columns=column_labels)
        df['fo'] = fo
        df['amplitude'] = amplitude

        stats_path = f'{os.path.splitext(save_path)[0]}.xlsx'
        df.to_excel(stats_path)
