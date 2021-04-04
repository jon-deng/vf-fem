"""
A basic test to see if forward.integrate will actually run
"""

import os
import unittest
from time import perf_counter

import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import dolfin as dfn
# import h5py

import femvf.statefile as sf
from femvf.forward import integrate, integrate_linear
from femvf.constants import PASCAL_TO_CGS

from femvf.models import load_fsi_model, load_fsai_model, Rayleigh, KelvinVoigt, Bernoulli, WRAnalog
from femvf import callbacks
from femvf import linalg

class ForwardConfig(unittest.TestCase):
    def setUp(self):
        """
        Set the solid mesh
        """
        dfn.set_log_level(30)
        np.random.seed(123)

        mesh_dir = '../meshes'

        mesh_base_filename = 'M5-3layers'
        self.mesh_path = os.path.join(mesh_dir, mesh_base_filename + '.xml')

    def config_fsi_model(self):
        ## Configure the model and its parameters
        model = load_fsi_model(self.mesh_path, None, SolidType=Rayleigh, FluidType=Bernoulli, coupling='explicit')

        # Set the control vector
        p_sub = 500

        control = model.get_control_vec()
        control['psub'][:] = p_sub * PASCAL_TO_CGS
        control['psup'][:] = 0.0 * PASCAL_TO_CGS
        controls = [control]

        # Set the properties
        y_gap = 0.01

        fl_props = model.fluid.get_properties_vec(set_default=True)
        fl_props['y_midline'][()] = np.max(model.solid.mesh.coordinates()[..., 1]) + y_gap

        sl_props = model.solid.get_properties_vec(set_default=True)
        xy = model.solid.scalar_fspace.tabulate_dof_coordinates()
        x = xy[:, 0]
        y = xy[:, 1]
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        sl_props['emod'][:] = 1/2*5.0e3*PASCAL_TO_CGS*((x-x_min)/(x_max-x_min) + (y-y_min)/(y_max-y_min)) + 2.5e3*PASCAL_TO_CGS
        sl_props['rayleigh_m'][()] = 0
        sl_props['rayleigh_k'][()] = 4e-3
        sl_props['k_collision'][()] = 1e11
        sl_props['y_collision'][()] = fl_props['y_midline'] - y_gap*1/2
        props = linalg.concatenate(sl_props, fl_props)

        # Set the initial state
        xy = model.solid.vector_fspace.tabulate_dof_coordinates()
        x = xy[:, 0]
        y = xy[:, 1]
        u0 = dfn.Function(model.solid.vector_fspace).vector()

        # model.fluid.set_properties(fluid_props)
        # qp0, *_ = model.fluid.solve_qp0()

        ini_state = model.get_state_vec()
        ini_state.set(0.0)
        ini_state['u'][:] = u0
        # ini_state['q'][()] = qp0['q']
        # ini_state['p'][:] = qp0['p']
        
        return model, ini_state, controls, props

    def config_fsai_model(self):
        ## Configure the model and its parameters
        acoustic = WRAnalog(44)
        model = load_fsai_model(self.mesh_path, None, acoustic, SolidType=Rayleigh, FluidType=Bernoulli,
                                coupling='explicit')

        # Set the control vector
        p_sub = 500

        control = model.get_control_vec()
        control['psub'][:] = p_sub * PASCAL_TO_CGS
        controls = [control]

        # Set the properties
        y_gap = 0.01

        fl_props = model.fluid.get_properties_vec(set_default=True)
        fl_props['y_midline'][()] = np.max(model.solid.mesh.coordinates()[..., 1]) + y_gap

        sl_props = model.solid.get_properties_vec(set_default=True)
        xy = model.solid.scalar_fspace.tabulate_dof_coordinates()
        x = xy[:, 0]
        y = xy[:, 1]
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        sl_props['emod'][:] = 1/2*5.0e3*PASCAL_TO_CGS*((x-x_min)/(x_max-x_min) + (y-y_min)/(y_max-y_min)) + 2.5e3*PASCAL_TO_CGS
        sl_props['rayleigh_m'][()] = 0
        sl_props['rayleigh_k'][()] = 4e-3
        sl_props['k_collision'][()] = 1e11
        sl_props['y_collision'][()] = fl_props['y_midline'] - y_gap*1/2

        ac_props = model.acoustic.get_properties_vec(set_default=True)
        ac_props['area'][:] = 4.0
        ac_props['length'][:] = 12.0
        ac_props['soundspeed'][:] = 340*100

        props = linalg.concatenate(sl_props, fl_props, ac_props)

        # Set the initial state
        u0 = dfn.Function(model.solid.vector_fspace).vector()

        # model.fluid.set_properties(fluid_props)
        # qp0, *_ = model.fluid.solve_qp0()

        ini_state = model.get_state_vec()
        ini_state.set(0.0)
        ini_state['u'][:] = u0
        # ini_state['q'][()] = qp0['q']
        # ini_state['p'][:] = qp0['p']
        
        return model, ini_state, controls, props

class TestIntegrate(ForwardConfig):
    
    def test_integrate_fsai(self):
        model, ini_state, controls, props = self.config_fsai_model()
        # model, ini_state, controls, props = self.config_fsi_model()

        times = linalg.BlockVec((np.linspace(0, 0.01, 100),), ('times',))

        # Set the total length of the WRAnalog to match the specified time step
        # dt = times[1]-times[0]
        # C, N = model.acoustic.properties['soundspeed'][0], model.acoustic.properties['area'].size
        # props['length'][:] = (0.5*dt*C) * N

        save_path = 'out/test_forward.h5'
        if os.path.isfile(save_path):
            os.remove(save_path)

        ## Run the simulation
        # Set values to export
        cbs = {'glottal_width': callbacks.safe_glottal_width}
        print("Running forward model")
        runtime_start = perf_counter()
        with sf.StateFile(model, save_path, mode='w') as f:
            fin_state, info = integrate(model, f, ini_state, controls, props, times, export_callbacks=cbs)

        runtime_end = perf_counter()
        print(f"Runtime {runtime_end-runtime_start:.2f} seconds")

        ## Plot the resulting glottal width
        # psubs = None
        # with sf.StateFile(model, save_path, mode='r') as f:
        #     for n in range
        #     psubs = f.get_control

        fig, ax = plt.subplots(1, 1)
        breakpoint()
        ax.plot(times['times'], info['glottal_width'])
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Glottal width [cm]")

        plt.show()
        fig.savefig('out/test_forward.png')

    def test_integrate_linear(self):
        """
        To test that the linearized forward integration is done properly:
            - Integrate the forward model twice at x and x+dx for some test change, dx
            - Compute the change in the final state using the difference
            - Compute the change in the final state using the linearized integration and compare
        """
        ## Set the linearization point
        NTIME = 50
        model, ini_state, controls, props = self.config_fsi_model()
        control = controls[0]
        times = linalg.BlockVec((np.linspace(0, 0.01, NTIME),), ('times',))

        ## Specify the test change in model parameters
        dini_state = model.get_state_vec()
        dcontrol = model.get_control_vec()
        dprops = model.get_properties_vec()
        dprops.set(0.0)
        # dtimes = linalg.BlockVec([np.linspace(0, 1e-6, NTIME)], ['times'])
        dtimes = linalg.BlockVec([np.linspace(0.0, 0.0, NTIME)], ['times'])
        dtimes['times'][-1] = 1e-10
        dini_state.set(0.0)
        for vec in [dini_state[label] for label in ['u', 'v', 'a']]:
            model.solid.bc_base.apply(vec)

        ## Integrate the model at x, and x+dx
        def _integrate(model, state, control, props, times, h5file, overwrite=False):
            if not overwrite and os.path.isfile(h5file):
                print("File already exists. Continuing with old file.")
            else:
                with sf.StateFile(model, h5file, mode='w') as f:
                    integrate(model, f, state, [control], props, times)

        xs = [ini_state, control, props, times]
        dxs = [dini_state, dcontrol, dprops, dtimes]

        h5file1 = 'out/test_forward_integrate_linear-1.h5'
        _integrate(model, *xs, h5file1)

        h5file2 = 'out/test_forward_integrate_linear-2.h5'
        _integrate(model, *[x+dx for x, dx in zip(xs, dxs)], h5file2, overwrite=True)

        dfin_state_fd = None
        with sf.StateFile(model, h5file1, mode='r') as f1, sf.StateFile(model, h5file2, mode='r') as f2:
            dfin_state_fd = f2.get_state(f2.size-1) - f1.get_state(f1.size-1)
        
        ## Integrate the linearized model
        dfin_state = None
        with sf.StateFile(model, h5file1, mode='r') as f:
            dfin_state = integrate_linear(
                model, f, dini_state, [dcontrol], dprops, dtimes)

        err = dfin_state - dfin_state_fd
        breakpoint()
        self.assertAlmostEqual(err.norm()/dfin_state.norm(), 0.0)

if __name__ == '__main__':
    test = TestIntegrate()
    test.setUp()
    test.test_integrate_fsai()
    test.test_integrate_linear()
    # unittest.main()
