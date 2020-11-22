import unittest
import os

import dolfin as dfn
import numpy as np

from femvf.model import load_fsai_model

from femvf.solids import Rayleigh, KelvinVoigt
from femvf.fluids import Bernoulli
from femvf.acoustics import WRA

from femvf.constants import PASCAL_TO_CGS
from femvf import linalg

class TestFSAIModel(unittest.TestCase):
    def setUp(self):
        """
        Set the solid mesh
        """
        dfn.set_log_level(30)

        mesh_dir = '../meshes'

        mesh_base_filename = 'M5-3layers'
        self.mesh_path = os.path.join(mesh_dir, mesh_base_filename + '.xml')

    def config_fsai_model(self):
        ## Configure the model and its parameters
        acoustic = WRA(44)
        model = load_fsai_model(self.mesh_path, None, acoustic, Solid=Rayleigh, Fluid=Bernoulli,
                                coupling='explicit')

        # Set the control vector
        p_sub = 500

        control = model.get_control_vec()
        control['psub'][:] = p_sub * PASCAL_TO_CGS
        controls = [control]

        model.set_control(control)

        # Set the properties
        y_gap = 0.01
        alpha, k, sigma = -3000, 50, 0.002

        fl_props = model.fluid.get_properties_vec(set_default=True)
        fl_props['y_midline'][()] = np.max(model.solid.mesh.coordinates()[..., 1]) + y_gap
        fl_props['alpha'][()] = alpha
        fl_props['k'][()] = k
        fl_props['sigma'][()] = sigma

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

    def test_apply_dres_dstate1(self):
        model, ini_state, controls, props = self.config_fsai_model()
        model.set_properties(props)

        fin_state = ini_state.copy()

        dx_fd = model.get_state_vec()
        dx_fd.set(0.0)
        # dx_fd['u'] = 0.0
        dx_fd['q'] = 1e-2
        # dx_fd['p'][:] = 1e-2
        # dx_fd['pinc'][:] = 1e-2
        # dx_fd['pref'][:] = 1e-2
        model.solid.bc_base.apply(dx_fd['u'])

        breakpoint()
        model.set_fin_state(fin_state)
        res0 = model.res()

        model.set_fin_state(fin_state + dx_fd)
        res1 = model.res()
        dres = res1-res0

        model.set_fin_state(fin_state)
        dx = model.solve_dres_dstate1(dres)
        print(f"||dres|| = {linalg.dot(dres, dres)}")
        print(f"||dx|| = {linalg.dot(dx, dx)}")
        print(f"||dx_fd|| = {linalg.dot(dx_fd, dx_fd)}")
        for name in ('u', 'v', 'a'):
            print(name, dx[name].norm('l2'), dx_fd[name].norm('l2'))

        for name in ('q', 'p', 'pinc', 'pref'):
            print(name, np.linalg.norm(dx[name]), np.linalg.norm(dx_fd[name]))
        breakpoint()

    def test_solve_dres_dstate1(self):
        model, ini_state, controls, props = self.config_fsai_model()
        model.set_properties(props)

        fin_state = ini_state.copy()

        dx_fd = model.get_state_vec()
        dx_fd.set(0.0)
        # dx_fd['u'] = 0.0
        # dx_fd['q'] = 1e-2
        # dx_fd['p'] = 1e-2
        # dx_fd['pref'][:2] = 1e-2
        model.solid.bc_base.apply(dx_fd['u'])

        model.set_fin_state(fin_state)
        res0 = model.res()

        model.set_fin_state(fin_state + dx_fd)
        breakpoint()
        res1 = model.res()
        dres = res1-res0

        model.set_fin_state(fin_state)
        dx = model.solve_dres_dstate1(dres)
        print(f"||dres|| = {linalg.dot(dres, dres)}")
        print(f"||dx|| = {linalg.dot(dx, dx)}")
        print(f"||dx_fd|| = {linalg.dot(dx_fd, dx_fd)}")
        for name in ('u', 'v', 'a'):
            print(name, dx[name].norm('l2'), dx_fd[name].norm('l2'))

        for name in ('q', 'p', 'pinc', 'pref'):
            print(name, np.linalg.norm(dx[name]), np.linalg.norm(dx_fd[name]))
        breakpoint()

class TestAcoustic(unittest.TestCase):

    def setUp(self):
        self.model = WRA(12)

    def test_res(self):
        pass

    def test_apply_dres_dcontrol(self):
        ## Set the linearization point
        state0 = self.model.get_state_vec() 
        state1 = self.model.get_state_vec()
        control_ = self.model.get_control_vec()
        props = self.model.get_properties_vec()

        state0['pref'][:] = 1
        self.model.set_ini_state(state0)
        self.model.set_fin_state(state1)
        self.model.set_control(control_)
        self.model.set_properties(props)

        dcontrol = self.model.get_control_vec()
        dcontrol['qin'] = 2.0

        dres = self.model.apply_dres_dcontrol(dcontrol)

        res0 = self.model.res()
        self.model.set_control(control_ + dcontrol)
        res1 = self.model.res()
        dres_fd = res1-res0

        print(dres.to_ndarray())
        print(dres_fd.to_ndarray())

if __name__ == '__main__':
    test = TestFSAIModel()
    test.setUp()
    test.test_apply_dres_dstate1()

    # test = TestAcoustic()
    # test.setUp()
    # test.test_apply_dres_dcontrol()
