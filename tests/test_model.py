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

class TestModel(unittest.TestCase):
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

        model.set_fin_control(control)

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

    def test_solve_dres_dstate1(self):
        model, ini_state, controls, props = self.config_fsai_model()
        model.set_properties(props)

        b = model.get_state_vec()
        b.set(1.0)
        x = model.solve_dres_dstate1(b)


if __name__ == '__main__':
    test = TestModel()
    test.setUp()
    test.test_solve_dres_dstate1()
