import unittest
import os

import dolfin as dfn
import numpy as np

from femvf.models import (load_fsai_model, Rayleigh, KelvinVoigt, Bernoulli, WRAnalog)
from femvf.constants import PASCAL_TO_CGS
from blockarray import linalg

from modeldefs import (
    load_fsai_rayleigh_model, load_fsi_kelvinvoigt_model, load_fsi_rayleigh_model
)

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
        acoustic = WRAnalog(44)
        model = load_fsai_model(self.mesh_path, None, acoustic, SolidType=Rayleigh, FluidType=Bernoulli,
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
        sl_props['kcontact'][()] = 1e11
        sl_props['ycontact'][()] = fl_props['y_midline'] - y_gap*1/2

        ac_props = model.acoustic.get_properties_vec(set_default=True)
        ac_props['area'][:] = 4.0
        ac_props['length'][:] = 12.0
        ac_props['soundspeed'][:] = 340*100

        props = vec.concatenate_vec([sl_props, fl_props, ac_props])

        # Set the initial state
        u0 = dfn.Function(model.solid.vector_fspace).vector()

        # model.fluid.set_props(fluid_props)
        # qp0, *_ = model.fluid.solve_qp0()

        ini_state = model.get_state_vec()
        ini_state.set(0.0)
        ini_state['u'][:] = u0
        # ini_state['q'][()] = qp0['q']
        # ini_state['p'][:] = qp0['p']

        return model, ini_state, controls, props

    def test_apply_dres_dstate1(self):
        model, ini_state, controls, props = self.config_fsai_model()
        model.set_props(props)

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
        model.set_props(props)

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
        self.model = WRAnalog(12)

    def test_res(self):
        pass

class TestModelResidualSensitivity(unittest.TestCase):
    def setUp(self):
        """
        This code must create the model and set the parameters for the time step to be tested
        """
        self.model, self.props = load_fsi_rayleigh_model(coupling='explicit')

        self.state1 = self.model.get_state_vec()
        self.state0 = self.model.get_state_vec()
        self.control = self.model.get_control_vec()
        self.dt = 1e-4

        bc_base = self.model.solid.bc_base
        self.state0['u'][:] = 1e-4
        self.state0['v'][:] = 1e-4
        self.state0['a'][:] = 1e-4
        bc_base.apply(self.state0['u'])
        bc_base.apply(self.state0['v'])
        bc_base.apply(self.state0['a'])

        self.state1['u'][:] = 2.5e-4
        self.state1['v'][:] = 2.5e-4
        self.state1['a'][:] = 2.5e-4
        bc_base.apply(self.state1['u'])
        bc_base.apply(self.state1['v'])
        bc_base.apply(self.state1['a'])

        self.state0['p'][:] = 800.0 * PASCAL_TO_CGS
        self.state1['p'][:] = 800.0 * PASCAL_TO_CGS
        self.control['psub'][:] = 800 * PASCAL_TO_CGS

    ## Convenience functions to represent the residual being tested
    def res(self, state1, state0, control, props, dt):
        self.set_linearization(state1, state0, control, props, dt)
        return self.model.assem_res()

    def set_linearization(self, state1, state0, control, props, dt):
        self.model.set_fin_state(state1)
        self.model.set_ini_state(state0)
        self.model.set_control(control)
        self.model.set_props(props)
        self.model.dt = dt

    def dres_dxx(self, dstate0, dcontrol, dprops, ddt):
        """
        Compute the action of the sensitivity
        """
        dres = (
            self.model.apply_dres_dstate0(dstate0)
            + self.model.apply_dres_dcontrol(dcontrol)
            + self.model.apply_dres_dp(dprops)
            + self.model.apply_dres_ddt(ddt))
        return dres

    def dres_dxx_adj(self, dres):
        """
        Compute the action of the sensitivity
        """
        dstate0 = self.model.apply_dres_dstate0_adj(dres)
        dcontrol = self.model.apply_dres_dcontrol_adj(dres)
        dprops = self.model.apply_dres_dp_adj(dres)
        ddt = self.model.apply_dres_ddt_adj(dres)
        return dstate0, dcontrol, dprops, ddt

    ## Test sensitivities of the residual
    def test_solve_dres_dstate1(self):
        """
        To test the sensitivity, compute `dres` using a specified `dstate` through
        a finite difference. Then try to recover the specified `dstate` by solving with the jacobian
        from `dres` and compare whether the results are sufficiently similar
        """
        # Define step vector for state1
        dstate1 = self.model.get_state_vec()
        bc_base = self.model.solid.bc_base
        dstate1['u'][:] = 1e-8
        dstate1['v'][:] = 1e-8
        dstate1['a'][:] = 1e-8
        dstate1['q'][:] = 10.0
        dstate1['p'][:] = 10.0
        for name in ('u', 'v', 'a'):
            bc_base.apply(dstate1[name])

        args = (self.state0, self.control, self.props, self.dt)
        dres = self.res(self.state1+dstate1, *args) - self.res(self.state1, *args)

        self.set_linearization(self.state1, self.state0, self.control, self.props, self.dt)
        dstate1_jac = self.model.solve_dres_dstate1(dres)

        err = dstate1_jac - dstate1
        breakpoint()
        self.assertAlmostEqual(err.norm(), 0.0)

    def test_solve_dres_dstate1_adj(self):
        raise NotImplementedError

    def test_apply_dres_dxx(self):
        """
        To test the sensitivity, compute the residual sensitivity, `dres` in two
        ways for an input `dxx`:
            1. using finite differences
            2. using the jacobian (which is being tested)
        and compare whether the results are sufficiently similar
        """
        # Define step vectors to use for state, control, etc.
        dstate0 = self.model.get_state_vec()
        dcontrol = self.model.get_control_vec()
        dprops = self.model.get_properties_vec()
        ddt = 1e-9

        bc_base = self.model.solid.bc_base
        dstate0['u'][:] = 1e-4
        dstate0['v'][:] = 1e-4
        dstate0['a'][:] = 1e-4
        dstate0['q'][:] = 0.0
        dstate0['p'][:] = 1e4
        for name in ('u', 'v', 'a'):
            bc_base.apply(dstate0[name])

        # Define the generic testing method for a combination of all steps
        def _test(xxs, dxxs):
            self.set_linearization(self.state1, *xxs)
            dres = self.dres_dxx(*dxxs)
            dres_fd = (
                self.res(self.state1, *[xx+dxx for xx, dxx in zip(xxs, dxxs)])
                - self.res(self.state1, *xxs))

            err = dres - dres_fd
            print(dres['u'].norm('l2'), dres_fd['u'].norm('l2'))
            breakpoint()
            self.assertAlmostEqual(err.norm(), 0)

        ## Test dt steps
        xxs = (self.state0, self.control, self.props, self.dt)
        dxxs = (0.0*dstate0, 0.0*dcontrol, 0.0*dprops, ddt)
        _test(xxs, dxxs)

        ## Test dstate steps
        xxs = (self.state0, self.control, self.props, self.dt)
        dxxs = (dstate0, 0.0*dcontrol, 0.0*dprops, 0*ddt)
        _test(xxs, dxxs)

    def test_solve_dres_dxx_adj(self):
        raise NotImplementedError


if __name__ == '__main__':
    # test = TestFSAIModel()
    # test.setUp()
    # test.test_apply_dres_dstate1()

    test = TestModelResidualSensitivity()
    test.setUp()
    test.test_apply_dres_dxx()
    # test.test_solve_dres_dstate1()
    # test_apply_dres_dxx_adj()
