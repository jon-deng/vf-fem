"""
Contains the Model class that couples fluid and solid behaviour

TODO: Think of a consistent strategy to handle the coupling between domains
Currently a lot of code is being duplicated to do the coupling through different
approaches (very confusing)
"""

from os import path
import warnings
import numpy as np
import dolfin as dfn
from petsc4py import PETSc
# import ufl

from ..equations.solid import newmark
from ..fsi import FSIMap
from . import base, solid as smd, fluid as fmd, acoustic as amd
from femvf import meshutils
from blockarray import linalg
from blockarray import blockvec as bvec
from femvf.solverconst import DEFAULT_NEWTON_SOLVER_PRM

class FSIModel(base.Model):
    """
    Represents a coupled system of a solid and a fluid model

    Parameters
    ----------
    solid, fluid

    Attributes
    ----------
    solid : Solid
        A solid model object
    fluid : Fluid
        A fluid model object

    fsi_verts : array_like
        A list of vertex indices on the pressure driven surface (FSI). These must be are ordered in
        increasing streamwise direction since the fluid mesh numbering is ordered like that too.
    fsi_coordinates
    """
    def __init__(self, solid: smd.Solid, fluid: fmd.QuasiSteady1DFluid, solid_fsi_dofs, fluid_fsi_dofs):
        self.solid = solid
        self.fluid = fluid

        ## Specify state, controls, and properties
        self.state0 = bvec.concatenate_vec([self.solid.state0, self.fluid.state0])
        self.state1 = bvec.concatenate_vec([self.solid.state1, self.fluid.state1])

        # The control is just the subglottal and supraglottal pressures
        self.control = self.fluid.control[1:].copy()

        _self_properties = bvec.BlockVector((np.array([1.0]),), (1,), (('ymid',),))
        self.props = bvec.concatenate_vec([self.solid.props, self.fluid.props, _self_properties])

        ## FSI related stuff
        self._solid_area = dfn.Function(self.solid.forms['fspace.scalar']).vector()
        # self._dsolid_area = dfn.Function(self.solid.forms['fspace.scalar']).vector()

        self.fsimap = FSIMap(
            self.fluid.state1['p'].size, self._solid_area.size(), fluid_fsi_dofs, solid_fsi_dofs
            )

    ## These have to be defined to exchange data between fluid/solid domains
    def set_ini_solid_state(self, uva0):
        raise NotImplementedError("")

    def set_fin_solid_state(self, uva1):
        raise NotImplementedError("")

    def set_ini_fluid_state(self, qp0):
        raise NotImplementedError("")

    def set_fin_fluid_state(self, qp1):
        raise NotImplementedError("")

    ## Methods for settings parameters of the model
    @property
    def dt(self):
        return self.solid.dt

    @dt.setter
    def dt(self, value):
        self.solid.dt = value
        self.fluid.dt = value

    def set_ini_state(self, state):
        self.set_ini_solid_state(state[:3])
        self.set_ini_fluid_state(state[3:])

    def set_fin_state(self, state):
        self.set_fin_solid_state(state[:3])
        self.set_fin_fluid_state(state[3:])

    def set_control(self, control):
        self.control[:] = control

        for key, value in control.items():
            self.fluid.control[key][:] = value

    def set_props(self, props):
        self.props[:] = props

        self.solid.set_props(props[:self.solid.props.size])
        self.fluid.set_props(props[self.solid.props.size:])

    # Additional more friendly method for setting parameters (use the above defined methods)
    def set_params_fromfile(self, f, n, update_props=True):
        """
        Set all parameters needed to integrate the model from a recorded value.

        Iteration `n` is the implicit relation
        :math:`f^{n}(u_n, u_{n-1}, p)`
        that gives the displacement at index `n`, given the state at `n-1` and all additional
        parameters.

        Parameters
        ----------
        f : statefileutils.StateFile
        n : int
            Index of iteration to set
        """
        # Get data from the state file
        if update_props:
            self.set_props(f.get_props())

        state = f.get_state(n)
        control = None
        if f.variable_controls:
            control = f.get_control(n)
        else:
            control = f.get_control(0)
        self.set_fin_state(state)
        self.set_control(control)

    def set_iter_params_fromfile(self, f, n, update_props=True):
        """
        Set all parameters needed to integrate the model and an initial guess, based on a recorded
        iteration.

        Iteration `n` is the implicit relation
        :math:`f^{n}(u_n, u_{n-1}, p)`
        that gives the displacement at index `n`, given the state at `n-1` and all additional
        parameters.

        Parameters
        ----------
        statefile : statefileutils.StateFile
        n : int
            Index of iteration to set
        """
        # Get data from the state file
        if update_props:
            self.set_props(f.get_props())

        state0 = f.get_state(n-1)
        state1 = f.get_state(n)

        control0, control1 = None, None
        if f.variable_controls:
            control0 = f.get_control(n-1)
            control1 = f.get_control(n)
        else:
            control0 = f.get_control(0)
            control1 = f.get_control(0)

        dt = f.get_time(n) - f.get_time(n-1)

        # Assign the values to the model
        self.set_iter_params(state0, control0, state1, control1, dt)

    ## Methods for getting vectors
    def get_state_vec(self):
        return bvec.concatenate_vec([self.solid.get_state_vec(), self.fluid.get_state_vec()])

    def get_control_vec(self):
        ret = self.control.copy()
        ret.set(0.0)
        return ret

    def get_properties_vec(self):
        return self.props.copy()

    ## Residual functions
    # The below residual function definitions are common to both explicit and implicit FSI models
    def apply_dres_dcontrol(self, x):
        ## Implement here since both FSI models should use this rule
        dres = self.get_state_vec()
        _, _, dq_dpsub, dp_dpsub, dq_dpsup, dp_dpsup = self.fluid.flow_sensitivity(*self.fluid.control.vecs, self.fluid.props)

        # Only the flow rate and pressure residuals are sensitive to the
        # controls
        dres.set(0.0)
        dres['q'][:] = dp_dpsub*x['psub'] + dp_dpsup*x['psup']
        dres['p'][:] = dq_dpsub*x['psub'] + dq_dpsup*x['psup']
        return -dres

    def apply_dres_dcontrol_adj(self, x):
        ## Implement here since both FSI models should use this rule
        b = self.get_control_vec()
        _, _, dq_dpsub, dp_dpsub, dq_dpsup, dp_dpsup = self.fluid.flow_sensitivity(*self.fluid.control.vecs, self.fluid.props)

        b['psub'][:] = np.dot(x['p'], dp_dpsub) + np.dot(x['q'], dq_dpsub)
        b['psup'][:] = np.dot(x['p'], dp_dpsup) + np.dot(x['q'], dq_dpsup)
        return -b

    def apply_dres_ddt(self, x):
        dres = self.get_state_vec()
        dres[:3] = self.solid.apply_dres_ddt(x)
        return dres

    def apply_dres_ddt_adj(self, x):
        x_solid = x[:3]
        return self.solid.apply_dres_ddt_adj(x_solid)

class ExplicitFSIModel(FSIModel):
    def set_ini_solid_state(self, uva0):
        """Set the initial solid state"""
        self.solid.set_ini_state(uva0)

    def set_fin_solid_state(self, uva1):
        """Set the final solid state and communicate FSI interactions"""
        self.solid.set_fin_state(uva1)

        # For explicit coupling, the final fluid area corresponds to the final solid deformation
        self._solid_area[:] = 2*(self.props['ymid'][0] - (self.solid.XREF.vector() + self.solid.state1['u'])[1::2])
        fl_control = self.fluid.control.copy()
        self.fsimap.map_solid_to_fluid(self._solid_area, fl_control['area'][:])
        self.fluid.set_control(fl_control)

    def set_ini_fluid_state(self, qp0):
        """Set the fluid state and communicate FSI interactions"""
        self.fluid.set_ini_state(qp0)

        # For explicit coupling, the final solid pressure corresponds to the initial fluid pressure
        sl_control = self.solid.control.copy()
        self.fsimap.map_fluid_to_solid(qp0[1], sl_control['p'])
        self.solid.set_control(sl_control)

    def set_fin_fluid_state(self, qp1):
        """Set the final fluid state"""
        self.fluid.set_fin_state(qp1)

    ## Solver functions
    def res(self):
        """
        Return the residual vector, F
        """
        dt = self.solid.dt
        u1, v1, a1 = self.solid.u1.vector(), self.solid.v1.vector(), self.solid.a1.vector()
        u0, v0, a0 = self.solid.u0.vector(), self.solid.v0.vector(), self.solid.a0.vector()
        res = self.get_state_vec()

        res['u'][:] = dfn.assemble(self.solid.forms['form.un.f1'])
        self.solid.bc_base.apply(res['u'])
        res['v'][:] = v1 - newmark.newmark_v(u1, u0, v0, a0, dt)
        res['a'][:] = a1 - newmark.newmark_a(u1, u0, v0, a0, dt)

        qp, *_ = self.fluid.solve_state1(self.fluid.state0)
        res['q'][:] = self.fluid.state1['q'] - qp['q']
        res['p'][:] = self.fluid.state1['p'] - qp['p']
        return res

    # Forward solver methods
    def solve_state1(self, ini_state, newton_solver_prm=None):
        """
        Solve for the final state given an initial guess
        """
        # Set the initial guess for the final state
        self.set_fin_state(ini_state)

        if newton_solver_prm is None:
            newton_solver_prm = DEFAULT_NEWTON_SOLVER_PRM
        uva1, solid_info = self.solid.solve_state1(ini_state[:3], newton_solver_prm)

        self.set_fin_solid_state(uva1)
        qp1, fluid_info = self.fluid.solve_state1(ini_state[3:])

        step_info = solid_info
        step_info.update({'fluid_info': fluid_info})

        return bvec.concatenate_vec([uva1, qp1]), step_info

    def solve_dres_dstate1(self, b):
        """
        Solve, dF/du x = f
        """
        x = self.get_state_vec()

        x[:3] = self.solid.solve_dres_dstate1(b[:3])

        dq_du, dp_du = self.fluid.solve_dqp1_du1_solid(self, adjoint=False)
        dfq2_du2 = 0.0 - dq_du
        dfp2_du2 = 0.0 - dp_du

        x['q'][:] = b['q'] - dfq2_du2.inner(x['u'])
        # _bp = dfp2_du2.getVecRight()
        # dfp2_du2.multTranspose(x['u'].vec(), _bp)
        # breakpoint()
        x['p'][:] = b['p'] - dfp2_du2*x['u'].vec()
        return x

    def solve_dres_dstate1_adj(self, x):
        """
        Solve, dF/du^T x = f
        """
        ## Assemble sensitivity matrices
        dq_du, dp_du = self.fluid.solve_dqp1_du1_solid(self, adjoint=True)
        dfq2_du2 = 0 - dq_du
        dfp2_du2 = 0 - dp_du

        ## Do the linear algebra that solves for x
        b = self.get_state_vec()
        b_qp, b_uva = b[3:], b[:3]

        # This performs the fluid part of the solve
        b_qp[:] = x[3:]

        # This is the solid part of the
        bp_vec = dfp2_du2.getVecRight()
        bp_vec[:] = b_qp['p']
        rhs = x[:3].copy()
        rhs['u'] -= (dfq2_du2*b_qp['q'] + dfn.PETScVector(dfp2_du2*bp_vec))
        b_uva[:] = self.solid.solve_dres_dstate1_adj(rhs)

        return b

    def apply_dres_dstate0(self, x):
        dfu2_dp1 = dfn.assemble(self.solid.forms['form.bi.df1_dp1'], tensor=dfn.PETScMatrix())
        self.solid.bc_base.zero(dfu2_dp1)

        # Map the fluid pressure state to the solid sides forcing pressure
        dp_vec = dfn.PETScVector(dfu2_dp1.mat().getVecRight())
        solid_dofs, fluid_dofs = self.get_fsi_scalar_dofs()
        dp_vec[solid_dofs] = x['p'][fluid_dofs]

        b = self.get_state_vec()
        # compute solid blocks
        b[:3] = self.solid.apply_dres_dstate0(x[:3])
        b['u'][:] += dfu2_dp1 * dp_vec

        # compute fluid blocks
        b[3:] = self.fluid.apply_dres_dstate0(x[3:]) #+ solid effect is 0
        return b

    def apply_dres_dstate0_adj(self, x):
        dfu2_dp1 = dfn.assemble(self.solid.forms['form.bi.df1_dp1_adj'], tensor=dfn.PETScMatrix())

        solid_dofs, fluid_dofs = self.get_fsi_scalar_dofs()
        dfu2_dp1 = dfn.as_backend_type(dfu2_dp1).mat()
        dfu2_dp1 = linalg.reorder_mat_rows(dfu2_dp1, solid_dofs, fluid_dofs, fluid_dofs.size)
        matvec_adj_p_rhs = dfu2_dp1*dfn.as_backend_type(x['u']).vec()

        b = self.get_state_vec()
        # Set uva blocks of b
        b[:3] = self.solid.apply_dres_dstate0_adj(x[:3])
        b['q'][:] = 0.0
        b['p'][:] = matvec_adj_p_rhs
        return b

    def apply_dres_dcontrol(self, x):
        dres = self.get_state_vec()
        dres.set(0.0)
        return dres

    def apply_dres_dp(self, x):
        dres = self.get_state_vec()
        dres.set(0.0)
        return dres

    def apply_dres_dp_adj(self, x):
        bsolid = self.solid.apply_dres_dp_adj(x[:3])
        bfluid = self.fluid.apply_dres_dp_adj(x[3:])
        return bvec.concatenate_vec([bsolid, bfluid])

class ImplicitFSIModel(FSIModel):
    ## These must be defined to properly exchange the forcing data between the solid and domains
    def set_ini_fluid_state(self, qp0):
        self.fluid.set_ini_state(qp0)

        p0_solid = self.map_fsi_scalar_from_fluid_to_solid(qp0[1])
        control = bvec.BlockVector((p0_solid,), labels=self.solid.control.labels)

    def set_fin_fluid_state(self, qp1):
        self.fluid.set_fin_state(qp1)

        p1_solid = self.map_fsi_scalar_from_fluid_to_solid(qp1[1])
        control = bvec.BlockVector((p1_solid,), labels=self.solid.control.labels)
        self.solid.set_control(control)

    ## Forward solver methods
    def res(self):
        """
        Return the residual vector, F
        """
        dt = self.solid.dt
        u1, v1, a1 = self.solid.u1.vector(), self.solid.v1.vector(), self.solid.a1.vector()
        u0, v0, a0 = self.solid.u0.vector(), self.solid.v0.vector(), self.solid.a0.vector()
        res = self.get_state_vec()

        res['u'][:] = dfn.assemble(self.solid.forms['form.un.f1'])
        self.solid.bc_base.apply(res['u'])
        res['v'][:] = v1 - newmark.newmark_v(u1, u0, v0, a0, dt)
        res['a'][:] = a1 - newmark.newmark_a(u1, u0, v0, a0, dt)

        qp, *_ = self.fluid.solve_state1(self.fluid.state0)
        res['q'][:] = self.fluid.state1['q'] - qp['q']
        res['p'][:] = self.fluid.state1['p'] - qp['p']
        return res

    def solve_state1(self, ini_state, newton_solver_prm=None):
        """
        Solve for the final state given an initial guess

        This uses a fixed-point iteration where the solid is solved, then the fluid and so-on.
        """
        if newton_solver_prm is None:
            newton_solver_prm = DEFAULT_NEWTON_SOLVER_PRM

        # Set initial guesses for the states at the next time
        # uva1 = solid.get_state_vec()
        uva1 = ini_state[:3].copy()
        qp1 = ini_state[3:].copy()

        # Calculate the initial residual
        self.set_fin_solid_state(uva1)
        self.set_fin_fluid_state(qp1)
        res0 = dfn.assemble(self.solid.f1)
        self.solid.bc_base.apply(res0)

        # Set tolerances for the fixed point iterations
        fluid_info = None
        nit, max_nit = 0, 10
        abs_tol, rel_tol = newton_solver_prm['absolute_tolerance'], newton_solver_prm['relative_tolerance']
        abs_err_prev, abs_err, rel_err = 1.0, np.inf, np.inf
        while abs_err > abs_tol and rel_err > rel_tol and nit < max_nit:
            # Solve the solid with the previous iteration's fluid pressures
            uva1, _ = self.solid.solve_state1(uva1, newton_solver_prm)

            # Compute new fluid pressures for the updated solid position
            self.set_fin_solid_state(uva1)
            qp1, fluid_info = self.fluid.solve_state1(qp1)
            self.set_fin_fluid_state(qp1)

            # Calculate the error in the solid residual with the updated pressures
            res = dfn.assemble(self.solid.f1)
            self.solid.bc_base.apply(res)

            abs_err = res.norm('l2')
            rel_err = abs_err/abs_err_prev

            nit += 1
        res = dfn.assemble(self.solid.forms['form.un.f1'])
        self.solid.bc_base.apply(res)

        step_info = {'fluid_info': fluid_info,
                     'num_iter': nit, 'abs_err': abs_err, 'rel_err': rel_err}

        return bvec.concatenate_vec([uva1, qp1]), step_info

    def solve_dres_dstate1(self, b):
        """
        Solve, dF/du x = f
        """
        dt = self.solid.dt
        x = self.get_state_vec()

        solid = self.solid

        dfu1_du1 = dfn.assemble(solid.df1_du1)
        dfv2_du2 = 0 - newmark.newmark_v_du1(dt)
        dfa2_du2 = 0 - newmark.newmark_a_du1(dt)

        dq_du, dp_du = self.fluid.solve_dqp1_du1_solid(self, adjoint=False)
        dfq2_du2 = 0 - dq_du
        dfp2_du2 = 0 - dp_du

        self.solid.bc_base.apply(dfu1_du1)
        dfn.solve(dfu1_du1, x['u'], b['u'], 'petsc')
        x['v'][:] = b['v'] - dfv2_du2*x['u']
        x['a'][:] = b['a'] - dfa2_du2*x['u']

        x['q'][:] = b['q'] - dfq2_du2.inner(x['u'])
        x['p'][:] = b['p'] - dfn.PETScVector(dfp2_du2*x['u'].vec())
        return x

    ## Adjoint solver methods
    def solve_dres_dstate1_adj(self, b):
        """
        Solve, dF/du^T x = f
        """
        ## Assemble sensitivity matrices
        # self.set_iter_params(**it_params)
        dt = self.solid.dt

        dfu2_du2 = self.solid.cached_form_assemblers['bilin.df1_du1_adj'].assemble()
        dfv2_du2 = 0 - newmark.newmark_v_du1(dt)
        dfa2_du2 = 0 - newmark.newmark_a_du1(dt)
        dfu2_dp2 = dfn.assemble(self.solid.forms['form.bi.df1_dp1_adj'])

        # map dfu2_dp2 to have p on the fluid domain
        solid_dofs, fluid_dofs = self.get_fsi_scalar_dofs()
        dfu2_dp2 = dfn.as_backend_type(dfu2_dp2).mat()
        dfu2_dp2 = linalg.reorder_mat_rows(dfu2_dp2, solid_dofs, fluid_dofs, self.fluid.state1['p'].size)

        dq_du, dp_du = self.fluid.solve_dqp1_du1_solid(self, adjoint=True)
        dfq2_du2 = 0 - dq_du
        dfp2_du2 = 0 - dp_du

        ## Do the linear algebra that solves for the adjoint states
        adj_uva = self.solid.get_state_vec()
        adj_qp = self.fluid.get_state_vec()

        adj_u_rhs, adj_v_rhs, adj_a_rhs, adj_q_rhs, adj_p_rhs = b

        # adjoint states for v, a, and q are explicit so we can solve for them
        self.solid.bc_base.apply(adj_v_rhs)
        adj_uva['v'][:] = adj_v_rhs

        self.solid.bc_base.apply(adj_a_rhs)
        adj_uva['a'][:] = adj_a_rhs

        # TODO: how to apply fluid boundary conditions in a generic way?
        adj_qp['q'][:] = adj_q_rhs

        adj_u_rhs -= dfv2_du2*adj_uva['v'] + dfa2_du2*adj_uva['a'] + dfq2_du2*adj_qp['q']

        bc_dofs = np.array(list(self.solid.bc_base.get_boundary_values().keys()), dtype=np.int32)
        self.solid.bc_base.apply(dfu2_du2, adj_u_rhs)
        dfp2_du2.zeroRows(bc_dofs, diag=0.0)
        # self.solid.bc_base.zero_columns(dfu2_du2, adj_u_rhs.copy(), diagonal_value=1.0)

        # solve the coupled system for pressure and displacement residuals
        dfu2_du2_mat = dfn.as_backend_type(dfu2_du2).mat()
        blocks = [[dfu2_du2_mat, dfp2_du2], [dfu2_dp2, 1.0]]

        dfup2_dup2 = linalg.form_block_matrix(blocks)
        adj_up, rhs = dfup2_dup2.getVecs()

        # calculate rhs vectors
        rhs[:adj_u_rhs.size()] = adj_u_rhs
        rhs[adj_u_rhs.size():] = adj_p_rhs

        # Solve the block linear system with LU factorization
        ksp = PETSc.KSP().create()
        ksp.setType(ksp.Type.PREONLY)

        pc = ksp.getPC()
        pc.setType(pc.Type.LU)

        ksp.setOperators(dfup2_dup2)
        ksp.solve(rhs, adj_up)

        adj_uva['u'][:] = adj_up[:adj_u_rhs.size()]
        adj_qp['p'][:] = adj_up[adj_u_rhs.size():]

        return bvec.concatenate_vec([adj_uva, adj_qp])

    def apply_dres_dstate0_adj(self, x):
        adj_u2, adj_v2, adj_a2, adj_q2, adj_p2 = x

        ## Assemble sensitivity matrices
        # dt2 = it_params2['dt']
        solid = self.solid
        dt = self.dt
        # model.set_iter_params(**it_params2)

        dfu2_du1 = solid.cached_form_assemblers['bilin.df1_du0_adj'].assemble()
        dfu2_dv1 = solid.cached_form_assemblers['bilin.df1_dv0_adj'].assemble()
        dfu2_da1 = solid.cached_form_assemblers['bilin.df1_da0_adj'].assemble()

        dfv2_du1 = 0 - newmark.newmark_v_du0(dt)
        dfv2_dv1 = 0 - newmark.newmark_v_dv0(dt)
        dfv2_da1 = 0 - newmark.newmark_v_da0(dt)
        dfa2_du1 = 0 - newmark.newmark_a_du0(dt)
        dfa2_dv1 = 0 - newmark.newmark_a_dv0(dt)
        dfa2_da1 = 0 - newmark.newmark_a_da0(dt)

        ## Do the matrix vector multiplication that gets the RHS for the adjoint equations
        # Allocate a vector the for fluid side mat-vec multiplication
        b = x.copy()
        b['u'][:] = (dfu2_du1*adj_u2 + dfv2_du1*adj_v2 + dfa2_du1*adj_a2)
        b['v'][:] = (dfu2_dv1*adj_u2 + dfv2_dv1*adj_v2 + dfa2_dv1*adj_a2)
        b['a'][:] = (dfu2_da1*adj_u2 + dfv2_da1*adj_v2 + dfa2_da1*adj_a2)
        b['q'][:] = 0
        b['p'][:] = 0

        return b

    def apply_dres_dp_adj(self, x):
        # bsolid = self.solid.get_properties_vec()
        # bfluid = self.fluid.get_properties_vec()
        bsolid = self.solid.apply_dres_dp_adj(x[:3])
        bfluid = self.fluid.apply_dres_dp_adj(x[3:])
        return bvec.concatenate_vec([bsolid, bfluid])

    def apply_dres_dcontrol_adj(self, x):
        return super().apply_dres_dcontrol_adj(x)

class FSAIModel(FSIModel):
    """
    Represents a fluid-structure-acoustic interaction system
    """
    def __init__(self, solid, fluid, acoustic, fsi_verts):
        self.solid = solid
        self.fluid = fluid
        self.acoustic = acoustic

        state = bvec.concatenate_vec([solid.get_state_vec(), fluid.get_state_vec(), acoustic.get_state_vec()])
        self.state0 = state
        self.state1 = state.copy()

        control = bvec.BlockVector((np.array([1.0]),), labels=[('psub',)])
        self.control = control.copy()

        self.props = bvec.concatenate_vec(
            [solid.props, fluid.props, acoustic.props])

        self._dt = 1.0

        self.fsi_verts = fsi_verts

    @property
    def dt(self):
        return self.solid.dt

    @dt.setter
    def dt(self, value):
        #  the acoustic domain time step is ignored for WRAnalog
        for domain in (self.solid, self.fluid):
            domain.dt = value

    ## Parameter setting methods
    def set_ini_state(self, state):
        sl_nblock = len(self.solid.state0.size)
        fl_nblock = len(self.fluid.state0.size)
        ac_nblock = len(self.acoustic.state0.size)

        sl_state = state[:sl_nblock]
        fl_state = state[sl_nblock:sl_nblock+fl_nblock]
        ac_state = state[sl_nblock+fl_nblock:sl_nblock+fl_nblock+ac_nblock]

        self.set_ini_solid_state(sl_state)
        self.set_ini_fluid_state(fl_state)
        self.set_ini_acoustic_state(ac_state)

    def set_fin_state(self, state):
        sl_nblock = len(self.solid.state0.size)
        fl_nblock = len(self.fluid.state0.size)
        ac_nblock = len(self.acoustic.state0.size)

        sl_state = state[:sl_nblock]
        fl_state = state[sl_nblock:sl_nblock+fl_nblock]
        ac_state = state[sl_nblock+fl_nblock:sl_nblock+fl_nblock+ac_nblock]

        self.set_fin_solid_state(sl_state)
        self.set_fin_fluid_state(fl_state)
        self.set_fin_acoustic_state(ac_state)

    def set_control(self, control):
        fl_control = self.fluid.control.copy()
        fl_control['psub'][:] = control['psub']
        self.fluid.set_control(fl_control)

    def set_props(self, props):
        sl_nblock = len(self.solid.props.size)
        fl_nblock = len(self.fluid.props.size)
        ac_nblock = len(self.acoustic.props.size)

        sl_props = props[:sl_nblock]
        fl_props = props[sl_nblock:sl_nblock+fl_nblock]
        ac_props = props[sl_nblock+fl_nblock:sl_nblock+fl_nblock+ac_nblock]

        self.solid.set_props(sl_props)
        self.fluid.set_props(fl_props)
        self.acoustic.set_props(ac_props)

    ## Coupling methods
    def set_ini_solid_state(self, sl_state0):
        self.solid.set_ini_state(sl_state0)

    def set_ini_fluid_state(self, fl_state0):
        self.fluid.set_ini_state(fl_state0)

        # for explicit coupling
        sl_control = self.solid.control.copy()
        fsi_sdofs = self.solid.vert_to_sdof[self.fsi_verts].copy()
        sl_control = fl_state_to_sl_control(fl_state0, sl_control, fsi_sdofs)
        self.solid.set_control(sl_control)

    def set_ini_acoustic_state(self, ac_state0):
        self.acoustic.set_ini_state(ac_state0)

    def set_fin_solid_state(self, sl_state1):
        self.solid.set_fin_state(sl_state1)

        fl_control = self.fluid.control.copy()

        fsi_ref_config = self.solid.mesh.coordinates()[self.fsi_verts].reshape(-1)
        fsi_vdofs = self.solid.vert_to_vdof.reshape(-1, 2)[self.fsi_verts].reshape(-1).copy()
        fl_control = sl_state_to_fl_control(sl_state1, fl_control, fsi_ref_config, fsi_vdofs)

        self.fluid.set_control(fl_control)

    def set_fin_fluid_state(self, fl_state1):
        self.fluid.set_fin_state(fl_state1)

        ac_control = fl_state_to_ac_control(fl_state1, self.acoustic.control.copy())
        self.acoustic.set_control(ac_control)

    def set_fin_acoustic_state(self, ac_state1):
        self.acoustic.set_fin_state(ac_state1)

        control = ac_state_to_fl_control(ac_state1, self.fluid.control.copy())
        self.fluid.set_control(control)

    ## Empty parameter vectors
    def get_control_vec(self):
        ret = self.control.copy()
        ret.set(0.0)
        return ret

    def get_state_vec(self):
        ret = self.state0.copy()
        ret.set(0.0)
        return ret

    def get_properties_vec(self, set_default=True):
        ret = self.props.copy()
        if not set_default:
            ret.set(0.0)
        return ret

    ## Solver methods
    def res(self):
        res_sl = self.solid.res()
        res_fl = self.fluid.res()
        res_ac = self.acoustic.res()
        return bvec.concatenate_vec([res_sl, res_fl, res_ac])

    def solve_state1(self, ini_state, newton_solver_prm=None):
        if newton_solver_prm is None:
            newton_solver_prm = DEFAULT_NEWTON_SOLVER_PRM

        ## Solve solid displacements at n
        self.set_fin_state(ini_state)
        ini_slstate = ini_state[:3]
        sl_state1, solver_info = self.solid.solve_state1(ini_slstate, newton_solver_prm)
        self.set_fin_solid_state(sl_state1)
        # print(self.solid.res().norm())
        # if self.solid.res().norm() > 1e-4:
        #     breakpoint()

        # self.set_fin_state(fin_state)
        # res_norm = self.res().norm()
        # print(res_norm)
        # if res_norm > 1:
        #     breakpoint()

        def make_linearized_flow_residual(qac):
            """
            Represents the coupled linearized subproblem @ qac
            res(qac) = qac - qbern(psup(qac))
            solve_jac(res) = res/(1 - jac(qbern(psup(qac))))
            """
            # Linearize the fluid/acoustic models about `qac`
            ac_control = self.acoustic.get_control_vec()
            ac_control['qin'][:] = qac['qin']
            self.acoustic.set_control(ac_control)
            ac_state1, _ = self.acoustic.solve_state1()
            self.set_fin_acoustic_state(ac_state1)
            fl_state1, _ = self.fluid.solve_state1(self.fluid.state0)

            dqbern_dpsup = self.fluid.flow_sensitivity(*self.fluid.control.vecs, self.fluid.props)[4]
            dpsup_dqac = self.acoustic.z[0]
            def res():
                qbern = fl_state1[0]
                return qac - bvec.BlockVector((qbern,), labels=[('qin',)])

            def solve_jac(res):
                dres_dq = 1-dqbern_dpsup*dpsup_dqac
                return res/dres_dq

            return res, solve_jac

        q, info = smd.newton_solve(bvec.BlockVector((ini_state['q'],), labels=[('qin',)]), make_linearized_flow_residual)

        self.acoustic.set_control(q)
        ac_state1, _ = self.acoustic.solve_state1()
        fl_state1, fluid_info = self.fluid.solve_state1(self.fluid.state0)

        step_info = {'fluid_info': fluid_info, **info}
        fin_state = bvec.concatenate_vec([sl_state1, fl_state1, ac_state1])

        return fin_state, step_info

    def _form_dflac_dflac(self):
        b = self.state0

        ## Solve the coupled fluid/acoustic system
        # First compute some sensitivities that are needed
        *_, dq_dpsup, dp_dpsup = self.fluid.flow_sensitivity(*self.fluid.control.vecs, self.fluid.props)

        # solve the coupled system for pressure and acoustic residuals
        dfq_dq = 1.0
        dfp_dp = 1.0
        dfpinc_dpinc = PETSc.Mat().createAIJ((b['pinc'].size, b['pinc'].size), nnz=b['pinc'].size)
        diag = PETSc.Vec().createSeq(b['pinc'].size)
        diag.set(1.0)
        dfpinc_dpinc.setDiagonal(diag)
        dfpref_dpref = 1.0

        # dfluid / dacoustic
        dfq_dpref = PETSc.Mat().createAIJ((b['q'].size, b['pref'].size), nnz=1)
        dfq_dpsup = -dq_dpsup
        dpsup_dpref = 1.0 # Supraglottal pressure is equal to very first reflected pressure
        dfq_dpref.setValue(0, 0, dfq_dpsup*dpsup_dpref)

        dfp_dpref = PETSc.Mat().createAIJ((b['p'].size, b['pref'].size), nnz=b['p'].size)
        dfp_dpsup = -dp_dpsup
        dfp_dpref.setValues(np.arange(b['p'].size, dtype=np.int32), 0, dfp_dpsup*dpsup_dpref)

        # dacoustic / dfluid
        dfpref_dq = PETSc.Mat().createAIJ((b['pref'].size, b['q'].size), nnz=2)
        dcontrol = self.acoustic.get_control_vec()
        dcontrol.set(0.0)
        dcontrol['qin'][:] = 1.0
        dfpref_dqin = self.acoustic.apply_dres_dcontrol(dcontrol)['pref'][:2]
        dqin_dq = 1.0
        dfpref_dq.setValues(np.array([0, 1], dtype=np.int32), 0, dfpref_dqin*dqin_dq)

        for mat in (dfq_dpref, dfp_dpref, dfpref_dq):
            mat.assemble()

        blocks = [[   dfq_dq,    0.0,          0.0,    dfq_dpref],
                  [      0.0, dfp_dp,          0.0,    dfp_dpref],
                  [      0.0,    0.0, dfpinc_dpinc,          0.0],
                  [dfpref_dq,    0.0,          0.0, dfpref_dpref]]

        A = linalg.form_block_matrix(blocks)
        return A

    def solve_dres_dstate1(self, b):
        """
        Solve, dF/du x = f
        """
        x = self.get_state_vec()
        ## Assmeble any needed sensitivity matrices
        dq_du, dp_du = self.fluid.solve_dqp1_du1_solid(self, adjoint=False)
        dfq2_du2 = 0 - dq_du
        dfp2_du2 = 0 - dp_du

        ## Solve the solid system portion
        x[:3] = self.solid.solve_dres_dstate1(b[:3])

        ## Solve the coupled fluid/acoustic system
        A = self._form_dflac_dflac()
        adj_z, rhs = A.getVecs()

        # Get only the q, p, pinc, pref (fluid/acoustic) portions of the residual
        b_flac = b[3:].copy()
        b_flac['q'] -= (dfq2_du2*x['u']).sum()
        b_flac['p'] -= dfp2_du2*x['u'].vec()
        rhs[:] = b_flac.to_ndarray()

        ksp = PETSc.KSP().create()
        ksp.setType(ksp.Type.PREONLY)

        pc = ksp.getPC()
        pc.setType(pc.Type.LU)

        ksp.setOperators(A)
        ksp.solve(rhs, adj_z)

        x[3:].set_vec(adj_z)
        # print(linalg.dot(x[3:], x[3:]))

        return x

    # Adjoint solver methods
    def solve_dres_dstate1_adj(self, b):
        """
        Solve, dF/du^T x = f
        """
        x = self.get_state_vec()

        ## Solve the coupled fluid/acoustic system first
        A = self._form_dflac_dflac()
        adj_z, rhs = A.getVecs()

        # Get only the q, p, pinc, pref (fluid/acoustic) portions of the residual
        rhs[:] = b[3:].to_ndarray()

        ksp = PETSc.KSP().create()
        ksp.setType(ksp.Type.PREONLY)

        pc = ksp.getPC()
        pc.setType(pc.Type.LU)

        ksp.setOperators(A)
        ksp.solveTranspose(rhs, adj_z)

        x[3:].set_vec(adj_z)

        ## Assemble sensitivity matrices
        dq_du, dp_du = self.fluid.solve_dqp1_du1_solid(self, adjoint=True)
        dfq2_du2 = 0 - dq_du
        dfp2_du2 = 0 - dp_du

        _adj_p = dfp2_du2.getVecRight()
        _adj_p[:] = x['p']

        b_uva = b[:3].copy()
        b_uva['u'] -= (dfq2_du2*x['q'] + dfn.PETScVector(dfp2_du2*_adj_p))

        x_uva = x[:3]
        x_uva[:] = self.solid.solve_dres_dstate1_adj(b_uva)

        return x

    def apply_dres_dstate0_adj(self, x):
        dfu2_dp1 = dfn.assemble(self.solid.forms['form.bi.df1_dp1_adj'])

        solid_dofs, fluid_dofs = self.get_fsi_scalar_dofs()
        dfu2_dp1 = dfn.as_backend_type(dfu2_dp1).mat()
        dfu2_dp1 = linalg.reorder_mat_rows(dfu2_dp1, solid_dofs, fluid_dofs, fluid_dofs.size)
        matvec_adj_p_rhs = dfu2_dp1*dfn.as_backend_type(x['u']).vec()

        b = self.get_state_vec()
        # Set uva blocks of b
        b[:3] = self.solid.apply_dres_dstate0_adj(x[:3])
        b['q'][:] = 0.0
        b['p'][:] = matvec_adj_p_rhs

        # Set acoustics state blocks of b
        b[-2:] = self.acoustic.apply_dres_dstate0_adj(x[-2:])
        return b

    def apply_dres_dp_adj(self, x):
        bsl = self.solid.apply_dres_dp_adj(x[:3])
        bfl = self.fluid.apply_dres_dp_adj(x[3:5])
        bac = self.acoustic.apply_dres_dp_adj(x[5:])
        return bvec.concatenate_vec([bsl, bfl, bac])

    def apply_dres_dcontrol_adj(self, x):
        ## Implement here since both FSI models should use this rule
        b = self.get_control_vec()
        _, _, dq_dpsub, dp_dpsub, dq_dpsup, dp_dpsup = self.fluid.flow_sensitivity(*self.fluid.control.vecs, self.fluid.props)

        b['psub'][:] = np.dot(x['p'], dp_dpsub) + np.dot(x['q'][0], dq_dpsub)
        return -b
