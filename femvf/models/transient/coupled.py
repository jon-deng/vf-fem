"""
Contains the Model class that couples fluid and solid behaviour

TODO: Think of a consistent strategy to handle the coupling between domains
Currently a lot of code is being duplicated to do the coupling through different
approaches (very confusing)
"""

from typing import List, TypeVar, Union
from numpy.typing import ArrayLike

import itertools

import numpy as np
import dolfin as dfn
from petsc4py import PETSc

from blockarray import blockvec as bv, blockmat as bm
from blockarray import linalg
from blockarray import subops
from femvf.solverconst import DEFAULT_NEWTON_SOLVER_PRM
from nonlineq import iterative_solve

from femvf.equations import newmark
from ..fsi import FSIMap, make_coupling_stuff
from . import base, solid as tsmd, fluid as tfmd, acoustic as amd


class BaseTransientFSIModel(base.BaseTransientModel):
    """
    Represents a coupled system of a solid and fluid(s) models

    Parameters
    ----------
    solid, fluid

    Attributes
    ----------
    solid : Solid
        A solid model object
    fluids : List[Fluid]
        A collection of 1D fluid model objects
    solid_fsi_dofs, fluid_fsi_dofs : Union[List[ArrayLike], ArrayLike]
        A collection of corresponding DOF arrays for fluid/structure interaction
        on the solid and fluid models, respectively.
        If there are `n` fluid models, then there must be `n` DOF arrays on both
        solid and fluid models.
        Note that while each DOF array
    """

    def __init__(
        self,
        solid: tsmd.Model,
        fluids: Union[List[tfmd.Model], tfmd.Model],
        solid_fsi_dofs: Union[List[ArrayLike], ArrayLike],
        fluid_fsi_dofs: Union[List[ArrayLike], ArrayLike],
    ):
        if isinstance(fluids, list):
            fluids = tuple(fluids)
            fluid_fsi_dofs = tuple(fluid_fsi_dofs)
            solid_fsi_dofs = tuple(solid_fsi_dofs)
        elif isinstance(fluids, tuple):
            pass
        else:
            fluids = (fluids,)
            fluid_fsi_dofs = (fluid_fsi_dofs,)
            solid_fsi_dofs = (solid_fsi_dofs,)
        self.solid = solid
        self.fluids = fluids

        ## Specify state, controls, and properties
        fluid_state0 = bv.concatenate_with_prefix(
            [fluid.state0 for fluid in fluids], 'fluid'
        )
        fluid_state1 = bv.concatenate_with_prefix(
            [fluid.state1 for fluid in fluids], 'fluid'
        )
        self.state0 = bv.concatenate([self.solid.state0, fluid_state0])
        self.state1 = bv.concatenate([self.solid.state1, fluid_state1])
        self.fluid_state0 = fluid_state0
        self.fluid_state1 = fluid_state1

        # The control is just the subglottal and supraglottal pressures
        self.control = bv.concatenate_with_prefix(
            [fluid.control[1:] for fluid in self.fluids], 'fluid'
        )

        _self_properties = bv.BlockVector((np.array([1.0]),), (1,), (('ymid',),))
        fluid_props = bv.concatenate_with_prefix(
            [fluid.prop for fluid in self.fluids], 'fluid'
        )
        self.prop = bv.concatenate([self.solid.prop, fluid_props, _self_properties])

        ## FSI related stuff
        (
            fsimaps,
            solid_area,
            dflcontrol_dslstate,
            dslcontrol_dflstate,
            dflcontrol_dslprops,
        ) = make_coupling_stuff(solid, fluids, solid_fsi_dofs, fluid_fsi_dofs)
        self._fsimaps = fsimaps
        self._solid_area = solid_area
        self._dflcontrol_dslstate = dflcontrol_dslstate
        self._dslcontrol_dflstate = dslcontrol_dflstate
        self._dflcontrol_dslprops = dflcontrol_dslprops

        # Make null `BlockMatrix`s relating fluid/solid states
        mats = [
            [subops.zero_mat(slvec.size, flvec.size) for flvec in fluid_state0.blocks]
            for slvec in self.solid.state0.blocks
        ]
        self._null_dslstate_dflstate = bm.BlockMatrix(mats)
        mats = [
            [
                subops.zero_mat(flvec.size, slvec.size)
                for slvec in self.solid.state0.blocks
            ]
            for flvec in fluid_state0.blocks
        ]
        self._null_dflstate_dslstate = bm.BlockMatrix(mats)

    @property
    def fsimaps(self):
        """
        Return the FSI map object
        """
        return self._fsimaps

    # These have to be defined to exchange data between fluid/solid domains
    # Explicit/implicit coupling methods may define these in different ways to
    # achieve the desired coupling
    def _set_ini_solid_state(self, uva0: bv.BlockVector):
        raise NotImplementedError("Subclasses must implement this method")

    def _set_fin_solid_state(self, uva1: bv.BlockVector):
        raise NotImplementedError("Subclasses must implement this method")

    def _set_ini_fluid_state(self, qp0: bv.BlockVector):
        raise NotImplementedError("Subclasses must implement this method")

    def _set_fin_fluid_state(self, qp1: bv.BlockVector):
        raise NotImplementedError("Subclasses must implement this method")

    ## Parameter setting methods
    @property
    def dt(self):
        return self.solid.dt

    @dt.setter
    def dt(self, value):
        self.solid.dt = value
        for fluid in self.fluids:
            fluid.dt = value

    def set_ini_state(self, state):
        sl_state, fl_state = bv.chunk(
            state, (self.solid.state0.size, self.fluid_state0.size)
        )

        self._set_ini_solid_state(sl_state)
        self._set_ini_fluid_state(fl_state)

    def set_fin_state(self, state):
        sl_state, fl_state = bv.chunk(
            state, (self.solid.state1.size, self.fluid_state1.size)
        )

        self._set_fin_solid_state(sl_state)
        self._set_fin_fluid_state(fl_state)

    def set_control(self, control):
        self.control[:] = control

        chunk_sizes = len(self.fluids) * [2]
        control_chunks = bv.chunk(self.control, chunk_sizes)

        for n, control in enumerate(control_chunks):
            for key, value in control.sub_items():
                key_postfix = '.'.join(key.split('.')[1:])
                self.fluids[n].control[key_postfix][:] = value

    def set_prop(self, prop):
        self.prop[:] = prop

        # The final `+ [1]` accounts for the 'ymid' property
        chunk_sizes = [model.prop.size for model in (self.solid,) + self.fluids] + [1]
        prop_chunks = bv.chunk(self.prop, chunk_sizes)[:-1]
        prop_setters = [self.solid.set_prop] + [fluid.set_prop for fluid in self.fluids]

        for set_prop, prop in zip(prop_setters, prop_chunks):
            set_prop(prop)

        # self.solid.set_prop(prop[:self.solid.prop.size])
        # self.fluid.set_prop(prop[self.solid.prop.size:-1])


# TODO: The `assem_*` type methods are incomplete as I haven't had to use them
class ExplicitFSIModel(BaseTransientFSIModel):

    # The functions below are set to represent an explicit (staggered) coupling
    # This means for a timestep:
    # - The fluid loading on the solid domain is based on the fluid loads
    #   at the beginning of the time step. i.e. the fluid loading is
    #   constant/known
    #     - Setting the initial fluid state changes the solid control
    # - The geometry of the fluid domain is based on the deformation of
    #   the solid in the current time step. i.e. the geometry of the fluid
    #   changes based on the computed deformation for the current time step
    #    - Setting the final solid state updates the fluid control
    def _set_ini_solid_state(self, uva0):
        self.solid.set_ini_state(uva0)

    def _set_fin_solid_state(self, uva1):
        self.solid.set_fin_state(uva1)

        # For explicit coupling, the final fluid area corresponds to the final solid deformation
        ndim = self.solid.residual.mesh().topology().dim()
        self._solid_area[:] = 2 * (
            self.prop['ymid'][0]
            - (self.solid.XREF + self.solid.state1.sub['u'])[1::ndim]
        )
        for n, (fluid, fsimap) in enumerate(zip(self.fluids, self.fsimaps)):
            fl_control = fluid.control.copy()
            fsimap.map_solid_to_fluid(self._solid_area, fl_control.sub['area'][:])
            fluid.set_control(fl_control)

    def _set_ini_fluid_state(self, qp0):
        # For explicit coupling, the final/current solid pressure corresponds to
        # the initial/previous fluid pressure
        sl_control = self.solid.control.copy()
        sl_control['p'] = 0
        qp0_parts = bv.chunk(qp0, tuple(fluid.state0.size for fluid in self.fluids))
        for fluid, fsimap, qp0_part in zip(self.fluids, self.fsimaps, qp0_parts):
            fluid.set_ini_state(qp0_part)
            fsimap.map_fluid_to_solid(qp0_part[1], sl_control.sub['p'])
        self.solid.set_control(sl_control)

    def _set_fin_fluid_state(self, qp1):
        """Set the final fluid state"""
        qp1_parts = bv.chunk(qp1, tuple(fluid.state1.size for fluid in self.fluids))
        for fluid, qp1_part in zip(self.fluids, qp1_parts):
            fluid.set_fin_state(qp1_part)

    ## Residual and derivative assembly functions
    def assem_res(self):
        """
        Return the residual
        """
        res_sl = self.solid.assem_res()
        res_fl = self.fluid.assem_res()
        return bv.concatenate((res_sl, res_fl))

    def assem_dres_dstate0(self):
        # TODO: Make this correct
        drsl_dxsl = self.solid.assem_dres_dstate0()
        drsl_dxfl = linalg.mult_mat_mat(
            self.solid.assem_dres_dcontrol(), self._dslcontrol_dflstate
        )
        drfl_dxfl = self.fluid.assem_dres_dstate0()
        drfl_dxsl = linalg.mult_mat_mat(
            self.fluid.assem_dres_dcontrol(), self._dflcontrol_dslstate
        )
        bmats = [[drsl_dxsl, drsl_dxfl], [drfl_dxsl, drfl_dxfl]]
        return bm.concatenate(bmats)

    def assem_dres_dstate1(self):
        drsl_dxsl = self.solid.assem_dres_dstate0()
        drsl_dxfl = linalg.mult_mat_mat(
            self.solid.assem_dres_dcontrol(), self._dslcontrol_dflstate
        )
        drfl_dxfl = self.fluid.assem_dres_dstate0()
        drfl_dxsl = linalg.mult_mat_mat(
            self.fluid.assem_dres_dcontrol(), self._dflcontrol_dslstate
        )
        bmats = [[drsl_dxsl, drsl_dxfl], [drfl_dxsl, drfl_dxfl]]
        return bm.concatenate(bmats)

    # Forward solver methods
    def solve_state1(self, ini_state, options=None):
        """
        Solve for the final state given an initial guess
        """
        # Set the initial guess for the final state
        self.set_fin_state(ini_state)

        uva1, solid_info = self.solid.solve_state1(ini_state[:3], options)

        self._set_fin_solid_state(uva1)

        fluids_solve_res = [
            fluid.solve_state1(ini_state[3:], options) for fluid in self.fluids
        ]
        qp1s = [solve_res[0] for solve_res in fluids_solve_res]
        fluids_solve_info = [solve_res[1] for solve_res in fluids_solve_res]

        step_info = solid_info
        step_info.update(
            {
                f'fluid{ii}_info': fluid_info
                for ii, fluid_info in enumerate(fluids_solve_info)
            }
        )

        return bv.concatenate([uva1] + qp1s, labels=self.state1.labels), step_info

    def solve_dres_dstate1(self, b):
        """
        Solve, dF/du x = f
        """
        x = self.state0.copy()

        x[:3] = self.solid.solve_dres_dstate1(b[:3])

        dq_du, dp_du = self.fluid.solve_dqp1_du1_solid(self, adjoint=False)
        dfq2_du2 = 0.0 - dq_du
        dfp2_du2 = 0.0 - dp_du

        x['q'][:] = b['q'] - dfq2_du2.inner(x['u'])
        x['p'][:] = b['p'] - dfp2_du2 * x['u'].vec()
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
        b = self.state0.copy()
        b_qp, b_uva = b[3:], b[:3]

        # This performs the fluid part of the solve
        b_qp[:] = x[3:]

        # This is the solid part of the
        bp_vec = dfp2_du2.getVecRight()
        bp_vec[:] = b_qp['p']
        rhs = x[:3].copy()
        rhs['u'] -= dfq2_du2 * b_qp['q'] + dfn.PETScVector(dfp2_du2 * bp_vec)
        b_uva[:] = self.solid.solve_dres_dstate1_adj(rhs)

        return b


class ImplicitFSIModel(BaseTransientFSIModel):
    ## These must be defined to properly exchange the forcing data between the solid and domains
    def _set_ini_fluid_state(self, qp0):
        qp0_parts = bv.chunk(qp0, tuple(fluid.state0.size for fluid in self.fluids))
        for fluid, qp0_part in zip(self.fluids, qp0_parts):
            fluid.set_ini_state(qp0_part)

    def _set_fin_fluid_state(self, qp1):
        sl_control = self.solid.control.copy()
        sl_control['p'] = 0
        qp1_parts = bv.chunk(qp1, tuple(fluid.state1.size for fluid in self.fluids))
        for fluid, fsimap, qp1_part in zip(self.fluids, self.fsimaps, qp1_parts):
            fluid.set_fin_state(qp1_part)
            fsimap.map_fluid_to_solid(qp1_part[1], sl_control.sub['p'])
        self.solid.set_control(sl_control)

    def _set_ini_solid_state(self, uva0):
        """Set the initial solid state"""
        self.solid.set_ini_state(uva0)

    def _set_fin_solid_state(self, uva1):
        self.solid.set_fin_state(uva1)

        # For both implicit/explicit coupling, the final fluid area corresponds to the final solid deformation
        ndim = self.solid.residual.mesh().topology().dim()
        self._solid_area[:] = 2 * (
            self.prop['ymid'][0] - (self.solid.XREF + self.solid.state1['u'])[1::ndim]
        )
        for n, (fluid, fsimap) in enumerate(zip(self.fluids, self.fsimaps)):
            fl_control = fluid.control
            fsimap.map_solid_to_fluid(self._solid_area, fl_control['area'][:])
            fluid.set_control(fl_control)

    ## Forward solver methods
    def assem_res(self):
        """
        Return the residual vector, F
        """
        res_sl = self.solid.assem_res()
        res_fl = self.fluid.assem_res()
        return bv.concatenate(res_sl, res_fl)

    def solve_state1(self, ini_state, options=None):
        """
        Solve for the final state given an initial guess

        This uses a fixed-point iteration where the solid is solved, then the fluid and so-on.
        """
        if options is None:
            options = DEFAULT_NEWTON_SOLVER_PRM

        def iterative_subproblem(x):
            uva1_0 = x[:3]
            qp1_0 = x[3:]

            def solve(res):
                # Solve the solid with the previous iteration's fluid pressures
                uva1, _ = self.solid.solve_state1(uva1_0, options)

                # Compute new fluid pressures for the updated solid position
                self._set_fin_solid_state(uva1)
                qp1, fluid_info = self.fluid.solve_state1(qp1_0)
                self._set_fin_fluid_state(qp1_0)
                return bv.concatenate((uva1, qp1))

            def assem_res():
                return self.assem_res()

            return assem_res, solve

        return iterative_solve(
            ini_state, iterative_subproblem, norm=bv.norm, params=options
        )

    def solve_dres_dstate1(self, b):
        """
        Solve, dF/du x = f
        """
        dt = self.solid.dt
        x = self.state0.copy()

        solid = self.solid

        dfu1_du1 = self.solid.cached_form_assemblers['form.bi.df1_du1'].assemble()
        dfv2_du2 = 0 - newmark.newmark_v_du1(dt)
        dfa2_du2 = 0 - newmark.newmark_a_du1(dt)

        dq_du, dp_du = self.fluid.solve_dqp1_du1_solid(self, adjoint=False)
        dfq2_du2 = 0 - dq_du
        dfp2_du2 = 0 - dp_du

        self.solid.forms['bc.dirichlet'].apply(dfu1_du1)
        dfn.solve(dfu1_du1, x['u'], b['u'], 'petsc')
        x['v'][:] = b['v'] - dfv2_du2 * x['u']
        x['a'][:] = b['a'] - dfa2_du2 * x['u']

        x['q'][:] = b['q'] - dfq2_du2.inner(x['u'])
        x['p'][:] = b['p'] - dfn.PETScVector(dfp2_du2 * x['u'].vec())
        return x

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
        dfu2_dp2 = self.solid.cached_form_assemblers['form.bi.df1_dp1_adj'].assemble()

        # map dfu2_dp2 to have p on the fluid domain
        solid_dofs, fluid_dofs = self.get_fsi_scalar_dofs()
        dfu2_dp2 = dfn.as_backend_type(dfu2_dp2).mat()
        dfu2_dp2 = linalg.reorder_mat_rows(
            dfu2_dp2, solid_dofs, fluid_dofs, self.fluid.state1['p'].size
        )

        dq_du, dp_du = self.fluid.solve_dqp1_du1_solid(self, adjoint=True)
        dfq2_du2 = 0 - dq_du
        dfp2_du2 = 0 - dp_du

        ## Do the linear algebra that solves for the adjoint states
        adj_uva = self.solid.state0.copy()
        adj_qp = self.fluid.state0.copy()

        adj_u_rhs, adj_v_rhs, adj_a_rhs, adj_q_rhs, adj_p_rhs = b

        # adjoint states for v, a, and q are explicit so we can solve for them
        self.solid.forms['bc.dirichlet'].apply(adj_v_rhs)
        adj_uva['v'][:] = adj_v_rhs

        self.solid.forms['bc.dirichlet'].apply(adj_a_rhs)
        adj_uva['a'][:] = adj_a_rhs

        # TODO: how to apply fluid boundary conditions in a generic way?
        adj_qp['q'][:] = adj_q_rhs

        adj_u_rhs -= (
            dfv2_du2 * adj_uva['v'] + dfa2_du2 * adj_uva['a'] + dfq2_du2 * adj_qp['q']
        )

        bc_dofs = np.array(
            list(self.solid.forms['bc.dirichlet'].get_boundary_values().keys()),
            dtype=np.int32,
        )
        self.solid.forms['bc.dirichlet'].apply(dfu2_du2, adj_u_rhs)
        dfp2_du2.zeroRows(bc_dofs, diag=0.0)
        # self.solid.forms['bc.dirichlet'].zero_columns(dfu2_du2, adj_u_rhs.copy(), diagonal_value=1.0)

        # solve the coupled system for pressure and displacement residuals
        dfu2_du2_mat = dfn.as_backend_type(dfu2_du2).mat()
        blocks = [[dfu2_du2_mat, dfp2_du2], [dfu2_dp2, 1.0]]

        dfup2_dup2 = linalg.form_block_matrix(blocks)
        adj_up, rhs = dfup2_dup2.getVecs()

        # calculate rhs vectors
        rhs[: adj_u_rhs.size()] = adj_u_rhs
        rhs[adj_u_rhs.size() :] = adj_p_rhs

        # Solve the block linear system with LU factorization
        ksp = PETSc.KSP().create()
        ksp.setType(ksp.Type.PREONLY)

        pc = ksp.getPC()
        pc.setType(pc.Type.LU)

        ksp.setOperators(dfup2_dup2)
        ksp.solve(rhs, adj_up)

        adj_uva['u'][:] = adj_up[: adj_u_rhs.size()]
        adj_qp['p'][:] = adj_up[adj_u_rhs.size() :]

        return bv.concatenate([adj_uva, adj_qp])


class FSAIModel(BaseTransientFSIModel):
    """
    Represents a fluid-structure-acoustic interaction system
    """

    def __init__(self, solid, fluid, acoustic, fsi_verts):
        self.solid = solid
        self.fluid = fluid
        self.acoustic = acoustic

        state = bv.concatenate(
            [solid.state0.copy(), fluid.state0.copy(), acoustic.state0.copy()]
        )
        self.state0 = state
        self.state1 = state.copy()

        control = bv.BlockVector((np.array([1.0]),), labels=[('psub',)])
        self.control = control.copy()

        self.prop = bv.concatenate([solid.prop, fluid.prop, acoustic.prop])

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
        fl_state = state[sl_nblock : sl_nblock + fl_nblock]
        ac_state = state[sl_nblock + fl_nblock : sl_nblock + fl_nblock + ac_nblock]

        self._set_ini_solid_state(sl_state)
        self._set_ini_fluid_state(fl_state)
        self.set_ini_acoustic_state(ac_state)

    def set_fin_state(self, state):
        sl_nblock = len(self.solid.state0.size)
        fl_nblock = len(self.fluid.state0.size)
        ac_nblock = len(self.acoustic.state0.size)

        sl_state = state[:sl_nblock]
        fl_state = state[sl_nblock : sl_nblock + fl_nblock]
        ac_state = state[sl_nblock + fl_nblock : sl_nblock + fl_nblock + ac_nblock]

        self._set_fin_solid_state(sl_state)
        self._set_fin_fluid_state(fl_state)
        self.set_fin_acoustic_state(ac_state)

    def set_control(self, control):
        fl_control = self.fluid.control.copy()
        fl_control['psub'][:] = control['psub']
        self.fluid.set_control(fl_control)

    def set_prop(self, prop):
        sl_nblock = len(self.solid.prop.size)
        fl_nblock = len(self.fluid.prop.size)
        ac_nblock = len(self.acoustic.prop.size)

        sl_props = prop[:sl_nblock]
        fl_props = prop[sl_nblock : sl_nblock + fl_nblock]
        ac_props = prop[sl_nblock + fl_nblock : sl_nblock + fl_nblock + ac_nblock]

        self.solid.set_prop(sl_props)
        self.fluid.set_prop(fl_props)
        self.acoustic.set_prop(ac_props)

    ## Coupling methods
    def _set_ini_solid_state(self, sl_state0):
        self.solid.set_ini_state(sl_state0)

    def _set_ini_fluid_state(self, fl_state0):
        self.fluid.set_ini_state(fl_state0)

        # for explicit coupling
        sl_control = self.solid.control.copy()
        fsi_sdofs = self.solid.vert_to_sdof[self.fsi_verts].copy()
        sl_control = fl_state_to_sl_control(fl_state0, sl_control, fsi_sdofs)
        self.solid.set_control(sl_control)

    def set_ini_acoustic_state(self, ac_state0):
        self.acoustic.set_ini_state(ac_state0)

    def _set_fin_solid_state(self, sl_state1):
        self.solid.set_fin_state(sl_state1)

        fl_control = self.fluid.control.copy()

        fsi_ref_config = self.solid.mesh.coordinates()[self.fsi_verts].reshape(-1)
        fsi_vdofs = (
            self.solid.vert_to_vdof.reshape(-1, 2)[self.fsi_verts].reshape(-1).copy()
        )
        fl_control = sl_state_to_fl_control(
            sl_state1, fl_control, fsi_ref_config, fsi_vdofs
        )

        self.fluid.set_control(fl_control)

    def _set_fin_fluid_state(self, fl_state1):
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
        ret[:] = 0.0
        return ret

    def get_state_vec(self):
        ret = self.state0.copy()
        ret[:] = 0.0
        return ret

    def get_properties_vec(self, set_default=True):
        ret = self.prop.copy()
        if not set_default:
            ret[:] = 0.0
        return ret

    ## Solver methods
    def assem_res(self):
        res_sl = self.solid.res()
        res_fl = self.fluid.res()
        res_ac = self.acoustic.res()
        return bv.concatenate([res_sl, res_fl, res_ac])

    def solve_state1(self, ini_state, newton_solver_prm=None):
        if newton_solver_prm is None:
            newton_solver_prm = DEFAULT_NEWTON_SOLVER_PRM

        ## Solve solid displacements at n
        self.set_fin_state(ini_state)
        ini_slstate = ini_state[:3]
        sl_state1, solver_info = self.solid.solve_state1(ini_slstate, newton_solver_prm)
        self._set_fin_solid_state(sl_state1)

        def make_linearized_flow_residual(qac):
            """
            Represents the coupled linearized subproblem @ qac
            res(qac) = qac - qbern(psup(qac))
            solve_jac(res) = res/(1 - jac(qbern(psup(qac))))
            """
            # Linearize the fluid/acoustic models about `qac`
            ac_control = self.acoustic.control.copy()
            ac_control['qin'][:] = qac['qin']
            self.acoustic.set_control(ac_control)
            ac_state1, _ = self.acoustic.solve_state1()
            self.set_fin_acoustic_state(ac_state1)
            fl_state1, _ = self.fluid.solve_state1(self.fluid.state0)

            dqbern_dpsup = self.fluid.flow_sensitivity(
                *self.fluid.control.vecs, self.fluid.prop
            )[4]
            dpsup_dqac = self.acoustic.z[0]

            def res():
                qbern = fl_state1[0]
                return qac - bv.BlockVector((qbern,), labels=[('qin',)])

            def solve_jac(res):
                dres_dq = 1 - dqbern_dpsup * dpsup_dqac
                return res / dres_dq

            return res, solve_jac

        q, info = tsmd.newton_solve(
            bv.BlockVector((ini_state['q'],), labels=[('qin',)]),
            make_linearized_flow_residual,
        )

        self.acoustic.set_control(q)
        ac_state1, _ = self.acoustic.solve_state1()
        fl_state1, fluid_info = self.fluid.solve_state1(self.fluid.state0)

        step_info = {'fluid_info': fluid_info, **info}
        fin_state = bv.concatenate([sl_state1, fl_state1, ac_state1])

        return fin_state, step_info

    def _form_dflac_dflac(self):
        b = self.state0

        ## Solve the coupled fluid/acoustic system
        # First compute some sensitivities that are needed
        *_, dq_dpsup, dp_dpsup = self.fluid.flow_sensitivity(
            *self.fluid.control.vecs, self.fluid.prop
        )

        # solve the coupled system for pressure and acoustic residuals
        dfq_dq = 1.0
        dfp_dp = 1.0
        dfpinc_dpinc = PETSc.Mat().createAIJ(
            (b['pinc'].size, b['pinc'].size), nnz=b['pinc'].size
        )
        diag = PETSc.Vec().createSeq(b['pinc'].size)
        diag[:] = 1.0
        dfpinc_dpinc.setDiagonal(diag)
        dfpref_dpref = 1.0

        # dfluid / dacoustic
        dfq_dpref = PETSc.Mat().createAIJ((b['q'].size, b['pref'].size), nnz=1)
        dfq_dpsup = -dq_dpsup
        dpsup_dpref = (
            1.0  # Supraglottal pressure is equal to very first reflected pressure
        )
        dfq_dpref.setValue(0, 0, dfq_dpsup * dpsup_dpref)

        dfp_dpref = PETSc.Mat().createAIJ(
            (b['p'].size, b['pref'].size), nnz=b['p'].size
        )
        dfp_dpsup = -dp_dpsup
        dfp_dpref.setValues(
            np.arange(b['p'].size, dtype=np.int32), 0, dfp_dpsup * dpsup_dpref
        )

        # dacoustic / dfluid
        dfpref_dq = PETSc.Mat().createAIJ((b['pref'].size, b['q'].size), nnz=2)
        dcontrol = self.acoustic.control.copy()
        dcontrol[:] = 0.0
        dcontrol['qin'][:] = 1.0
        dfpref_dqin = self.acoustic.apply_dres_dcontrol(dcontrol)['pref'][:2]
        dqin_dq = 1.0
        dfpref_dq.setValues(np.array([0, 1], dtype=np.int32), 0, dfpref_dqin * dqin_dq)

        for mat in (dfq_dpref, dfp_dpref, dfpref_dq):
            mat.assemble()

        blocks = [
            [dfq_dq, 0.0, 0.0, dfq_dpref],
            [0.0, dfp_dp, 0.0, dfp_dpref],
            [0.0, 0.0, dfpinc_dpinc, 0.0],
            [dfpref_dq, 0.0, 0.0, dfpref_dpref],
        ]

        A = linalg.form_block_matrix(blocks)
        return A

    def solve_dres_dstate1(self, b):
        """
        Solve, dF/du x = f
        """
        x = self.state0.copy()
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
        b_flac['q'] -= (dfq2_du2 * x['u']).sum()
        b_flac['p'] -= dfp2_du2 * x['u'].vec()
        rhs[:] = b_flac.to_ndarray()

        ksp = PETSc.KSP().create()
        ksp.setType(ksp.Type.PREONLY)

        pc = ksp.getPC()
        pc.setType(pc.Type.LU)

        ksp.setOperators(A)
        ksp.solve(rhs, adj_z)

        x[3:] = adj_z
        # print(linalg.dot(x[3:], x[3:]))

        return x

    # Adjoint solver methods
    def solve_dres_dstate1_adj(self, b):
        """
        Solve, dF/du^T x = f
        """
        x = self.state0.copy()

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

        x[3:] = adj_z

        ## Assemble sensitivity matrices
        dq_du, dp_du = self.fluid.solve_dqp1_du1_solid(self, adjoint=True)
        dfq2_du2 = 0 - dq_du
        dfp2_du2 = 0 - dp_du

        _adj_p = dfp2_du2.getVecRight()
        _adj_p[:] = x['p']

        b_uva = b[:3].copy()
        b_uva['u'] -= dfq2_du2 * x['q'] + dfn.PETScVector(dfp2_du2 * _adj_p)

        x_uva = x[:3]
        x_uva[:] = self.solid.solve_dres_dstate1_adj(b_uva)

        return x
