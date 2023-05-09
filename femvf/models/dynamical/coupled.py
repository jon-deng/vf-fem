"""
Contains class definitions for coupled dynamical systems models
"""

import dolfin as dfn
import numpy as np
from petsc4py import PETSc as PETSc

from blockarray import blockmat as bm, blockvec as bv, subops, linalg as bla

from .base import BaseDynamicalModel
from .fluid import BaseDynamical1DFluid
from .solid import DynamicalSolid, LinearizedSolidDynamicalSystem

from ..fsi import FSIMap

# pylint: disable=missing-function-docstring

class BaseDynamicalFSIModel(BaseDynamicalModel):
    """
    Class representing a fluid-solid coupled dynamical system
    """

    def __init__(self,
        solid_model: DynamicalSolid,
        fluid_model: BaseDynamical1DFluid,
        solid_fsi_dofs, fluid_fsi_dofs):
        self.solid = solid_model
        self.fluid = fluid_model
        self.models = (self.solid, self.fluid)

        self.state = bv.concatenate_vec(
            [bv.convert_subtype_to_petsc(model.state) for model in self.models]
        )
        self.statet = bv.concatenate_vec(
            [bv.convert_subtype_to_petsc(model.statet) for model in self.models]
        )

        is_linearized = isinstance(solid_model, LinearizedSolidDynamicalSystem)
        if is_linearized:
            self.dstate = bv.concatenate_vec(
                [bv.convert_subtype_to_petsc(model.dstate) for model in self.models]
            )
            self.dstatet = bv.concatenate_vec(
                [bv.convert_subtype_to_petsc(model.dstatet) for model in self.models]
            )

        # This selects only psub and psup from the fluid control
        self.control = bv.convert_subtype_to_petsc(self.fluid.control[1:])

        _ymid_props = bv.BlockVector([np.array([1.0])], labels=[['ymid']])
        self.prop = bv.concatenate_vec(
            [bv.convert_subtype_to_petsc(model.prop) for model in self.models]
            + [bv.convert_subtype_to_petsc(_ymid_props)]
        )

        ## -- FSI --
        # Below here is all extra stuff needed to do the coupling between fluid/solid
        self.solid_area = dfn.Function(self.solid.residual.form['coeff.fsi.p1'].function_space()).vector()
        if is_linearized:
            self.dsolid_area = dfn.Function(self.solid.residual.form['coeff.fsi.p1'].function_space()).vector()
        # have to compute dslarea_du here as sensitivity of solid area wrt displacement function

        self.solid_xref = self.solid.XREF

        # solid and fluid fsi dofs should be created when the two models are created
        self.fsimap = FSIMap(
            self.fluid.state['p'].size, self.solid_area.size(),
            fluid_fsi_dofs, solid_fsi_dofs
        )

        # These are jacobians of the mapping of scalars at the FSI interface from one domain to the
        # other
        self._dsolid_dfluid_scalar = self.fsimap.dsolid_dfluid
        self._dfluid_dsolid_scalar = self.fsimap.dfluid_dsolid

        # The matrix here is d(p)_solid/d(q, p)_fluid
        dslp_dflq = subops.zero_mat(self.solid.control['p'].size, self.fluid.state['q'].size)
        dslp_dflp = self._dsolid_dfluid_scalar
        mats = [[dslp_dflq, dslp_dflp]]
        self.dslcontrol_dflstate = bv.BlockMatrix(mats, labels=(('p',), ('q', 'p')))

        # The matrix here is d(area)_fluid/d(u, v)_solid
        # pylint: disable=no-member
        mats = [
            [subops.zero_mat(nrow, ncol)
            for ncol in self.solid.state.bshape[0]]
            for nrow in self.fluid.control.bshape[0]]
        dslarea_dslu = PETSc.Mat().createAIJ(
            (self.solid_area.size(), self.solid.state['u'].size),
            nnz=2
        )
        for ii in range(dslarea_dslu.size[0]):
            # Each solid area is only sensitive to the y component of u, so that's set here
            # REFINE: can only set sensitivites for relevant DOFS; only DOFS on the surface have an
            # effect
            dslarea_dslu.setValues([ii], [2*ii, 2*ii+1], [0, -2])
        dslarea_dslu.assemble()
        dflarea_dslarea = self._dfluid_dsolid_scalar
        dflarea_dslu = subops.mult_mat_mat(dflarea_dslarea, dslarea_dslu)
        # NOTE: It's hardcoded/assumed that the 'area' control is at index 0 in
        # `self.fluid.control`
        row_flarea = self.fluid.control.labels[0].index('area')
        col_slu = self.solid.state.labels[0].index('u')
        mats[row_flarea][col_slu] = dflarea_dslu
        self.dflcontrol_dslstate = bv.BlockMatrix(
            mats, labels=self.fluid.control.labels+self.solid.state.labels
        )

        # Sensitivity of fluid area wrt, mesh displacement (if present)
        mats = [
            [subops.zero_mat(nrow, ncol) for ncol in self.solid.prop.bshape[0]]
            for nrow in self.fluid.control.bshape[0]
        ]
        if 'umesh' in self.solid.prop:
            dflarea_dumesh = dflarea_dslu.copy()
            row_flarea = self.fluid.control.labels[0].index('area')
            col_slu = self.solid.prop.labels[0].index('umesh')
            mats[row_flarea][col_slu] = dflarea_dumesh
        self.dflcontrol_dslprops = bv.BlockMatrix(
            mats, labels=self.fluid.control.labels+self.solid.prop.labels
        )

        # Make null BlockMats relating fluid/solid states
        mats = [
            [subops.zero_mat(slvec.size, flvec.size) for flvec in self.fluid.state.blocks]
            for slvec in self.solid.state.blocks]
        self.null_dslstate_dflstate = bv.BlockMatrix(mats)
        mats = [
            [subops.zero_mat(flvec.size, slvec.size) for slvec in self.solid.state.blocks]
            for flvec in self.fluid.state.blocks]
        self.null_dflstate_dslstate = bv.BlockMatrix(mats)

    def set_state(self, state):
        self.state[:] = state
        block_sizes = [model.state.size for model in self.models]
        sub_states = bv.split_bvec(state, block_sizes)
        for model, sub_state in zip(self.models, sub_states):
            model.set_state(sub_state)

        self._transfer_solid_to_fluid()

    def _transfer_solid_to_fluid(self):
        """
        Update fluid controls from the solid state
        """
        ## The below are needed to communicate FSI interactions
        # Set solid_area
        self.solid_area[:] = 2*(self.prop['ymid'][0] - (self.solid.XREF + self.solid.state.sub['u'])[1::2])

        # map solid_area to fluid area
        fluid_control = self.fluid.control.copy()
        self.fsimap.map_solid_to_fluid(self.solid_area, fluid_control['area'])
        self.fluid.set_control(fluid_control)

        # map fluid pressure to solid pressure
        solid_control = self.solid.control.copy()
        self.fsimap.map_fluid_to_solid(self.fluid.state['p'], solid_control['p'])
        self.solid.set_control(solid_control)

    def set_dstate(self, dstate):
        self.dstate[:] = dstate
        block_sizes = [model.dstate.size for model in self.models]
        sub_states = bv.split_bvec(dstate, block_sizes)
        for model, sub_state in zip(self.models, sub_states):
            model.set_dstate(sub_state)

        ## The below are needed to communicate FSI interactions
        # map linearized state to linearized solid area
        self.dsolid_area[:] = -2*(self.dstate['u'][1::2])

        # map linearized solid area to fluid area
        dfluid_control = self.fluid.dcontrol.copy()
        dfluid_control['area'][:] = subops.mult_mat_vec(
            self._dfluid_dsolid_scalar,
            subops.convert_vec_to_petsc(self.dsolid_area))
        self.fluid.set_dcontrol(dfluid_control)

        # map linearized fluid pressure to solid pressure
        dsolid_control = self.solid.control.copy()
        dsolid_control['p'][:] = subops.mult_mat_vec(
            self._dsolid_dfluid_scalar,
            subops.convert_vec_to_petsc(self.fluid.dstate['p']))
        self.solid.set_dcontrol(dsolid_control)

    # Since the fluid has no time dependence there should be no need to set FSI interactions here
    # for the specialized 1D Bernoulli model so I've left it empty for now
    def set_statet(self, statet):
        self.statet[:] = statet
        block_sizes = [model.statet.size for model in self.models]
        sub_states = bv.split_bvec(statet, block_sizes)
        for model, sub_state in zip(self.models, sub_states):
            model.set_statet(sub_state)

    def set_dstatet(self, dstatet):
        self.dstatet[:] = dstatet
        block_sizes = [model.dstatet.size for model in self.models]
        sub_states = bv.split_bvec(dstatet, block_sizes)
        for model, sub_state in zip(self.models, sub_states):
            model.set_dstatet(sub_state)

    def set_control(self, control):
        self.control[:] = control
        fl_control = self.fluid.control.copy()
        fl_control[1:][:] = control # Set psub/psup of the coupled model to the fluid model control
        self.fluid.set_control(fl_control)

    def set_prop(self, prop):
        self.prop[:] = prop
        # You have to add a final size of 1 block to account for the final ymid
        # property block
        block_sizes = [model.prop.size for model in self.models] + [1]
        sub_props = bv.split_bvec(prop, block_sizes)
        for model, sub_prop in zip(self.models, sub_props):
            model.set_prop(sub_prop)

        # NOTE: You have to update the fluid control on a property due to shape
        # changes
        self._transfer_solid_to_fluid()

    def assem_res(self):
        return bv.concatenate_vec([bv.convert_subtype_to_petsc(model.assem_res()) for model in self.models])

    def assem_dres_dstate(self):
        dslres_dslx = bm.convert_subtype_to_petsc(self.models[0].assem_dres_dstate())
        dslres_dflx = bla.mult_mat_mat(
            bm.convert_subtype_to_petsc(self.models[0].assem_dres_dcontrol()),
            self.dslcontrol_dflstate)

        dflres_dflx = bm.convert_subtype_to_petsc(self.models[1].assem_dres_dstate())
        dflres_dslx = bla.mult_mat_mat(
            bm.convert_subtype_to_petsc(self.models[1].assem_dres_dcontrol()),
            self.dflcontrol_dslstate)
        bmats = [
            [dslres_dslx, dslres_dflx],
            [dflres_dslx, dflres_dflx]]
        return bm.concatenate_mat(bmats)

    def assem_dres_dstatet(self):
        # Because the fluid models is quasi-steady, there are no time varying FSI quantities
        # As a result, the off-diagonal block terms here are just zero
        dslres_dslx = bm.convert_subtype_to_petsc(self.models[0].assem_dres_dstatet())
        # dfsolid_dxfluid = self.models[0].assem_dres_dcontrolt() * self.dslcontrolt_dflstatet
        dslres_dflx = bm.convert_subtype_to_petsc(self.null_dslstate_dflstate)

        dflres_dflx = bm.convert_subtype_to_petsc(self.models[1].assem_dres_dstatet())
        # dffluid_dxsolid = self.models[1].assem_dres_dcontrolt() * self.dflcontrolt_dslstatet
        dflres_dslx = bm.convert_subtype_to_petsc(self.null_dflstate_dslstate)
        bmats = [
            [dslres_dslx, dslres_dflx],
            [dflres_dslx, dflres_dflx]]
        return bm.concatenate_mat(
            bmats, labels=(self.state.labels[0], self.state.labels[0]))

    def assem_dres_dprop(self):

        ## Solid residual sensitivities
        dslres_dslprops = bm.convert_subtype_to_petsc(self.solid.assem_dres_dprop())

        submats = [
            subops.zero_mat(slsubvec.size, propsubvec.size)
            for slsubvec in self.solid.state
            for propsubvec in self.fluid.prop
        ]
        dslres_dflprops = bm.BlockMatrix(
            submats,
            shape=self.solid.state.f_shape+self.fluid.prop.f_shape,
            labels=self.solid.state.labels+self.fluid.prop.labels
        )

        submats = [
            subops.zero_mat(slsubvec.size, self.prop['ymid'].size)
            for slsubvec in self.solid.state
        ]
        dslres_dymid = bm.BlockMatrix(
            submats,
            shape=self.solid.state.shape+(1,),
            labels=self.solid.state.labels+(('ymid',),)
        )

        ## Fluid residual sensitivities
        submats = [
            subops.zero_mat(flsubvec.size, propsubvec.size)
            for flsubvec in self.fluid.state
            for propsubvec in self.solid.prop
        ]
        dflres_dslprops = bla.mult_mat_mat(
            bm.convert_subtype_to_petsc(self.fluid.assem_dres_dcontrol()),
            self.dflcontrol_dslprops
        )

        dflres_dflprops = bm.convert_subtype_to_petsc(
            self.fluid.assem_dres_dprop()
        )

        submats = [
            subops.zero_mat(flsubvec.size, self.prop['ymid'].size)
            for flsubvec in self.fluid.state
        ]
        dflres_dymid = bm.BlockMatrix(
            submats,
            shape=self.fluid.state.f_shape+(1,),
            labels=self.fluid.state.labels+(('ymid',),)
        )

        bmats = [
            [dslres_dslprops, dslres_dflprops, dslres_dymid],
            [dflres_dslprops, dflres_dflprops, dflres_dymid]
        ]
        return bm.concatenate_mat(bmats)

    def assem_dres_dcontrol(self):
        _mats = [[subops.zero_mat(m, n) for n in self.control.bshape[0]] for m in self.solid.state.bshape[0]]
        dslres_dg = bv.BlockMatrix(_mats, labels=self.solid.state.labels+self.control.labels)

        dflres_dflg = bm.convert_subtype_to_petsc(self.fluid.assem_dres_dcontrol())
        _mats = [[row[kk] for kk in range(1, dflres_dflg.shape[1])] for row in dflres_dflg]
        dflres_dg =  bv.BlockMatrix(_mats, labels=self.fluid.state.labels+self.control.labels)
        return bm.concatenate_mat([[dslres_dg], [dflres_dg]])

    # TODO: Need to implement for optimization strategies
    # def assem_dres_dprops(self):
    #     dfsolid_dxsolid = self.models[0].assem_dres_dprops()
    #     dfsolid_dxfluid =

    #     dffluid_dxfluid = self.models[1].assem_dres_dprops()
    #     dffluid_dxsolid =
    #     bmats = [
    #         [dfsolid_dxsolid, dfsolid_dxfluid],
    #         [dffluid_dxsolid, dffluid_dxfluid]]
    #     return bm.concatenate_mat(bmats)
