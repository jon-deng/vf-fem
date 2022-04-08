"""
Contains class definitions for coupled dynamical systems models
"""

import dolfin as dfn
import numpy as np
from petsc4py import PETSc as PETSc

import blocktensor.linalg as bla
from blocktensor import subops as gops
from blocktensor.mat import convert_bmat_to_petsc
from blocktensor.vec import convert_bvec_to_petsc, split_bvec, BlockVector

from .base import DynamicalSystem
from .fluid import BaseFluid1DDynamicalSystem
from .solid import BaseSolidDynamicalSystem

# pylint: disable=missing-function-docstring

class FSIDynamicalSystem(DynamicalSystem):
    """
    Class representing a fluid-solid coupled dynamical system
    """

    def __init__(self,
        solid_model: BaseSolidDynamicalSystem,
        fluid_model: BaseFluid1DDynamicalSystem,
        solid_fsi_dofs, fluid_fsi_dofs):
        self.solid = solid_model
        self.fluid = fluid_model
        self.models = (self.solid, self.fluid)

        self.state = bla.concatenate_vec(
            [convert_bvec_to_petsc(model.state) for model in self.models])
        self.statet = bla.concatenate_vec(
            [convert_bvec_to_petsc(model.statet) for model in self.models])

        self.dstate = bla.concatenate_vec(
            [convert_bvec_to_petsc(model.dstate) for model in self.models])
        self.dstatet = bla.concatenate_vec(
            [convert_bvec_to_petsc(model.dstatet) for model in self.models])

        # This selects only psub and psup from the fluid control
        self.control = convert_bvec_to_petsc(self.fluid.control[1:])

        _ymid_props = BlockVector([np.array([1.0])], labels=[['ymid']])
        self.properties = bla.concatenate_vec(
            [convert_bvec_to_petsc(model.properties) for model in self.models]
            + [_ymid_props])

        ## -- FSI --
        # Below here is all extra stuff needed to do the coupling between fluid/solid
        self.solid_area = dfn.Function(self.solid.forms['fspace.scalar']).vector()
        self.dsolid_area = dfn.Function(self.solid.forms['fspace.scalar']).vector()
        # have to compute dslarea_du here as sensitivity of solid area wrt displacement function

        self.solid_xref = self.solid.XREF.vector()

        # solid and fluid fsi dofs should be created when the two models are created
        self.fsimap = FSIMap(
            self.fluid.state['p'].size, self.solid_area.size(), fluid_fsi_dofs, solid_fsi_dofs)

        # These are jacobians of the mapping of scalars at the FSI interface from one domain to the
        # other
        self._dsolid_dfluid_scalar = self.fsimap.dsolid_dfluid
        self._dfluid_dsolid_scalar = self.fsimap.dfluid_dsolid

        # The matrix here is d(p)_solid/d(q, p)_fluid
        dslp_dflq = bla.zero_mat(self.solid.control['p'].size(), self.fluid.state['q'].size)
        dslp_dflp = self._dsolid_dfluid_scalar
        mats = [[dslp_dflq, dslp_dflp]]
        self.dslcontrol_dflstate = bla.BlockMatrix(mats, labels=(('p',), ('q', 'p')))

        # The matrix here is d(area)_fluid/d(u, v)_solid
        # pylint: disable=no-member
        mats = [
            [bla.zero_mat(nrow, ncol)
            for ncol in self.solid.state.bshape[0]]
            for nrow in self.fluid.control.bshape[0]]
        dslarea_dslu = PETSc.Mat().createAIJ([self.solid_area.size(), self.solid.state['u'].size()])
        dslarea_dslu.setUp() # should set preallocation manually in the future
        for ii in range(dslarea_dslu.size[0]):
            # Each solid area is only sensitive to the y component of u, so that's set here
            # REFINE: can only set sensitivites for relevant DOFS; only DOFS on the surface have an
            # effect
            dslarea_dslu.setValues([ii], [2*ii, 2*ii+1], [0, -2])
        dslarea_dslu.assemble()
        dflarea_dslarea = self._dfluid_dsolid_scalar
        dflarea_dslu = gops.mult_mat_mat(dflarea_dslarea, dslarea_dslu)
        mats[0][0] = dflarea_dslu
        self.dflcontrol_dslstate = bla.BlockMatrix(mats, labels=(self.fluid.control.keys, self.solid.state.keys))

        # Make null BlockMats relating fluid/solid states
        mats = [
            [bla.zero_mat(slvec.size(), flvec.size) for flvec in self.fluid.state.vecs]
            for slvec in self.solid.state.vecs]
        self.null_dslstate_dflstate = bla.BlockMatrix(mats)
        mats = [
            [bla.zero_mat(flvec.size, slvec.size()) for slvec in self.solid.state.vecs]
            for flvec in self.fluid.state.vecs]
        self.null_dflstate_dslstate = bla.BlockMatrix(mats)

    def set_state(self, state):
        self.state[:] = state
        block_sizes = [model.state.size for model in self.models]
        sub_states = split_bvec(state, block_sizes)
        for model, sub_state in zip(self.models, sub_states):
            model.set_state(sub_state)

        ## The below are needed to communicate FSI interactions
        # Set solid_area
        self.solid_area[:] = 2*(self.properties['ymid'][0] - (self.solid_xref + self.solid.state['u'])[1::2])

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
        sub_states = split_bvec(dstate, block_sizes)
        for model, sub_state in zip(self.models, sub_states):
            model.set_dstate(sub_state)

        ## The below are needed to communicate FSI interactions
        # map linearized state to linearized solid area
        self.dsolid_area[:] = -2*(self.dstate['u'][1::2])

        # map linearized solid area to fluid area
        dfluid_control = self.fluid.dcontrol.copy()
        dfluid_control['area'][:] = gops.mult_mat_vec(
            self._dfluid_dsolid_scalar,
            gops.convert_vec_to_petsc(self.dsolid_area))
        self.fluid.set_dcontrol(dfluid_control)

        # map linearized fluid pressure to solid pressure
        dsolid_control = self.solid.control.copy()
        dsolid_control['p'][:] = gops.mult_mat_vec(
            self._dsolid_dfluid_scalar,
            gops.convert_vec_to_petsc(self.fluid.dstate['p']))
        self.solid.set_dcontrol(dsolid_control)

    # Since the fluid has no time dependence there should be no need to set FSI interactions here
    # for the specialized 1D Bernoulli model so I've left it empty for now
    def set_statet(self, statet):
        self.statet[:] = statet
        block_sizes = [model.statet.size for model in self.models]
        sub_states = split_bvec(statet, block_sizes)
        for model, sub_state in zip(self.models, sub_states):
            model.set_statet(sub_state)

    def set_dstatet(self, dstatet):
        self.dstatet[:] = dstatet
        block_sizes = [model.dstatet.size for model in self.models]
        sub_states = split_bvec(dstatet, block_sizes)
        for model, sub_state in zip(self.models, sub_states):
            model.set_dstatet(sub_state)

    def set_control(self, control):
        self.control[:] = control
        fl_control = self.fluid.control.copy()
        fl_control[1:][:] = control # Set psub/psup of the coupled model to the fluid model control
        self.fluid.set_control(fl_control)

    def set_properties(self, props):
        self.properties[:] = props
        # You have to add a final size of 1 block to account for the final ymid
        # property block
        block_sizes = [model.properties.size for model in self.models] + [1]
        sub_props = split_bvec(props, block_sizes)
        for model, sub_prop in zip(self.models, sub_props):
            model.set_properties(sub_prop)

    def assem_res(self):
        return bla.concatenate_vec([convert_bvec_to_petsc(model.assem_res()) for model in self.models])

    def assem_dres_dstate(self):
        dslres_dslx = convert_bmat_to_petsc(self.models[0].assem_dres_dstate())
        dslres_dflx = bla.mult_mat_mat(
            convert_bmat_to_petsc(self.models[0].assem_dres_dcontrol()),
            self.dslcontrol_dflstate)

        dflres_dflx = convert_bmat_to_petsc(self.models[1].assem_dres_dstate())
        dflres_dslx = bla.mult_mat_mat(
            convert_bmat_to_petsc(self.models[1].assem_dres_dcontrol()),
            self.dflcontrol_dslstate)
        bmats = [
            [dslres_dslx, dslres_dflx],
            [dflres_dslx, dflres_dflx]]
        return bla.concatenate_mat(bmats)

    def assem_dres_dstatet(self):
        # Because the fluid models is quasi-steady, there are no time varying FSI quantities
        # As a result, the off-diagonal block terms here are just zero
        dslres_dslx = convert_bmat_to_petsc(self.models[0].assem_dres_dstatet())
        # dfsolid_dxfluid = self.models[0].assem_dres_dcontrolt() * self.dslcontrolt_dflstatet
        dslres_dflx = convert_bmat_to_petsc(self.null_dslstate_dflstate)

        dflres_dflx = convert_bmat_to_petsc(self.models[1].assem_dres_dstatet())
        # dffluid_dxsolid = self.models[1].assem_dres_dcontrolt() * self.dflcontrolt_dslstatet
        dflres_dslx = convert_bmat_to_petsc(self.null_dflstate_dslstate)
        bmats = [
            [dslres_dslx, dslres_dflx],
            [dflres_dslx, dflres_dflx]]
        return bla.concatenate_mat(
            bmats, labels=(self.state.labels[0], self.state.labels[0]))

    def assem_dres_dprops(self):
        raise NotImplementedError("Not implemented yet")

    def assem_dres_dcontrol(self):
        _mats = [[bla.zero_mat(m, n) for n in self.control.rbshape[0]] for m in self.solid.state.rbshape[0]]
        dslres_dg = bla.BlockMatrix(_mats, labels=(self.solid.state.keys, self.control.keys))

        dflres_dflg = convert_bmat_to_petsc(self.fluid.assem_dres_dcontrol())
        _mats = [[row[kk] for kk in range(1, dflres_dflg.rshape[1])] for row in dflres_dflg]
        dflres_dg =  bla.BlockMatrix(_mats, labels=(self.fluid.state.keys, self.control.keys))
        return bla.concatenate_mat([[dslres_dg], [dflres_dg]])

    # TODO: Need to implement for optimization strategies
    # def assem_dres_dprops(self):
    #     dfsolid_dxsolid = self.models[0].assem_dres_dprops()
    #     dfsolid_dxfluid =

    #     dffluid_dxfluid = self.models[1].assem_dres_dprops()
    #     dffluid_dxsolid =
    #     bmats = [
    #         [dfsolid_dxsolid, dfsolid_dxfluid],
    #         [dffluid_dxsolid, dffluid_dxfluid]]
    #     return bla.concatenate_mat(bmats)
