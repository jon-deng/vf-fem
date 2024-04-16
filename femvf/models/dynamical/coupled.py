"""
Contains class definitions for coupled dynamical systems models
"""

from typing import List, Union
from numpy.typing import NDArray

import dolfin as dfn
import numpy as np
from petsc4py import PETSc as PETSc

from blockarray import blockmat as bm, blockvec as bv, subops, linalg as bla

from .base import BaseDynamicalModel, BaseLinearizedDynamicalModel
from .fluid import Model, LinearizedModel
from .solid import Model, LinearizedModel

from ..fsi import make_coupling_stuff

# pylint: disable=missing-function-docstring

class BaseDynamicalFSIModel(BaseDynamicalModel):
    """
    Class representing a fluid-solid coupled dynamical system
    """

    def __init__(
            self,
            solid: Model,
            fluids: Union[List[Model], Model],
            solid_fsi_dofs: Union[List[NDArray], NDArray],
            fluid_fsi_dofs: Union[List[NDArray], NDArray]
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

        self._models = (self.solid,) +  tuple(self.fluids)

        _to_petsc = bv.convert_subtype_to_petsc
        self._fl_state = bv.concatenate_with_prefix(
            [_to_petsc(fluid.state) for fluid in fluids], 'fluid'
        )
        self.state = bv.concatenate(
            [_to_petsc(solid.state), self._fl_state]
        )

        self._fl_statet = bv.concatenate_with_prefix(
            [_to_petsc(fluid.statet) for fluid in fluids], 'fluid'
        )
        self.statet = bv.concatenate(
            [_to_petsc(solid.statet), self._fl_statet]
        )

        # This selects only psub and psup from the fluid control
        self._fl_control = bv.concatenate_with_prefix(
            [_to_petsc(fluid.control[['psub', 'psup']]) for fluid in fluids],
            prefix='fluid'
        )
        self.control = self._fl_control

        _ymid_props = bv.BlockVector([np.array([1.0])], labels=[['ymid']])
        self._fl_prop = bv.concatenate_with_prefix(
            [_to_petsc(fluid.prop) for fluid in fluids], 'fluid'
        )
        self.prop = bv.concatenate(
            [_to_petsc(solid.prop),
                self._fl_prop,
                bv.convert_subtype_to_petsc(_ymid_props)
            ]
        )

        ## -- FSI --
        fsimaps, solid_area, dflcontrol_dslstate, dslcontrol_dflstate, dflcontrol_dslprops = \
            make_coupling_stuff(solid, fluids, solid_fsi_dofs, fluid_fsi_dofs)
        self._fsimaps = fsimaps
        self._solid_area = solid_area
        self._dflcontrol_dslstate = dflcontrol_dslstate
        self._dslcontrol_dflstate = dslcontrol_dflstate
        self._dflcontrol_dslprops = dflcontrol_dslprops

        # Make null BlockMats relating fluid/solid states
        # Make null `BlockMatrix`s relating fluid/solid states
        mats = [
            [subops.zero_mat(slvec.size, flvec.size) for flvec in self._fl_state.blocks]
            for slvec in self.solid.state.blocks
        ]
        self._null_dslstate_dflstate = bm.BlockMatrix(mats)
        mats = [
            [subops.zero_mat(flvec.size, slvec.size) for slvec in self.solid.state.blocks]
            for flvec in self._fl_state.blocks
        ]
        self._null_dflstate_dslstate = bm.BlockMatrix(mats)

    def set_state(self, state):
        self.state[:] = state
        block_sizes = [model.state.size for model in self._models]
        sub_states = bv.chunk(state, block_sizes)
        for model, sub_state in zip(self._models, sub_states):
            model.set_state(sub_state)

        self._transfer_solid_to_fluid()
        self._transfer_fluid_to_solid()

    def _transfer_solid_to_fluid(self):
        """
        Update fluid controls from the solid state
        """
        ## The below are needed to communicate FSI interactions
        # Set solid_area
        dim = self.solid.residual.mesh().topology().dim()
        self._solid_area[:] = 2*(
            self.prop['ymid'][0]
            - (self.solid.XREF + self.solid.state.sub['u'])[1::dim]
        )

        # map solid_area to fluid area
        for fsimap, fluid in zip(self._fsimaps, self.fluids):
            control = fluid.control.copy()
            fsimap.map_solid_to_fluid(self._solid_area, control['area'])
            fluid.set_control(control)

    def _transfer_fluid_to_solid(self):
        """
        Update solid controls from the fluid state
        """
        # map fluid pressure to solid pressure
        control = self.solid.control.copy()
        for fsimap, fluid in zip(self._fsimaps, self.fluids):
            fsimap.map_fluid_to_solid(fluid.state['p'], control['p'])
        self.solid.set_control(control)

    # Since the fluid has no time dependence there should be no need to set FSI interactions here
    # for the specialized 1D Bernoulli model so I've left it empty for now
    def set_statet(self, statet):
        self.statet[:] = statet
        block_sizes = [model.statet.size for model in self._models]
        sub_states = bv.chunk(statet, block_sizes)
        for model, sub_state in zip(self._models, sub_states):
            model.set_statet(sub_state)

    def set_control(self, control):
        self.control[:] = control
        for n, fluid in enumerate(self.fluids):
            fl_control = fluid.control.copy()
            # Set psub/psup of the coupled model to the fluid model control
            keys = ['psub', 'psup']
            _keys = [f'fluid{n}.{key}' for key in keys]
            fl_control[keys][:] = control[_keys]
            fluid.set_control(fl_control)

    def set_prop(self, prop):
        self.prop[:] = prop
        # You have to add a final size of 1 block to account for the final ymid
        # property block
        block_sizes = [model.prop.size for model in self._models] + [1]
        sub_props = bv.chunk(prop, block_sizes)
        for model, sub_prop in zip(self._models, sub_props):
            model.set_prop(sub_prop)

        # NOTE: You have to update the fluid control on a property due to shape
        # changes
        self._transfer_solid_to_fluid()

    def assem_res(self):
        return bv.concatenate(
            [bv.convert_subtype_to_petsc(model.assem_res())
                for model in self._models
            ]
        )

    def assem_dres_dstate(self):
        dslres_dslx = bm.convert_subtype_to_petsc(self.solid.assem_dres_dstate())
        dslres_dflx = bla.mult_mat_mat(
            bm.convert_subtype_to_petsc(self.solid.assem_dres_dcontrol()),
            self._dslcontrol_dflstate
        )

        # TODO: This probably won't work in 3D since there would be multiple fluid models
        dflres_dflx = bm.convert_subtype_to_petsc(self._models[1].assem_dres_dstate())
        dflres_dslx = bla.mult_mat_mat(
            bm.convert_subtype_to_petsc(self._models[1].assem_dres_dcontrol()),
            self._dflcontrol_dslstate
        )
        bmats = [
            [dslres_dslx, dslres_dflx],
            [dflres_dslx, dflres_dflx]
        ]
        return bm.concatenate(bmats)

    def assem_dres_dstatet(self):
        # Because the fluid models is quasi-steady, there are no time varying FSI quantities
        # As a result, the off-diagonal block terms here are just zero
        dslres_dslx = bm.convert_subtype_to_petsc(self._models[0].assem_dres_dstatet())
        # dfsolid_dxfluid = self._models[0].assem_dres_dcontrolt() * self.dslcontrolt_dflstatet
        dslres_dflx = bm.convert_subtype_to_petsc(self._null_dslstate_dflstate)

        dflres_dflx = bm.convert_subtype_to_petsc(self._models[1].assem_dres_dstatet())
        # dffluid_dxsolid = self._models[1].assem_dres_dcontrolt() * self.dflcontrolt_dslstatet
        dflres_dslx = bm.convert_subtype_to_petsc(self._null_dflstate_dslstate)
        bmats = [
            [dslres_dslx, dslres_dflx],
            [dflres_dslx, dflres_dflx]
        ]
        return bm.concatenate(
            bmats, labels=(self.state.labels[0], self.state.labels[0])
        )

    def assem_dres_dprop(self):

        ## Solid residual sensitivities
        dslres_dslprops = bm.convert_subtype_to_petsc(self.solid.assem_dres_dprop())

        submats = [
            subops.zero_mat(slsubvec.size, propsubvec.size)
            for slsubvec in self.solid.state
            for propsubvec in self._fl_prop
        ]
        dslres_dflprops = bm.BlockMatrix(
            submats,
            shape=self.solid.state.f_shape+self._fl_prop.f_shape,
            labels=self.solid.state.labels+self._fl_prop.labels
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
        dflres_dflcontrol = bm.concatenate_diag(
            [fluid.assem_dres_dcontrol() for fluid in self.fluids]
        )
        dflres_dslprops = bla.mult_mat_mat(
            bm.convert_subtype_to_petsc(dflres_dflcontrol),
            self._dflcontrol_dslprops
        )

        dflres_dflprops = bm.convert_subtype_to_petsc(
            bm.concatenate_diag(
                [fluid.assem_dres_dprop() for fluid in self.fluids]
            )
        )

        submats = [
            subops.zero_mat(flsubvec.size, self.prop['ymid'].size)
            for flsubvec in self._fl_state
        ]
        dflres_dymid = bm.BlockMatrix(
            submats,
            shape=self._fl_state.f_shape+(1,),
            labels=self._fl_state.labels+(('ymid',),)
        )

        bmats = [
            [dslres_dslprops, dslres_dflprops, dslres_dymid],
            [dflres_dslprops, dflres_dflprops, dflres_dymid]
        ]
        return bm.concatenate(bmats)

    def assem_dres_dcontrol(self):
        _mats = [[subops.zero_mat(m, n) for n in self.control.bshape[0]] for m in self.solid.state.bshape[0]]
        dslres_dg = bm.BlockMatrix(_mats, labels=self.solid.state.labels+self.control.labels)

        # dflres_dflg = bm.convert_subtype_to_petsc(self.fluid.assem_dres_dcontrol())
        dflres_dflg = bm.concatenate_diag(
            [fluid.assem_dres_dcontrol() for fluid in self.fluids]
        )
        _mats = [[row[kk] for kk in range(1, dflres_dflg.shape[1])] for row in dflres_dflg]
        # breakpoint()
        dflres_dg = bm.convert_subtype_to_petsc(bm.BlockMatrix(_mats, labels=self._fl_state.labels+self.control.labels))
        return bm.concatenate([[dslres_dg], [dflres_dg]])


class BaseLinearizedDynamicalFSIModel(BaseLinearizedDynamicalModel, BaseDynamicalFSIModel):
    """
    Class representing a fluid-solid coupled dynamical system
    """

    def __init__(self,
            solid: LinearizedModel,
            fluids: [List[LinearizedModel]],
            solid_fsi_dofs, fluid_fsi_dofs
        ):
        super().__init__(solid, fluids, solid_fsi_dofs, fluid_fsi_dofs)

        self.dstate = bv.concatenate(
            [bv.convert_subtype_to_petsc(self.solid.dstate),
                bv.concatenate_with_prefix(
                    [bv.convert_subtype_to_petsc(fluid.dstate) for fluid in self.fluids],
                    prefix='fluid'
                )
            ]
        )
        self.dstatet = bv.concatenate(
            [bv.convert_subtype_to_petsc(self.solid.dstatet),
                bv.concatenate_with_prefix(
                    [bv.convert_subtype_to_petsc(fluid.dstatet) for fluid in self.fluids],
                    prefix='fluid'
                )
            ]
        )

        self._dsolid_area = dfn.Function(self.solid.residual.form['coeff.fsi.p1'].function_space()).vector()

    def set_dstate(self, dstate):
        self.dstate[:] = dstate
        block_sizes = [model.dstate.size for model in self._models]
        sub_states = bv.chunk(dstate, block_sizes)
        for model, sub_state in zip(self._models, sub_states):
            model.set_dstate(sub_state)

        self._transfer_linearized_solid_to_fluid()
        self._transfer_linearized_fluid_to_solid()

    def set_dstatet(self, dstatet):
        self.dstatet[:] = dstatet
        block_sizes = [model.dstatet.size for model in self._models]
        sub_states = bv.chunk(dstatet, block_sizes)
        for model, sub_state in zip(self._models, sub_states):
            model.set_dstatet(sub_state)

    def _transfer_linearized_solid_to_fluid(self):
        """
        Update fluid controls from the solid state
        """
        ## The below are needed to communicate FSI interactions
        # map linearized state to linearized solid area
        dim = self.solid.residual.mesh().topology().dim()
        self._dsolid_area[:] = -2*(self.dstate['u'][1::dim])

        # map linearized solid area to fluid area
        for fsimap, fluid in zip(self._fsimaps, self.fluids):
            dfl_control = fluid.dcontrol.copy()
            dfl_control['area'][:] = subops.mult_mat_vec(
                fsimap.dfluid_dsolid,
                subops.convert_vec_to_petsc(self._dsolid_area)
            )
            fluid.set_dcontrol(dfl_control)

    def _transfer_linearized_fluid_to_solid(self):
        """
        Update fluid controls from the solid state
        """
        # map linearized fluid pressure to solid pressure
        dsolid_control = self.solid.control.copy()
        for fsimap, fluid in zip(self._fsimaps, self.fluids):
            dsolid_control['p'][:] = subops.mult_mat_vec(
                fsimap.dsolid_dfluid,
                subops.convert_vec_to_petsc(fluid.dstate['p'])
            )
        self.solid.set_dcontrol(dsolid_control)

    def set_dcontrol(self, dcontrol):
        raise NotImplementedError()