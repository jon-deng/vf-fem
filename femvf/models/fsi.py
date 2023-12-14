"""
This module contains functionality for coupling fluid/solid domains
"""

from typing import Union, List, Any
from numpy.typing import ArrayLike

import itertools

import dolfin as dfn
from petsc4py import PETSc

from .transient import solid as tsmd, fluid as tfmd
from .dynamical import solid as dsmd, fluid as dfmd
from blockarray import subops, blockmat as bm, blockvec as bv

class FSIMap:
    """
    Represents a mapping between two domains (fluid and solid)

    This mapping involves a 1-to-1 correspondence between DOFs of vectors on the two domains
    """
    def __init__(
            self,
            ndof_fluid: int, ndof_solid: int,
            fluid_dofs: ArrayLike, solid_dofs: ArrayLike,
            comm=None
        ):
        """
        Parameters
        ----------
        ndof_fluid, ndof_solid : int
            number of DOFS on the fluid and solid domains
        fluid_dofs, solid_dofs : array
            arrays of corresponding dofs on the fluid and solid side domains
        comm : None or PETSc.Comm
            MPI communicator. Not really used here since I never run stuff in parallel.
        """
        self.N_FLUID = ndof_fluid
        self.N_SOLID = ndof_solid

        self.dofs_fluid = fluid_dofs
        self.dofs_solid = solid_dofs

        self.fluid_to_solid_idx = {idxf: idxs for idxf, idxs in zip(fluid_dofs, solid_dofs)}
        self.solid_to_fluid_idx = {idxs: idxf for idxf, idxs in zip(fluid_dofs, solid_dofs)}

        self.dsolid_dfluid = self.assem_dsolid_dfluid(comm)
        self.dfluid_dsolid = self.assem_dfluid_dsolid(comm)

    def map_fluid_to_solid(self, fluid_vec, solid_vec):
        solid_vec[self.dofs_solid] = fluid_vec[self.dofs_fluid]

    def map_solid_to_fluid(self, solid_vec, fluid_vec):
        fluid_vec[self.dofs_fluid] = solid_vec[self.dofs_solid]

    def assem_dsolid_dfluid(self, comm=None):
        # pylint: disable=no-member
        A = PETSc.Mat().createAIJ([self.N_SOLID, self.N_FLUID], comm=comm)
        A.setUp()
        for jj, ii in self.fluid_to_solid_idx.items():
            A.setValue(ii, jj, 1)
        A.assemble()
        return A

    def assem_dfluid_dsolid(self, comm=None):
        # pylint: disable=no-member
        A = PETSc.Mat().createAIJ([self.N_FLUID, self.N_SOLID], comm=comm)
        A.setUp()
        for jj, ii in self.solid_to_fluid_idx.items():
            A.setValue(ii, jj, 1)
        A.assemble()
        return A

SolidModel = Union[tsmd.Model, dsmd.Model]
FluidModel = Union[tfmd.Model, dfmd.Model]

def _state_from_dynamic_or_transient_model(model: Union[SolidModel, FluidModel]):
    if isinstance(model, (tfmd.Model, tsmd.Model)):
        return model.state0
    elif isinstance(model, (dfmd.Model, dsmd.Model)):
        return model.state
    else:
        raise ValueError("Unknown `model` type")

def coupling_stuff(
        solid: SolidModel,
        fluids: Union[List[FluidModel], FluidModel],
        solid_fsi_dofs: Union[List[ArrayLike], ArrayLike],
        fluid_fsi_dofs: Union[List[ArrayLike], ArrayLike]
    ):

    # Load a solid and fluid state(s) vectors because these have different
    # attribute names between dynamical/transient model types
    sl_state = _state_from_dynamic_or_transient_model(solid)
    fl_states = [
        _state_from_dynamic_or_transient_model(fluid) for fluid in fluids
    ]
    fl_state = bv.concatenate_with_prefix(fl_states, 'fluid')

    solid_area = dfn.Function(
        solid.residual.form['coeff.fsi.p1'].function_space()
    ).vector()

    # n_flq = fl_states[0]['q'].size
    # n_flp = fl_states[0]['p'].size
    n_slp = solid.control['p'].size
    n_slu = sl_state['u'].size

    fsimaps = tuple(
        FSIMap(
            fl_state['p'].size,
            solid_area.size(),
            fluid_dofs, solid_dofs
        )
        for fl_state, fluid_dofs, solid_dofs in zip(fl_states, fluid_fsi_dofs, solid_fsi_dofs)
    )

    ## Construct the derivative of the solid control w.r.t fluid state
    dslp_dflq_coll = [
        subops.zero_mat(n_slp, _fl_state['q'].size) for _fl_state in fl_states
    ]
    dslp_dflp_coll = [fsimap.dsolid_dfluid for fsimap in fsimaps]
    mats = tuple(
        mat for dslp_dflq, dslp_dflp in zip(dslp_dflq_coll, dslp_dflp_coll)
        for mat in [dslp_dflq, dslp_dflp]
    )

    ret_bshape = solid.control.bshape + fl_state.bshape
    dslcontrol_dflstate = bm.BlockMatrix(
        mats,
        shape=solid.control.shape+fl_state.shape,
        labels=solid.control.labels+fl_state.labels
    )
    assert dslcontrol_dflstate.bshape == ret_bshape

    ## Construct the derivative of the fluid control w.r.t solid state
    # To do this, first build the sensitivty of area wrt displacement:
    # TODO: Many of the lines here are copy-pasted from `femvf.dynamical.coupled`
    # and should be refactored to a seperate function
    dslarea_dslu = PETSc.Mat().createAIJ(
        (n_slp, n_slu), nnz=2
    )
    for ii in range(dslarea_dslu.size[0]):
        # Each solid area is only sensitive to the y component of u, so
        # that's set here
        # TODO: can only set sensitivites for relevant DOFS; only DOFS on
        # the surface have an effect
        dslarea_dslu.setValues([ii], [2*ii, 2*ii+1], [0, -2])
    dslarea_dslu.assemble()

    dflarea_dslarea_coll = [fsimap.dfluid_dsolid for fsimap in fsimaps]
    dflarea_dslu_coll = [dflarea_dslarea*dslarea_dslu for dflarea_dslarea in dflarea_dslarea_coll]
    fluid_control = bv.concatenate_with_prefix([fluid.control for fluid in fluids], 'fluid')
    ret_bshape = fluid_control.bshape+sl_state.bshape
    mats = [
        subops.zero_mat(nrow, ncol) for nrow, ncol
        in itertools.product(*ret_bshape)
    ]
    # Set the block components `[f'fluid{ii}.area', 'u']`
    labels = [f'fluid{ii}.area' for ii in range(len(fluids))]
    rows = [fluid_control.labels[0].index(label) for label in labels]
    ncol = sl_state.size
    col = sl_state.labels[0].index('u')
    for ii, dflarea_dslu in zip(rows, dflarea_dslu_coll):
        mats[ii*ncol+col] = dflarea_dslu

    # breakpoint()
    dflcontrol_dslstate = bm.BlockMatrix(
        mats,
        shape=fluid_control.shape+sl_state.shape,
        labels=fluid_control.labels+sl_state.labels
    )
    assert dflcontrol_dslstate.bshape == ret_bshape

    return fsimaps, solid_area, dflcontrol_dslstate, dslcontrol_dflstate

