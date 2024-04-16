"""
This module contains functionality for coupling fluid/solid domains
"""

from typing import Union, List, Any
from numpy.typing import NDArray

import itertools

import dolfin as dfn
from petsc4py import PETSc

from .transient import solid as tsmd, fluid as tfmd
from .dynamical import solid as dsmd, fluid as dfmd
from blockarray import subops, blockmat as bm, blockvec as bv

class FSIMap:
    """
    Represents a mapping between two domains (fluid and solid)

    This mapping involves a 1-to-1 correspondence between DOFs of vectors on the
    two domains.

    To illustrate this, consider this example.
    The solid domain has a pressure vector with `ndof_solid` coefficents while
    the fluid domain has a pressure vector with `ndof_fluid` coefficients.
    The vectors `fluid_dofs` and `solid_dofs` have the same size and specify
    which elements of the fluid domain pressure vector map to which elements
    of the solid domain pressure vector.

    Parameters
    ----------
    ndof_fluid, ndof_solid : int
        The number of DOFs for the FSI vectors on the fluid and solid domains
    fluid_dofs, solid_dofs : NDArray[int]
        arrays of corresponding dofs on the fluid and solid side domains
    comm : None or PETSc.Comm
        MPI communicator. Not really used here since I never run stuff in parallel.
    """
    def __init__(
            self,
            ndof_fluid: int, ndof_solid: int,
            fluid_dofs: NDArray[int], solid_dofs: NDArray[int],
            comm=None
        ):
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
    elif isinstance(model, (dfmd.Model, dfmd.LinearizedModel, dsmd.Model, dsmd.LinearizedModel)):
        return model.state
    else:
        raise ValueError("Unknown `model` type")

def make_coupling_stuff(
        solid: SolidModel,
        fluids: Union[List[FluidModel], FluidModel],
        solid_fsi_dofs: Union[List[NDArray[int]], NDArray[int]],
        fluid_fsi_dofs: Union[List[NDArray[int]], NDArray[int]]
    ):
    """
    Return coupling matrices, etc.
    """

    # Load a solid and fluid state(s) vectors because these have different
    # attribute names between dynamical/transient model types
    sl_state = _state_from_dynamic_or_transient_model(solid)
    fl_states = [
        _state_from_dynamic_or_transient_model(fluid) for fluid in fluids
    ]

    solid_area = dfn.Function(
        solid.residual.form['coeff.fsi.p1'].function_space()
    ).vector()

    fsimaps = make_fsimaps(solid, fluids, solid_fsi_dofs, fluid_fsi_dofs)

    ## Construct the derivative of the solid control w.r.t fluid state
    dslcontrol_dflstate = make_dslcontrol_dflstate(
        solid.control, fl_states, fsimaps
    )
    # assert dslcontrol_dflstate.bshape == ret_bshape

    ## Construct the derivative of the fluid control w.r.t solid inputs
    ndim = solid.residual.mesh().topology().dim()
    n_u = sl_state['u'].size
    n_area = n_u//ndim
    dslarea_dslu = make_dslarea_dslu(n_area, n_u, ndim)

    fl_controls = [fluid.control for fluid in fluids]
    dflcontrol_dslstate = make_dflcontrol_dslstate(
        fl_controls, sl_state, fsimaps, dslarea_dslu
    )

    dflcontrol_dslprop = make_dflcontrol_dslprop(
        fl_controls, solid.prop, fsimaps, dslarea_dslu
    )

    return (fsimaps, solid_area, dflcontrol_dslstate, dslcontrol_dflstate, dflcontrol_dslprop)

def make_fsimaps(
        solid: SolidModel,
        fluids: Union[List[FluidModel], FluidModel],
        solid_fsi_dofs: Union[List[NDArray[int]], NDArray[int]],
        fluid_fsi_dofs: Union[List[NDArray[int]], NDArray[int]]
    ):
    """
    Return `FSIMap` instances for multiple fluids coupled to a single solid domain
    """
    fl_states = [
        _state_from_dynamic_or_transient_model(fluid) for fluid in fluids
    ]
    # fl_state = bv.concatenate_with_prefix(fl_states, 'fluid')

    solid_area = dfn.Function(
        solid.residual.form['coeff.fsi.p1'].function_space()
    ).vector()

    fsimaps = tuple(
        FSIMap(
            fl_state['p'].size,
            solid_area.size(),
            fluid_dofs, solid_dofs
        )
        for fl_state, fluid_dofs, solid_dofs in zip(fl_states, fluid_fsi_dofs, solid_fsi_dofs)
    )
    return fsimaps

def make_dslcontrol_dflstate(sl_control, fl_states, fsimaps):
    """
    Return the sensitivity of the solid control to fluid state vector
    """
    n_slp = sl_control['p'].size
    dslp_dflq_coll = [
        subops.zero_mat(n_slp, _fl_state['q'].size) for _fl_state in fl_states
    ]
    dslp_dflp_coll = [fsimap.dsolid_dfluid for fsimap in fsimaps]
    mats = tuple(
        mat for dslp_dflq, dslp_dflp in zip(dslp_dflq_coll, dslp_dflp_coll)
        for mat in [dslp_dflq, dslp_dflp]
    )

    fl_state = bv.concatenate_with_prefix(fl_states, 'fluid')
    ret_bshape = sl_control.bshape + fl_state.bshape
    dslcontrol_dflstate = bm.BlockMatrix(
        mats,
        shape=sl_control.shape+fl_state.shape,
        labels=sl_control.labels+fl_state.labels
    )
    assert dslcontrol_dflstate.bshape == ret_bshape

    return dslcontrol_dflstate

def make_dflcontrol_dslstate(fl_controls, sl_state, fsimaps, dslarea_dslu):
    """
    Return the sensitivity of the fluid control to solid state vector
    """
    # To do this, first build the sensitivty of area wrt displacement:
    dflarea_dslarea_coll = [fsimap.dfluid_dsolid for fsimap in fsimaps]
    dflarea_dslu_coll = [dflarea_dslarea*dslarea_dslu for dflarea_dslarea in dflarea_dslarea_coll]
    fluid_control = bv.concatenate_with_prefix(fl_controls, 'fluid')
    ret_bshape = fluid_control.bshape+sl_state.bshape
    mats = [
        subops.zero_mat(nrow, ncol) for nrow, ncol
        in itertools.product(*ret_bshape)
    ]
    # Set the block components `[f'fluid{ii}.area', 'u']`
    labels = [f'fluid{ii}.area' for ii in range(len(fl_controls))]
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

    return dflcontrol_dslstate

def make_dflcontrol_dslprop(fl_controls, sl_prop, fsimaps, dslarea_dslu):
    fl_control = bv.concatenate_with_prefix(fl_controls, prefix='fluid')
    mats = [
        subops.zero_mat(nrow, ncol)
        for nrow, ncol
        in itertools.product(
            fl_control.bshape[0], sl_prop.bshape[0],
        )
    ]
    if 'umesh' in sl_prop:
        dflarea_dslarea_coll = [fsimap.dfluid_dsolid for fsimap in fsimaps]
        dflarea_dslu_coll = [
            dflarea_dslarea*dslarea_dslu
            for dflarea_dslarea in dflarea_dslarea_coll
        ]

        # Set the block components `[f'fluid{ii}.area', 'umesh']`
        labels = [f'fluid{ii}.area' for ii in range(len(fl_controls))]
        fluid_control = bv.concatenate_with_prefix(fl_controls, 'fluid')
        rows = [fluid_control.labels[0].index(label) for label in labels]
        ncol = sl_prop.size
        col = sl_prop.labels[0].index('umesh')
        for ii, dflarea_dslu in zip(rows, dflarea_dslu_coll):
            mats[ii*ncol+col] = dflarea_dslu

    dflcontrol_dslprop = bv.BlockMatrix(
        mats,
        shape=fl_control.shape+sl_prop.shape,
        labels=fl_control.labels+sl_prop.labels
    )
    return dflcontrol_dslprop

def make_dslarea_dslu(n_area, n_dis, ndim=2):
    """
    Return the sensitivity of the channel area to the displacement vector
    """
    # To do this, first build the sensitivty of area wrt displacement:
    dslarea_dslu = PETSc.Mat().createAIJ(
        (n_area, n_dis), nnz=2
    )
    for ii in range(dslarea_dslu.size[0]):
        # Each solid area is only sensitive to the y component of u, so
        # that's set here
        # TODO: can only set sensitivites for relevant DOFS; only DOFS on
        # the surface have an effect
        dslarea_dslu.setValues([ii], [ndim*ii, ndim*ii+1], [0, -2])
    dslarea_dslu.assemble()
    return dslarea_dslu