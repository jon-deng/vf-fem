"""
This module defines the basic interface for a transient model.
"""

from typing import TypeVar, Union

from blockarray import subops
from blockarray import blockvec as bv, blockmat as bm

T = TypeVar('T')
Vector = Union[subops.DfnVector, subops.GenericSubarray, subops.PETScVector]
Matrix = Union[subops.DfnMatrix, subops.GenericSubarray, subops.PETScMatrix]

BlockVec = bv.BlockVector[Vector]
BlockMat = bm.BlockMatrix[Matrix]

class Model:
    """
    This object represents the equations defining a system over one time step.

    The residual represents the error in the equations for a time step given by:
        F(u1, u0, g, p, dt)
        where
        u1, u0 : final/initial states of the system
        g : control vector at the current time (i.e. index 1)
        p : properties (constant in time)
        dt : time step

    Derivatives of F w.r.t u1, u0, g, p, dt and adjoint of those operators should all be
    defined.
    """
    ## Parameter setting functions
    def set_ini_state(self, state0: BlockVec):
        raise NotImplementedError()

    def set_fin_state(self, state1: BlockVec):
        raise NotImplementedError()

    def set_control(self, control: BlockVec):
        raise NotImplementedError()

    def set_props(self, props: BlockVec):
        raise NotImplementedError()

    ## Residual and sensitivity methods
    def assem_res(self) -> BlockVec:
        """
        Return the (nonlinear) residual for the current time step
        """
        raise NotImplementedError()

    def assem_dres_dstate0(self) -> BlockMat:
        raise NotImplementedError()

    def assem_dres_dstate1(self) -> BlockMat:
        raise NotImplementedError()

    def assem_dres_dcontrol(self) -> BlockMat:
        raise NotImplementedError()

    def assem_dres_dprops(self) -> BlockMat:
        raise NotImplementedError()

    ## Solver methods
    def solve_state1(self, state1: BlockVec) -> BlockVec:
        """
        Solve for the final state such that the residual = 0
        """
        raise NotImplementedError()

