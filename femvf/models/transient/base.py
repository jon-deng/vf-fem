"""
This module defines the basic interface for a transient model.
"""

from typing import TypeVar, Union

from blockarray import subops
from blockarray import blockvec as bv, blockmat as bm

T = TypeVar('T')
Vector = Union[subops.DfnVector, subops.GenericSubarray, subops.PETScVector]
Matrix = Union[subops.DfnMatrix, subops.GenericSubarray, subops.PETScMatrix]

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
    def set_ini_state(self, state0: bv.BlockVector[Vector]):
        raise NotImplementedError()

    def set_fin_state(self, state1: bv.BlockVector[Vector]):
        raise NotImplementedError()

    def set_control(self, control: bv.BlockVector[Vector]):
        raise NotImplementedError()

    def set_props(self, props: bv.BlockVector[Vector]):
        raise NotImplementedError()

    ## Residual and sensitivity methods
    def assem_res(self) -> bv.BlockVector[Vector]:
        """
        Return the (nonlinear) residual for the current time step
        """
        raise NotImplementedError()

    def assem_dres_dstate0(self) -> bm.BlockMatrix[Matrix]:
        raise NotImplementedError()

    def assem_dres_dstate1(self) -> bm.BlockMatrix[Matrix]:
        raise NotImplementedError()

    def assem_dres_dcontrol(self) -> bm.BlockMatrix[Matrix]:
        raise NotImplementedError()

    def assem_dres_dprops(self) -> bm.BlockMatrix[Matrix]:
        raise NotImplementedError()

    ## Solver methods
    def solve_state1(self, state1: bv.BlockVector[Vector]) -> bv.BlockVector[Vector]:
        """
        Solve for the final state such that the residual = 0
        """
        raise NotImplementedError()

    def solve_dres_dstate1(self, b: bv.BlockVector[Vector]) -> bv.BlockVector[Vector]:
        """
        Solve dF/du1 x = b
        """
        raise NotImplementedError()

    def solve_dres_dstate1_adj(self, x: bv.BlockVector[Vector]) -> bv.BlockVector[Vector]:
        """
        Solve dF/du1^T b = x
        """
        raise NotImplementedError()
