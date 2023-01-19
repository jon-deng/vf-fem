"""
This module defines the basic interface for a transient model.
"""

from typing import TypeVar, Union, Tuple, Mapping, Optional, Any

from blockarray import subops
from blockarray import blockvec as bv, blockmat as bm

T = TypeVar('T')
Vector = Union[subops.DfnVector, subops.GenericSubarray, subops.PETScVector]
Matrix = Union[subops.DfnMatrix, subops.GenericSubarray, subops.PETScMatrix]

BlockVec = bv.BlockVector[Vector]
BlockMat = bm.BlockMatrix[Matrix]

class BaseTransientModel:
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

    @property
    def dt(self):
        """
        Return/set the time step
        """
        raise NotImplementedError(f"Subclass {type(self)} must implement this function")

    def set_ini_state(self, state0: BlockVec):
        """
        Set the initial state (`self.state0`)

        Parameters
        ----------
        state0: BlockVec
            The state to set
        """
        raise NotImplementedError(f"Subclass {type(self)} must implement this function")

    def set_fin_state(self, state1: BlockVec):
        """
        Set the final state (`self.state1`)

        Parameters
        ----------
        state1: BlockVec
            The state to set
        """
        raise NotImplementedError(f"Subclass {type(self)} must implement this function")

    def set_control(self, control: BlockVec):
        """
        Set the control (`self.control`)

        Parameters
        ----------
        control: BlockVec
            The controls to set
        """
        raise NotImplementedError(f"Subclass {type(self)} must implement this function")

    def set_prop(self, prop: BlockVec):
        """
        Set the properties (`self.prop`)

        Parameters
        ----------
        prop: BlockVec
            The properties to set
        """
        raise NotImplementedError(
            f"Subclass {type(self)} must implement this function"
        )

    ## Residual and sensitivity methods
    def assem_res(self) -> BlockVec:
        """
        Return the residual of the current time step
        """
        raise NotImplementedError(f"Subclass {type(self)} must implement this function")

    def assem_dres_dstate0(self) -> BlockMat:
        """
        Return the residual sensitivity to the initial state for the time step
        """
        raise NotImplementedError(f"Subclass {type(self)} must implement this function")

    def assem_dres_dstate1(self) -> BlockMat:
        """
        Return the residual sensitivity to the final state for the time step
        """
        raise NotImplementedError(f"Subclass {type(self)} must implement this function")

    def assem_dres_dcontrol(self) -> BlockMat:
        """
        Return the residual sensitivity to the control for the time step
        """
        raise NotImplementedError(f"Subclass {type(self)} must implement this function")

    def assem_dres_dprops(self) -> BlockMat:
        """
        Return the residual sensitivity to the properties for the time step
        """
        raise NotImplementedError(f"Subclass {type(self)} must implement this function")

    ## Solver methods
    def solve_state1(
            self,
            state1: BlockVec,
            options: Optional[Mapping[str, Any]]
        ) -> Tuple[BlockVec, Mapping[str, Any]]:
        """
        Solve for the final state for the time step

        Parameters
        ----------
        state1: BlockVec
            An initial guess for the final state. For nonlinear models, this
            serves as the initial guess for an iterative procedure.

        Returns
        -------
        BlockVec
            The final state at the end of the time step
        dict
            A dictionary of information about the solution. This depends on the
            solver but usually includes information like: the number of
            iterations, residual error, etc.
        """
        raise NotImplementedError(f"Subclass {type(self)} must implement this function")

    # TODO: If you want to take derivatives of Transient models, you will need
    # to implement solvers for the jacobian and its adjoint
    # (`solve_dres_dstate1` and `solve_dres_dstate1_adj`).
    # In addition you'll need adjoints of all the `assem_*` functions
    # Currently some of these functions are left over from a previous implementation
    # but no longer work due to code changes.
