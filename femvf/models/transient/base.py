"""
Generic model definition
"""

import numpy as np

from blocktensor.vec import BlockVector

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
    ## Get empty vectors
    def get_state_vec(self):
        """
        Return empty flow speed and pressure state vectors
        """
        raise NotImplementedError("")

    def get_properties_vec(self, set_default=True):
        """
        Return a BlockVector representing the properties of the fluid
        """
        raise NotImplementedError("")

    def get_control_vec(self):
        ret = self.control.copy()
        ret.set(0.0)
        return ret

    ## Residual and sensitivity methods
    def res(self):
        """
        Return the (nonlinear) residual for the current time step
        """
        raise NotImplementedError

    def solve_state1(self, state1):
        """
        Solve for the final state such that the residual = 0
        """
        raise NotImplementedError

    def solve_dres_dstate1(self, b):
        """
        Solve dF/du1 x = b
        """
        raise NotImplementedError

    def solve_dres_dstate1_adj(self, x):
        """
        Solve dF/du1^T b = x
        """
        raise NotImplementedError

    def apply_dres_dstate0(self, x):
        raise NotImplementedError

    def apply_dres_dstate0_adj(self, b):
        raise NotImplementedError

    def apply_dres_dcontrol(self, x):
        raise NotImplementedError

    def apply_dres_dcontrol_adj(self, x):
        raise NotImplementedError

    def apply_dres_dp(self, x):
        raise NotImplementedError

    def apply_dres_dp_adj(self, x):
        raise NotImplementedError

    def apply_dres_ddt(self, x):
        raise NotImplementedError

    def apply_dres_ddt_adj(self, b):
        raise NotImplementedError
