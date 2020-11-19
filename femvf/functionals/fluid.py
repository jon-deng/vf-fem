"""
This module contains definitions of various functionals.

A functional is mapping from the time history of all states and parameters to a real number
.. ::math {(u, v, a, q, p; t, params)_n for n in {0, 1, ..., N-1}} .

A functional should take in the entire time history of states from a forward model run and return a
real number.

For computing the sensitivity of the functional through the discrete adjoint method, you also need
the sensitivity of the functional with respect to the n'th state. This function has the signature

dfunctional_du(model, n, f, ....) -> float, dict

, where the parameters have the same uses as defined previously. Parameter `n` is the state to
compute the sensitivity with respect to.
"""

import numpy as np
# import matplotlib.pyplot as plt
import scipy.signal as sig

# import dolfin as dfn
# import ufl

from .abstract import AbstractFunctional
from ..fluids import QuasiSteady1DFluid, smoothmin, dsmoothmin_df

class FluidFunctional(AbstractFunctional):
    """
    This class provides an interface/method to define basic solid functionals

    To define a new FluidFunctional you need to implement the `eval`, `eval_du` ... `eval_dp`,
    `form_definitions` methods.
    """
    def __init__(self, model):
        super().__init__(model, ())

        if isinstance(model, QuasiSteady1DFluid):
            self.fluid = model
        else:
            self.fluid = model.fluid

    # These are written to handle the case where you have a coupled model input
    # then the provided eval_dsl_state only supplies the solid portion and needs to be 
    # extended
    def eval_dstate(self, f, n):
        vecs = [self.model.solid.get_state_vec(), self.eval_dfl_state(f, n)]

        if hasattr(self.model, 'acoustic'):
            vecs.append(self.model.acoustic.get_state_vec())
        return linalg.concatenate(*vecs)

    def eval_dprops(self, f):
        vecs = [self.model.solid.get_properties_vec(), self.eval_dfl_props(f)]

        if hasattr(self.model, 'acoustic'):
            vecs.append(self.model.acoustic.get_properties_vec())
        return linalg.concatenate(*vecs)

class FinalSurfacePressureNorm(FluidFunctional):
    r"""
    Return the l2 norm of pressure at the final time

    This returns :math:`\sum{||\vec{u}||}_2`.
    """
    func_types = ()
    default_constants = {}

    def eval(self, f):
        self.model.set_params_fromfile(f, f.size-1)
        # self.forms['pressure'].vector()[:] = f.get_fluid_state(f.size-1)[0]

        return dfn.assemble(self.forms['res'])

    def eval_dfl_state(self, f, n):
        dqp = self.fluid.get_state_vec()

        if n == f.size-1:
            self.model.set_params_fromfile(f, f.size-1)
            dqp['p'][:] = dfn.assemble(self.forms['dres_dpressure'],
                                       tensor=dfn.PETScVector())

        return dqp

    def eval_dfl_props(self, f):
        dfluid = self.fluid.get_properties_vec()
        dfluid.set(0.0)
        return dfluid

    def eval_dt0(self, f, n):
        return 0.0

    def eval_ddt(self, f, n):
        return 0.0

class FinalFlowRateNorm(FluidFunctional):
    """Represents the norm of the final flow rate"""
    func_types = ()

    def eval(self, f):
        qp = f.get_fluid_state(f.size-1)

        return qp[0]**2

    def eval_dfl_state(self, f, n):
        dqp = self.fluid.get_state_vec()

        if n == f.size-1:
            qp = f.get_fluid_state(n)
            dqp['q'][:] = 2*qp[0]

        return dqp

    def eval_dfl_props(self, f):
        dfluid = self.fluid.get_properties_vec()
        dfluid.set(0.0)
        return dfluid

    def eval_dt0(self, f, n):
        return 0.0

    def eval_ddt(self, f, n):
        return 0.0
