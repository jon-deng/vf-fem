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
import scipy.signal as sig

from .abstract import AbstractFunctional
from ..models import acoustic

class AcousticFunctional(AbstractFunctional):
    """
    This class provides an interface/method to define basic solid functionals

    To define a new FluidFunctional you need to implement the `eval`, `eval_du` ... `eval_dp`,
    `form_definitions` methods.
    """
    def __init__(self, model):
        super().__init__(model, ())

        if isinstance(model, acoustic.Acoustic1D):
            self.acoustic = model
        else:
            self.acoustic = model.acoustic

    def eval_dac_state(self, f, n):
        raise NotImplementedError

    def eval_dac_props(self, f):
        raise NotImplementedError

class RmsRadiatedPressure(AcousticFunctional):
    """The norm of the final flow rate"""
    def eval(self, f):
        # dt must be a constant for acoustic simulations
        dt = self.model.dt
        T = (f.size-1)*dt

        # compute the mean-square using trapezoidal integration
        prad_ms = 0
        for n in range(f.size-1):
            prad_a = f.get_state(n)['pref'][-1]
            prad_b = f.get_state(n+1)['pref'][-1]

            prad_ms += (prad_a**2 + prad_b**2)/2 * dt

        return (prad_ms/T)**0.5

    def eval_dac_state(self, f, n):
        prad_rms = self(f)
        prad_ms = prad_rms**2

        dt = self.model.dt
        
        dprad_ms = 0
        prad_n = f.get_state(n)['pref'][-1]
        if n == 0 or n == f.size-1:
            dprad_ms = prad_n * dt
        else:
            dprad_ms = 2*prad_n*dt

        dac = self.model.acoustic.get_state_vec()
        dac['pref'][-1] = 0.5*prad_ms**-0.5 * dprad_ms

        return dac

    def eval_dfl_props(self, f):
        dfluid = self.acoustic.get_properties_vec()
        dfluid.set(0.0)
        return dfluid

    def eval_dt0(self, f, n):
        return 0.0

    def eval_ddt(self, f, n):
        return 0.0
