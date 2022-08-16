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

from .base import AbstractFunctional
from blockarray import blockvec

class AcousticFunctional(AbstractFunctional):
    """
    This class provides an interface/method to define basic solid functionals

    To define a new FluidFunctional you need to implement the `eval`, `eval_du` ... `eval_dp`,
    `form_definitions` methods.
    """
    def __init__(self, model):
        super().__init__(model, ())

    def eval_dstate(self, f, n):
        vecs = [self.model.solid.state0.copy(), self.model.fluid.state0.copy(), self.eval_dac_state(f, n)]
        return vec.concatenate_vec(vecs)

    def eval_dprops(self, f):
        dsolid = self.model.solid.props.copy()
        dsolid[:] = 0.0

        dfluid = self.model.fluid.props.copy()
        dfluid[:] = 0.0

        vecs = [dsolid, dfluid, self.eval_dac_props(f)]

        return vec.concatenate_vec(vecs)

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

        dac = self.model.acoustic.state0.copy()
        dac['pref'][-1] = 0.5*prad_ms**-0.5 * dprad_ms

        return dac

    def eval_dac_props(self, f):
        dfluid = self.acoustic.props.copy()
        dfluid[:] = 0.0
        return dfluid

    def eval_dt0(self, f, n):
        return 0.0

    def eval_ddt(self, f, n):
        return 0.0

class AcousticPower(AcousticFunctional):
    """The norm of the final flow rate"""
    def eval(self, f):
        # dt must be a constant for acoustic simulations
        props = f.get_props()
        RHO, CSPEED = props['rho_air'][0], props['soundspeed'][0]
        AMOUTH = props['area'][-1]
        ZMOUTH = RHO*CSPEED/AMOUTH
        dt = self.model.dt
        T = (f.size-1)*dt

        # Compute the average power using trapezoidal integration
        work = 0
        for n in range(f.size-1):
            prad0 = f.get_state(n)['pref'][-1]
            fmouth0 = f.get_state(n)['pinc'][-2]
            bmouth0 = f.get_state(n)['pref'][-2]

            power0 = prad0*(fmouth0-bmouth0)/ZMOUTH

            prad1 = f.get_state(n+1)['pref'][-1]
            fmouth1 = f.get_state(n+1)['pinc'][-2]
            bmouth1 = f.get_state(n+1)['pref'][-2]

            power1 = prad1*(fmouth1-bmouth1)/ZMOUTH

            work += (power1 + power0)/2 * dt

        return work/T

    def eval_dac_state(self, f, n):
        props = f.get_props()
        RHO, CSPEED = props['rho_air'], props['soundspeed']
        AMOUTH = props['area'][-1]
        ZMOUTH = RHO*CSPEED/AMOUTH
        dt = self.model.dt
        T = (f.size-1)*dt

        # Compute the average power using trapezoidal integration
        work = 0

        prad = f.get_state(n)['pref'][-1]
        fmouth = f.get_state(n)['pinc'][-2]
        bmouth = f.get_state(n)['pref'][-2]
        # power = prad*(fmouth-bmouth)/ZMOUTH
        # work += (power1 + power0)/2 * dt
        dwork_dprad = (fmouth-bmouth)/ZMOUTH/2*dt
        dwork_dfmouth = prad/ZMOUTH/2*dt
        dwork_dbmouth = -prad/ZMOUTH/2*dt

        dac = self.model.acoustic.state0.copy()
        dac[:] = 0.0
        if n > 0:
            dac['pref'][-1] += dwork_dprad/T
            dac['pinc'][-2] += dwork_dfmouth/T
            dac['pref'][-2] += dwork_dbmouth/T
        if n < f.size-1:
            dac['pref'][-1] += dwork_dprad/T
            dac['pinc'][-2] += dwork_dfmouth/T
            dac['pref'][-2] += dwork_dbmouth/T
        return dac

    def eval_dac_props(self, f):
        dfluid = self.model.acoustic.props.copy()
        dfluid[:] = 0.0
        return dfluid

    def eval_dt0(self, f, n):
        return 0.0

    def eval_ddt(self, f, n):
        return 0.0
