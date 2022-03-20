"""
This module contains definitions of basic fluid functionals.

These act only on fluid states
"""

import numpy as np
# import matplotlib.pyplot as plt
import scipy.signal as sig

# import dolfin as dfn
# import ufl

from .base import AbstractFunctional
from blocktensor import linalg

class FluidFunctional(AbstractFunctional):
    """
    This class provides an interface/method to define basic fluid functionals

    To define a new FluidFunctional you need to implement the `eval`, `eval_du` ... `eval_dp`,
    `form_definitions` methods.
    """
    def __init__(self, model):
        super().__init__(model, ())

    # These are written to handle the case where you have a coupled model input
    # The provided eval_dfl_state only supplies the fluid state component so needs to be 
    # extended to include states of the remaining components (which are just zero)
    def eval_dstate(self, f, n):
        vecs = [self.model.solid.get_state_vec(), self.eval_dfl_state(f, n)]

        if hasattr(self.model, 'acoustic'):
            vecs.append(self.model.acoustic.get_state_vec())
        return linalg.concatenate_vec(vecs)

    def eval_dprops(self, f):
        dsolid = self.model.solid.get_properties_vec()
        dsolid.set(0.0)
        vecs = [dsolid, self.eval_dfl_props(f)]

        if hasattr(self.model, 'acoustic'):
            vecs.append(self.model.acoustic.get_properties_vec())
        return linalg.concatenate_vec(vecs)

    def eval_dfl_state(self, f, n):
        raise NotImplementedError

    def eval_dfl_props(self, f):
        raise NotImplementedError

class FinalPressureNorm(FluidFunctional):
    r"""
    Return the l2 norm of pressure at the final time

    This returns :math:`\sum{||\vec{u}||}_2`.
    """
    def eval(self, f):
        # self.model.set_params_fromfile(f, f.size-1)
        state = f.get_state(f.size-1)

        return np.linalg.norm(state['p'])**2

    def eval_dfl_state(self, f, n):
        dqp = self.model.fluid.get_state_vec()

        if n == f.size-1:
            state = f.get_state(n)
            dqp['p'][:] = 2*state['p']

        return dqp

    def eval_dfl_props(self, f):
        dfluid = self.model.fluid.get_properties_vec()
        dfluid.set(0.0)
        return dfluid

    def eval_dt0(self, f, n):
        return 0.0

    def eval_ddt(self, f, n):
        return 0.0

class FinalFlowRateNorm(FluidFunctional):
    """The norm of the final flow rate"""
    def eval(self, f):
        # breakpoint()
        qp = f.get_state(f.size-1)[3:5]

        return qp['q'][0]

    def eval_dfl_state(self, f, n):
        dqp = self.model.fluid.get_state_vec()

        if n == f.size-1:
            # qp = f.get_state(n)[3:5]
            dqp['q'][:] = 1.0

        return dqp

    def eval_dfl_props(self, f):
        dfluid = self.model.fluid.get_properties_vec()
        dfluid.set(0.0)
        return dfluid

    def eval_dt0(self, f, n):
        return 0.0

    def eval_ddt(self, f, n):
        return 0.0

class AvgSubglottalPower(FluidFunctional):
    """
    Return the average subglottal power input
    """
    def __init__(self, model):
        super().__init__(model)

        self.constants = {'n_start': 0}

    def eval(self, f):
        N_START = self.constants['n_start']
        N_STATE = f.size
        times = f.get_times()

        work = 0
        for ii in range(N_START+1, N_STATE):
            # Set form coefficients to represent the equation mapping state ii->ii+1
            state0 = f.get_state(ii-1)
            state1 = f.get_state(ii)
            
            q0, psub0 = state0['q'][0], state0['p'][0]
            q1, psub1 = state1['q'][0], state1['p'][0]
            dt = times[ii] - times[ii-1]

            work += 0.5*(q0*psub0+q1*psub1)*dt

        return work/(times[N_STATE-1]-times[N_START])

    def eval_dfl_state(self, f, n):
        dqp = self.model.fluid.get_state_vec()
        dqp.set(0.0)

        N_START = self.constants['n_start']
        N_STATE = f.size

        if n >= N_START:
            times = f.get_times()
            state_n = f.get_state(n)
            q, psub = state_n['q'][0], state_n['p'][0]

            # derivative from 'left' quadrature interval
            if n != N_START:
                dt = times[n] - times[n-1]
                dqp['q'][:] += 0.5*psub*dt
                dqp['p'][0] += 0.5*q*dt

            # derivative from 'right' quadrature interval
            if n != N_STATE-1:
                dt = times[n+1] - times[n]
                dqp['q'][:] += 0.5*psub*dt
                dqp['p'][0] += 0.5*q*dt

            dqp = dqp/(times[N_STATE-1] - times[N_START])

        return dqp

    def eval_dfl_props(self, f):
        dfluid = self.model.fluid.get_properties_vec()
        dfluid.set(0.0)
        return dfluid

    def eval_dt0(self, f, n):
        return 0.0

    def eval_ddt(self, f, n):
        # TODO: This has not been fixed
        ddt = 0.0

        N_START = self.constants['n_start']

        def power(n):
            psub_n = f.get_control(n)['psub'][0]
            qn = f.get_state(n)['q'][0]
            return qn*psub_n

        if n > N_START:
            # dt = f.get_time(n) - f.get_time(n-1)
            # work = 0.5*(power(n-1)+power(n))*dt
            ddt += 0.5*(power(n-1)+power(n))

        return ddt

class AvgAcousticPower(FluidFunctional):
    """
    Return the mean acoustic power.
    """
    def __init__(self, model):
        super().__init__(model)
        self.constants = {
            'n_start': 0,
            # The constants below are density, sound speed, and piston radius (mouth opening)
            'rho': 0.001225,
            'c': 350*1e2,
            'a': 0.5,
            'tukey_alpha': 0.05}

    def eval(self, f):
        ## Load the flow rate vector
        n_start = self.constants['n_start']
        n_final = f.size-1

        time = f.get_times()[n_start:n_final]

        q = np.array([f.get_state(i)['q'][0] for i in range(n_start, n_final)])

        ## Multiply the flow rate vector data by a tukey window
        tukey_window = sig.tukey(q.size, alpha=self.constants['tukey_alpha'])
        qw = tukey_window*q

        ## Calculate the DFT of flow rate
        dft_qw = np.fft.fft(qw, n=qw.size)
        dft_freq = np.fft.fftfreq(qw.size, d=time[1]-time[0])

        ## Calculate the normalized radiation impedance, which is
        # complex pressure/flow rate (rather than complex pressure/velocity) 
        rho = self.constants['rho']
        c = self.constants['c']
        a = self.constants['a']

        k = 2*np.pi*dft_freq/c
        z = 1/2*(k*a)**2 + 1j*8*k*a/3/np.pi
        z_radiation = z * rho*c/(np.pi*a**2)

        ## Compute power spectral density of acoustic power
        # psd_acoustic = np.real(z_radiation) * (dft_q * np.conj(dft_q))
        psd_acoustic = np.real(z_radiation) * np.abs(dft_qw)**2

        # By Plancherel's theorem
        # sum(p[k] q*[k]) = 1/N sum(P[k] Q*[k])
        # Divide by the number of power samples to get the average power
        res = np.sum(psd_acoustic)/qw.size/qw.size
        return res

    def eval_dfl_state(self, f, n):
        dqp = self.model.fluid.get_state_vec()

        ## Load the flow rate
        n_start = self.constants['n_start']
        n_final = f.size-1

        time = f.get_times()[n_start:n_final]

        q = np.array([f.get_state(i)['q'][0] for i in range(n_start, n_final)])

        ## Multiply the flow rate vector data by a tukey window
        tukey_window = sig.tukey(q.size, alpha=self.constants['tukey_alpha'])
        q_tukey = tukey_window * q

        ## Calculate the normalized radiation impedance, so it's equal to pressure/flow rate
        rho = self.constants['rho']
        c = self.constants['c']
        a = self.constants['a']

        dft_q_tukey = np.fft.fft(q_tukey, n=q.size)
        dft_freq = np.fft.fftfreq(q.size, d=time[1]-time[0])

        k = 2*np.pi*dft_freq/c
        z = 1/2*(k*a)**2 + 1j*8*k*a/3/np.pi
        z_rad = z * rho*c/(np.pi*a**2)

        ## Calculate the needed derivative terms (if applicable)
        N = q.size
        n_ = n-n_start
        # breakpoint()
        if n_ >= 0 and n_ < q.size:
            # dft_q[n_] = sum(dft_factor * q)
            # psd_acoustic = np.real(z_radiation) * np.abs(dft_q_tukey)**2
            dft_factor = np.exp(1j*2*np.pi*n_*np.arange(N)/N)
            dpsd_dq_tukey = np.real(z_rad) * 2 * np.real(dft_q_tukey * dft_factor)
            dpsd_dq = dpsd_dq_tukey * tukey_window[n_]

            dqp['q'][:] = np.sum(dpsd_dq) / N**2

        return dqp

    def eval_dfl_props(self, f):
        dfluid = self.model.fluid.get_properties_vec()
        dfluid.set(0.0)
        return dfluid

    def eval_ddt(self, f, n):
        # TODO: This should be non-zer0
        return 0.0

    def eval_dp(self, f):
        return None
