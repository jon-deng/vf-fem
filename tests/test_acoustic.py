import unittest

from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt

from femvf.models.acoustic import WRAnalog

class TestWRA(unittest.TestCase):

    def setUp(self):
        self.NTUBE = 44
        self.CSOUND = 343*100

        self.LENGTH = 17.46
        # self.DT = 2*self.LENGTH/self.NTUBE/self.CSOUND
        # print(self.DT)

        self.model = WRAnalog(self.NTUBE)

    def test_uniform(self):
        ## Set properties of the uniform tube
        props = self.model.get_properties_vec(set_default=True)
        props['soundspeed'][:] = 343*1e2
        props['proploss'][:] = 1.0
        props['rhoac'][:] = 1.2e-3
        props['length'][:] = self.LENGTH
        # self.model.dt = self.LENGTH/self.model.NTUBE
        # Uniform tract
        props['area'][:] = 4.0
        # Idealized /a/ 
        # props['area'][:22] = 0.5
        # props['area'][22:] = 3
        self.model.set_properties(props)

        # Set the impulse flow input
        state0 = self.model.get_state_vec()
        state0.set(0.0)
        qin = 1.0
        state0['pref'][:2] = self.model.inputq(qin, state0['pinc'][:2])
        control = self.model.get_control_vec()
        control.set(0.0)
        # control['qin'][:] = 0.0

        ## Set the input function and integrate over time
        tstart = perf_counter()
        times = self.model.dt*np.arange(2**14)
        pinc = np.zeros((times.size, state0['pinc'].size))
        pref = np.zeros((times.size, state0['pref'].size))
        # pref[0, :] = state0['pref']
        for n in range(1, times.size):
            self.model.set_ini_state(state0)
            self.model.set_control(control)
            
            state1, _ = self.model.solve_state1()
            pinc[n, :] = state1['pinc']
            pref[n, :] = state1['pref']

            state0 = state1
        print(f"Duration {perf_counter()-tstart:.4f} s")

        ## Numerical impulse response from WRA radiated pressure
        prad = pref[:, -1]
        N = prad.size
        ft_prad = np.fft.fft(prad)
        ft_freq = np.fft.fftfreq(N, d=self.model.dt)

        fig, ax = plt.subplots(1, 1)
        ax.plot(ft_freq[0:N//2], 20*np.log10(np.abs(ft_prad[0:N//2])), label='WRA')

        ## Theoretical impulse response from uniform tube anlalytical solution
        omega = ft_freq*2*np.pi
        RHO = props['rhoac']
        LENGTH, CSOUND = props['length'], props['soundspeed']
        AMOUTH = props['area'][-1]
        PISTON_RAD = np.sqrt(AMOUTH/np.pi)
        ZMOUTH = RHO*CSOUND/AMOUTH

        R = 128/(9*np.pi**2) * ZMOUTH
        L = 8*PISTON_RAD/(3*np.pi*CSOUND) * ZMOUTH
        z_rad = (1j*omega*R*L)/(R+1j*omega*L)

        z0 = ZMOUTH
        gamma_ref = (z_rad-z0)/(z_rad+z0)
        # To derive this formula, see the Lecture 6 of the acoustics class notes
        z_tube = z0*(1+gamma_ref)*np.exp(-1j*omega*LENGTH/CSOUND)/(1-gamma_ref*np.exp(-2j*omega*LENGTH/CSOUND)) 
        # breakpoint()

        qin_vec = np.zeros(times.size)
        qin_vec[0] = qin
        ft_qin = np.fft.fft(qin_vec)
        ax.plot(ft_freq[0:N//2], 20*np.log10(np.abs((ft_qin*z_tube)[0:N//2])), label='Ideal')

        ax.set_xticks(np.arange(0, ft_freq.max(), 500))
        ax.set_ylabel("Frequency response [db]") 
        ax.set_xlabel("Frequency [Hz]")
        ax.set_xlim(0, 5000)
        ax.grid()
        ax.legend()

        fig.tight_layout()
        fig.savefig(f'out/acoustics-uniform-impulseresponse-2e{np.log2(times.size):.0f}.png')


if __name__ == '__main__':
    test = TestWRA()
    test.setUp()
    test.test_uniform()
    