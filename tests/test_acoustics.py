import unittest

from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt

from femvf.acoustics import WRA

class TestWRA(unittest.TestCase):

    def setUp(self):
        self.NTUBE = 44
        self.CSOUND = 343*100

        self.LENGTH = 17.46
        # self.DT = 2*self.LENGTH/self.NTUBE/self.CSOUND
        # print(self.DT)

        model = WRA(self.NTUBE)
        self.model = model

    def test_uniform(self):
        ## Set properties of the uniform tube
        props = self.model.get_properties_vec(set_default=True)
        props['soundspeed'][:] = 343*1e2
        props['area'][:] = 4.0
        props['proploss'][:] = 1.0
        props['rhoac'][:] = 1.2e-3
        props['length'][:] = self.LENGTH

        self.model.set_properties(props)

        state0 = self.model.get_state_vec()
        state0.set(0.0)

        control = self.model.get_control_vec()
        control.set(0.0)

        qin = 1.0
        state0['pref'][:2] = self.model.inputq(qin, state0['pinc'][:2])

        tstart = perf_counter()
        times = self.model.dt*np.arange(2**12)
        pinc = np.zeros((times.size, state0['pinc'].size))
        pref = np.zeros((times.size, state0['pref'].size))
        for n in range(1, times.size):
            self.model.set_ini_state(state0)
            self.model.set_fin_control(control)
            state1 = self.model.solve_state1()
            pinc[n, :] = state1['pinc']
            pref[n, :] = state1['pref']

            state0 = state1
        print(f"Duration {perf_counter()-tstart:.4f} s")

        # Compute the impulse response of the radiated pressure
        prad = pref[:, -1]
        N = prad.size
        ft_prad = np.fft.fft(prad)
        ft_freq = np.fft.fftfreq(N, d=self.model.dt)

        fig, ax = plt.subplots(1, 1)
        ax.plot(ft_freq[0:N//2], 20*np.log10(np.abs(ft_prad[0:N//2])))

        ax.set_ylabel("Frequency response [db]")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_xlim(0, 5000)
        ax.grid()
        fig.savefig('out/acoustics-uniform-impulseresponse.png')


if __name__ == '__main__':
    test = TestWRA()
    test.setUp()
    test.test_uniform()