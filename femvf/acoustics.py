"""
Contains functionality for simulating acoustics
"""

import numpy as np
import jax
from jax import numpy as jnp

from femvf import linalg

class Acoustic1D:
    def __init__(self, num_tube):
        assert num_tube%2 == 0

        # self._dt = 0.0

        # pinc (interlaced f1, b2 partial pressures) are incident pressures
        # pref (interlaced b1, f2 partial pressures) are reflected pressures
        pinc = np.zeros((num_tube//2 + 1)*2)
        pref = np.zeros((num_tube//2 + 1)*2)
        self.state0 = linalg.BlockVec((pinc, pref), ('pinc', 'pref'))

        self.state1 = self.state0.copy()

        # The control is the input flow rate
        qin = np.zeros((1,))
        self.control0 = linalg.BlockVec((qin,), ('qin',))
        self.control1 = self.control0.copy()

        length = np.ones(1)
        area = np.ones(num_tube)
        gamma = np.ones(num_tube)
        rho = 1.225*1e-3*np.ones(1)
        c = 340*100*np.ones(1)
        rrad = np.ones(1)
        lrad = np.ones(1)
        self.properties = linalg.BlockVec(
            (length, area, gamma, rho, c, rrad, lrad), 
            ('length', 'area', 'proploss', 'rhoac', 'soundspeed', 'rrad', 'lrad'))

    ## Setting parameters of the acoustic model
    @property
    def dt(self):
        NTUBE = self.properties['area'].size
        length = self.properties['length'][0]
        C = self.properties['soundspeed'][0]
        return (2*length/NTUBE) / C

    @dt.setter
    def dt(self, value):
        # Note the acoustic model can't change the time step without changing the length of the tract
        # because the tract length and time step are linked throught the speed of sound
        raise NotImplementedError("You can't set the time step of a WRA tube")

    def set_ini_state(self, state):
        for key, value in state.items():
            self.state0[key][:] = value

    def set_fin_state(self, state):
        for key, value in state.items():
            self.state0[key][:] = value

    def set_ini_control(self, control):
        for key, value in control.items():
            self.control0[key][:] = value

    def set_fin_control(self, control):
        for key, value in control.items():
            self.control1[key][:] = value

    def set_time_step(self, dt):
        self.dt = dt

    def set_properties(self, props):
        for key, value in props.items():
            self.properties[key][:] = value

    ## Getting empty vectors
    def get_state_vec(self):
        ret = self.state0.copy()
        ret.set(0.0)
        return ret

    def get_control_vec(self):
        ret = self.control0.copy()
        ret.set(0.0)
        return ret

    def get_properties_vec(self, set_default=True):
        return self.properties.copy()

class WRA(Acoustic1D):

    def set_properties(self, props):
        super().set_properties(props)

        # Reset the WRA 'reflect' function when properties of the tract are updated
        # The reflection function behaviour only changes if the properties are changed
        # so must be reset here
        self.init_wra()

    def init_wra(self):
        dt = self.dt
        cspeed = self.properties['soundspeed'][0]
        rho = self.properties['rhoac'][0]
        area = self.properties['area'].copy()
        gamma = self.properties['proploss'].copy()

        ## Set radiation proeprties
        # Ignore below for now?
        R = self.properties['rrad'][0]
        L = self.properties['lrad'][0]

        # Formula given by Story and Flanagan
        PISTON_RAD = np.sqrt(area[-1]/np.pi)
        R = 128/(9*np.pi**2)
        L = 16/dt*PISTON_RAD/(3*np.pi*cspeed)

        NUM_TUBE = area.size

        a1 = np.concatenate([[1.0], area[:-1:2]])
        a2 = np.concatenate([area[1::2], [1.0]])

        gamma1 = np.concatenate([[1.0], gamma[:-1:2]])
        gamma2 = np.concatenate([gamma[1::2], [1.0]])

        self.reflect, self.reflect00, self.inputq = wra(dt, a1, a2, gamma1, gamma2, NUM_TUBE, cspeed, rho, R=R, L=L)

    ## Solver functions
    def solve_state1(self):
        qin = self.control1['qin'][0]
        pinc, pref = self.state0.vecs
        pinc_1, pref_1 = self.reflect(pinc, pref, qin)

        state1 = linalg.BlockVec((pinc_1, pref_1), self.state0.keys)
        info = {}
        return state1, info

    def res(self):
        return self.state1 - self.solve_state1()

    def dres_dstate1_adj(self, x):
        return x

    def dres_dstate0_adj(self, x):
        args = (*self.state0.vecs, *self.control1.vecs)
        Atrans = jax.linear_transpose(self.reflect, *args)

        bvecs = [np.array(vec) for vec in Atrans(x)[:-1]]
        return -linalg.BlockVec(bvecs, self.state0.keys)

    def dres_dcontrol1_adj(self, x):
        args = (*self.state0.vecs, *self.control1.vecs)
        _, Atrans = jax.linear_transpose(self.reflect, *args)
        
        # collect only the linearized controls
        bvecs = [np.array(vec) for vec in Atrans(x)[len(self.state0.vecs):]]
        return -linalg.BlockVec(bvecs, self.control1.keys)

    def dres_dcontrol(self, x):
        args = (*self.state0.vecs, *self.control1.vecs)
        _, A = jax.linearize(self.reflect, *args)

        x_ = linalg.concatenate(self.get_state_vec(), x)
        bvecs = [np.array(vec) for vec in A(*x_.vecs)]

        return -linalg.BlockVec(bvecs, self.state1.keys)

    def dres_dcontrol0_adj(self, x):
        b = self.control0.copy()
        b.set(0.0)
        return b

    def dres_dproperties_adj(self, x):
        b = self.get_properties_vec()
        b.set(0.0)
        return b

def wra(dt, a1, a2, gamma1, gamma2, N, C, RHO, R=1.0, L=1.0):
    assert gamma1.size == N//2+1
    assert gamma2.size == N//2+1
    
    assert a1.size == N//2+1
    assert a2.size == N//2+1

    z1 = RHO*C/a1
    z2 = RHO*C/a2

    def inputq(q, pinc):
        q = jnp.squeeze(q)
        z = z2[0]
        gamma = gamma2[0]

        f1, b2 = pinc[0], pinc[1]
        b2 = gamma * b2

        f2 = z*q + b2
        b1 = b2 + f2 - f1
        return jnp.array([b1, f2])

    def dinputq(q, bi, z, gamma):
        dfr_dq = z
        dfr_dbi = gamma*1.0
        return dfr_dq, dfr_dbi


    def radiation(pinc, pinc_prev, pref_prev):
        gamma = gamma1[-1]
        f1prev = pinc_prev[0]
        b1prev, f2prev = pref_prev[0], pref_prev[1]
        f1 = pinc[0]

        f1 = gamma * f1

        _a1 = -R+L-R*L
        _a2 = -R-L + R*L

        _b1 = -R+L + R*L
        _b2 = R+L + R*L

        b1 = 1/_b2*(f1*_a2 + f1prev*_a1 + b1prev*_b1)
        f2 = 1/_b2*(f2prev*_b1 + f1*(_b2+_a2) + f1prev*(_a1-_b1))
        return jnp.array([b1, f2])

    def dradiation(f1, f1prev, b1prev, f2prev, gamma):
        _a1 = -R+L-R*L
        _a2 = -R-L + R*L

        _b1 = -R+L + R*L
        _b2 = R+L + R*L

        df2_df1 = gamma*(1/_b2)*(_b2+_a2)
        df2_df1prev = gamma*(1/_b2)*(_a1-_b1)
        df2_db1prev = 0.0
        df2_df2prev = gamma*(1/_b2)*(_b1)

        db1_df1 = gamma*(1/_b2)*(_a2)
        db1_df1prev = gamma*(1/_b2)*(_a1)
        db1_db1prev = gamma*(1/_b2)*(_b1)
        db1_df2prev = 0.0
        return (df2_df1, df2_df1prev, df2_db1prev, df2_df2prev), \
               (db1_df1, db1_df1prev, db1_db1prev, db1_df2prev)


    def reflect00(pinc, pinc_prev, pref_prev, q):
        # Note that int, inp, rad refer to interior, input, and radiation junction
        # locations, respectively
        f1, b2 = pinc[:-1:2], pinc[1::2]
        
        f1 = gamma1 * f1
        b2 = gamma2 * b2
        
        r1 = (z2-z1)/(z2+z1)

        f2int = (f1 + (f1-b2)*r1)[1:-1]
        b1int = (b2 + (f1-b2)*r1)[1:-1]
        pref_int = jnp.stack([b1int, f2int], axis=-1).reshape(-1)

        ## Input boundary
        pinc_inp = pinc[:2]
        pref_inp = inputq(q, pinc_inp) * np.ones(1)

        ## Radiation boundary
        pinc_rad = pinc[-2:]
        pinc_rad_prev = pinc_prev[-2:]
        pref_rad_prev = pref_prev[-2:]
        pref_rad = radiation(pinc_rad, pinc_rad_prev, pref_rad_prev)

        pref = jnp.concatenate([pref_inp, pref_int, pref_rad])
        return pref

    def dreflect00(f1, b2, f1prev, b2prev, b1prev, f2prev, q):
        r1 = (z2-z1)/(z2+z1)

        df2_df1 = gamma1*(1 + r1)
        df2_db2 = gamma2*(-r1)

        db1_df1 = gamma1*(r1)
        db1_db2 = gamma2*(1 - r1)

        ## Input boundary
        df2_dq, df2_db2[0] = dinputq(q, b2[0], z2[0], gamma2[0])
        db1_dq = 0

        ## Radiation boundary
        _df2, _db1 = dradiation(f1[-1], f1prev[-1], b1prev[-1], f2prev[-1], gamma1[-1])
        df2_dfm, df2_dfmprev, df2_dbmprev, df2_dfradprev = _df2
        db1_dfm, db1_dfmprev, db1_dbmprev, db1_dfradprev = _db1

        df2_df1[-1] = df2_dfm
        db1_df1[-1] = db1_dfm

        df2_df1prev = np.zeros(f1prev.shape)
        df2_db1prev = np.zeros(f1prev.shape)
        df2_df2prev = np.zeros(f1prev.shape)
        df2_db2prev = 0.0

        db1_df1prev = np.zeros(f1prev.shape)
        db1_db1prev = np.zeros(f1prev.shape)
        db1_df2prev = np.zeros(f1prev.shape)
        db1_db2prev = 0.0

        db1_df1prev[-1] = db1_dfmprev
        db1_db1prev[-1] = db1_dbmprev
        db1_df2prev[-1] = db1_dfradprev

        df2_df1prev[-1] = df2_dfmprev
        df2_db1prev[-1] = df2_dbmprev
        df2_df2prev[-1] = df2_dfradprev

        df2 = (df2_df1, df2_db2, df2_df1prev, df2_db1prev, df2_df2prev, df2_db2prev, df2_dq)
        db1 = (db1_df1, db1_db2, db1_df1prev, db1_db1prev, db1_df2prev, db1_db2prev, db1_dq)
        return db1, df2 


    def reflect05(pinc):
        z1_ = z2[:-1]
        z2_ = z1[1:]
        
        gamma1_ = gamma2[:-1]
        gamma2_ = gamma1[1:]

        f1 = pinc[:-1:2]
        b2 = pinc[1::2]
        
        f1 = gamma1_ * f1
        b2 = gamma2_ * b2
        r = (z2_-z1_)/(z2_+z1_)

        b1 = b2 + (f1-b2)*r
        f2 = f1 + (f1-b2)*r
        pref = jnp.stack([b1, f2], axis=-1).reshape(-1)
        return pref

    def dreflect05(f1, b2, gamma1, gamma2, z1, z2):
        r = (z2-z1)/(z2+z1)

        df2_df1 = (1.0 + r)*gamma1
        df2_db2 = (-r)*gamma2

        db1_df1 = (r)*gamma1
        db1_db2 = (1.0-r)*gamma2
        return (db1_df1, db1_db2), (df2_df1, df2_db2)
    
    @jax.jit
    def reflect(pinc, pref, q):
        f1, b2 = pinc[:-1:2], pinc[1::2]
        b1, f2 = pref[:-1:2], pref[1::2]

        # f2 and b1 (reflected @ 0.0) -> f1, b2 (incident @ 0.5)
        f1_05 = f2[:-1]
        b2_05 = b1[1:]
        pinc_05 = jnp.stack([f1_05, b2_05], axis=-1).reshape(-1)

        pref_05 = reflect05(pinc_05)
        b1_05, f2_05 = pref_05[:-1:2], pref_05[1::2]
        
        # f2_05 and b1_05 (reflected @ 0.5) -> f1, b2 (incident @ 1.0)
        f1inp, b2rad = np.zeros(1), np.zeros(1)
        f1_1 = jnp.concatenate([f1inp, f2_05])
        b2_1 = jnp.concatenate([b1_05, b2rad])
        pinc_1 = jnp.stack([f1_1, b2_1], axis=-1).reshape(-1)
        
        pref_1 = reflect00(pinc_1, pinc, pref, q)
        return pinc_1, pref_1

    def dreflect(f1, b2, b1, f2, q):
        df1_1_df1 = np.zeros(f1.shape)
        df1_1_db2 = np.zeros(f1.shape)
        df1_1_db1 = np.zeros(f1.shape)
        df1_1_df2 = np.zeros(f1.shape)

        db1_1_df1 = np.zeros(f1.shape)
        db1_1_db2 = np.zeros(f1.shape)
        db1_1_db1 = np.zeros(f1.shape)
        db1_1_df2 = np.zeros(f1.shape)

        df2_1_df1 = np.zeros(f1.shape)
        df2_1_db2 = np.zeros(f1.shape)
        df2_1_db1 = np.zeros(f1.shape)
        df2_1_df2 = np.zeros(f1.shape)

        db2_1_df1 = np.zeros(f1.shape)
        db2_1_db2 = np.zeros(f1.shape)
        db2_1_db1 = np.zeros(f1.shape)
        db2_1_df2 = np.zeros(f1.shape)

        z1_05 = z2[:-1]
        z2_05 = z1[1:]
        
        gamma1_05 = gamma2[:-1]
        gamma2_05 = gamma1[1:]

        f1_05 = f2[:-1]
        b2_05 = b1[1:]

        db1_05, df2_05 = dreflect05(f1_05, b2_05, gamma1_05, gamma2_05, z1_05, z2_05)

        # f1_1 = np.concatenate([f1inp, f2_05])
        # b2_1 = np.concatenate([b1_05, b2rad])
        df1_1_df2[1:] = df2_05[0]
        df1_1_db1[1:] = df2_05[1]
        
        db2_1_df2[:-1] = db1_05[0]
        db2_1_db1[:-1] = db1_05[1]
        pass

    return reflect, reflect00, inputq
