"""
Contains functionality for simulating acoustics
"""

import numpy as np
import jax
from jax import numpy as jnp

from femvf import linalg

class WaveAnalog1D:
    def __init__(self, num_tube, dt):
        assert num_tube%2 == 0

        self.dt = dt

        f1 = np.zeros(num_tube//2 + 1)
        b2 = np.zeros(num_tube//2 + 1)
        b1 = np.zeros(num_tube//2 + 1)
        f2 = np.zeros(num_tube//2 + 1)
        self.state0 = linalg.BlockVec((f1, b2, b1, f2), ('f1', 'b2', 'b1', 'f2'))

        self.state1 = self.state0.copy()

        # The control is the input flow rate
        qin = np.zeros((1,))
        self.control0 = linalg.BlockVec((qin,), ('qin',))
        self.control1 = self.control0.copy()

        area = np.ones(num_tube)
        gamma = np.zeros(num_tube)
        rho = 1.225*1e-3*np.ones(1)
        c = 340*100*np.ones(1)
        re_rad = np.ones(1)
        im_rad = np.ones(1)
        self.properties = linalg.BlockVec(
            (area, gamma, rho, c, re_rad, im_rad), 
            ('area', 'proploss', 'rhoac', 'soundspeed', 're_zrad', 'im_zrad'))

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


    def get_state_vec(self):
        ret = self.state0.copy()
        ret.set(0.0)
        return ret

    def get_control_vec(self):
        ret = self.control0.copy()
        ret.set(0.0)
        return ret

    def get_properties_vec(self):
        return self.properties.copy()

class WRA(WaveAnalog1D):

    def set_properties(self, props):
        super().set_properties(props)

        # Reset the WRA 'reflect' function when properties of the tract are updated
        # The reflection function behaviour only changes if the properties are changed
        # so must be reset here
        self.init_wra()

    def init_wra(self):
        dt = self.properties['dt'][0]
        cspeed = self.properties['soundspeed'][0]
        rho = self.properties['rhoac'][0]
        area = self.properties['area'].copy()
        gamma = self.properties['proploss'].copy()

        im_zrad = self.properties['im_zrad'][0]
        re_zrad = self.properties['re_zrad'][0]

        NUM_TUBE = area.size

        a1 = np.concatenate([0], area[:-1:2])
        a2 = np.concatenate(area[1::2], [0])

        gamma1 = np.concatenate([0], gamma[:-1:2])
        gamma2 = np.concatenate(gamma[1::2], [0])

        self.reflect, *_ = wra(dt, a1, a2, gamma1, gamma2, NUM_TUBE, cspeed, rho, R=re_zrad, L=im_zrad)

    def res(self):
        qin = self.control0['qin'][0]
        f1, b2, b1, f2 = self.state0.vecs
        f1_1, b2_1, b1_1, f2_1 = self.reflect(f1, b2, b1, f2, qin)

        state1_guess = linalg.BlockVec((f1_1, b2_1, b1_1, f2_1), self.state0.keys)
        return self.state1 - state1_guess

    def dres_dstate1_adj(self, x):
        return x

    def dres_dstate0_adj(self, x):
        args = (*self.state0.vecs, *self.control1.vecs)
        Atrans = jax.linear_transpose(self.reflect, *args)

        bvecs = [np.array(vec) for vec in Atrans(x)[:-1]]
        return -linalg.BlockVec(bvecs, self.state0.keys)

    def dres_dcontrol1_adj(self, x):
        args = (*self.state0.vecs, *self.control1.vecs)
        Atrans = jax.linear_transpose(self.reflect, *args)
        
        bvecs = [np.array(vec) for vec in Atrans(x)[-1:]]
        return -linalg.BlockVec(bvecs, self.control1.keys)

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

    # T = dt

    def inputq(q, bi, z, gamma):
        bi = gamma * bi

        fr = z*q + bi
        return fr

    def dinputq(q, bi, z, gamma):
        dfr_dq = z
        dfr_dbi = gamma*1.0
        return dfr_dq, dfr_dbi


    def radiation(f1, f1prev, b1prev, f2prev, gamma):
        f1 = gamma * f1

        _a1 = -R+L-R*L
        _a2 = -R-L + R*L

        _b1 = -R+L + R*L
        _b2 = R+L + R*L

        b1 = 1/_b2*(f1*_a2 + f1prev*_a1 + b1prev*_b1)
        f2 = 1/_b2*(f2prev*_b1 + f1*(_b2+_a2) + f1prev*(_a1-_b1))
        return f2, b1

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


    def reflect00(f1, b2, f1prev, b2prev, b1prev, f2prev, q):
        # Note that int, inp, rad refer to interior, input, and radiation junction
        # locations, respectively
        f1 = gamma1 * f1
        b2 = gamma2 * b2
        
        r1 = (z2-z1)/(z2+z1)

        f2int = (f1 + (f1-b2)*r1)[1:-1]
        b1int = (b2 + (f1-b2)*r1)[1:-1]

        ## Input boundary
        f2inp = inputq(q, b2[0], z2[0], gamma2[0]) * np.ones(1)
        b1inp = np.zeros(1)

        ## Radiation boundary
        f2rad, b1rad = radiation(f1[-1], f1prev[-1], b1prev[-1], f2prev[-1], gamma1[-1])

        f2 = np.concatenate((f2inp, f2int, f2rad))
        b1 = np.concatenate((b1inp, b1int, b1rad))
        return b1, f2

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


    def reflect05(f1, b2, gamma1, gamma2, z1, z2):
        assert f1.size == N//2
        assert b2.size == N//2
        
        f1 = gamma1 * f1
        b2 = gamma2 * b2
        r = (z2-z1)/(z2+z1)

        b1 = b2 + (f1-b2)*r
        f2 = f1 + (f1-b2)*r
        return b1, f2

    def dreflect05(f1, b2, gamma1, gamma2, z1, z2):
        r = (z2-z1)/(z2+z1)

        df2_df1 = (1.0 + r)*gamma1
        df2_db2 = (-r)*gamma2

        db1_df1 = (r)*gamma1
        db1_db2 = (1.0-r)*gamma2
        return (db1_df1, db1_db2), (df2_df1, df2_db2)
    

    def reflect(f1, b2, b1, f2, q):
        # f2 and b1 (reflected @ 0.0) -> f1, b2 (incident @ 0.5)
        f1_05 = f2[:-1]
        b2_05 = b1[1:]
        
        z1_05 = z2[:-1]
        z2_05 = z1[1:]
        
        gamma1_05 = gamma2[:-1]
        gamma2_05 = gamma1[1:]
        
        b1_05, f2_05 = reflect05(f1_05, b2_05, gamma1_05, gamma2_05, z1_05, z2_05)
        
        # f2_05 and b1_05 (reflected @ 0.5) -> f1, b2 (incident @ 1.0)
        f1inp = np.zeros(1)
        b2rad = np.zeros(1)
        f1_1 = np.concatenate((f1inp, f2_05))
        b2_1 = np.concatenate((b1_05, b2rad))
        
        b1_1, f2_1 = reflect00(f1_1, b2_1, f1, b2, b1, f2, q)
        return f1_1, b2_1, b1_1, f2_1

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

        # f1_1 = np.concatenate((f1inp, f2_05))
        # b2_1 = np.concatenate((b1_05, b2rad))
        df1_1_df2[1:] = df2_05[0]
        df1_1_db1[1:] = df2_05[1]
        
        db2_1_df2[:-1] = db1_05[0]
        db2_1_db1[:-1] = db1_05[1]
        pass

    return reflect, reflect00
