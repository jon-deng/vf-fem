"""
Contains a bunch of different functionals
"""

import os.path as path

import numpy as np
import dolfin as dfn
import ufl

from petsc4py import PETSc

from . import statefileutils as sfu
from . import forms as frm

# Form definitions needed for the fluid work functional
frm_fluidwork = ufl.dot(frm.fluid_force, frm.u1-frm.u0) * frm.ds(frm.domainid_pressure)
frm_dfluidwork_du0 = ufl.derivative(frm_fluidwork, frm.u0, frm.test)
frm_dfluidwork_dp = ufl.derivative(frm_fluidwork, frm.pressure, frm.scalar_test)
frm_dfluidwork_du1 = ufl.derivative(frm_fluidwork, frm.u1, frm.test)

## Functionals defined for a single state index
def fluidwork(n, h5file, h5group='/'):
    """
    Returns the fluid work from n-1 to n.

    Parameters
    ----------
    n : int
        State index
    h5path : str
        Path to the file containing states from the forward run
    """
    # Set form coefficients to represent the fluidwork at index n
    sfu.set_form_states(n, h5file, h5group=h5group, u0=frm.u0, v0=frm.v0, a0=frm.a0, u1=frm.u1)
    fluid_props = sfu.get_fluid_properties(n, h5file, group=h5group)
    frm.set_pressure(fluid_props)

    return dfn.assemble(frm_fluidwork)

def dfluidwork_du(n, h5file, h5group='/'):
    """
    Returns the derivative of fluid work w.r.t state n.

    Since fluid work over both time intervals [n-1, n] and [n, n+1] involve state n, this derivative
    incorporates both timesteps.

    Parameters
    ----------
    n : int
        State index
    h5path : str
        Path to the file containing states from the forward run
    """
    out = 0
    num_states = sfu.get_num_states(h5file, group=h5group)

    # Set form coefficients to represent the fluidwork over n to n+1
    if n < -1 or n < num_states-1:
        sfu.set_states(n+1, h5file, group=h5group, u0=frm.u0, v0=frm.v0, a0=frm.a0, u1=frm.u1)
        fluid_props = sfu.get_fluid_properties(n, h5file, group=h5group)
        frm.set_pressure(fluid_props)
        dp_du, _ = frm.set_flow_sensitivity(fluid_props)

        out += dfn.assemble(frm_dfluidwork_du0)

        # Correct dfluidwork_du0 since pressure depends on u0
        dfluidwork_dp = dfn.assemble(frm_dfluidwork_dp, tensor=dfn.PETScVector()).vec()

        dfluidwork_du0_correction = dfn.as_backend_type(out).vec().copy()
        dp_du.multTranspose(dfluidwork_dp, dfluidwork_du0_correction)

        out += dfn.Vector(dfn.PETScVector(dfluidwork_du0_correction))

    # Set form coefficients to represent the fluidwork over n-1 to n
    if n > 0:
        sfu.set_states(n, h5file, group=h5group, u0=frm.u0, v0=frm.v0, a0=frm.a0, u1=frm.u1)
        fluid_props = sfu.get_fluid_properties(n-1, h5file, group=h5group)
        frm.set_pressure(fluid_props)

        out += dfn.assemble(frm_dfluidwork_du1)

    return out

def vocaleff(n, h5file, h5group='/'):
    """
    Returns the vocal efficiency over the timestep from n to n+1.

    h5path : str
        Path to the file containing states from the forward run
    """
    # Set form coefficients to represent the fluidwork at index n
    sfu.set_states(n, h5file, group=h5group, u0=frm.u0, v0=frm.v0, a0=frm.a0, u1=frm.u1)
    fluid_props = sfu.get_fluid_properties(n, h5file, group=h5group)
    info = frm.set_pressure(fluid_props)

    dt = float(frm.dt)
    inputwork = dt*info['flow_rate']*fluid_props['p_sub']
    cost = dfn.assemble(frm_fluidwork)/inputwork
    return cost

def dvocaleff_du(n, h5file, h5group='/'):
    """
    Returns the derivative of the cost function with respect to u_n.

    h5path : str
        Path to the file containing states from the forward run
    """
    out = 0
    num_states = sfu.get_num_states(h5file, group=h5group)

    # Set form coefficients to represent step from n to n+1
    if n <= num_states-2 or n <= -2:
        ## Calculate the derivative of cost w.r.t u_n due to work from n to n+1.
        # Set the initial conditions for the forms properly
        sfu.set_states(n+1, h5file, group=h5group, u0=frm.u0, v0=frm.v0, a0=frm.a0, u1=frm.u1)
        fluid_props = sfu.get_fluid_properties(n, h5file, group=h5group)
        info = frm.set_pressure(fluid_props)
        dp_du, dq_du = frm.set_flow_sensitivity(fluid_props)

        fluidwork = dfn.assemble(frm_fluidwork)

        dfluidwork_du0 = dfn.assemble(frm_dfluidwork_du0)

        # Correct dfluidwork_du0 since pressure depends on u0
        dfluidwork_dp = dfn.assemble(frm_dfluidwork_dp, tensor=dfn.PETScVector()).vec()

        dfluidwork_du0_correction = dfn.as_backend_type(dfluidwork_du0).vec().copy()
        dp_du.multTranspose(dfluidwork_dp, dfluidwork_du0_correction)

        dfluidwork_du0 += dfn.Vector(dfn.PETScVector(dfluidwork_du0_correction))

        dt = frm.dt.values()[0]
        p_sub = fluid_props['p_sub']
        inputwork = dt*info['flow_rate']*p_sub
        dinputwork_du0 = dt*p_sub*dfn.PETScVector(dq_du)

        out += dfluidwork_du0/inputwork - fluidwork/inputwork**2 * dinputwork_du0

    # Set form coefficients to represent step from n-1 to n
    if n >= 1:
        # Set the initial conditions for the forms properly
        sfu.set_states(n, h5file, group=h5group, u0=frm.u0, v0=frm.v0, a0=frm.a0, u1=frm.u1)
        fluid_props = sfu.get_fluid_properties(n-1, h5file, group=h5group)
        info = frm.set_pressure(fluid_props)

        fluidwork = dfn.assemble(frm_fluidwork)
        dfluidwork_du1 = dfn.assemble(frm_dfluidwork_du1)

        dt = frm.dt.values()[0]
        inputwork = dt*info['flow_rate']*fluid_props['p_sub']

        out += dfluidwork_du1/inputwork

    return out

def fdr(n, h5file, h5group='/'):
    """
    Returns the flow declination rate at n.

    This uses finite differencing
    """
    # Set form coefficients to represent the equation at state ii
    sfu.set_states(n+2, h5file, group=h5group, u0=frm.u0, v0=frm.v0, a0=frm.a0, u1=frm.u1)
    fluid_props = sfu.get_fluid_properties(n, h5file, group=h5group)
    info = frm.set_pressure(fluid_props)

    t_plus = sfu.get_time(n+1, h5file, group=h5group)
    q_plus = info['flow_rate']

    sfu.set_states(n+1, h5file, group=h5group, u0=frm.u0, v0=frm.v0, a0=frm.a0, u1=frm.u1)
    fluid_props = sfu.get_fluid_properties(n, h5file, group=h5group)
    info = frm.set_pressure(fluid_props)

    t_minus = sfu.get_time(n, h5file, group=h5group)
    q_minus = info['flow_rate']

    return (q_plus-q_minus)/(t_plus-t_minus)

def dfdr_du(n, h5file, h5group='/'):
    """
    Returns the flow declination rate at n.

    This uses finite differencing
    """
    # Set form coefficients to represent the equation at state ii
    sfu.set_states(n+2, h5file, group=h5group, u0=frm.u0, v0=frm.v0, a0=frm.a0, u1=frm.u1)
    fluid_props = sfu.get_fluid_properties(n, h5file, group=h5group)

    q_plus = frm.set_pressure(fluid_props)['flow_rate']
    dq_plus_du = frm.set_flow_sensitivity(fluid_props)[1]
    t_plus = sfu.get_time(n+1, h5file, group=h5group)

    sfu.set_states(n+1, h5file, group=h5group, u0=frm.u0, v0=frm.v0, a0=frm.a0, u1=frm.u1)
    fluid_props = sfu.get_fluid_properties(n, h5file, group=h5group)

    q_minus = frm.set_pressure(fluid_props)['flow_rate']
    dq_minus_du = frm.set_flow_sensitivity(fluid_props)[1]
    t_minus = sfu.get_time(n, h5file, group=h5group)

    # if n

    return (dq_plus_du-dq_minus_du)/(t_plus-t_minus)

## Functionals defined over the entire state history
def totalfluidwork(n, h5file, h5group='/'):
    """
    Returns the total work done by the fluid on the vocal folds.
    """
    res = 0
    num_states = sfu.get_num_states(h5file, group=h5group)
    for ii in range(num_states-1):
        # Set form coefficients to represent the equation at state ii
        sfu.set_states(ii+1, h5file, group=h5group, u0=frm.u0, v0=frm.v0, a0=frm.a0, u1=frm.u1)
        fluid_props = sfu.get_fluid_properties(ii, h5file, group=h5group)
        frm.set_pressure(fluid_props)

        res += dfn.assemble(frm_fluidwork)

    return res

def totalinputwork(n, h5file, h5group='/'):
    """
    Returns the total work input into the fluid.
    """
    res = 0
    num_states = sfu.get_num_states(h5file, group=h5group)
    for ii in range(num_states-1):
        # Set form coefficients to represent the equation at state ii
        sfu.set_states(ii+1, h5file, group=h5group, u0=frm.u0, v0=frm.v0, a0=frm.a0, u1=frm.u1)
        fluid_props = sfu.get_fluid_properties(ii, h5file, group=h5group)
        info = frm.set_pressure(fluid_props)

        res += float(frm.dt)*info['flow_rate']*fluid_props['p_sub']

    return res

def totalvocaleff(n, h5file, h5group='/'):
    """
    Returns the total vocal efficiency.

    This is the ratio of the total work done by the fluid on the folds to the total input work on
    the fluid.
    """
    return totalfluidwork(n, h5file, h5group=h5group)/totalinputwork(n, h5file, h5group=h5group)

    # return totalfluidwork/totalinputwork

def dtotalvocaleff_du(n, h5file, h5group='/', cache_totalfluidwork=None, cache_totalinputwork=None):
    """
    Returns the derivative of the total vocal efficiency w.r.t state n.

    This is the ratio of the total work done by the fluid on the folds to the total input work on
    the fluid.
    """
    tfluidwork = cache_totalfluidwork
    if tfluidwork is None:
        tfluidwork = totalfluidwork(n, h5file, h5group=h5group)

    tinputwork = cache_totalinputwork
    if tinputwork is None:
        tinputwork = totalinputwork(n, h5file, h5group=h5group)

    dtotalfluidwork_du = dfluidwork_du(n, h5file, h5group=h5group)

    dtotalinputwork_du = 0
    # Set form coefficients to represent step from n to n+1
    num_states = sfu.get_num_states(h5file, group=h5group)
    if n < num_states-1 or n < -1:
        sfu.set_states(n+1, h5file, group=h5group, u0=frm.u0, v0=frm.v0, a0=frm.a0, u1=frm.u1)
        fluid_props = sfu.get_fluid_properties(n, h5file, group=h5group)
        _, dq_du = frm.set_flow_sensitivity(fluid_props)

        dt = frm.dt.values()[0]
        dtotalinputwork_du = dt*fluid_props['p_sub']*dfn.PETScVector(dq_du)

    return dtotalfluidwork_du/tinputwork - tfluidwork/tinputwork**2*dtotalinputwork_du

def mfdr(n, h5file, h5group='/', min_time=0.03):
    """
    Returns the maximum flow declination rate at a time > min_time (s).
    """
    flow_rate = []

    num_states = sfu.get_num_states(h5file, group=h5group)
    for ii in range(num_states-1):
        # Set form coefficients to represent the equation at state ii
        sfu.set_states(ii+1, h5file, group=h5group, u0=frm.u0, v0=frm.v0, a0=frm.a0, u1=frm.u1)
        fluid_props = sfu.get_fluid_properties(ii, h5file, group=h5group)
        info = frm.set_pressure(fluid_props)

        flow_rate.append(info['flow_rate'])
    flow_rate = np.array(flow_rate)

    times = sfu.get_times(h5file, group=h5group)[:-1]
    dflow_rate_dt = (flow_rate[1:]-flow_rate[:-1]) / (times[1:] - times[:-1])

    idx_start = 50 # TODO: should base this on min_time
    idx_min = np.argmin(dflow_rate_dt[idx_start:]) + idx_start

    res = dflow_rate_dt[idx_min]

    info['idx_mfdr'] = idx_min
    return res, info

def dmfdr_du(n, h5file, h5group='/', min_time=0.03, cache_idx_mfdr=None):
    """
    Returns the sensitivity of the maximum flow declination rate at a time > min_time (s).
    """

    # First make a petsc vector
    res = None

    if cache_idx_mfdr is None:
        cache_idx_mfdr = mfdr(n, h5file, h5group=h5group, min_time=min_time)[1]['idx_mfdr']

    if n == cache_idx_mfdr or n == cache_idx_mfdr+1:
        # First calculate flow rates at n and n+1
        sfu.set_states(n+2, h5file, group=h5group, u0=frm.u0, v0=frm.v0, a0=frm.a0, u1=frm.u1)
        fluid_props = sfu.get_fluid_properties(n, h5file, group=h5group)

        q1 = frm.set_pressure(fluid_props)['flow_rate']
        dq1_du = frm.set_flow_sensitivity(fluid_props)[1]
        t1 = sfu.get_time(n+1, h5file, group=h5group)

        sfu.set_states(n+1, h5file, group=h5group, u0=frm.u0, v0=frm.v0, a0=frm.a0, u1=frm.u1)
        fluid_props = sfu.get_fluid_properties(n, h5file, group=h5group)

        q0 = frm.set_pressure(fluid_props)['flow_rate']
        dq0_du = frm.set_flow_sensitivity(fluid_props)[1]
        t0 = sfu.get_time(n, h5file, group=h5group)

        # fdr = (q1-q0) / (t1-t0)

        dfdr_du0 = -dfn.PETScVector(dq0_du) / (t1-t0)
        dfdr_du1 = dfn.PETScVector(dq1_du) / (t1-t0)

        if n == cache_idx_mfdr:
            res = dfdr_du0
        elif n == cache_idx_mfdr+1:
            res = dfdr_du1
    else:
        ndof = h5file[path.join(h5group, 'u')].shape[1]
        res = PETSc.Vec().create(PETSc.COMM_SELF).createSeq(ndof)

        res.setValues(ndof, 0)
        res.assemblyBegin()
        res.assemblyEnd()

        res = dfn.PETScVector(res)

    return res
