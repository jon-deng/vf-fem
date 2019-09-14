"""
This module contains definitions of various functionals.

A functional should take in the entire time history of states from a forward model run and return a
real number. Each functional has a general signature

functional(model, h5file, h5group='/', **kwargs) -> float, dict

, where h5file is an hdf5 file reference that contains the states history, h5group is the group path
in the file where the states are stored, and **kwargs are and specific keyword arguments. The
functional returns it's value as the first argument and a dictionary of additional info as a second
argument.

For computing the sensitivity of the functional through the discrete adjoint method, you also need
the sensitivity of the functional with respect to the n'th state. This function has the signature

dfunctional_du(model, n, statefile, **kwargs) -> float, dict

, where the parameters have the same uses as defined previously.
"""

import os.path as path

import numpy as np
import dolfin as dfn
import ufl

from petsc4py import PETSc

# from . import statefileutils as sfu
from . import forms

# Form definitions needed for the fluid work functional
# frm_fluidwork = ufl.dot(model.fluid_force, model.u1-model.u0) * model.ds(model.domainid_pressure)
# dfluid_work_du0 = ufl.derivative(frm_fluidwork, model.u0, model.test)
# dfluid_work_dp = ufl.derivative(frm_fluidwork, model.pressure, model.scalar_test)
# dfluid_work_du1 = ufl.derivative(frm_fluidwork, model.u1, model.test)

## Functionals defined for a single state index
def fluidwork(model, n, statefile):
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
    statefile.set_iteration_states(n, u0=model.u0, v0=model.v0, a0=model.a0, u1=model.u1)
    fluid_props = statefile.get_fluid_properties(n)
    model.set_pressure(fluid_props)

    return dfn.assemble(model.fluid_work)

def vocaleff(model, n, statefile):
    """
    Returns the vocal efficiency over the timestep from n to n+1.

    h5path : str
        Path to the file containing states from the forward run
    """
    # Set form coefficients to represent the fluidwork at index n
    statefile.set_iteration_states(n, u0=model.u0, v0=model.v0, a0=model.a0, u1=model.u1)
    fluid_props = statefile.get_fluid_properties(n)
    info = model.set_pressure(fluid_props)

    dt = float(model.dt)
    inputwork = dt*info['flow_rate']*fluid_props['p_sub']
    cost = dfn.assemble(model.fluid_work)/inputwork
    return cost

def dvocaleff_du(model, n, statefile):
    """
    Returns the derivative of the cost function with respect to u_n.

    h5path : str
        Path to the file containing states from the forward run
    """
    out = 0
    num_states = statefile.get_num_states()

    # Set form coefficients to represent step from n to n+1
    if n <= num_states-2 or n <= -2:
        ## Calculate the derivative of cost w.r.t u_n due to work from n to n+1.
        # Set the initial conditions for the forms properly
        statefile.set_iteration_states(n+1, u0=model.u0, v0=model.v0, a0=model.a0, u1=model.u1)
        statefile.set_time_step(n+1, dt=model.dt)
        fluid_props = statefile.get_fluid_properties(n)
        info = model.set_pressure(fluid_props)
        dp_du, dq_du = model.get_flow_sensitivity()

        fluidwork = dfn.assemble(model.fluid_work)

        dfluidwork_du0 = dfn.assemble(model.dfluid_work_du0)

        # Correct dfluidwork_du0 since pressure depends on u0
        dfluidwork_dp = dfn.assemble(model.dfluid_work_dp, tensor=dfn.PETScVector()).vec()

        dfluidwork_du0_correction = dfn.as_backend_type(dfluidwork_du0).vec().copy()
        dp_du.multTranspose(dfluidwork_dp, dfluidwork_du0_correction)

        dfluidwork_du0 += dfn.Vector(dfn.PETScVector(dfluidwork_du0_correction))

        dt = model.dt.values()[0]
        p_sub = fluid_props['p_sub']
        inputwork = dt*info['flow_rate']*p_sub
        dinputwork_du0 = dt*p_sub*dfn.PETScVector(dq_du)

        out += dfluidwork_du0/inputwork - fluidwork/inputwork**2 * dinputwork_du0

    # Set form coefficients to represent step from n-1 to n
    if n >= 1:
        # Set the initial conditions for the forms properly
        statefile.set_iteration_states(n, u0=model.u0, v0=model.v0, a0=model.a0, u1=model.u1)
        statefile.set_time_step(n, dt=model.dt)
        fluid_props = statefile.get_fluid_properties(n-1)
        info = model.set_pressure(fluid_props)

        fluidwork = dfn.assemble(model.fluid_work)
        dfluidwork_du1 = dfn.assemble(model.dfluid_work_du1)

        dt = model.dt.values()[0]
        inputwork = dt*info['flow_rate']*fluid_props['p_sub']

        out += dfluidwork_du1/inputwork

    return out

def fdr(model, n, statefile):
    """
    Returns the flow declination rate at n.

    This uses finite differencing
    """
    # Set form coefficients to represent the equation at state ii
    statefile.set_iteration_states(n+2, u0=model.u0, v0=model.v0, a0=model.a0, u1=model.u1)
    statefile.set_time_step(n+2, dt=model.dt)
    fluid_props = statefile.get_fluid_properties(n)
    info = model.set_pressure(fluid_props)

    t_plus = statefile.get_time(n+1)
    q_plus = info['flow_rate']

    statefile.set_iteration_states(n+1, u0=model.u0, v0=model.v0, a0=model.a0, u1=model.u1)
    statefile.set_time_step(n+1, dt=model.dt)
    fluid_props = statefile.get_fluid_properties(n)
    info = model.set_pressure(fluid_props)

    t_minus = statefile.get_time(n)
    q_minus = info['flow_rate']

    return (q_plus-q_minus)/(t_plus-t_minus)

def dfdr_du(model, n, statefile):
    """
    Returns the flow declination rate at n.

    This uses finite differencing
    """
    # Set form coefficients to represent the equation at state ii
    statefile.set_iteration_states(n+2, u0=model.u0, v0=model.v0, a0=model.a0, u1=model.u1)
    fluid_props = statefile.get_fluid_properties(n)

    q_plus = model.set_pressure(fluid_props)['flow_rate']
    dq_plus_du = model.get_flow_sensitivity()[1]
    t_plus = statefile.get_time(n+1)

    statefile.set_iteration_states(n+1, u0=model.u0, v0=model.v0, a0=model.a0, u1=model.u1)
    fluid_props = statefile.get_fluid_properties(n)

    q_minus = model.set_pressure(fluid_props)['flow_rate']
    dq_minus_du = model.get_flow_sensitivity()[1]
    t_minus = statefile.get_time(n)

    # if n

    return (dq_plus_du-dq_minus_du)/(t_plus-t_minus)

## Functionals defined over the entire state history
def totalfluidwork(model, statefile):
    """
    Returns the total work done by the fluid on the vocal folds.
    """
    res = 0
    info = {}
    num_states = statefile.get_num_states()
    for ii in range(num_states-1):
        # Set form coefficients to represent the equation at state ii
        model.set_iteration_fromfile(statefile, ii+1)

        res += dfn.assemble(model.fluid_work)

    return res, info

def dtotalfluidwork_du(model, n, statefile):
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
    info = {}
    num_states = statefile.get_num_states()

    # Set form coefficients to represent the fluidwork over n to n+1
    if n < -1 or n < num_states-1:
        model.set_iteration_fromfile(statefile, n+1)
        dp_du, _ = model.get_flow_sensitivity()

        out += dfn.assemble(model.dfluid_work_du0)

        # Correct dfluidwork_du0 since pressure depends on u0
        dfluidwork_dp = dfn.assemble(model.dfluid_work_dp, tensor=dfn.PETScVector()).vec()

        dfluidwork_du0_correction = dfn.as_backend_type(out).vec().copy()
        dp_du.multTranspose(dfluidwork_dp, dfluidwork_du0_correction)

        out += dfn.Vector(dfn.PETScVector(dfluidwork_du0_correction))

    # Set form coefficients to represent the fluidwork over n-1 to n
    if n > 0:
        model.set_iteration_fromfile(statefile, n)

        out += dfn.assemble(model.dfluid_work_du1)

    return out, info

def totalinputwork(model, statefile):
    """
    Returns the total work input into the fluid.
    """
    res = 0
    info = {}

    num_states = statefile.get_num_states()
    for ii in range(num_states-1):
        # Set form coefficients to represent the equation at state ii
        fluid_info, fluid_props = model.set_iteration_fromfile(statefile, ii+1)

        res += float(model.dt)*fluid_info['flow_rate']*fluid_props['p_sub']

    return res, info

def totalvocaleff(model, statefile):
    """
    Returns the total vocal efficiency.

    This is the ratio of the total work done by the fluid on the folds to the total input work on
    the fluid.
    """
    res = totalfluidwork(model, statefile)[0]/totalinputwork(model, statefile)[0]
    info = {}
    return res, info

def dtotalvocaleff_du(model, n, statefile, cache_totalfluidwork=None, cache_totalinputwork=None):
    """
    Returns the derivative of the total vocal efficiency w.r.t state n.

    This is the ratio of the total work done by the fluid on the folds to the total input work on
    the fluid.

    # TODO : Something is slightly wrong with this one. You can tell from comparing with FD. The
    # error is small but is not propto step size!
    """
    info = {}

    tfluidwork = cache_totalfluidwork
    if tfluidwork is None:
        tfluidwork = totalfluidwork(model, statefile)[0]

    tinputwork = cache_totalinputwork
    if tinputwork is None:
        tinputwork = totalinputwork(model, statefile)[0]

    dtotalfluidwork_du_ = dtotalfluidwork_du(model, n, statefile)[0]

    dtotalinputwork_du = 0
    # Set form coefficients to represent step from n to n+1
    num_states = statefile.get_num_states()
    if n < num_states-1 or n < -1:
        _, fluid_props = model.set_iteration_fromfile(statefile, n+1)
        _, dq_du = model.get_flow_sensitivity()

        dt = model.dt.values()[0]
        dtotalinputwork_du = dt*fluid_props['p_sub']*dq_du

    return dtotalfluidwork_du_/tinputwork - tfluidwork/tinputwork**2*dtotalinputwork_du, info

def mfdr(model, statefile, min_time=0.03):
    """
    Returns the maximum flow declination rate at a time > min_time (s).
    """
    flow_rate = []
    info = {}

    num_states = statefile.get_num_states()
    for ii in range(num_states-1):
        # Set form coefficients to represent the equation at state ii
        info, _ = model.set_iteration_fromfile(statefile, ii+1)

        flow_rate.append(info['flow_rate'])
    flow_rate = np.array(flow_rate)

    times = statefile.get_solution_times()[:-1]
    dflow_rate_dt = (flow_rate[1:]-flow_rate[:-1]) / (times[1:] - times[:-1])

    idx_start = 50 # TODO: should base this on min_time
    idx_min = np.argmin(dflow_rate_dt[idx_start:]) + idx_start

    res = dflow_rate_dt[idx_min]

    info['idx_mfdr'] = idx_min
    return res, info

def dmfdr_du(model, n, statefile, min_time=0.03, cache_idx_mfdr=None):
    """
    Returns the sensitivity of the maximum flow declination rate at a time > min_time (s).
    """
    res = None
    info = {}

    if cache_idx_mfdr is None:
        cache_idx_mfdr = mfdr(model, statefile, min_time=min_time)[1]['idx_mfdr']

    if n == cache_idx_mfdr or n == cache_idx_mfdr+1:
        # First calculate flow rates at n and n+1
        fluid_info, _ = model.set_iteration_fromfile(statefile, n+2)

        q1 = fluid_info['flow_rate']
        dq1_du = model.get_flow_sensitivity()[1]
        t1 = statefile.get_time(n+1)

        fluid_info, _ = model.set_iteration_fromfile(statefile, n+1)

        q0 = fluid_info['flow_rate']
        dq0_du = model.get_flow_sensitivity()[1]
        t0 = statefile.get_time(n)

        # fdr = (q1-q0) / (t1-t0)

        dfdr_du0 = -dq0_du / (t1-t0)
        dfdr_du1 = dq1_du / (t1-t0)

        if n == cache_idx_mfdr:
            res = dfdr_du0
        elif n == cache_idx_mfdr+1:
            res = dfdr_du1
    else:
        res = dfn.Function(model.vector_function_space).vector()

    return res, info

def wss_gwidth(model, statefile, weights=None, meas_indices=None,
               meas_glottal_widths=None):
    """
    Returns the weighted sum of squared differences between a measurement/model glottal widths.
    """
    wss = 0
    info = {}

    u = dfn.Function(model.vector_function_space)
    v = dfn.Function(model.vector_function_space)
    a = dfn.Function(model.vector_function_space)

    # Set default values when kwargs are not provided
    num_states = statefile.get_num_states()
    if weights is None:
        weights = np.ones(num_states) / num_states
    if meas_indices is None:
        meas_indices = np.arange(num_states)
    if meas_glottal_widths is None:
        meas_glottal_widths = np.zeros(num_states)

    assert meas_indices.size == meas_glottal_widths.size

    # Loop through every state
    for ii, gw_meas, weight in zip(meas_indices, meas_glottal_widths, weights):

        u, v, a = statefile.get_state(ii, function_space=model.vector_function_space)
        model.set_initial_state(u, v, a)

        # Find the maximum y coordinate on the surface
        cur_surface = model.get_surface_state()[0]
        idx_surface = np.argmax(cur_surface[:, 1])

        # Find the maximum y coordinate on the surface
        fluid_props = statefile.get_fluid_properties(0)
        gw_modl = 2 * (fluid_props['y_midline'] - cur_surface[idx_surface, 1])

        wss += weight * (gw_modl - gw_meas)**2

    return wss, info

def dwss_gwidth_du(model, n, statefile, weights=None, meas_indices=None,
                   meas_glottal_widths=None):
    """
    Returns the sensitivy of the wss difference of measurement/model glottal width w.r.t state n.
    """
    dwss_du = dfn.Function(model.vector_function_space).vector()
    info = {}

    # Set default values when kwargs are not provided
    num_states = statefile.get_num_states()
    if weights is None:
        weights = np.ones(num_states) / num_states
    if meas_indices is None:
        meas_indices = np.arange(num_states)
    if meas_glottal_widths is None:
        meas_glottal_widths = np.zeros(num_states)

    assert meas_indices.size == meas_glottal_widths.size

    # The sensitivity is only non-zero if n corresponds to a measurement index
    if n in set(meas_indices):
        weight = weights[n]
        gw_meas = meas_glottal_widths[n]

        u, v, a = statefile.get_state(n, function_space=model.vector_function_space)
        model.set_initial_state(u, v, a)

        # Find the surface vertex corresponding to where the glottal width is measured
        # This is numbered according to the 'local' numbering scheme of the surface vertices i.e.
        # 0 is the most upstream node, 1 the next node etc.
        cur_surface = model.get_surface_state()[0]
        idx_surface = np.argmax(cur_surface[:, 1])

        # Find the maximum y coordinate on the surface
        # TODO: The midline shouldn't vary but maybe it can in the future.
        fluid_props = statefile.get_fluid_properties(0)
        gw_modl = 2 * (fluid_props['y_midline'] - cur_surface[idx_surface, 1])
        dgw_modl_du_width = -2

        # Find the vertex number according to the mesh vertex numbering scheme
        idx_body = model.surface_vertices[idx_surface]

        # Finally convert it to the DOF number of the u-dof that actually influences glottal width
        dof_width = model.vert_to_vdof[idx_body, 1]

        # wss = weight * (gw_modl - gw_meas)**2
        dwss_du[dof_width] = 2*weight*(gw_modl - gw_meas)*dgw_modl_du_width
    else:
        # In this case the derivative is simply 0 so the default value is right
        pass

    return dwss_du, info

def dwss_gwidth_dt(model, n, statefile, weights=None, meas_indices=None,
                   meas_glottal_widths=None):
    """
    Returns the weighted sum of squared difference between a measurement and a model's glottal width.
    """
    dwss_dt = 0
    info = {}

    # Set default values when kwargs are not provided
    num_states = statefile.get_num_states()
    if weights is None:
        weights = np.ones(num_states) / num_states
    if meas_indices is None:
        meas_indices = np.arange(num_states)
    if meas_glottal_widths is None:
        meas_glottal_widths = np.zeros(num_states)

    assert meas_indices.size == meas_glottal_widths.size

    # Loop through every state
    for ii, gw_meas, weight in zip(meas_indices, meas_glottal_widths, weights):
        u, v, a = statefile.get_state(ii, function_space=model.vector_function_space)
        model.set_initial_state(u, v, a)

        cur_surface = model.get_surface_state()[0]

        # Find the maximum y coordinate on the surface
        idx_surface = np.argmax(cur_surface[:, 1])

        # Find the vertex number according to the mesh vertex numbering scheme
        idx_body = model.surface_vertices[idx_surface]

        # Finally convert it to the DOF number of the u-dof that actually influences glottal width
        dof_width = model.vert_to_vdof[idx_body, 1]

        # Find the maximum y coordinate on the surface
        gw_modl = 2 * (model.y_midline - cur_surface[idx_surface, 1])
        dgw_modl_dt = -2 * v[dof_width]

        wss += weight * (gw_modl - gw_meas)**2
        dwss_dt += weight * 2 * (gw_modl - gw_meas) * dgw_modl_dt

    return dwss_dt, info
