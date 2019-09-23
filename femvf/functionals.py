"""
This module contains definitions of various functionals.

A functional should take in the entire time history of states from a forward model run and return a
real number. Each functional has a general signature

functional(model, f, **kwargs) -> float, dict

, where `model` is a `ForwardModel` instance, `f` is a `StateFile` instance, and **kwargs are
specific keyword arguments. The functional returns it's value as the first argument and a dictionary
of additional info as a second argument.

For computing the sensitivity of the functional through the discrete adjoint method, you also need
the sensitivity of the functional with respect to the n'th state. This function has the signature

dfunctional_du(model, n, f, **kwargs) -> float, dict

, where the parameters have the same uses as defined previously. Parameter `n` is the state to
compute the sensitivity with respect to.
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

def totalfluidwork(model, f, **kwargs):
    """
    Returns the total work done by the fluid on the vocal folds.
    """
    res = 0
    info = {}

    n_start = kwargs.get('n_start', 0)
    num_states = f.get_num_states()
    for ii in range(n_start, num_states-1):
        # Set form coefficients to represent the equation at state ii to ii+1
        model.set_iteration_fromfile(f, ii+1)

        res += dfn.assemble(model.fluid_work)

    return res, info

def dtotalfluidwork_du(model, n, f, **kwargs):
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
    num_states = f.get_num_states()
    n_start = kwargs.get('n_start', 0)

    if n < n_start:
        out += dfn.Function(model.vector_function_space).vector()
    else:
        # Add the sensitivity component due to work from n to n+1
        if n < num_states-1:
            model.set_iteration_fromfile(f, n+1)
            dp_du, _ = model.get_flow_sensitivity()

            out += dfn.assemble(model.dfluid_work_du0)

            # Correct dfluidwork_du0 since pressure depends on u0
            dfluidwork_dp = dfn.assemble(model.dfluid_work_dp, tensor=dfn.PETScVector()).vec()

            dfluidwork_du0_correction = dfn.as_backend_type(out).vec().copy()
            dp_du.multTranspose(dfluidwork_dp, dfluidwork_du0_correction)

            out += dfn.Vector(dfn.PETScVector(dfluidwork_du0_correction))

        # Add the sensitviity component due to work from n-1 to n
        if n > n_start:
            model.set_iteration_fromfile(f, n)

            out += dfn.assemble(model.dfluid_work_du1)

    return out, info

def totalflow(model, f, **kwargs):
    """
    Returns the total volume of flow that passes through the glottis.
    """
    num_states = f.get_num_states()
    n_start = kwargs.get('n_start', 0)

    totalflow = 0
    info = {}
    for ii in range(n_start, num_states-1):
        fluid_info, _ = model.set_iteration_fromfile(f, ii+1)

        totalflow += fluid_info['flow_rate']*model.dt.values()[0]

    return totalflow, info

def dtotalflow_du(model, n, f, **kwargs):
    """
    Returns the sensitivity of total volume of flow that passes through the glottis.
    """
    dtotalflow_dun = None
    info = {}
    n_start = kwargs.get('n_start', 0)

    num_states = f.get_num_states()
    if n < n_start or n == num_states-1:
        dtotalflow_dun = dfn.Function(model.vector_function_space).vector()
    else:
        model.set_iteration_fromfile(f, n+1)
        _, dq_dun = model.get_flow_sensitivity()
        dtotalflow_dun = dq_dun * model.dt.values()[0]

    return dtotalflow_dun, info

def totalinputwork(model, f, **kwargs):
    """
    Returns the total work input into the fluid.
    """
    ret = 0
    info = {}
    n_start = kwargs.get('n_start', 0)

    num_states = f.get_num_states()
    for ii in range(n_start, num_states-1):
        # Set form coefficients to represent the equation at state ii
        fluid_info, fluid_props = model.set_iteration_fromfile(f, ii+1)

        ret += model.dt.values()[0]*fluid_info['flow_rate']*fluid_props['p_sub']

    return ret, info

def dtotalinputwork_du(model, n, f, **kwargs):
    """
    Returns the derivative of input work w.r.t state n.

    Since fluid work over both time intervals [n-1, n] and [n, n+1] involve state n, this derivative
    incorporates both timesteps.

    Parameters
    ----------
    n : int
        State index
    h5path : str
        Path to the file containing states from the forward run
    """
    ret = 0

    n_start = kwargs.get('n_start', 0)
    num_states = f.get_num_states()

    if n >= n_start and n < num_states-1:
        _, fluid_props = model.set_iteration_fromfile(f, n+1)
        _, dq_du = model.get_flow_sensitivity()

        ret += model.dt.values()[0]*fluid_props['p_sub']*dq_du
    else:
        ret += dfn.Function(model.vector_function_space).vector()

    return ret, {}

def totalvocaleff(model, f, **kwargs):
    """
    Returns the total vocal efficiency.

    This is the ratio of the total work done by the fluid on the folds to the total input work on
    the fluid.
    """
    # n_start = kwargs.get('n_start', 0)
    totalfluidwork_ = totalfluidwork(model, f, **kwargs)[0]
    totalinputwork_ = totalinputwork(model, f, **kwargs)[0]

    res = totalfluidwork_/totalinputwork_
    info = {'totalfluidwork': totalfluidwork_, 'totalinputwork': totalinputwork_}
    return res, info

def dtotalvocaleff_du(model, n, f, **kwargs):
    """
    Returns the derivative of the total vocal efficiency w.r.t state n.

    This is the ratio of the total work done by the fluid on the folds to the total input work on
    the fluid.

    # TODO : Something is slightly wrong with this one. You can tell from comparing with FD. The
    # error is small but is not propto step size!
    """
    info = {}
    n_start = kwargs.get('n_start', 0)

    tfluidwork = kwargs.get('totalfluidwork', None)
    if tfluidwork is None:
        tfluidwork = totalfluidwork(model, f, **kwargs)[0]

    tinputwork = kwargs.get('totalinputwork', None)
    if tinputwork is None:
        tinputwork = totalinputwork(model, f, **kwargs)[0]

    dtotalfluidwork_dun = dtotalfluidwork_du(model, n, f, **kwargs)[0]
    dtotalinputwork_dun = dtotalinputwork_du(model, n, f, **kwargs)[0]

    # import ipdb; ipdb.set_trace()
    if n < n_start:
        return dfn.Function(model.vector_function_space).vector(), info
    else:
        return dtotalfluidwork_dun/tinputwork - tfluidwork/tinputwork**2*dtotalinputwork_dun, info

def mfdr(model, f, **kwargs):
    """
    Returns the maximum flow declination rate at a time > min_time (s).
    """
    flow_rate = []
    info = {}

    num_states = f.get_num_states()
    for ii in range(num_states-1):
        # Set form coefficients to represent the equation at state ii
        info, _ = model.set_iteration_fromfile(f, ii+1)

        flow_rate.append(info['flow_rate'])
    flow_rate = np.array(flow_rate)

    times = f.get_solution_times()[:-1]
    dflow_rate_dt = (flow_rate[1:]-flow_rate[:-1]) / (times[1:] - times[:-1])

    idx_start = 50 # TODO: should base this on min_time
    idx_min = np.argmin(dflow_rate_dt[idx_start:]) + idx_start

    res = dflow_rate_dt[idx_min]

    info['idx_mfdr'] = idx_min
    return res, info

def dmfdr_du(model, n, f, **kwargs):
    """
    Returns the sensitivity of the maximum flow declination rate at a time > min_time (s).
    """
    res = None
    info = {}

    idx_mfdr = kwargs.get('idx_mfdr', None)
    if idx_mfdr is None:
        idx_mfdr = mfdr(model, f, **kwargs)[1]['idx_mfdr']

    if n == idx_mfdr or n == idx_mfdr+1:
        # First calculate flow rates at n and n+1
        # fluid_info, _ = model.set_iteration_fromfile(f, n+2)

        # q1 = fluid_info['flow_rate']
        dq1_du = model.get_flow_sensitivity()[1]
        t1 = f.get_time(n+1)

        # fluid_info, _ = model.set_iteration_fromfile(f, n+1)

        # q0 = fluid_info['flow_rate']
        dq0_du = model.get_flow_sensitivity()[1]
        t0 = f.get_time(n)

        dfdr_du0 = -dq0_du / (t1-t0)
        dfdr_du1 = dq1_du / (t1-t0)

        if n == idx_mfdr:
            res = dfdr_du0
        elif n == idx_mfdr+1:
            res = dfdr_du1
    else:
        res = dfn.Function(model.vector_function_space).vector()

    return res, info

def wss_gwidth(model, f, **kwargs):
    """
    Returns the weighted sum of squared differences between a measurement/model glottal widths.
    """
    wss = 0
    info = {}

    u = dfn.Function(model.vector_function_space)
    v = dfn.Function(model.vector_function_space)
    a = dfn.Function(model.vector_function_space)

    # Set default values when kwargs are not provided
    num_states = f.get_num_states()
    weights = kwargs.get('weights', np.ones(num_states) / num_states)
    meas_indices = kwargs.get('meas_indices', np.arange(num_states))
    meas_glottal_widths = kwargs.get('meas_glottal_widths', np.zeros(num_states))

    assert meas_indices.size == meas_glottal_widths.size

    # Loop through every state
    for ii, gw_meas, weight in zip(meas_indices, meas_glottal_widths, weights):

        u, v, a = f.get_state(ii, function_space=model.vector_function_space)
        model.set_initial_state(u, v, a)

        # Find the maximum y coordinate on the surface
        cur_surface = model.get_surface_state()[0]
        idx_surface = np.argmax(cur_surface[:, 1])

        # Find the maximum y coordinate on the surface
        fluid_props = f.get_fluid_properties(0)
        gw_modl = 2 * (fluid_props['y_midline'] - cur_surface[idx_surface, 1])

        wss += weight * (gw_modl - gw_meas)**2

    return wss, info

def dwss_gwidth_du(model, n, f, **kwargs):
    """
    Returns the sensitivy of the wss difference of measurement/model glottal width w.r.t state n.
    """
    dwss_du = dfn.Function(model.vector_function_space).vector()
    info = {}

    # Set default values when kwargs are not provided
    num_states = f.get_num_states()
    weights = kwargs.get('weights', np.ones(num_states) / num_states)
    meas_indices = kwargs.get('meas_indices', np.arange(num_states))
    meas_glottal_widths = kwargs.get('meas_glottal_widths', np.zeros(num_states))

    assert meas_indices.size == meas_glottal_widths.size

    # The sensitivity is only non-zero if n corresponds to a measurement index
    if n in set(meas_indices):
        weight = weights[n]
        gw_meas = meas_glottal_widths[n]

        u, v, a = f.get_state(n, function_space=model.vector_function_space)
        model.set_initial_state(u, v, a)

        # Find the surface vertex corresponding to where the glottal width is measured
        # This is numbered according to the 'local' numbering scheme of the surface vertices i.e.
        # 0 is the most upstream node, 1 the next node etc.
        cur_surface = model.get_surface_state()[0]
        idx_surface = np.argmax(cur_surface[:, 1])

        # Find the maximum y coordinate on the surface
        # TODO: The midline shouldn't vary but maybe it can in the future.
        fluid_props = f.get_fluid_properties(0)
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

def dwss_gwidth_dt(model, n, f, **kwargs):
    """
    Returns the weighted sum of squared differences between a measurement / model's glottal widths.
    """
    dwss_dt = 0
    info = {}

    # Set default values when kwargs are not provided
    num_states = f.get_num_states()
    weights = kwargs.get('weights', np.ones(num_states) / num_states)
    meas_indices = kwargs.get('meas_indices', np.arange(num_states))
    meas_glottal_widths = kwargs.get('meas_glottal_widths', np.zeros(num_states))

    assert meas_indices.size == meas_glottal_widths.size

    # Loop through every state
    for ii, gw_meas, weight in zip(meas_indices, meas_glottal_widths, weights):
        u, v, a = f.get_state(ii, function_space=model.vector_function_space)
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
