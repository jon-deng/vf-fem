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
