"""
Contains a bunch of different functionals
"""

import dolfin as dfn
import ufl

import statefileutils as sfu
import forms as frm

# Form definitions needed for the fluid work functional
frm_fluidwork = ufl.dot(frm.fluid_force, frm.u1-frm.u0) * frm.ds(frm.domainid_pressure)
frm_dfluidwork_du0 = ufl.derivative(frm_fluidwork, frm.u0, frm.test)
frm_dfluidwork_dp = ufl.derivative(frm_fluidwork, frm.pressure, frm.scalar_test)
frm_dfluidwork_du1 = ufl.derivative(frm_fluidwork, frm.u1, frm.test)

def fluidwork(n, h5path, fluid_props, h5group='/'):
    """
    Returns the fluid work over the timestep from n to n+1.

    Parameters
    ----------
    n : int
        State index
    h5path : str
        Path to the file containing states from the forward run
    """
    # Set form coefficients to represent the fluidwork at index n
    sfu.set_form_states(n, h5path, h5group=h5group, u0=frm.u0, v0=frm.v0, a0=frm.a0, u1=frm.u1)
    frm.set_pressure(fluid_props)

    return dfn.assemble(frm_fluidwork)

def dfluidwork_du(n, h5path, fluid_props, h5group='/'):
    """
    Returns the derivative of fluid work w.r.t state n.

    Parameters
    ----------
    n : int
        State index
    h5path : str
        Path to the file containing states from the forward run
    """
    out = 0
    num_states = sfu.get_num_states(h5path, group=h5group)

    # Set form coefficients to represent the fluidwork over n to n+1
    if n <= -2 or n <= num_states-2:
        sfu.set_states(n+1, h5path, group=h5group, u0=frm.u0, v0=frm.v0, a0=frm.a0, u1=frm.u1)
        frm.set_pressure(fluid_props)
        dp_du, _ = frm.set_flow_sensitivity(fluid_props)

        out += dfn.assemble(frm_dfluidwork_du0)

        # Correct dfluidwork_du0 since pressure depends on u0
        dfluidwork_dp = dfn.assemble(frm_dfluidwork_dp, tensor=dfn.PETScVector()).vec()

        dfluidwork_du0_correction = dfn.as_backend_type(out).vec().copy()
        dp_du.multTranspose(dfluidwork_dp, dfluidwork_du0_correction)

        out += dfn.Vector(dfn.PETScVector(dfluidwork_du0_correction))

    # Set form coefficients to represent the fluidwork over n-1 to n
    if n >= 1:
        sfu.set_states(n, h5path, group=h5group, u0=frm.u0, v0=frm.v0, a0=frm.a0, u1=frm.u1)
        frm.set_pressure(fluid_props)

        out += dfn.assemble(frm_dfluidwork_du1)

    return out

def vocaleff(n, h5path, h5group='/'):
    """
    Returns the vocal efficiency over the timestep from n to n+1.

    h5path : str
        Path to the file containing states from the forward run
    """
    # Set form coefficients to represent the fluidwork at index n
    sfu.set_states(n, h5path, group=h5group, u0=frm.u0, v0=frm.v0, a0=frm.a0, u1=frm.u1)
    fluid_props = sfu.get_fluid_properties(n, h5path, group=h5group)
    info = frm.set_pressure(fluid_props)

    dt = float(frm.dt)
    inputwork = dt*info['flow_rate']*fluid_props['p_sub']
    cost = dfn.assemble(frm_fluidwork)/inputwork
    return cost

def dvocaleff_du(n, h5path, h5group='/'):
    """
    Returns the derivative of the cost function with respect to u_n.

    h5path : str
        Path to the file containing states from the forward run
    """
    out = 0
    num_states = sfu.get_num_states(h5path, group=h5group)

    # Set form coefficients to represent step from n to n+1
    if n <= num_states-2 or n <= -2:
        ## Calculate the derivative of cost w.r.t u_n due to work from n to n+1.
        # Set the initial conditions for the forms properly
        sfu.set_states(n+1, h5path, group=h5group, u0=frm.u0, v0=frm.v0, a0=frm.a0, u1=frm.u1)
        fluid_props = sfu.get_fluid_properties(n, h5path, group=h5group)
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
        sfu.set_states(n, h5path, group=h5group, u0=frm.u0, v0=frm.v0, a0=frm.a0, u1=frm.u1)
        fluid_props = sfu.get_fluid_properties(n-1, h5path, group=h5group)
        info = frm.set_pressure(fluid_props)

        fluidwork = dfn.assemble(frm_fluidwork)
        dfluidwork_du1 = dfn.assemble(frm_dfluidwork_du1)

        dt = frm.dt.values()[0]
        inputwork = dt*info['flow_rate']*fluid_props['p_sub']

        out += dfluidwork_du1/inputwork

    return out

def totalfluidwork(n, h5path, h5group='/'):
    """
    Returns the total work done by the fluid.
    
    This is the ratio of the total work done by the fluid on the folds to the total input work on
    the fluid.
    """
    totalfluidwork = 0
    num_states = sfu.get_num_states(h5path, group=h5group)
    for ii in range(num_states-1):
        # Set form coefficients to represent the equation at state ii
        sfu.set_states(ii+1, h5path, group=h5group, u0=frm.u0, v0=frm.v0, a0=frm.a0, u1=frm.u1)
        fluid_props = sfu.get_fluid_properties(ii, h5path, group=h5group)
        info = frm.set_pressure(fluid_props)

        totalfluidwork += dfn.assemble(frm_fluidwork)

    return totalfluidwork

def totalvocaleff(n, h5path, h5group='/'):
    """
    Returns the total vocal efficiency.
    
    This is the ratio of the total work done by the fluid on the folds to the total input work on
    the fluid.
    """
    totalfluidwork = 0
    totalinputwork = 0
    num_states = sfu.get_num_states(h5path, group=h5group)
    for ii in range(num_states-1):
        # Set form coefficients to represent the equation at state ii
        sfu.set_states(ii+1, h5path, group=h5group, u0=frm.u0, v0=frm.v0, a0=frm.a0, u1=frm.u1)
        fluid_props = sfu.get_fluid_properties(ii, h5path, group=h5group)
        info = frm.set_pressure(fluid_props)

        dt = float(frm.dt)
        totalinputwork += dt*info['flow_rate']*fluid_props['p_sub']
        totalfluidwork += dfn.assemble(frm_fluidwork)

    return totalfluidwork/totalinputwork

def dtotalvocaleff_du(n, h5path, h5group='/'):
    totalfluidwork = 0
    totalinputwork = 0
    num_states = sfu.get_num_states(h5path, group=h5group)
    for ii in range(num_states-1):
        # Set form coefficients to represent the equation at state ii
        sfu.set_states(ii+1, h5path, group=h5group, u0=frm.u0, v0=frm.v0, a0=frm.a0, u1=frm.u1)
        fluid_props = sfu.get_fluid_properties(ii, h5path, group=h5group)
        info = frm.set_pressure(fluid_props)

        dt = float(frm.dt)
        totalinputwork += dt*info['flow_rate']*fluid_props['p_sub']
        totalfluidwork += dfn.assemble(frm_fluidwork)

    return dtotalfluidwork_du/totalinputwork - totalfluidwork/totalinputwork**2*dtotalinputwork_du

