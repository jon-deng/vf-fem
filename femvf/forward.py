"""
Forward model

Uses CGS (cm-g-s) units unless otherwise stated
"""

from functools import partial

import numpy as np
import dolfin as dfn
from petsc4py import PETSc

from . import statefile as sf
from . import linalg

# TODO: Allow negative indexes in get functions (negative indexes index in reverse order)
# @profile
def integrate( 
    model, f, ini_state, controls, props, times, 
    idx_meas=None, newton_solver_prm=None, write=True
    ):
    """
    Integrate the model over each time in `times` for the specified parameters

    Parameters
    ----------
    model : model.ForwardModel
    f : sf.StateFile
    ini_state : BlockVec
        Initial state of the system (for example: displacement, velocity, acceleration)
    controls : list(BlockVec)
        List of control BlockVec with on entry for each integration time. If there is only one 
        control in the list, then the controls are considered to be constant in time.
    props : BlockVec
        Properties vector for the system
    times : BlockVec
        Array of discrete integration times. Each time point is one integration point so the time
        between successive time points is a time step.
    idx_meas : np.ndarray
        Array of integers marking which integration points correspond to something (usu. measurements)
    h5file : str
        Path to h5 file where states should be saved

    Returns
    -------
    fin_state
    info : dict
        Any exported quantites are contained in here
    """
    if idx_meas is None:
        idx_meas = np.array([])

    # Check integration times are specified correctly
    times_bvec = times
    times = times_bvec[0]
    if times[-1] <= times[0]:
        raise ValueError("The final time point must be greater than the initial one."
                         f"The input intial/final times were {times[0]}/{times[-1]}")
    if times.size <= 1:
        raise ValueError("There must be at least 2 time integration points.")

    # Initialize datasets and save the initial state to the h5 file
    if write:
        f.init_layout()
        append_step_result(f, ini_state, controls[0], times[0], 
                           {'num_iter': 0, 'abs_err': 0, 'rel_err': 0})
        f.append_properties(props)
        if 0 in idx_meas:
            f.append_meas_index(0)

    # Integrate the system over the specified times and record final state for each step
    # model.set_ini_state(ini_state)
    # model.set_properties(props)
    _controls = controls[1:] if len(controls) > 1 else controls
    fin_state, step_info = integrate_steps(
        model, f, ini_state, _controls, props, times, 
        idx_meas=idx_meas, newton_solver_prm=newton_solver_prm, write=write)

    return fin_state, step_info

def integrate_extend(
    model, f, controls, times,
    idx_meas=None, newton_solver_prm=None, write=True):
    props = f.get_properties()
    _controls = controls[1:] if len(controls) > 1 else controls

    N = f.size
    ini_state = f.get_state(N-1)
    ini_time = f.get_time(N-1)
    times += ini_time
    
    fin_state, step_info = integrate_steps(
        model, f, ini_state, _controls, props, times.vecs[0],
        idx_meas=idx_meas, newton_solver_prm=newton_solver_prm, write=write)
    return fin_state, step_info

def integrate_steps(
    model, f, ini_state, controls, props, times, 
    idx_meas=None, newton_solver_prm=None, write=True):
    """
    
    """
    # Setting the properties is mandatory because they are constant for each
    # time step (only need to set it once at the beginnning)
    state0 = ini_state
    model.set_properties(props)
    for n in range(1, times.size):
        control1 = controls[min(n, len(controls)-1)]
        dt = times[n] - times[n-1]
        
        state1, step_info = integrate_step(model, state0, control1, props, dt)

        # Write the solution outputs to the h5 file
        if write:
            append_step_result(f, state1, control1, times[n], step_info)
            if n in idx_meas:
                f.append_meas_index(n)

        # Update initial conditions for the next time step
        state0 = state1

    return state1, step_info

def integrate_linear(model, f, dini_state, dcontrols, dprops, dtimes):
    """
    Integrate the linearized forward equations and return the final linearized state

    Returns
    -------
    dfin_state : linalg.BlockVec
    """
    model.set_properties(f.get_properties())

    dfin_state_n = dini_state
    ts = f.get_times()
    for n in range(1, f.size):
        # Set the linearization point
        # This represents the residual F^n
        model.set_ini_state(f.get_state(n-1))
        model.set_fin_state(f.get_state(n))
        model.set_control(f.get_control(n))
        model.dt = ts[n] - ts[n-1]

        # Compute the action of the n'th time step
        # note that the input "dx^{n-1}" vector is the previous output "dx"
        _dini_state = dfin_state_n
        _dcontrol = dcontrols[min(n, len(dcontrols)-1)]
        _ddt = dtimes[0][n]-dtimes[0][n-1]
        dres_n = (model.apply_dres_dstate0(_dini_state) 
                  + model.apply_dres_dcontrol(_dcontrol)
                  + model.apply_dres_dp(dprops)
                  + model.apply_dres_ddt(_ddt))
        dfin_state_n = model.solve_dres_dstate1(-dres_n)

    return dfin_state_n


def integrate_step(model, ini_state, control, props, dt, set_props=False):
    """
    Integrate a model over a single time step
    """
    model.dt = dt
    model.set_ini_state(ini_state)
    model.set_control(control)
    if set_props:
        model.set_properties(props)
    
    fin_state, step_info = model.solve_state1(ini_state)
    return fin_state, step_info

def append_step_result(f, state, control, time, step_info):
    f.append_state(state)
    f.append_control(control)
    f.append_time(time)
    f.append_solver_info(step_info)