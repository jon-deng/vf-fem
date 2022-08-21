"""
Forward model

Uses CGS (cm-g-s) units unless otherwise stated
"""
from typing import List, Optional, Mapping, Any, Tuple
from tqdm import tqdm

import numpy as np
from blockarray import blockvec as bv

from .models.transient.base import BaseTransientModel
from . import statefile as sf

# TODO: Allow negative indexes in get functions (negative indexes index in reverse order)
# @profile
Options = Mapping[str, Any]
Info = Options
def integrate(
        model: BaseTransientModel,
        f: sf.StateFile,
        ini_state: bv.BlockVector,
        controls: List[bv.BlockVector],
        props: bv.BlockVector,
        times: bv.BlockVector,
        idx_meas: Optional[np.ndarray]=None,
        newton_solver_prm: Optional[Mapping[str, Any]]=None,
        write: bool=True,
        use_tqdm: bool=False
    ) -> Tuple[bv.BlockVector, Mapping[str, Any]]:
    """
    Integrate the model over each time in `times` for the specified parameters

    Parameters
    ----------
    model : model.ForwardModel
    f : sf.StateFile
    ini_state : BlockVector
        Initial state of the system (for example: displacement, velocity, acceleration)
    controls : list(BlockVector)
        List of control BlockVector with on entry for each integration time. If there is only one
        control in the list, then the controls are considered to be constant in time.
    props : BlockVector
        Properties vector for the system
    times : BlockVector
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
    times_vec = times[0]
    if times_vec[-1] < times_vec[0]:
        raise ValueError("The final time point must be greater or equal to the initial one."
                         f"The input intial/final times were {times_vec[0]}/{times_vec[-1]}")
    if times_vec.size < 1:
        raise ValueError("There must be at least 1 time integration points.")

    # Initialize datasets and save the initial state to the h5 file
    if write:
        f.init_layout()
        append_step_result(
            f, ini_state, controls[0], times_vec[0],
            {'num_iter': 0, 'abs_err': 0, 'rel_err': 0}
        )
        f.append_properties(props)
        if 0 in idx_meas:
            f.append_meas_index(0)

    # Integrate the system over the specified times and record final state for each step
    # model.set_ini_state(ini_state)
    # model.set_props(props)
    fin_state, step_info = integrate_steps(
        model, f, ini_state, controls, props, times_vec,
        idx_meas=idx_meas, newton_solver_prm=newton_solver_prm, write=write,
        use_tqdm=use_tqdm
    )

    return fin_state, step_info

def integrate_extend(
        model: BaseTransientModel,
        f: sf.StateFile,
        controls: bv.BlockVector,
        times: bv.BlockVector,
        idx_meas: np.ndarray=None,
        newton_solver_prm: Optional[Options]=None,
        write: bool=True
    ) -> Tuple[bv.BlockVector, Info]:
    """
    See `integrate`
    """
    props = f.get_props()
    _controls = controls[1:] if len(controls) > 1 else controls

    N = f.size
    ini_state = f.get_state(N-1)
    ini_time = f.get_time(N-1)
    times += ini_time

    fin_state, step_info = integrate_steps(
        model, f, ini_state, _controls, props, times.vecs[0],
        idx_meas=idx_meas, newton_solver_prm=newton_solver_prm, write=write
    )
    return fin_state, step_info

def integrate_steps(
        model: BaseTransientModel,
        f: sf.StateFile,
        ini_state: bv.BlockVector,
        controls: List[bv.BlockVector],
        props: bv.BlockVector,
        times: bv.BlockVector,
        idx_meas: Optional[np.ndarray]=None,
        newton_solver_prm: Optional[Mapping[str, Any]]=None,
        write: bool=True,
        use_tqdm: bool=False
    ) -> Tuple[bv.BlockVector, Mapping[str, Any]]:
    """
    See `integrate`
    """
    if idx_meas is None:
        idx_meas = np.array([])

    # Setting the properties is mandatory because they are constant for each
    # time step (only need to set it once at the beginnning)
    state0 = ini_state
    model.set_props(props)
    step_info = {}

    time_indices = tqdm(range(1, times.size)) if use_tqdm else range(1, times.size)
    for n in time_indices:
        control1 = controls[min(n, len(controls)-1)]
        dt = times[n] - times[n-1]

        state1, step_info = integrate_step(
            model, state0, control1, props, dt, options=newton_solver_prm
        )
        # if n%100 == 0:
        #     print(step_info['num_iter'])
        # Write the solution outputs to the h5 file
        if write:
            append_step_result(f, state1, control1, times[n], step_info)
            if n in idx_meas:
                f.append_meas_index(n)

        # Update initial conditions for the next time step
        state0 = state1

    return state0, step_info

def integrate_linear(
        model: BaseTransientModel,
        f: sf.StateFile,
        dini_state: bv.BlockVector,
        dcontrols: List[bv.BlockVector],
        dprops: bv.BlockVector,
        dtimes: bv.BlockVector
    ) -> bv.BlockVector:
    """
    Integrate linearized forward equations

    The integration is done about a sequence of states obtained from integrating
    the model forward in time. This sequence of states is stored in the `f`
    instance.

    Parameters
    ----------
    model :
        The transient model to integrate
    f :
        A statefile containing a history of states to linearize about
    dini_state, dcontrols, dprops :
        The linear perturbation in initial state, controls, and properties
    dtimes :
        The linear perturbation in times

    Returns
    -------
    dfin_state : vec.BlockVector
    """
    model.set_props(f.get_props())

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


def integrate_step(
        model: BaseTransientModel,
        ini_state: bv.BlockVector,
        control: bv.BlockVector,
        props: bv.BlockVector,
        dt: float,
        set_props: bool=False,
        options: Options=None
    ) -> Tuple[bv.BlockVector, Mapping[str, Any]]:
    """
    Integrate a model over a single time step

    See `integrate` for more details
    """
    model.dt = dt
    model.set_ini_state(ini_state)
    model.set_control(control)
    if set_props:
        model.set_props(props)

    fin_state, step_info = model.solve_state1(ini_state, options=options)
    return fin_state, step_info

def append_step_result(
        f: sf.StateFile,
        state: bv.BlockVector,
        control: bv.BlockVector,
        time: float,
        step_info: Info
    ):
    """
    Append the result of an integration step to a statefile
    """
    f.append_state(state)
    f.append_control(control)
    f.append_time(time)
    f.append_solver_info(step_info)
