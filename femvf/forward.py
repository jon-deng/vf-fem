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

# TODO: change args to require statefile context where outputs will be written to, 
# instead of having function open a file
# TODO: Allow negative indexes in get functions (negative indexes index in reverse order)
# @profile
def integrate(
    model, ini_state, controls, props, times, idx_meas=None,
    h5file='tmp.h5', h5group='/', newton_solver_prm=None, export_callbacks=None
    ):
    """
    Integrate the model over each time in `times` for the specified parameters

    Parameters
    ----------
    model : model.ForwardModel
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
    info : dict
        Any exported quantites are contained in here
    """
    # Initialize storage of exported quantites
    if export_callbacks is None:
        export_callbacks = {}
    info = {key: [] for key in export_callbacks}

    if idx_meas is None:
        idx_meas = np.array([])

    variable_controls = False
    if len(controls) > 1:
        variable_controls = True
        assert len(controls) == times.size

    # Check integration times are specified fine
    times_bvec = times
    times = times_bvec[0]
    if times[-1] <= times[0]:
        raise ValueError("The final time point must be greater than the initial one."
                         f"The input intial/final times were {times[0]}/{times[-1]}")
    if times.size <= 1:
        raise ValueError("There must be at least 2 time integration points.")

    model.set_ini_state(ini_state)
    model.set_properties(props)

    ## Allocate functions to store states
    state0 = ini_state.copy()
    control0 = controls[0]

    for key, func in export_callbacks.items():
        info[key].append(func(model, state0, control0, props, times[0]))

    ## Initialize datasets and save initial states to the h5 file
    with sf.StateFile(model, h5file, group=h5group, mode='a') as f:
        f.init_layout()

        f.append_state(state0)
        f.append_control(control0)
        f.append_properties(props)
        f.append_time(times[0])
        
        if 0 in idx_meas:
            f.append_meas_index(0)

    ## Integrate the system over the specified times
    with sf.StateFile(model, h5file, group=h5group, mode='a') as f:
        for n in range(1, times.size):
            if variable_controls:
                control1 = controls[n]
            else:
                control1 = control0

            dt = times[n] - times[n-1]

            model.dt = dt
            model.set_ini_state(state0)
            model.set_control(control1)
            
            state1, step_info = model.solve_state1(state0)
            # model.set_fin_state(state1)
            for key, func in export_callbacks.items():
                info[key].append(func(model, state1, control1, props, times[n]))

            # Write the solution outputs to a file
            f.append_state(state1)
            f.append_time(times[n])
            if n in idx_meas:
                f.append_meas_index(n)

            # Update initial conditions for the next time step
            state0 = state1
            control0 = control1

        info['times'] = np.array(times)
        for key, value in info.items():
            f.root_group[f'exports/{key}'] = value

    return h5file, h5group, info

def integrate_linear(
    model, f, dini_state, dcontrols, dprops, dtimes, idx_meas=None,
    h5file='tmp.h5', h5group='/', newton_solver_prm=None, export_callbacks=None
    ):
    """
    Integrate the linearized forward equations and return the final linearized state

    Returns
    -------
    dfin_state : linalg.BlockVec
    """
    model.set_properties(f.get_properties())

    dfin_state_n = dini_state
    for n in range(1, f.size):
        # Set the linearization point
        # This represents the residual F^n
        model.set_ini_state(f.get_state(n-1))
        model.set_fin_state(f.get_state(n))
        model.set_control(f.get_control(n))

        # Compute the action of the n'th time step
        # note that the input "dx^{n-1}" vector is the previous output "dx"
        _dini_state = dfin_state_n
        _dcontrol = dcontrols[min(n, len(dcontrols)-1)]
        _dt = dtimes[0][n]-dtimes[0][n-1]
        dres_n = (model.apply_dres_dstate0(_dini_state) 
                  + model.apply_dres_dcontrol(_dcontrol)
                  + model.apply_dres_dp(dprops)
                  + model.apply_dres_ddt(_dt))
        dfin_state_n = model.solve_dres_dstate1(dres_n)

    return dfin_state_n
        
def newton_solve(u, du, jac, res, bcs, **kwargs):
    """
    Solves the system using a newton method.

    Parameters
    ----------
    u : dfn.cpp.la.Vector
        The initial guess of the solution.
    du : dfn.cpp.la.Vector
        A vector for storing increments in the solution.
    res : callable(dfn.GenericVector) -> dfn.cpp.la.Vector
    jac : callable(dfn.GenericVector) -> dfn.cpp.la.Matrix
    bcs : list of dfn.DirichletBC

    Returns
    -------
    u1 : dfn.Function
    """
    omega = kwargs.get('relaxation', 1.0)
    linear_solver = kwargs.get('linear_solver', 'petsc')
    abs_tol = kwargs.get('abs_tol', 1e-8)
    rel_tol = kwargs.get('rel_tol', 1e-6)
    maxiter = kwargs.get('maxiter', 25)

    abs_err = 1.0
    rel_err = 1.0

    _res = res(u)
    for bc in bcs:
        bc.apply(_res)
    res_norm0 = _res.norm('l2')
    res_norm1 = 1.0

    ii = 0
    while abs_err > abs_tol and rel_err > rel_tol and ii < maxiter:
        _jac = jac(u)
        for bc in bcs:
            bc.apply(_jac, _res)

        dfn.solve(_jac, du, _res, linear_solver)

        u[:] = u - omega*du

        _res = res(u)
        for bc in bcs:
            bc.apply(_res)
        res_norm1 = _res.norm('l2')

        rel_err = abs((res_norm1 - res_norm0)/res_norm0)
        abs_err = res_norm1

        ii += 1

    info = {'niter': ii, 'abs_err': abs_err, 'rel_err': rel_err}
    return u, info

def newmark_error_estimate(a1, a0, dt, beta=1/4):
    """
    Return an estimate of the truncation error in `u` over the step.

    Error is esimated using eq (18) in [1]. Note that their paper defines $\beta2$ as twice $\beta$
    in the original newmark notation (used here). Therefore the beta term is multiplied by 2.

    [1] A simple error estimator and adaptive time stepping procedure for dynamic analysis.
    O. C. Zienkiewicz and Y. M. Xie. Earthquake Engineering and Structural Dynamics, 20:871-887
    (1991).

    Parameters
    ----------
    a1 : dfn.Vector()
        The newmark prediction of acceleration at :math:`n+1`
    a0 : dfn.Vector()
        The newmark prediction of acceleration at :math:`n`
    dt : float
        The time step integrated over
    beta : float
        The newmark-beta method :math:`beta` parameter

    Returns
    -------
    dfn.Vector()
        An estimate of the error in :math:`u_{n+1}`
    """
    return 0.5*dt**2*(2*beta - 1/3)*(a1-a0)
