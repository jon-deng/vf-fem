"""
Forward model

Uses CGS (cm-g-s) units unless otherwise stated
"""

from functools import partial

import numpy as np
import dolfin as dfn
from petsc4py import PETSc

from . import solids
from . import statefile as sf
from . import linalg

DEFAULT_NEWTON_SOLVER_PRM = {
    'linear_solver': 'petsc',
    'absolute_tolerance': 1e-7,
    'relative_tolerance': 1e-9,
    'max_num_it': 5}

FIXEDPOINT_SOLVER_PRM = {
    'absolute_tolerance': 1e-8,
    'relative_tolerance': 1e-11}

# @profile
def integrate(model, ini_state, controls, props, times, idx_meas=None,
              h5file='tmp.h5', h5group='/', newton_solver_prm=None, callbacks=None):
    """
    Integrate the model over each time in `times` for the specified parameters

    Parameters
    ----------
    model : model.ForwardModel
    uva : tuple of dfn.Vector
        Initial solid state (displacement, velocity, acceleration)
    solid_props, fluid_props : femvf.Properties
        Solid / fluid parameter vectors
    times : array of float
        Array of discrete integration times. Each time point is one integration point so the time
        between successive time points is a time step.
    idx_meas : array of int
        Array marking which integration points correspond to something (usu. measurements)
    h5file : str
        Path to h5 file where states should be saved

    Returns
    -------
    info : dict
        Info about the run
    """
    if callbacks is None:
        callbacks = {}
    info = {key: [] for key in callbacks}

    if idx_meas is None:
        idx_meas = np.array([])

    variable_controls = False
    if isinstance(controls, list):
        variable_controls = True
        assert len(controls) == times.size

    # Check integration times are specified fine
    if times[-1] < times[0]:
        raise ValueError("The final time point must be greater than the initial one."
                         f"The input intial/final times were {times[0]}/{times[-1]}")
    if times.size <= 1:
        raise ValueError("There must be at least 2 time integration points.")

    model.set_ini_state(ini_state)
    model.set_properties(props)

    ## Allocate functions to store states
    state0 = ini_state.copy()
    control0 = None
    if variable_controls:
        control0 = controls[0]
    else:
        control0 = controls

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
                control0 = controls[n-1]
                control1 = controls[n]
            else:
                control0 = controls
                control1 = controls
            dt = times[n] - times[n-1]

            model.set_time_step(dt)
            model.set_ini_state(state0)
            model.set_ini_control(control0)
            model.set_fin_control(control1)
            state1, step_info = model.solve_state1(state0)

            model.set_fin_state(state1)
            for key, callback in callbacks.items():
                info[key].append(callback(model))

            # Write the solution outputs to a file
            f.append_state(state1)
            f.append_time(times[n])
            if n in idx_meas:
                f.append_meas_index(n)

            # Update initial conditions for the next time step
            state0 = state1

    return info

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

def gw_callback(model):
    _, info = model.fluid.solve_qp1()
    return info['a_min']
