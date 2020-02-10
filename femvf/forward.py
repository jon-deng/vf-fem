"""
Forward model

Uses CGS (cm-g-s) units unless otherwise stated
"""

from math import isclose, ceil, floor, remainder

import numpy as np
from matplotlib import pyplot as plt
import dolfin as dfn

from . import forms
from . import statefile as sf
from . import vis

# from .collision import detect_collision
from .misc import get_dynamic_fluid_props
# @profile
def forward(model, timing_props, solid_props, fluid_props,
            h5file='tmp.h5', h5group='/', abs_tol=1e-5, abs_tol_bounds=(0, 1.2),
            show_figure=False, figure_path=None):
    """
    Solves the forward model over specific time instants.

    The `model` is solved over specific time instants. All intermediate solution states are saved to
    an hdf5 file.

    Parameters
    ----------
    model : forms.ForwardModel
        An object representing the forward model.
    timing_props : properties.TimingProperties
        A timing properties object
    solid_props : properties.SolidProperties
        A solid properties object
    fluid_props : properties.FluidProperties
        A fluid properties object
    h5file : string
        Path to an hdf5 file where solution information will be appended.
    h5group : string
        A group in the h5 file to save solution information under.
    abs_tol : float or None
        A desired tolerance that the norm of the displacement solution should meet
    abs_tol_bounds : tuple of float
        Bounds on the solution norm tolerance. Time steps are adjusted so that the local error in
        :math:`u_{n+1}` is between `abs_tol_bounds[0]*abs_tol` and `abs_tol_bounds[1]*abs_tol`.
    show_figure : bool
        Determines whether to display figures of the solution or not.
    figure_path : string
        A path to save figures to. The figures will have a postfix of the iteration number and
        extension added.

    Returns
    -------
    info : dict
        A dictionary of info about the run.
    """
    model.set_fluid_props(fluid_props)
    model.set_solid_props(solid_props)
    t0, tmeas, dt_max = timing_props['t0'], timing_props['tmeas'], timing_props['dt_max']

    info = {}

    # Allocate functions for states
    u0 = dfn.Function(model.vector_function_space)
    v0 = dfn.Function(model.vector_function_space)
    a0 = dfn.Function(model.vector_function_space)

    u1 = dfn.Function(model.vector_function_space)
    v1 = dfn.Function(model.vector_function_space)
    a1 = dfn.Function(model.vector_function_space)

    ## Allocate a figure for plotting
    fig, axs = None, None
    if show_figure:
        fig, axs = vis.init_figure(model, fluid_props)

    # Get the solution times
    tmeas = np.array(tmeas)
    assert tmeas.size >= 2
    assert tmeas[-1] > tmeas[0]

    ## Initialize datasets to save in h5 file
    with sf.StateFile(h5file, group=h5group, mode='a') as f:
        f.init_layout(model, x0=(u0, v0, a0), fluid_props=fluid_props, solid_props=solid_props)
        f.append_time(t0)


    ## Loop through solution times and write solution variables to the h5file.

    # TODO: Hardcoded the calculation of glottal width here, but it should be an option you
    # can pass in along with other functionals of interest you may want to calculate a time-history
    # of
    idx_separation = []
    idx_min_area = []
    glottal_width = []
    flow_rate = []
    pressure = []
    with sf.StateFile(h5file, group=h5group, mode='a') as f:
        t_current = t0
        n_state = 0

        for t_target in tmeas:

            # keep incrementing until you reach the target time
            dt_proposal = dt_max
            while not isclose(t_current, t_target, rel_tol=1e-7, abs_tol=10*2**-52):
                assert t_current < t_target

                # if t_current > 0.04374:
                #     breakpoint()
                x0 = (u0, v0, a0)

                # Increment the state using a target time step. If the previous time step was
                # refined to be smaller than the max time step, then try using that time step again.
                # If the local error is super low, the refinement time step will be predicted to be
                # high and so it will go back to the max time step.
                dt_target = min(dt_proposal, dt_max, t_target - t_current)
                x1, dt_actual, step_info = adaptive_step(model, x0, dt_target, abs_tol=abs_tol,
                                                         abs_tol_bounds=abs_tol_bounds)
                n_state += 1
                t_current += dt_actual

                dt_proposal = dt_actual

                idx_separation.append(step_info['fluid_info']['idx_sep'])
                idx_min_area.append(step_info['fluid_info']['idx_min'])
                glottal_width.append(step_info['fluid_info']['a_min'])
                flow_rate.append(step_info['fluid_info']['flow_rate'])
                pressure.append(step_info['fluid_info']['pressure'])

                ## Write the solution outputs to a file
                f.append_time(t_current)
                f.append_state(x1)

                ## Update initial conditions for the next time step
                u0.assign(x1[0])
                v0.assign(x1[1])
                a0.assign(x1[2])

                ## Plot the solution
                if show_figure:
                    fig, axs = vis.update_figure(fig, axs, model, t_current, (u0, v0, a0), step_info['fluid_info'],
                                                 solid_props, fluid_props)
                    plt.pause(0.001)

                    if figure_path is not None:
                        ext = '.png'
                        fig.savefig(f'{figure_path}_{n_state}{ext}')

            f.append_meas_index(n_state)

        # Write the final functionals
        _x1 = [comp.vector() for comp in x1]
        step_info = model.set_params(_x1)
        glottal_width.append(step_info['a_min'])
        flow_rate.append(step_info['flow_rate'])

        info['meas_ind'] = f.get_meas_indices()
        info['time'] = f.get_solution_times()
        info['glottal_width'] = np.array(glottal_width)
        info['flow_rate'] = np.array(flow_rate)
        info['idx_separation'] = np.array(idx_separation)
        info['idx_min_area'] = np.array(idx_min_area)
        info['pressure'] = np.array(pressure)
        info['h5file'] = h5file
        info['h5group'] = h5group

        return info

# @profile
def increment_forward(model, x0, dt):
    """
    Return the state at the end of `dt` `x1 = (u1, v1, a1)`.

    Parameters
    ----------
    model : forms.ForwardModel
    x0 : tuple of dfn.Function
        Initial states (u0, v0, a0) for the forward model
    dt : float
        The time step to increment over
    solid_props : properties.SolidProperties, optional
        A dictionary of solid properties
    fluid_props : properties.FluidProperties, optional
        A dictionary of fluid properties.

    Returns
    -------
    tuple of dfn.Function
        The next state (u1, v1, a1) of the forward model
    fluid_info : dict
        A dictionary containing information on the fluid solution. These include the flow rate,
        surface pressure, etc.
    """
    u0, v0, a0 = x0

    u1 = dfn.Function(model.vector_function_space)
    v1 = dfn.Function(model.vector_function_space)
    a1 = dfn.Function(model.vector_function_space)

    # Update form coefficients and initial guess
    _x0 = [comp.vector() for comp in x0]
    fluid_info = model.set_iter_params(x0=_x0, dt=dt, u1=u0.vector())

    # Solve the thing
    # TODO: Implement this manually so that linear/nonlinear solver is switched according to the
    # form. During collision the equations are non-linear but in all other cases they are currently
    # linear.
    newton_prm = {'linear_solver': 'petsc', 'absolute_tolerance': 1e-8, 'relative_tolerance': 1e-11}
    dfn.solve(model.f1 == 0, model.u1, bcs=model.bc_base, J=model.df1_du1,
              solver_parameters={"newton_solver": newton_prm})
    u1.assign(model.u1)

    # u1.assign(u0)
    # du = dfn.Function(model.vector_function_space)
    # _u1, _ = newton_solve(u1.vector(), du.vector(), model.assem_df1_du1, model.assem_f1,
    #                       [model.bc_base], **newton_prm)
    # u1.vector()[:] = _u1

    v1.vector()[:] = forms.newmark_v(u1.vector(), u0.vector(), v0.vector(), a0.vector(), model.dt)
    a1.vector()[:] = forms.newmark_a(u1.vector(), u0.vector(), v0.vector(), a0.vector(), model.dt)

    return (u1, v1, a1), fluid_info

def adaptive_step(model, x0, dt_max, abs_tol=1e-5, abs_tol_bounds=(0.8, 1.2)):
    """
    Integrate the model over `dt` using a smaller time step if needed.

    # TODO: `fluid_props` is assumed to be constant over the time step

    Parameters
    ----------
    model : forms.ForwardModel
    x0 : tuple of dfn.Function
        Initial states (u0, v0, a0) for the forward model.
    dt_max : float
        The maximum time step to increment over.
    solid_props : dict
        A dictionary of solid properties.
    fluid_props : dict
        A dictionary of fluid properties.
    adaptive : bool
        Setting `adaptive=False` will enforce a single integration over the interval `dt`.

    Returns
    -------
    tuple(3 * dfn.Function)
        The states at the end of the time step.
    float
        The time step integrated over.
    fluid_info : dict
        A dictionary containing information on the fluid solution. These include the flow rate,
        surface pressure, etc.
    """
    x1 = None
    dt = dt_max
    info = {}

    nrefine = -1
    refine = True
    while refine:
        nrefine += 1
        x1, fluid_info = increment_forward(model, x0, dt)
        info['fluid_info'] = fluid_info

        err = newmark_error_estimate(x1[2].vector(), x0[2].vector(), dt, beta=model.forms['coeff.beta'].values()[0])
        err_norm = err.norm('l2')
        info['err_norm'] = err_norm
        info['nrefine'] = nrefine

        # coll_verts = model.get_collision_verts()
        # print(err_norm)
        # print(coll_verts)

        if abs_tol is not None:
            # step control method that prevents crossing the midline in one step near collision
            refine, dt = refine_initial_collision(model, x0, x1, dt)

            # Step control method from [1]
            if err_norm > abs_tol_bounds[1]*abs_tol or err_norm < abs_tol_bounds[0]*abs_tol:
                dt = (abs_tol/err_norm)**(1/3) * dt
                refine = True
        else:
            refine = False

    return x1, dt, info

# @profile
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

def refine_initial_collision(model, x0, x1, dt):
    """
    Return whether to refine the time step, and a proposed time step to use.
    """
    refine = False
    dt_refine = dt

    u0, v0, a0 = x0
    u1, v1, a1 = x1

    # Refine the time step if there is a transition from no-collision to collision
    model.set_ini_state(u0.vector(), v0.vector(), a0.vector())
    ymax0 = model.get_ymax()

    y_collision = model.y_collision.values()[0]
    y_midline = model.fluid_props['y_midline']
    gap0 = y_collision - ymax0

    model.set_ini_state(u1.vector(), v1.vector(), a1.vector())
    ymax1 = model.get_ymax()
    gap1 = y_collision - ymax1

    # Refinement condition is based on initial collision penetration tolerance
    tol = 1/50 * (y_midline - y_collision)
    if gap0 >= 0 and gap1 < 0:
        if -gap1 > tol:
            refine = True
            dt_refine = 0.5 * dt

    return refine, dt_refine

# def local_refinement(err, dt, tol=1e-5):
#     """
#     Return an estimate of the time step needed to achieve a certain error level.
#     """

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
