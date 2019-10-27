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
from . import visualization as vis

# from .collision import detect_collision
from .misc import get_dynamic_fluid_props

def forward(model, t0, tmeas, dt_max, solid_props, fluid_props, adaptive=True,
            h5file='tmp.h5', h5group='/', show_figure=False, figure_path=None):
    """
    Solves the forward model over specific time instants.

    The `model` is solved over specific time instants. All intermediate solution states are saved to
    an hdf5 file.

    Parameters
    ----------
    model : forms.ForwardModel
        An object representing the forward model.
    t0 : float
        Simulation starting time.
    tmeas : array_like of float
        Specific times at which the model should be solved. There should be a minimum of two
        entries. The first/final entries are the first/final measurements. A common way to set this
        would be to set [0, tfinal] to record the first/final times and all steps in between.
    dt : float
        The time step in seconds.
    solid_props : dict
        A dictionary of solid properties.
    fluid_props : dict
        A dictionary of fluid properties.
    adaptive : bool
        Indicate whether or not to use adaptive time stepping.
    h5file : string
        Path to an hdf5 file where solution information will be appended.
    h5group : string
        A group in the h5 file to save solution information under.
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
    forward_info = {}

    # Allocate functions for states
    u0 = dfn.Function(model.vector_function_space)
    v0 = dfn.Function(model.vector_function_space)
    a0 = dfn.Function(model.vector_function_space)

    u1 = dfn.Function(model.vector_function_space)
    v1 = dfn.Function(model.vector_function_space)
    a1 = dfn.Function(model.vector_function_space)

    # Set solid material properties
    model.set_solid_properties(solid_props)

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
    glottal_width = []
    flow_rate = []
    with sf.StateFile(h5file, group=h5group, mode='a') as f:
        t_current = t0
        n_state = 0

        for t_target in tmeas:

            # Here we keep incrementing until we reach the target time
            while not isclose(t_current, t_target, rel_tol=1e-10, abs_tol=10*2**-52):
                assert t_current < t_target
                x0 = (u0, v0, a0)

                # Update properties
                fluid_props_ii = get_dynamic_fluid_props(fluid_props, t_current)

                # Increment the state
                dt_target = min(dt_max, t_target - t_current)
                x1, dt_actual, info = adaptive_step(model, x0, dt_target, solid_props, fluid_props,
                                                    adaptive=adaptive)
                n_state += 1
                t_current += dt_actual

                glottal_width.append(info['a_min'])
                flow_rate.append(info['flow_rate'])

                ## Write the solution outputs to a file
                f.append_time(t_current)
                f.append_state(x1)
                f.append_fluid_props(fluid_props_ii)

                ## Update initial conditions for the next time step
                u0.assign(x1[0])
                v0.assign(x1[1])
                a0.assign(x1[2])

                ## Plot the solution
                if show_figure:
                    fig, axs = vis.update_figure(fig, axs, model, t_current, (u0, v0, a0), info,
                                                 solid_props, fluid_props_ii)
                    plt.pause(0.001)

                    if figure_path is not None:
                        ext = '.png'
                        fig.savefig(f'{figure_path}_{n_state}{ext}')

            f.append_meas_index(n_state)

        # Write the final fluid properties
        fluid_props_ii = get_dynamic_fluid_props(fluid_props, tmeas[-1])
        f.append_fluid_props(fluid_props_ii)

        # Write the final functionals
        u1, v1, a1 = x1
        info = model.set_params((u1.vector(), v1.vector(), a1.vector()), fluid_props_ii,
                                solid_props)
        glottal_width.append(info['a_min'])
        flow_rate.append(info['flow_rate'])

        forward_info['glottal_width'] = np.array(glottal_width)
        forward_info['flow_rate'] = np.array(flow_rate)

        return forward_info

def increment_forward(model, x0, dt, solid_props, fluid_props):
    """
    Return the state at the end of `dt` `x1 = (u1, v1, a1)`.

    Parameters
    ----------
    model : forms.ForwardModel
    x0 : tuple of dfn.Function
        Initial states (u0, v0, a0) for the forward model
    dt : float
        The time step to increment over
    solid_props : dict
        A dictionary of solid properties
    fluid_props : dict
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
    fluid_info = model.set_iteration_params((u0.vector(), v0.vector(), a0.vector()), dt,
                                            fluid_props, solid_props, u1=u0.vector())

    # Solve the thing
    # TODO: Implement this manually so that linear/nonlinear solver is switched according to the
    # form. During collision the equations are non-linear but in all other cases they are currently
    # linear.
    newton_prm = {'linear_solver': 'petsc', 'absolute_tolerance': 1e-8, 'relative_tolerance': 1e-6}
    dfn.solve(model.fu_nonlin == 0, model.u1, bcs=model.bc_base, J=model.jac_fu_nonlin,
              solver_parameters={"newton_solver": newton_prm})

    u1.assign(model.u1)
    v1.vector()[:] = forms.newmark_v(u1.vector(), u0.vector(), v0.vector(), a0.vector(), model.dt)
    a1.vector()[:] = forms.newmark_a(u1.vector(), u0.vector(), v0.vector(), a0.vector(), model.dt)

    return (u1, v1, a1), fluid_info

def adaptive_step(model, x0, dt_max, solid_props, fluid_props, adaptive=True):
    """
    Integrate the model over `dt` using a smaller time step if needed.

    # TODO: `fluid_props` is assumed to be constant over the time step

    Parameters
    ----------
    model : forms.ForwardModel
    x0 : tuple of dfn.Function
        Initial states (u0, v0, a0) for the forward model.
    dt : float
        The time step to increment over.
    solid_props : dict
        A dictionary of solid properties.
    fluid_props : dict
        A dictionary of fluid properties.
    adaptive : bool
        Setting `adaptive=False` will enforce a single integration over the interval `dt`.

    Returns
    -------
    tuple(3 * dfn.Function)
        A list of intermediate states in integrating over dt.
    float
        A list of the corresponding time steps taken for each intermediate state.
    fluid_info : dict
        A dictionary containing information on the fluid solution. These include the flow rate,
        surface pressure, etc.
    """
    x1 = None
    dt = dt_max
    info = None

    refine = True
    while refine:
        x1, info = increment_forward(model, x0, dt, solid_props, fluid_props)

        refine = None
        if adaptive:
            refine, dt = refine_initial_collision(model, x0, x1, dt, solid_props, fluid_props)
        else:
            refine = False

    return x1, dt, info

def refine_initial_collision(model, x0, x1, dt, solid_props, fluid_props):
    """
    Return whether to refine the time step, and a proposed time step to use.
    """
    refine = False
    dt_refine = dt

    u0, v0, a0 = x0
    u1, v1, a1 = x1

    # Refine the time step if there is a transition from no-collision to collision
    model.set_initial_state(u0.vector(), v0.vector(), a0.vector())
    ymax0 = model.get_ymax()
    gap0 = solid_props['y_collision'] - ymax0

    model.set_initial_state(u1.vector(), v1.vector(), a1.vector())
    ymax1 = model.get_ymax()
    gap1 = solid_props['y_collision'] - ymax1

    # Initial collision penetration tolerance
    tol = 1/10 * (fluid_props['y_midline'] - solid_props['y_collision'])
    if gap0 >= 0 and gap1 < 0:
        if -gap1 > tol:
            refine = True

    if refine:
        dt_refine = 0.5 * dt

    return refine, dt_refine
