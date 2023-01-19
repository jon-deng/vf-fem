"""
Adjoint model.

The forward model is represented by a mapping from
..math: (x^0, g, p) -> (x^n; n>1)
i.e. the initial state and control/parameter vectors map to a sequence of output
states corresponding to the state vectors at a sequence of time steps. This relationship
comes from the recursive relation between states over a single time step, represented by the
model residual function.

I'm using CGS : cm-g-s units
"""

import numpy as np

from blockarray import blockvec as vec

def integrate(model, f, dfin_state):
    """
    Given a list of adjoint output state vectors, x^n (n>1), integrate the adjoint model

    Since there can be many output states, `dfin_state` is a callable that returns the
    appropriate dx_n vector at index n without having an explicit list of the vectors for all
    n.

    Parameters
    ----------
    model : .models.base.Model
    f : statefile.StateFile
    dfin_state : callable with signature dfin_state(f, i) -> dx^i vector
    """
    ## Set potentially constant values
    model.set_prop(f.get_prop())

    control0 = f.get_control(0)
    control1 = control0

    ## Allocate space for the adjoints of all the parameters
    adj_dt = []
    adj_props = model.prop.copy()
    adj_props[:] = 0.0
    adj_controls = [model.control.copy() for i in range(f.num_controls)]

    ## Load states/parameters
    N = f.size
    times = f.get_times()

    ## Loop through states for adjoint computation
    # Initialize the adj rhs
    adj_state1 = dfin_state(f, N-1)
    dres1 = None
    for ii in range(N-1, 0, -1):
        ## Linearize the model about time step `ii`
        dt1 = times[ii] - times[ii-1]
        state0, state1 = f.get_state(ii-1), f.get_state(ii)
        control1 = f.get_control(ii)

        model.set_ini_state(state0)
        model.set_fin_state(state1)
        model.set_control(control1)
        model.dt = dt1

        ## Perform adjoint calculations
        dres1 = model.solve_dres_dstate1_adj(adj_state1)

        # Update adjoint output variables using the adjoint
        # this logic assumes the last control applies over all remaining time steps (which is correct)
        adj_controls[min(ii, len(adj_controls)-1)] -= model.apply_dres_dcontrol_adj(dres1)
        adj_props -= model.apply_dres_dp_adj(dres1)
        adj_dt.insert(0, -model.apply_dres_ddt_adj(dres1))

        # Update the RHS for the next iteration
        adj_state1 = dfin_state(f, ii-1) - model.apply_dres_dstate0_adj(dres1)

    ## Compute adjoint input variables
    adj_ini_state = adj_state1

    # Calculate sensitivities w.r.t integration times
    grad_dt = np.array(adj_dt)

    # Convert adjoint variables in terms of time steps to variables in
    # terms of start/end integration times. This uses the fact that
    # dt^{n} = t^{n} - t^{n-1}, therefore
    # df/d[t^{n}, t^{n-1}] = df/ddt^{n} * ddt^{n}/d[t^{n}, t^{n-1}]
    adj_times = np.zeros(N)
    adj_times[1:] += grad_dt
    adj_times[:-1] -= grad_dt
    adj_times = vec.BlockVector((adj_times,), labels=(('times',),))

    return adj_ini_state, adj_controls, adj_props, adj_times

def integrate_grad(model, f, functional):
    """
    Returns the gradient of the cost function using the adjoint model.

    Parameters
    ----------
    model : model.ForwardModel
    f : statefile.StateFile
    functional : functional.Functional
    show_figures : bool
        Whether to display a figure showing the solution or not.

    Returns
    -------
    float
        The value of the functional
    grad_uva, grad_solid, grad_fluid, grad_times
        Gradients with respect to initial state, solid, fluid, and integration time points
    """
    functional_value = functional(f)

    def dfin_state(f, n):
        return functional.dstate(f, n)

    # The result of integrating the adjoint are the partial sensitivity components due only to the
    # model
    dini_state, dcontrols, dprop, dtimes = integrate(model, f, dfin_state)

    # To form the final gradient term, add partial sensitivity components due to the functional
    dprop += functional.dprop(f)

    ddts = [functional.ddt(f, n) for n in range(1, f.size)]
    dtimes_functional = vec.BlockVector([np.cumsum([0] + ddts)], labels=[['times']])
    dtimes += dtimes_functional

    return functional_value, dini_state, dcontrols, dprop, dtimes
