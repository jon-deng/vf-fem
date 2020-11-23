"""
Adjoint model.

I'm using CGS : cm-g-s units
"""

import numpy as np
import dolfin as dfn
import ufl
from petsc4py import PETSc

from .models.newmark import (newmark_v_du1, newmark_v_du0, newmark_v_dv0, newmark_v_da0, newmark_v_dt,
                             newmark_a_du1, newmark_a_du0, newmark_a_dv0, newmark_a_da0, newmark_a_dt)
from . import linalg

# @profile
def adjoint(model, f, functional):
    """
    Returns the gradient of the cost function using the adjoint model.

    Parameters
    ----------
    model : model.ForwardModel
    f : statefile.StateFile
    functional : functionals.Functional
    show_figures : bool
        Whether to display a figure showing the solution or not.

    Returns
    -------
    float
        The value of the functional
    grad_uva, grad_solid, grad_fluid, grad_times
        Gradients with respect to initial state, solid, fluid, and integration time points
    """
    ## Set potentially constant values
    # Set properties
    props = f.get_properties()
    model.set_properties(props)

    # Check whether controls are variable in time or constant
    variable_controls = f.variable_controls
    control0 = f.get_control(0)
    control1 = control0

    # run the functional once to initialize any cached values
    functional_value = functional(f)

    ## Allocate space for the adjoints of all the parameters
    adj_dt = []
    adj_props = model.get_properties_vec()
    adj_props.set(0.0)

    ## Load states/parameters
    N = f.size
    times = f.get_times()

    ## Initialize the adj rhs
    adj_state1_rhs = functional.dstate(f, N-1)
    
    ## Loop through states for adjoint computation
    # Note that ii corresponds to the time index of the adjoint state we are solving for.
    # In a given loop, adj^{ii+1} is known, and the iteration of the loop finds adj^{ii}
    for ii in range(N-1, 0, -1):
        ## Set the properties of the system to that of the `ii` iteration
        # Properties at index 2 through 1 were loaded during initialization, so we only need to read
        # index 0
        dt1 = times[ii] - times[ii-1]
        state0, state1 = f.get_state(ii-1), f.get_state(ii)
        if variable_controls:
            control1 = f.get_control(ii)

        model.set_ini_state(state0)
        model.set_fin_state(state1)
        model.set_control(control1)
        model.dt = dt1

        ## Do the adjoint calculations
        # breakpoint()
        adj_state1 = model.solve_dres_dstate1_adj(adj_state1_rhs)

        # Update gradients wrt parameters using the adjoint
        adj_props[:] = adj_props - model.apply_dres_dp_adj(adj_state1)

        adj_dt1 = solve_grad_dt(model, adj_state1) + functional.ddt(f, ii)
        adj_dt.insert(0, adj_dt1)

        # Find the RHS for the next iteration
        dcost_dstate0 = functional.dstate(f, ii-1)
        adj_state0_rhs = dcost_dstate0 - model.apply_dres_dstate0_adj(adj_state1)

        adj_state1_rhs = adj_state0_rhs

    ## Calculate gradients
    grad_state = adj_state1_rhs

    # Finally, if the functional is sensitive to the parameters, you have to add their sensitivity
    # components once
    grad_props = adj_props + functional.dprops(f)

    grad_controls = model.get_control_vec()

    # Calculate sensitivities w.r.t integration times
    grad_dt = np.array(adj_dt)

    grad_times = np.zeros(N)
    # the conversion below is becase dt = t1 - t0
    grad_times[1:] = grad_dt
    grad_times[:-1] -= grad_dt

    return functional_value, grad_state, grad_controls, grad_props, grad_times

def solve_grad_dt(model, adj_state1):
    """
    Calculate the gradient wrt dt
    """
    # model.set_iter_params(**iter_params1)
    dt1 = model.solid.dt
    uva0 = model.solid.state0
    uva1 = model.solid.state1

    dfu1_ddt = dfn.assemble(model.solid.forms['form.bi.df1_dt_adj'])
    dfv1_ddt = 0 - newmark_v_dt(uva1[0], *uva0, dt1)
    dfa1_ddt = 0 - newmark_a_dt(uva1[0], *uva0, dt1)

    adj_u1, adj_v1, adj_a1 = adj_state1[:3]
    adj_dt1 = -(dfu1_ddt*adj_u1).sum() - dfv1_ddt.inner(adj_v1) - dfa1_ddt.inner(adj_a1)
    return adj_dt1
