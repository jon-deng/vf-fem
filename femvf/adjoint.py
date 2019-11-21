"""
Adjoint model.

Makes a vocal fold go elggiw elggiw.

I'm using CGS : cm-g-s units
"""

import numpy as np

from matplotlib import tri
from matplotlib import pyplot as plt

import dolfin as dfn
import ufl

from . import forms
# import .forms as forms


# @profile
def decrement_adjoint(model, adj_x2, iter_params1, iter_params2, dcost_dx1):
    """
    Returns the adjoint at the previous time step.

    Each adjoint step is based on an indexing scheme where the postfix on a variable represents that
    variable at time index n + postfix. For example, variables x0, x1, and x2 correspond to states
    at n, n+1, and n+2.

    This is done because the adjoint calculation to solve for :math:`lambda_{n+1}` given
    :math:`lambda_{n+2}` requires the forward equations :math:`f^{n+2}=0`, and :math:`f^{n+1}=0`,
    which in turn requires states :math:`x^{n}`, :math:`x^{n+1}`, and :math:`x^{n+2}` to be defined.

    Note that :math:`f^{n+1} = f^{n+1}([u, v, a]^{n+1}, [u, v, a]^{n}) = 0` involves the FEM
    approximation and time stepping scheme that defines the state :math`x^{n+1} = (u, v, a)^{n+1}`
    implicitly, which could be linear or non-linear.

    Parameters
    ----------
    model : forms.ForwardModel
    adj_x2 : tuple of dfn.cpp.la.Vector
        A tuple (adj_u2, adj_v2, adj_a2) of 'initial' (time index 2) states for the adjoint model.
    iter_params1, iter_params2 : tuple
        iter_params1 is a tuple of:
        (x^{n}, t^{n+1}-t^{n}, u^{n+1})
        where x is itself a tuple of (u, v, a)
    h5path : string
        Path to an hdf5 file containing states from a forward run of the model.
    h5group : string
        The group where states are stored in the hdf5 file.

    Returns
    -------
    adj_x1 : tuple of dfn.Function
        The 'next' state (adj_u1, adj_v1, adj_a1) of the adjoint model.
    info : dict
        Additional info computed during the solve that might be useful.
    """
    adj_u2, adj_v2, adj_a2 = adj_x2
    dcost_du1, dcost_dv1, dcost_da1 = dcost_dx1

    ## Set form coefficients to represent f^{n+2} aka f2(x1, x2) -> x2
    dt1 = iter_params1[1]
    dt2 = iter_params2[1]
    model.set_iter_params(*iter_params2)

    # Assemble needed forms
    # breakpoint()
    df2_du1 = model.assem_df1_du0_adj()
    df2_dv1 = model.assem_df1_dv0_adj()
    df2_da1 = model.assem_df1_da0_adj()

    # Correct df2_du1 since pressure depends on u1 for explicit FSI forcing
    df2_dpressure = dfn.assemble(model.forms['bilin.df1_dpressure_adj'])
    dpressure_du1 = dfn.PETScMatrix(model.get_flow_sensitivity()[0])

    ## Set form coefficients to represent f^{n+1} aka f1(x0, x1) -> x1
    dt2 = iter_params2[1]
    model.set_iter_params(*iter_params1)

    # Assemble needed forms
    df1_du1 = model.assem_df1_du1_adj()

    ## Adjoint recurrence relations
    # Allocate adjoint states
    adj_u1 = dfn.Function(model.vector_function_space).vector()
    adj_v1 = dfn.Function(model.vector_function_space).vector()
    adj_a1 = dfn.Function(model.vector_function_space).vector()

    gamma, beta = model.gamma.values()[0], model.beta.values()[0]

    # Calculate 'source' terms for the adjoint calculation
    df2_du1_correction = dfn.Function(model.vector_function_space).vector()
    dpressure_du1.transpmult(df2_dpressure * adj_u2, df2_du1_correction)

    adj_v1_lhs = dcost_dv1
    adj_v1_lhs -= df2_dv1*adj_u2 - forms.newmark_v_dv0(dt2)*adj_v2 - forms.newmark_a_dv0(dt2)*adj_a2
    adj_v1 = adj_v1_lhs

    adj_a1_lhs = dcost_da1
    adj_a1_lhs -= df2_da1*adj_u2 - forms.newmark_v_da0(dt2)*adj_v2 - forms.newmark_a_da0(dt2)*adj_a2
    adj_a1 = adj_a1_lhs

    adj_u1_lhs = dcost_du1 + forms.newmark_v_du1(dt1)*adj_v1 + forms.newmark_a_du1(dt1)*adj_a1
    adj_u1_lhs -= df2_du1*adj_u2 + df2_du1_correction \
                  - forms.newmark_v_du0(dt2)*adj_v2 - forms.newmark_a_du0(dt2)*adj_a2
    model.bc_base.apply(df1_du1)
    model.bc_base.apply(adj_u1_lhs)
    dfn.solve(df1_du1, adj_u1, adj_u1_lhs, 'petsc')

    return (adj_u1, adj_v1, adj_a1)

# @profile
def adjoint(model, f, Functional, functional_kwargs, show_figure=False):
    """
    Returns the gradient of the cost function w.r.t elastic modulus using the adjoint model.

    Parameters
    ----------
    model : forms.ForwardModel
    f : statefile.StateFile
    Functional : class functionals.GenericFunctional
    functional_kwargs : dict
        Options to pass to the functional
    show_figures : bool
        Whether to display a figure showing the solution or not.

    Returns
    -------
    np.array of float
        The sensitivity of the functional wrt parameters.
    """
    # Assuming that fluid and solid properties are constant in time so set them once
    # and leave them
    # TODO: May want to use time varying fluid/solid props in future
    fluid_props = f.get_fluid_props(0)
    solid_props = f.get_solid_props()

    model.set_fluid_props(fluid_props)
    model.set_solid_props(solid_props)

    # Initialize the functional instance and run it once to initialize any cached values
    functional = Functional(model, f, **functional_kwargs)
    functional_value = functional()

    df1_dparam_form_adj = dfn.adjoint(ufl.derivative(model.f1, model.emod, model.scalar_trial))

    ## Preallocating vector
    # Temporary variables to shorten code
    def get_block_vec():
        vspace = model.vector_function_space
        return (dfn.Function(vspace).vector(),
                dfn.Function(vspace).vector(),
                dfn.Function(vspace).vector())

    # Adjoint states
    adj_x1 = get_block_vec()
    adj_x2 = get_block_vec()

    # Model states
    x0 = get_block_vec()
    x1 = get_block_vec()
    x2 = get_block_vec()

    # Allocate space for the gradient
    gradient = dfn.Function(model.scalar_function_space).vector()

    ## Initialize Adjoint states
    # Set form coefficients to represent f^{N-1} (the final forward increment model that solves
    # for the final state)
    # To initialize, we need to solve for \lambda^{N-1} i.e. `adj_u2`, `adj_v2`, `adj_a2` etc.
    N = f.size
    times = f.get_solution_times()

    f.get_state(N-1, out=x2)
    f.get_state(N-2, out=x1)
    dt2 = times[N-1]-times[N-2]

    iter_params2 = (x1, dt2, x2[0])
    iter_params3 = (x2, 0.0, None)

    dcost_du2 = functional.du(N-1, iter_params2, iter_params3)
    dcost_dv2 = functional.dv(N-1, iter_params2, iter_params3)
    dcost_da2 = functional.da(N-1, iter_params2, iter_params3)

    model.set_iter_params(*iter_params2)
    df2_du2 = model.assem_df1_du1_adj()

    # Initializing adjoint states:
    adj_a2_lhs = dcost_da2
    adj_x2[2][:] = adj_a2_lhs

    adj_v2_lhs = dcost_dv2
    adj_x2[1][:] = adj_v2_lhs

    adj_u2_lhs = dcost_du2 + forms.newmark_v_du1(dt2)*adj_x2[1] + forms.newmark_a_du1(dt2)*adj_x2[2]
    model.bc_base_adj.apply(df2_du2, adj_u2_lhs)
    dfn.solve(df2_du2, adj_x2[0], adj_u2_lhs)

    df2_dparam = dfn.assemble(df1_dparam_form_adj)
    gradient -= df2_dparam*adj_x2[0]

    ## Loop through states for adjoint computation
    # Note that ii corresponds to the time index of the adjoint state we are solving for.
    # In a given loop, adj^{ii+1} is known, and the iteration of the loop finds adj^{ii}
    for ii in range(N-2, 0, -1):
        # Properties at index 2 through 1 were loaded during initialization, so we only need to
        # read index 0
        x0 = f.get_state(ii-1, out=x0)

        dt1 = times[ii] - times[ii-1]

        # breakpoint()
        iter_params1 = (x0, dt1, x1[0])
        iter_params2 = (x1, dt2, x2[0])

        dcost_dx1 = []
        dcost_dx1.append(functional.du(ii, iter_params1, iter_params2))
        dcost_dx1.append(functional.dv(ii, iter_params1, iter_params2))
        dcost_dx1.append(functional.da(ii, iter_params1, iter_params2))

        (adj_u1, adj_v1, adj_a1) = decrement_adjoint(
            model, adj_x2, iter_params1, iter_params2, dcost_dx1)

        for comp, val in zip(adj_x1, [adj_u1, adj_v1, adj_a1]):
            comp[:] = val

        # Update gradient using adjoint states
        model.set_iter_params(*iter_params1)
        df1_dparam = dfn.assemble(df1_dparam_form_adj)

        gradient += -(df1_dparam*adj_x1[0]) #+ BLAH*adj_x1[1] + BLAH*adj_x1[2]

        dfunc_dparam = functional.dparam()
        if dfunc_dparam is not None:
            gradient += dfunc_dparam

        # Set initial states to the previous states for the start of the next iteration
        for comp1, comp2 in zip(x1, x2):
            comp2[:] = comp1

        for comp0, comp1 in zip(x0, x1):
            comp1[:] = comp0

        for comp1, comp2 in zip(adj_x1, adj_x2):
            comp2[:] = comp1

        dt2 = dt1

    return functional_value, gradient
