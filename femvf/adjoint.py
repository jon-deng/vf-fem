"""
Adjoint model.

Makes a vocal fold go elggiw elggiw.

I'm using CGS : cm-g-s units
"""
# TODO: Refactor this to use adj_qp (sensitive to fluid state)
# This means functionals have to provide a sensitivity to flow and pressure variables

import numpy as np

from matplotlib import tri
from matplotlib import pyplot as plt

import dolfin as dfn
import ufl

from . import forms
from .newmark import *

def adjoint(model, f, functional, show_figure=False):
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
    dict(str: dfn.Vector)
        The gradient of the functional wrt parameter labelled by `str`
    """
    # gradients = model.SolidProperties()

    # Assuming that fluid and solid properties are constant in time so set them once
    # and leave them
    # TODO: May want to use time varying fluid/solid props in future
    fluid_props = f.get_fluid_props(0)
    solid_props = f.get_solid_props()

    model.set_fluid_props(fluid_props)
    model.set_solid_props(solid_props)

    # Initialize the functional instance and run it once to initialize any cached values
    functional_value = functional(f)
    
    # Make adjoint forms for sensitivity of parameters
    df1_demod_form_adj = dfn.adjoint(ufl.derivative(model.f1, model.emod, model.scalar_trial))
    df1_ddt_form_adj = dfn.adjoint(ufl.derivative(model.f1, model.dt, model.scalar_trial))

    ## Preallocating vector
    # Temporary variables to shorten code
    def get_block_vec():
        vspace = model.vector_function_space
        return (dfn.Function(vspace).vector(),
                dfn.Function(vspace).vector(),
                dfn.Function(vspace).vector())

    # Adjoint states
    adj_uva1 = get_block_vec()
    adj_uva2 = get_block_vec()

    # Model states
    uva0 = get_block_vec()
    uva1 = get_block_vec()
    uva2 = get_block_vec()

    # Allocate space for the gradient
    grad = {'emod': dfn.Function(model.scalar_function_space).vector(),
            'dt': []}
    # gradient = dfn.Function(model.scalar_function_space).vector()

    ## Initialize Adjoint states
    # Set form coefficients to represent f^{N-1} (the final forward increment model that solves
    # for the final state)
    # To initialize, we need to solve for \lambda^{N-1} i.e. `adj_u2`, `adj_v2`, `adj_a2` etc.
    N = f.size
    times = f.get_solution_times()

    f.get_state(N-1, out=uva2)
    f.get_state(N-2, out=uva1)
    qp2 = f.get_fluid_state(N-1)
    qp1 = f.get_fluid_state(N-2)
    dt2 = times[N-1]-times[N-2]

    iter_params2 = {'uva0': uva1, 'qp0': qp1, 'dt': dt2, 'u1': uva2[0]}
    iter_params3 = {'uva0': uva2, 'qp0': qp2, 'dt': 0.0, 'u1': None}

    dcost_du2 = functional.du(f, N-1, iter_params2, iter_params3)
    dcost_dv2 = functional.dv(f, N-1, iter_params2, iter_params3)
    dcost_da2 = functional.da(f, N-1, iter_params2, iter_params3)

    model.set_iter_params(**iter_params2)
    df2_du2 = model.assem_df1_du1_adj()

    # Initializing adjoint states:
    adj_a2_lhs = dcost_da2
    adj_uva2[2][:] = adj_a2_lhs

    adj_v2_lhs = dcost_dv2
    adj_uva2[1][:] = adj_v2_lhs

    adj_u2_lhs = dcost_du2 + newmark_v_du1(dt2)*adj_uva2[1] + newmark_a_du1(dt2)*adj_uva2[2]
    model.bc_base_adj.apply(df2_du2, adj_u2_lhs)
    dfn.solve(df2_du2, adj_uva2[0], adj_u2_lhs)

    df2_demod = dfn.assemble(df1_demod_form_adj)
    df2_ddt = dfn.assemble(df1_ddt_form_adj)
    grad['emod'] -= df2_demod*adj_uva2[0]

    # The sum is done since the 'dt' at every DOF, must be the same; we can then collapse
    # each sensitivity into one
    grad_dt = - (df2_ddt*adj_uva2[0]).sum() \
              - newmark_v_dt(uva2[0], *uva1, dt2).inner(adj_uva2[1]) \
              - newmark_a_dt(uva2[0], *uva1, dt2).inner(adj_uva2[2])
    grad['dt'].insert(0, grad_dt)

    ## Loop through states for adjoint computation
    # Note that ii corresponds to the time index of the adjoint state we are solving for.
    # In a given loop, adj^{ii+1} is known, and the iteration of the loop finds adj^{ii}
    for ii in range(N-2, 0, -1):
        # Properties at index 2 through 1 were loaded during initialization, so we only need to
        # read index 0
        uva0 = f.get_state(ii-1, out=uva0)
        qp0 = f.get_fluid_state(ii-1)
        dt1 = times[ii] - times[ii-1]

        iter_params1 = {'uva0': uva0, 'qp0': qp0, 'dt': dt1, 'u1': uva1[0]}
        iter_params2 = {'uva0': uva1, 'qp0': qp1, 'dt': dt2, 'u1': uva2[0]}

        dcost_duva1 = [functional.du(f, ii, iter_params1, iter_params2),
                       functional.dv(f, ii, iter_params1, iter_params2),
                       functional.da(f, ii, iter_params1, iter_params2)]

        (adj_u1, adj_v1, adj_a1) = decrement_adjoint(
            model, adj_uva2, iter_params1, iter_params2, dcost_duva1)

        for comp, val in zip(adj_uva1, [adj_u1, adj_v1, adj_a1]):
            comp[:] = val

        # Update gradient using adjoint states
        model.set_iter_params(**iter_params1)
        df1_dparam = dfn.assemble(df1_demod_form_adj)
        df1_ddt= dfn.assemble(df1_ddt_form_adj)

        grad['emod'] -= df1_dparam*adj_uva1[0]
        grad_dt = - (df1_ddt*adj_uva1[0]).sum() \
                  - newmark_v_dt(uva1[0], *uva0, dt1).inner(adj_uva1[1]) \
                  - newmark_a_dt(uva1[0], *uva0, dt1).inner(adj_uva1[2])
        grad['dt'].insert(0, grad_dt)

        # Set initial states to the previous states for the start of the next iteration
        for comp1, comp2 in zip(uva1, uva2):
            comp2[:] = comp1

        for comp0, comp1 in zip(uva0, uva1):
            comp1[:] = comp0

        for comp1, comp2 in zip(adj_uva1, adj_uva2):
            comp2[:] = comp1

        qp2 = qp1
        qp1 = qp0

        dt2 = dt1
    
    # If the functional is sensitive to the parameters, you have to add its component once
    dfunc_dparam = functional.dp(f)
    if dfunc_dparam is not None:
        grad['emod'] += dfunc_dparam.get('emod', 0) # ['emod']

    # At the end of the `for` loop ii=1, and we can compute the sensitivity w.r.t initial state
    # The model parameters should already be set to compute `F1`, so we can directly assemble below
    grad_u0_par = -(model.assem_df1_du0_adj()*adj_uva1[0])
    grad_v0_par = -(model.assem_df1_dv0_adj()*adj_uva1[1])
    grad_a0 = -(model.assem_df1_da0_adj()*adj_uva1[2])

    df0_du0_adj = dfn.assemble(model.forms['form.bi.df0_du0_adj']) 
    df0_dv0_adj = dfn.assemble(model.forms['form.bi.df0_dv0_adj']) 
    df0_da0_adj = dfn.assemble(model.forms['form.bi.df0_da0_adj']) 
    adj_a0 = dfn.Function(model.vector_function_space).vector()
    dfn.solve(df0_da0_adj, adj_a0, grad_a0, 'petsc')
    grad['u0'] = grad_u0_par - df0_du0_adj*adj_a0 + dfunc_dparam['u0']
    grad['v0'] = grad_u0_par - df0_dv0_adj*adj_a0 + dfunc_dparam['v0']

    # Change grad_dt to an array
    grad['dt'] = np.array(grad['dt'])
    return functional_value, grad, functional

def decrement_adjoint(model, adj_uva2, iter_params1, iter_params2, dcost_duva1):
    """
    Returns the adjoint at the previous time step.

    Each adjoint step is based on an indexing scheme where the postfix on a variable represents that
    variable at time index n + postfix. For example, variables uva0, uva1, and uva2 correspond to states
    at n, n+1, and n+2.

    This is done because the adjoint calculation to solve for :math:`lambda_{n+1}` given
    :math:`lambda_{n+2}` requires the forward equations :math:`f^{n+2}=0`, and :math:`f^{n+1}=0`,
    which in turn requires states :math:`x^{n}`, :math:`x^{n+1}`, and :math:`x^{n+2}` to be defined.

    Note that :math:`f^{n+1} = f^{n+1}([u, v, a]^{n+1}, [u, v, a]^{n}) = 0` involves the FEM
    approximation and time stepping scheme that defines the state :math`x^{n+1} = (u, v, a)^{n+1}`
    implicitly, which could be linear or non-linear.

    Parameters
    ----------
    model : model.ForwardModel
    adj_uva2 : tuple of dfn.cpp.la.Vector
        A tuple (adj_u2, adj_v2, adj_a2) of 'initial' (time index 2) states for the adjoint model.
    iter_params1, iter_params2 : dict
        Dictionaries representing the parameters of ForwardModel.set_iter_params
    h5path : string
        Path to an hdf5 file containing states from a forward run of the model.
    h5group : string
        The group where states are stored in the hdf5 file.

    Returns
    -------
    adj_uva1 : tuple of dfn.Function
        The 'next' state (adj_u1, adj_v1, adj_a1) of the adjoint model.
    info : dict
        Additional info computed during the solve that might be useful.
    """
    adj_u2, adj_v2, adj_a2 = adj_uva2
    dcost_du1, dcost_dv1, dcost_da1 = dcost_duva1

    ## Set form coefficients to represent f^{n+2} aka f2(uva1, uva2) -> uva2
    dt1 = iter_params1['dt']
    dt2 = iter_params2['dt']
    model.set_iter_params(**iter_params2)

    # Assemble needed forms
    df2_du1 = model.assem_df1_du0_adj()
    df2_dv1 = model.assem_df1_dv0_adj()
    df2_da1 = model.assem_df1_da0_adj()

    # Correct df2_du1 since pressure depends on u1 for explicit FSI forcing
    df2_dpressure = dfn.assemble(model.forms['form.bi.df1_dpressure_adj'])
    dpressure_du1 = dfn.PETScMatrix(model.get_flow_sensitivity()[0])

    ## Set form coefficients to represent f^{n+1} aka f1(uva0, uva1) -> uva1
    model.set_iter_params(**iter_params1)

    # Assemble needed forms
    df1_du1 = model.assem_df1_du1_adj()

    ## Adjoint recurrence relations
    # Allocate adjoint states
    adj_u1 = dfn.Function(model.vector_function_space).vector()
    adj_v1 = dfn.Function(model.vector_function_space).vector()
    adj_a1 = dfn.Function(model.vector_function_space).vector()

    # gamma, beta = model.gamma.values()[0], model.beta.values()[0]

    # Calculate 'source' terms for the adjoint calculation
    df2_du1_correction = dfn.Function(model.vector_function_space).vector()
    dpressure_du1.transpmult(df2_dpressure * adj_u2, df2_du1_correction)

    adj_v1_lhs = dcost_dv1
    adj_v1_lhs -= df2_dv1*adj_u2 - newmark_v_dv0(dt2)*adj_v2 - newmark_a_dv0(dt2)*adj_a2
    adj_v1 = adj_v1_lhs

    adj_a1_lhs = dcost_da1
    adj_a1_lhs -= df2_da1*adj_u2 - newmark_v_da0(dt2)*adj_v2 - newmark_a_da0(dt2)*adj_a2
    adj_a1 = adj_a1_lhs

    adj_u1_lhs = dcost_du1 + newmark_v_du1(dt1)*adj_v1 + newmark_a_du1(dt1)*adj_a1
    adj_u1_lhs -= df2_du1*adj_u2 + df2_du1_correction \
                  - newmark_v_du0(dt2)*adj_v2 - newmark_a_du0(dt2)*adj_a2
    model.bc_base.apply(df1_du1)
    model.bc_base.apply(adj_u1_lhs)
    dfn.solve(df1_du1, adj_u1, adj_u1_lhs, 'petsc')

    return (adj_u1, adj_v1, adj_a1)

