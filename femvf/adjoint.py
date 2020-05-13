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

from . import solids
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
    # Assuming that fluid and solid properties are constant in time so set them once
    # and leave them
    # TODO: Only gradient wrt. solid properties are included right now
    # grad = model.solid.get_properties()
    # grad.vector[:] = 0

    fluid_props = f.get_fluid_props(0)
    solid_props = f.get_solid_props()

    model.set_fluid_props(fluid_props)
    model.set_solid_props(solid_props)

    # Initialize the functional instance and run it once to initialize any cached values
    functional_value = functional(f)

    # Make adjoint forms for sensitivity of parameters
    solid = model.solid
    df1_ddt_form_adj = dfn.adjoint(ufl.derivative(solid.f1, solid.forms['coeff.time.dt'], solid.scalar_trial))
    df1_dsolid_form_adj = get_df1_dsolid_forms(solid)

    ## Preallocating vector
    # Temporary variables to shorten code
    def get_block_vec():
        vspace = solid.vector_fspace
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

    ## Allocate space for the adjoints of all the parameters
    adj_dt = []
    adj_solid = {}
    for key in solid_props:
        field_or_const, value_shape = solid_props.TYPES[key]
        if field_or_const == 'const':
            adj_solid[key] = None
        elif value_shape == ():
            adj_solid[key] = dfn.Function(solid.scalar_fspace).vector()
        else:
            adj_solid[key] = dfn.Function(solid.vector_fspace).vector()

    ## Initialize Adjoint states
    # Set form coefficients to represent f^{N-1} (the final forward increment model that solves
    # for the final state)
    # To initialize, we need to solve for \lambda^{N-1} i.e. `adj_u2`, `adj_v2`, `adj_a2` etc.
    N = f.size
    times = f.get_times()

    f.get_state(N-1, out=uva2)
    f.get_state(N-2, out=uva1)
    qp2 = f.get_fluid_state(N-1)
    qp1 = f.get_fluid_state(N-2)
    dt2 = times[N-1]-times[N-2]

    iter_params3 = {'uva0': uva2, 'qp0': qp2, 'dt': 0.0, 'u1': None}
    iter_params2 = {'uva0': uva1, 'qp0': qp1, 'dt': dt2, 'u1': uva2[0]}

    dcost_du2, dcost_dv2, dcost_da2 = functional.duva(f, N-1, iter_params2, iter_params3)

    model.set_iter_params(**iter_params2)
    df2_du2 = model.assem_df1_du1_adj()

    # Initializing adjoint states:
    adj_a2_lhs = dcost_da2
    model.solid.bc_base.apply(adj_a2_lhs)
    adj_uva2[2][:] = adj_a2_lhs

    adj_v2_lhs = dcost_dv2
    model.solid.bc_base.apply(adj_v2_lhs)
    adj_uva2[1][:] = adj_v2_lhs

    adj_u2_lhs = dcost_du2 + newmark_v_du1(dt2)*adj_uva2[1] + newmark_a_du1(dt2)*adj_uva2[2]
    model.solid.bc_base.apply(df2_du2, adj_u2_lhs)
    dfn.solve(df2_du2, adj_uva2[0], adj_u2_lhs)


    # Update the adjoint w.r.t. solid parameters
    for key, vector in adj_solid.items():
        if vector is not None:
            # the solid parameters only affect the displacement residual, under the
            # displacement-based newmark scheme, hence why there is only mult. by
            # adj_uva2[0]
            df2_dkey = dfn.assemble(df1_dsolid_form_adj[key])
            vector[:] -= df2_dkey*adj_uva2[0]
    # grad['emod'] -= df2_demod*adj_uva2[0]

    # The sum is done since the 'dt' at every spatial DOF, must be the same;
    # we can collapse each sensitivity into one
    df2_ddt = dfn.assemble(df1_ddt_form_adj)
    adj_dt2 = - (df2_ddt*adj_uva2[0]).sum() \
              + newmark_v_dt(uva2[0], *uva1, dt2).inner(adj_uva2[1]) \
              + newmark_a_dt(uva2[0], *uva1, dt2).inner(adj_uva2[2])
    adj_dt.insert(0, adj_dt2)
    # grad['dt'].insert(0, grad_dt)

    ## Loop through states for adjoint computation
    # Note that ii corresponds to the time index of the adjoint state we are solving for.
    # In a given loop, adj^{ii+1} is known, and the iteration of the loop finds adj^{ii}
    for ii in range(N-2, 0, -1):
        # Properties at index 2 through 1 were loaded during initialization, so we only need to read index 0
        uva0 = f.get_state(ii-1, out=uva0)
        qp0 = f.get_fluid_state(ii-1)
        dt1 = times[ii] - times[ii-1]

        iter_params2 = {'uva0': uva1, 'qp0': qp1, 'dt': dt2, 'u1': uva2[0]}
        iter_params1 = {'uva0': uva0, 'qp0': qp0, 'dt': dt1, 'u1': uva1[0]}

        dcost_duva1 = functional.duva(f, ii, iter_params1, iter_params2)

        (adj_u1, adj_v1, adj_a1) = explicit_decrement_adjoint(
            model, adj_uva2, iter_params1, iter_params2, dcost_duva1)

        for comp, val in zip(adj_uva1, [adj_u1, adj_v1, adj_a1]):
            comp[:] = val

        # Update adjoint w.r.t parameters
        model.set_iter_params(**iter_params1)
        # df1_dparam = dfn.assemble(df1_demod_form_adj)

        # grad['emod'] -= df1_dparam*adj_uva1[0]
        # grad['dt'].insert(0, grad_dt)
        for key, vector in adj_solid.items():
            if vector is not None:
                df1_dkey = dfn.assemble(df1_dsolid_form_adj[key])
                vector -= df1_dkey*adj_uva1[0]

        df1_ddt = dfn.assemble(df1_ddt_form_adj)
        adj_dt1 = - (df1_ddt*adj_uva1[0]).sum() \
                  + newmark_v_dt(uva1[0], *uva0, dt1).inner(adj_uva1[1]) \
                  + newmark_a_dt(uva1[0], *uva0, dt1).inner(adj_uva1[2])
        adj_dt.insert(0, adj_dt1)

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

    # The model parameters should already be set to compute `F1`, so we can directly assemble below
    iter_params1 = {'uva0': uva1, 'qp0': qp1, 'dt': dt2, 'u1': uva2[0]}
    # iter_params0 = {'uva0': uva0, 'qp0': qp0, 'dt': dt1, 'u1': uva1[0]}
    # model.set_iter_params(**iter_params1)

    ## Calculate sensitivities wrt initial states
    df1_du0 = model.assem_df1_du0_adj()
    df1_dv0 = model.assem_df1_dv0_adj()
    df1_da0 = model.assem_df1_da0_adj()

    # Calculate a correction for df1_du0 due to the pressure loading
    df1_dpressure = dfn.assemble(solid.forms['form.bi.df1_dpressure_adj'])
    dpressure_du0 = dfn.PETScMatrix(model.get_flow_sensitivity()[0])
    df1_du0_correction = dfn.Function(model.solid.vector_fspace).vector()
    dpressure_du0.transpmult(df1_dpressure*adj_uva2[0], df1_du0_correction)

    # the `None` pass in is because there is no time step before solving time step 1
    # corresponding to iter_params1
    dcost_duva0 = functional.duva(f, 0, None, iter_params1)
    grad_u0_par = -(df1_du0*adj_uva2[0]) + dcost_duva0[0] \
                  + newmark_v_du0(dt2)*adj_uva2[1] \
                  + newmark_a_du0(dt2)*adj_uva2[2] - df1_du0_correction
    model.solid.bc_base.apply(grad_u0_par)

    grad_v0_par = -(df1_dv0*adj_uva2[0]) + dcost_duva0[1] \
                  + newmark_v_dv0(dt2)*adj_uva2[1] \
                  + newmark_a_dv0(dt2)*adj_uva2[2]
    model.solid.bc_base.apply(grad_v0_par)

    grad_a0_par = -(df1_da0*adj_uva2[0]) + dcost_duva0[2] \
                  + newmark_v_da0(dt2)*adj_uva2[1] \
                  + newmark_a_da0(dt2)*adj_uva2[2]
    model.solid.bc_base.apply(grad_a0_par)

    # Since we've integrated over the whole time in reverse, the adjoint are not gradients
    grad = {}
    grad['u0'] = grad_u0_par
    grad['v0'] = grad_v0_par
    grad['a0'] = grad_a0_par

    # Change grad_dt to an array
    grad['dt'] = np.array(adj_dt)

    for key, vector in adj_solid.items():
        if vector is not None:
            grad[key] = vector

    # Finally, if the functional is sensitive to the parameters, you have to add their sensitivity
    # components once
    dfunc_dparam = functional.dp(f)
    if dfunc_dparam is not None:
        for key, vector in adj_solid.items():
            if vector is not None:
                grad[key] += dfunc_dparam.get(key, 0)

    return functional_value, grad, functional

def explicit_decrement_adjoint(model, adj_uva2, iter_params1, iter_params2, dcost_duva1):
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
    adj_u1 = dfn.Function(model.solid.vector_fspace).vector()
    adj_v1 = dfn.Function(model.solid.vector_fspace).vector()
    adj_a1 = dfn.Function(model.solid.vector_fspace).vector()

    # gamma, beta = model.gamma.values()[0], model.beta.values()[0]

    # Calculate 'source' terms for the adjoint calculation
    df2_du1_correction = dfn.Function(model.solid.vector_fspace).vector()
    dpressure_du1.transpmult(df2_dpressure * adj_u2, df2_du1_correction)

    adj_v1_lhs = dcost_dv1
    adj_v1_lhs -= df2_dv1*adj_u2 - newmark_v_dv0(dt2)*adj_v2 - newmark_a_dv0(dt2)*adj_a2
    model.solid.bc_base.apply(adj_v1_lhs)
    adj_v1 = adj_v1_lhs

    adj_a1_lhs = dcost_da1
    adj_a1_lhs -= df2_da1*adj_u2 - newmark_v_da0(dt2)*adj_v2 - newmark_a_da0(dt2)*adj_a2
    model.solid.bc_base.apply(adj_a1_lhs)
    adj_a1 = adj_a1_lhs

    adj_u1_lhs = dcost_du1 + newmark_v_du1(dt1)*adj_v1 + newmark_a_du1(dt1)*adj_a1
    adj_u1_lhs -= df2_du1*adj_u2 + df2_du1_correction \
                  - newmark_v_du0(dt2)*adj_v2 - newmark_a_du0(dt2)*adj_a2
    model.solid.bc_base.apply(df1_du1, adj_u1_lhs)
    dfn.solve(df1_du1, adj_u1, adj_u1_lhs, 'petsc')

    return (adj_u1, adj_v1, adj_a1)

def implicit_decrement_adjoint(model, adj_uva2, iter_params1, iter_params2, dcost_duva1):
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
    df2_dp = dfn.assemble(model.forms['form.bi.df1_dpressure_adj'])
    dpressure_du1 = dfn.PETScMatrix(model.get_flow_sensitivity()[0])

    ## Set form coefficients to represent f^{n+1} aka f1(uva0, uva1) -> uva1
    model.set_iter_params(**iter_params1)

    # Assemble needed forms
    df1_du1 = model.assem_df1_du1_adj()

    ## Adjoint recurrence relations
    # Allocate adjoint states
    adj_u1 = dfn.Function(model.solid.vector_fspace).vector()
    adj_v1 = dfn.Function(model.solid.vector_fspace).vector()
    adj_a1 = dfn.Function(model.solid.vector_fspace).vector()

    # gamma, beta = model.gamma.values()[0], model.beta.values()[0]

    # Calculate 'source' terms for the adjoint calculation
    df2_du1_correction = dfn.Function(model.solid.vector_fspace).vector()
    dpressure_du1.transpmult(df2_dp * adj_u2, df2_du1_correction)

    adj_v1_lhs = dcost_dv1
    adj_v1_lhs -= df2_dv1*adj_u2 - newmark_v_dv0(dt2)*adj_v2 - newmark_a_dv0(dt2)*adj_a2
    model.solid.bc_base.apply(adj_v1_lhs)
    adj_v1 = adj_v1_lhs

    adj_a1_lhs = dcost_da1
    adj_a1_lhs -= df2_da1*adj_u2 - newmark_v_da0(dt2)*adj_v2 - newmark_a_da0(dt2)*adj_a2
    model.solid.bc_base.apply(adj_a1_lhs)
    adj_a1 = adj_a1_lhs

    adj_u1_lhs = dcost_du1 + newmark_v_du1(dt1)*adj_v1 + newmark_a_du1(dt1)*adj_a1
    adj_u1_lhs -= df2_du1*adj_u2 + df2_du1_correction \
                  - newmark_v_du0(dt2)*adj_v2 - newmark_a_du0(dt2)*adj_a2
    model.solid.bc_base.apply(df1_du1, adj_u1_lhs)
    dfn.solve(df1_du1, adj_u1, adj_u1_lhs, 'petsc')

    return (adj_u1, adj_v1, adj_a1)

def get_df1_dsolid_forms(solid):
    df1_dsolid = {}
    for key in solid.PROPERTY_TYPES:
        try:
            df1_dsolid[key] = dfn.adjoint(ufl.derivative(solid.f1, solid.forms[f'coeff.prop.{key}'], solid.scalar_trial))
        except RuntimeError:
            df1_dsolid[key] = None

        if df1_dsolid[key] is not None:
            try:
                dfn.assemble(df1_dsolid[key])
            except RuntimeError:
                df1_dsolid[key] = None
    return df1_dsolid

