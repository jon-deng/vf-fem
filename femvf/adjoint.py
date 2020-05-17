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

    iter_params3 = {'uva0': uva2, 'dt': 0.0, 'qp1': qp2, 'u1': None}
    iter_params2 = {'uva0': uva1, 'dt': dt2, 'qp1': qp1, 'u1': uva2[0]}

    dcost_du2, dcost_dv2, dcost_da2 = functional.duva(f, N-1, iter_params2, iter_params3)
    dcost_dq2, dcost_dp2 = functional.dqp(f, N-1, iter_params2, iter_params3)

    # Initializing adjoint states:
    adj_rhs = (dcost_du2, dcost_dv2, dcost_da2, dcost_dq2, dcost_dp2)
    adj_state2 = solve_adjoint_exp(model, adj_rhs, iter_params2)

    # Update the adjoint w.r.t. solid parameters
    for key, vector in adj_solid.items():
        if vector is not None:
            # the solid parameters only affect the displacement residual, under the
            # displacement-based newmark scheme, hence why there is only mult. by
            # adj_uva2[0]
            df2_dkey = dfn.assemble(df1_dsolid_form_adj[key])
            vector[:] -= df2_dkey*adj_state2[0]

    # The sum is done since the 'dt' at every spatial DOF, must be the same;
    # we can collapse each sensitivity into one
    df2_ddt = dfn.assemble(df1_ddt_form_adj)
    adj_dt2 = - (df2_ddt*adj_state2[0]).sum() \
              + newmark_v_dt(uva2[0], *uva1, dt2).inner(adj_state2[1]) \
              + newmark_a_dt(uva2[0], *uva1, dt2).inner(adj_state2[2])
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

        iter_params2 = {'uva0': uva1, 'dt': dt2, 'qp1': qp1, 'u1': uva2[0]}
        iter_params1 = {'uva0': uva0, 'dt': dt1, 'qp1': qp0, 'u1': uva1[0]}

        dcost_duva1 = functional.duva(f, ii, iter_params1, iter_params2)
        dcost_dqp1 = functional.dqp(f, ii, iter_params1, iter_params2)
        dcost_duvaqp1 = (*dcost_duva1, *dcost_dqp1)

        adj_state1_lhs = solve_adjrhs_recurence_exp(model, adj_state2, dcost_duvaqp1, iter_params2)
        adj_state1 = solve_adjoint_exp(model, adj_state1_lhs, iter_params1)

        # Update adjoint w.r.t parameters
        model.set_iter_params(**iter_params1)
        for key, vector in adj_solid.items():
            if vector is not None:
                df1_dkey = dfn.assemble(df1_dsolid_form_adj[key])
                vector -= df1_dkey*adj_state1[0]

        df1_ddt = dfn.assemble(df1_ddt_form_adj)
        adj_dt1 = - (df1_ddt*adj_state1[0]).sum() \
                  + newmark_v_dt(uva1[0], *uva0, dt1).inner(adj_state1[1]) \
                  + newmark_a_dt(uva1[0], *uva0, dt1).inner(adj_state1[2])
        adj_dt.insert(0, adj_dt1)

        # Set initial states to the previous states for the start of the next iteration
        uva2 = uva1
        uva1 = uva0
        adj_state2 = adj_state1

        qp2 = qp1
        qp1 = qp0

        dt2 = dt1

    # The model parameters should already be set to compute `F1`, so we can directly assemble below
    iter_params1 = {'uva0': uva1, 'dt': dt2, 'qp1': qp1, 'u1': uva2[0]}
    # iter_params0 = {'uva0': uva0, 'qp0': qp0, 'dt': dt1, 'u1': uva1[0]}
    # model.set_iter_params(**iter_params1)

    ## Calculate sensitivities wrt initial states
    adj_state1 = adj_state2
    dcost_state0 = (*functional.duva(f, 0, None, iter_params1),
                    *functional.dqp(f, 0, None, iter_params1))
    adj_state0 = solve_adjrhs_recurence_exp(model, adj_state1, dcost_state0, iter_params1)
    adj_u0, adj_v0, adj_a0, adj_q0, adj_p0 = adj_state0
    model.solid.bc_base.apply(adj_u0)
    model.solid.bc_base.apply(adj_v0)
    model.solid.bc_base.apply(adj_a0)

    # Since we've integrated over the whole time in reverse, the adjoint are not gradients
    grad = {}
    grad['u0'] = adj_u0
    grad['v0'] = adj_v0
    grad['a0'] = adj_a0

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


def decrement_adjoint_imp(model, adj_uva2, iter_params1, iter_params2, dcost_duva1):
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

def solve_adjoint_imp(model, adj_rhs, it_params, out=None):
    pass

def solve_adjrhs_recurrence_imp(model, adj_uva2, dcost_duva1, it_params2, out=None):
    pass

def decrement_adjoint_exp(model, adj_2, iter_params1, iter_params2, dcost_dstate1):
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
    adj_state1_lhs = solve_adjrhs_recurence_exp(model, adj_2, dcost_dstate1, iter_params2)

    adj_state1 = solve_adjoint_exp(model, adj_state1_lhs, iter_params1)

    return adj_state1

def solve_adjoint_exp(model, adj_rhs, it_params, out=None):
    """
    Solve for adjoint states given the RHS source vector

    Parameters
    ----------

    Returns
    -------
    """
    model.set_iter_params(**it_params)
    df2_du2 = model.assem_df1_du1_adj()
    df2_dp2 = dfn.assemble(model.forms['form.bi.df1_dpressure_adj'])
    dt = it_params['dt']

    if out is None:
        out = tuple([vec.copy() for vec in adj_rhs])
    adj_u, adj_v, adj_a, adj_q, adj_p = out

    adj_u_rhs, adj_v_rhs, adj_a_rhs, adj_q_rhs, adj_p_rhs = adj_rhs

    model.solid.bc_base.apply(adj_a_rhs)
    adj_a[:] = adj_a_rhs

    model.solid.bc_base.apply(adj_v_rhs)
    adj_v[:] = adj_v_rhs

    adj_u_rhs += newmark_v_du1(dt)*adj_v + newmark_a_du1(dt)*adj_a

    model.solid.bc_base.apply(df2_du2, adj_u_rhs)
    dfn.solve(df2_du2, adj_u, adj_u_rhs)

    # TODO: how should fluid boundary conditions be applied in a generic way?
    # the current way is specialized for the 1D case so most likely won't work in other cases
    # Below, I know there's subglottal pressure at 0 applied so I set that to be 0
    adj_q[:] = adj_q_rhs

    # Allocate a vector the for fluid side mat-vec multiplication
    _, matvec_adj_p_rhs = model.fluid.get_state_vecs()
    solid_dofs, fluid_dofs = model.get_fsi_scalar_dofs()
    matvec_adj_p_rhs[fluid_dofs] = (df2_dp2 * adj_u)[solid_dofs]

    adj_p_rhs -= matvec_adj_p_rhs
    adj_p_rhs[0] = 0 # Sets subglottal pressure boundary condition
    adj_p[:] = adj_p_rhs

    return adj_u, adj_v, adj_a, adj_q, adj_p

def solve_adjrhs_recurence_exp(model, adj_state2, dcost_dstate1, it_params2, out=None):
    """
    Solves the adjoint recurrence relations to return the rhs for adj_uva1

    Parameters
    ----------

    Returns
    -------
    """
    adj_u2, adj_v2, adj_a2, adj_q2, adj_p2 = adj_state2
    dcost_du1, dcost_dv1, dcost_da1, dcost_dq1, dcost_dp1 = dcost_dstate1

    ## Set form coefficients to represent f^{n+2} aka f2(uva1, uva2) -> uva2
    dt2 = it_params2['dt']
    model.set_iter_params(**it_params2)

    # Assemble needed forms
    df2_du1 = model.assem_df1_du0_adj()
    df2_dv1 = model.assem_df1_dv0_adj()
    df2_da1 = model.assem_df1_da0_adj()

    dq2_du1, dp2_du1 = model.get_flow_sensitivity()

    dpres2_du1 = dfn.Function(model.solid.vector_fspace).vector()
    dqres2_du1 = dfn.Function(model.solid.vector_fspace).vector()
    solid_dofs, fluid_dofs = model.get_fsi_vector_dofs()
    # breakpoint()
    dqres2_du1[solid_dofs] = (dq2_du1 * adj_q2)
    dpres2_du1[solid_dofs] = (dp2_du1.T @ adj_p2)[fluid_dofs]

    adj_u1_lhs = dcost_du1 - (df2_du1*adj_u2 - newmark_v_du0(dt2)*adj_v2 - newmark_a_du0(dt2)*adj_a2 - dqres2_du1 - dpres2_du1)
    adj_v1_lhs = dcost_dv1 - (df2_dv1*adj_u2 - newmark_v_dv0(dt2)*adj_v2 - newmark_a_dv0(dt2)*adj_a2)
    adj_a1_lhs = dcost_da1 - (df2_da1*adj_u2 - newmark_v_da0(dt2)*adj_v2 - newmark_a_da0(dt2)*adj_a2)
    adj_q1_lhs = dcost_dq1.copy()
    adj_p1_lhs = dcost_dp1.copy()

    return adj_u1_lhs, adj_v1_lhs, adj_a1_lhs, adj_q1_lhs, adj_p1_lhs

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

