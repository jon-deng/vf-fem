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

    ## Load states/parameters
    N = f.size
    times = f.get_times()

    uva2 = (None, None, None)
    uva1 = f.get_state(N-1, out=uva1)
    qp1 = f.get_fluid_state(N-1)
    qp0 = f.get_fluid_state(N-2)
    dt2 = None
    dt1 = times[N-1] - times[N-2]
    iter_params2 = {'uva0': uva1, 'qp0': qp1, 'dt': dt2, 'qp1': qp1, 'u1': uva2[0]}
    iter_params1 = {'uva0': uva0, 'qp0': qp0, 'dt': dt1, 'qp1': qp0, 'u1': uva1[0]}

    ## Initialize the adj rhs
    dcost_duva1 = functional.duva(f, N-1, iter_params1, iter_params2)
    dcost_dqp1 = functional.dqp(f, N-1, iter_params1, iter_params2)
    adj_state1_rhs = (*dcost_duva1, *dcost_dqp1)

    ## Loop through states for adjoint computation
    # Note that ii corresponds to the time index of the adjoint state we are solving for.
    # In a given loop, adj^{ii+1} is known, and the iteration of the loop finds adj^{ii}
    for ii in range(N-1, 0, -1):
        # Properties at index 2 through 1 were loaded during initialization, so we only need to read
        # index 0
        uva1 = f.get_state(ii, out=uva1)
        qp1 = f.get_fluid_state(ii)

        dt1 = times[ii] - times[ii-1]

        uva0 = f.get_state(ii-1, out=uva0)
        qp0 = f.get_fluid_state(ii-1)

        uva_n1 = f.get_state(ii-2)
        qp_n1 = f.get_fluid_state(ii-2)
        dt0 = times[ii-1] - times[ii-2]

        # Set the iter params according to an explicit iteration
        iter_params1 = {'uva0': uva0, 'qp0': qp0, 'dt': dt1, 'qp1': qp0, 'u1': uva1[0]}
        iter_params0 = {'uva0': uva_n1, 'qp0': qp_n1, 'dt': dt0, 'qp1': qp_n1, 'u1': uva0[0]}

        print(adj_state1_rhs[0].norm('l2'), np.linalg.norm(adj_state1_rhs[3]), np.linalg.norm(adj_state1_rhs[4]))
        adj_state1 = solve_adjoint_exp(model, adj_state1_rhs, iter_params1)
        print(adj_state1[0].norm('l2'), np.linalg.norm(adj_state1[3]), np.linalg.norm(adj_state1[4]))

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

        # Find the RHS for the next iteration
        dcost_duva0 = functional.duva(f, ii, iter_params0, iter_params1)
        dcost_dqp0 = functional.dqp(f, ii, iter_params0, iter_params1)
        dcost_dstate0 = (*dcost_duva0, *dcost_dqp0)
        # print(dcost_dstate0[0].norm('l2'))
        adj_state0_rhs = solve_adjrhs_recurrence_exp(model, adj_state1, dcost_dstate0, iter_params1)

        # Set initial states to the previous states for the start of the next iteration
        adj_state1_rhs = adj_state0_rhs

        # iter_params2 = iter_params1
        # adj_state2 = adj_state1
        # uva2 = uva1
        # uva1 = uva0
        # qp1 = qp0
        # dt2 = dt1

    ## Calculate sensitivities wrt initial states
    adj_u0, adj_v0, adj_a0, adj_q0, adj_p0 = adj_state1_rhs
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

def solve_adjoint_imp(model, adj_rhs, it_params, out=None):
    pass

def solve_adjrhs_recurrence_imp(model, adj_uva2, dcost_duva1, it_params2, out=None):
    pass

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
    dt = it_params['dt']

    if out is None:
        out = tuple([vec.copy() for vec in adj_rhs])
    adj_u, adj_v, adj_a, adj_q, adj_p = out

    adj_u_rhs, adj_v_rhs, adj_a_rhs, adj_q_rhs, adj_p_rhs = adj_rhs

    model.solid.bc_base.apply(adj_a_rhs)
    adj_a[:] = adj_a_rhs

    model.solid.bc_base.apply(adj_v_rhs)
    adj_v[:] = adj_v_rhs

    # TODO: how should fluid boundary conditions be applied in a generic way?
    # the current way is specialized for the 1D case so most likely won't work in other cases
    # Below, I know there's subglottal pressure at 0 applied so I set that to be 0
    adj_q[:] = adj_q_rhs

    adj_p_rhs[0] = 0 # Sets subglottal pressure boundary condition
    adj_p[:] = adj_p_rhs

    dq_du, dp_du = model.get_flow_sensitivity()
    dpres_du = dfn.Function(model.solid.vector_fspace).vector()
    dqres_du = dfn.Function(model.solid.vector_fspace).vector()
    solid_dofs, fluid_dofs = model.get_fsi_vector_dofs()

    dqres_du[solid_dofs] = (dq_du * adj_q)
    dpres_du[solid_dofs] = (dp_du.T @ adj_p)[fluid_dofs]

    adj_u_rhs += newmark_v_du1(dt)*adj_v + newmark_a_du1(dt)*adj_a + dqres_du + dpres_du

    model.solid.bc_base.apply(df2_du2, adj_u_rhs)
    dfn.solve(df2_du2, adj_u, adj_u_rhs)

    return adj_u, adj_v, adj_a, adj_q, adj_p

def solve_adjrhs_recurrence_exp(model, adj_state2, dcost_dstate1, it_params2, out=None):
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
    df2_dp1 = dfn.assemble(model.forms['form.bi.df1_dpressure_adj'])

    # Allocate a vector the for fluid side mat-vec multiplication
    _, matvec_adj_p_rhs = model.fluid.get_state_vecs()
    solid_dofs, fluid_dofs = model.get_fsi_scalar_dofs()
    matvec_adj_p_rhs[fluid_dofs] = (df2_dp1 * adj_u2)[solid_dofs]

    adj_u1_rhs = dcost_du1 - (df2_du1*adj_u2 - newmark_v_du0(dt2)*adj_v2 - newmark_a_du0(dt2)*adj_a2)
    adj_v1_rhs = dcost_dv1 - (df2_dv1*adj_u2 - newmark_v_dv0(dt2)*adj_v2 - newmark_a_dv0(dt2)*adj_a2)
    adj_a1_rhs = dcost_da1 - (df2_da1*adj_u2 - newmark_v_da0(dt2)*adj_v2 - newmark_a_da0(dt2)*adj_a2)
    adj_q1_rhs = dcost_dq1.copy()
    adj_p1_rhs = dcost_dp1.copy() + matvec_adj_p_rhs

    return adj_u1_rhs, adj_v1_rhs, adj_a1_rhs, adj_q1_rhs, adj_p1_rhs

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

