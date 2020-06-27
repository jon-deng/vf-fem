"""
Adjoint model.

Makes a vocal fold go elggiw elggiw.

I'm using CGS : cm-g-s units
"""

import numpy as np
import dolfin as dfn
import ufl
from petsc4py import PETSc

# from .newmark import *
# from .newmark import
from .newmark import (newmark_v_du1, newmark_v_du0, newmark_v_dv0, newmark_v_da0, newmark_v_dt,
                      newmark_a_du1, newmark_a_du0, newmark_a_dv0, newmark_a_da0, newmark_a_dt)
from . import linalg

def adjoint(model, f, functional, coupling='explicit'):
    """
    Returns the gradient of the cost function using the adjoint model.

    Parameters
    ----------
    model : model.ForwardModel
    f : statefile.StateFile
    functional : functionals.Functional
    coupling : 'implicit' or 'explicit'
        How the fluid and structure should be coupled
    show_figures : bool
        Whether to display a figure showing the solution or not.

    Returns
    -------
    float
        The value of the functional
    dict(str: dfn.Vector)
        The gradient of the functional wrt parameter labelled by `str`
        # TODO: Only gradient wrt. solid properties are included right now
    """
    solve_adj, solve_adj_rhs = None, None
    if coupling == 'explicit':
        solve_adj = solve_adj_exp
        solve_adj_rhs = solve_adj_rhs_exp
    elif coupling == 'implicit':
        solve_adj = solve_adj_imp
        solve_adj_rhs = solve_adj_rhs_imp
    else:
        raise ValueError("`coupling` can only be implicit or explicit you goofball")

    # Assumes fluid and solid properties are constant in time
    fluid_props = f.get_fluid_props(0)
    solid_props = f.get_solid_props()

    model.set_fluid_props(fluid_props)
    model.set_solid_props(solid_props)

    # run the functional once to initialize any cached values
    functional_value = functional(f)

    # Make adjoint forms for sensitivity of parameters
    solid = model.solid
    df1_dsolid_form_adj = get_df1_dsolid_forms(solid)

    ## Allocate space for the adjoints of all the parameters
    # TODO: This is kind of an ugly hardcoded solution. You should come up with a way to encapsulate
    # solid properties as a collection of vector/array objects. Since this is a gradient,
    # it can use the same data structure as the properties are stored in
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
    # adj_fluid = ....

    ## Load states/parameters
    uva0 = tuple([dfn.Function(solid.vector_fspace).vector() for i in range(3)])
    uva1 = tuple([dfn.Function(solid.vector_fspace).vector() for i in range(3)])
    N = f.size
    times = f.get_times()

    qp2 = (None, None)
    uva2 = (None, None, None)
    uva1 = f.get_state(N-1, out=uva1)
    uva0 = f.get_state(N-2, out=uva0)
    qp1 = f.get_fluid_state(N-1)
    qp0 = f.get_fluid_state(N-2)
    dt2 = None
    dt1 = times[N-1] - times[N-2]

    qp1_, qp2_ = None, None
    if coupling == 'explicit':
        qp2_ = qp1
        qp1_ = qp0
    else:
        qp2_ = qp2
        qp1_ = qp1
    iter_params2 = {'uva0': uva1, 'qp0': qp1, 'dt': dt2, 'qp1': qp2_, 'uva1': uva2}
    iter_params1 = {'uva0': uva0, 'qp0': qp0, 'dt': dt1, 'qp1': qp1_, 'uva1': uva1}

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

        # Set the iter params
        qp0_, qp1_ = None, None
        if coupling == 'explicit':
            qp1_ = qp0
            qp0_ = qp_n1
        else:
            qp1_ = qp1
            qp0_ = qp0
        iter_params1 = {'uva0': uva0, 'qp0': qp0, 'dt': dt1, 'qp1': qp1_, 'uva1': uva1}
        iter_params0 = {'uva0': uva_n1, 'qp0': qp_n1, 'dt': dt0, 'qp1': qp0_, 'uva1': uva0}

        model.set_iter_params(**iter_params1)
        res = dfn.assemble(model.solid.forms['form.un.f1'])
        model.solid.bc_base.apply(res)

        adj_state1 = solve_adj(model, adj_state1_rhs, iter_params1)

        # Update gradients wrt parameters using the adjoint
        adj_solid = solve_grad_solid(model, adj_state1, iter_params1, adj_solid, df1_dsolid_form_adj)
        adj_dt1 = solve_grad_dt(model, adj_state1, iter_params1) + functional.ddt(f, ii)
        adj_dt.insert(0, adj_dt1)

        # Find the RHS for the next iteration
        dcost_duva0 = functional.duva(f, ii-1, iter_params0, iter_params1)
        dcost_dqp0 = functional.dqp(f, ii-1, iter_params0, iter_params1)
        dcost_dstate0 = (*dcost_duva0, *dcost_dqp0)
        adj_state0_rhs = solve_adj_rhs(model, adj_state1, dcost_dstate0, iter_params1)

        # Set initial states to the previous states for the start of the next iteration
        adj_state1_rhs = adj_state0_rhs

        # iter_params2 = iter_params1
        # adj_state2 = adj_state1
        # uva2 = uva1
        # uva1 = uva0
        # qp1 = qp0
        # dt2 = dt1

    # Finally, if the functional is sensitive to the parameters, you have to add their sensitivity
    # components once
    dfunc_dsolid = functional.dsolid(f)
    for key, vector in adj_solid.items():
        if vector is not None:
            vector[:] += dfunc_dsolid[key]

    ## Calculate gradients
    # Calculate sensitivities wrt initial states
    adj_u0, adj_v0, adj_a0, adj_q0, adj_p0 = adj_state1_rhs
    model.solid.bc_base.apply(adj_u0)
    model.solid.bc_base.apply(adj_v0)
    model.solid.bc_base.apply(adj_a0)

    grad = {}
    grad_u0 = adj_u0
    grad_v0 = adj_v0
    grad_a0 = adj_a0

    if coupling == 'explicit':
        # model.set_ini_state(*uva0)
        dq_du, dp_du = model.solve_dqp0_du0_solid(adjoint=True)
        adj_p0_ = dp_du.getVecRight()
        adj_p0_[:] = adj_p0

        # have to convert (dp_du*adj_p0_) to dfn.PETScVector to add left and right terms
        grad_u0 += dfn.PETScVector(dp_du*adj_p0_) + dq_du*adj_q0

    grad_uva = (grad_u0, grad_v0, grad_a0)

    # Calculate sensitivities w.r.t solid properties
    grad_solid = model.solid.get_properties()
    grad_solid.vector[:] = 0
    for key, vector in adj_solid.items():
        if vector is not None:
            if grad_solid[key].shape == ():
                grad_solid[key][()] = vector
            else:
                grad_solid[key][:] = vector

    # Calculate sensitivties w.r.t fluid properties
    grad_fluid = model.fluid.get_properties()
    grad_fluid.vector[:] = 0

    # Calculate sensitivities w.r.t integration times
    grad_dt = np.array(adj_dt)

    grad_times = np.zeros(N)
    # the conversion below is becase dt = t1 - t0
    grad_times[1:] = grad_dt
    grad_times[:-1] -= grad_dt

    for key, vector in adj_solid.items():
        if vector is not None:
            grad[key] = vector

    return functional_value, grad_uva, grad_solid, grad_fluid, grad_times

def solve_grad_solid(model, adj_state1, iter_params1, grad_solid, df1_dsolid_form_adj):
    """
    Update the gradient wrt solid parameters
    """
    model.set_iter_params(**iter_params1)
    for key, vector in grad_solid.items():
        if vector is not None:
            df1_dkey = dfn.assemble(df1_dsolid_form_adj[key])
            vector -= df1_dkey*adj_state1[0]
    return grad_solid

def solve_grad_dt(model, adj_state1, iter_params1):
    """
    Calculate the gradietn wrt dt
    """
    model.set_iter_params(**iter_params1)
    uva0 = iter_params1['uva0']
    uva1 = iter_params1['uva1']
    dt1 = iter_params1['dt']

    dfu1_ddt = dfn.assemble(model.solid.forms['form.bi.df1_dt_adj'])
    dfv1_ddt = 0 - newmark_v_dt(uva1[0], *uva0, dt1)
    dfa1_ddt = 0 - newmark_a_dt(uva1[0], *uva0, dt1)

    adj_u1, adj_v1, adj_a1 = adj_state1[:3]
    adj_dt1 = -(dfu1_ddt*adj_u1).sum() - dfv1_ddt.inner(adj_v1) - dfa1_ddt.inner(adj_a1)
    return adj_dt1

def solve_adj_imp(model, adj_rhs, it_params, out=None):
    """
    Solve for adjoint states given the RHS source vector

    Parameters
    ----------

    Returns
    -------
    """
    ## Assemble sensitivity matrices
    model.set_iter_params(**it_params)
    dt = it_params['dt']

    dfu2_du2 = model.assem_df1_du1_adj()
    dfv2_du2 = 0 - newmark_v_du1(dt)
    dfa2_du2 = 0 - newmark_a_du1(dt)
    dfu2_dp2 = dfn.assemble(model.solid.forms['form.bi.df1_dp1_adj'])

    # map dfu2_dp2 to have p on the fluid domain
    solid_dofs, fluid_dofs = model.get_fsi_scalar_dofs()
    dfu2_dp2 = dfn.as_backend_type(dfu2_dp2).mat()
    dfu2_dp2 = linalg.reorder_mat_rows(dfu2_dp2, solid_dofs, fluid_dofs, model.fluid.p1.size)

    dq_du, dp_du = model.solve_dqp1_du1_solid(adjoint=True)
    dfq2_du2 = 0 - dq_du
    dfp2_du2 = 0 - dp_du

    ## Do the linear algebra that solves for the adjoint states
    if out is None:
        out = tuple([vec.copy() for vec in adj_rhs])
    adj_u, adj_v, adj_a, adj_q, adj_p = out

    adj_u_rhs, adj_v_rhs, adj_a_rhs, adj_q_rhs, adj_p_rhs = adj_rhs

    # adjoint states for v, a, and q are explicit so we can solve for them
    model.solid.bc_base.apply(adj_v_rhs)
    adj_v[:] = adj_v_rhs

    model.solid.bc_base.apply(adj_a_rhs)
    adj_a[:] = adj_a_rhs

    # TODO: how to apply fluid boundary conditions in a generic way?
    adj_q[:] = adj_q_rhs

    adj_u_rhs -= dfv2_du2*adj_v + dfa2_du2*adj_a + dfq2_du2*adj_q

    bc_dofs = np.array(list(model.solid.bc_base.get_boundary_values().keys()), dtype=np.int32)
    model.solid.bc_base.apply(dfu2_du2, adj_u_rhs)
    dfp2_du2.zeroRows(bc_dofs, diag=0.0)
    # model.solid.bc_base.zero_columns(dfu2_du2, adj_u_rhs.copy(), diagonal_value=1.0)

    # solve the coupled system for pressure and displacement residuals
    dfu2_du2_mat = dfn.as_backend_type(dfu2_du2).mat()
    blocks = [[dfu2_du2_mat, dfp2_du2], [dfu2_dp2, 1.0]]

    dfup2_dup2 = linalg.form_block_matrix(blocks)
    adj_up, rhs = dfup2_dup2.getVecs()

    # calculate rhs vectors
    rhs[:adj_u_rhs.size()] = adj_u_rhs
    rhs[adj_u_rhs.size():] = adj_p_rhs

    # Solve the block linear system with LU factorization
    ksp = PETSc.KSP().create()
    ksp.setType(ksp.Type.PREONLY)

    pc = ksp.getPC()
    pc.setType(pc.Type.LU)

    ksp.setOperators(dfup2_dup2)
    ksp.solve(rhs, adj_up)

    adj_u[:] = adj_up[:adj_u_rhs.size()]
    adj_p[:] = adj_up[adj_u_rhs.size():]

    return adj_u, adj_v, adj_a, adj_q, adj_p

def solve_adj_rhs_imp(model, adj_state2, dcost_dstate1, it_params2, out=None):
    """
    Solves the adjoint recurrence relations to return the rhs

    ## Set form coefficients to represent f^{n+2} aka f2(uva1, uva2) -> uva2

    Parameters
    ----------

    Returns
    -------
    """
    adj_u2, adj_v2, adj_a2, adj_q2, adj_p2 = adj_state2
    dcost_du1, dcost_dv1, dcost_da1, dcost_dq1, dcost_dp1 = dcost_dstate1

    ## Assemble sensitivity matrices
    dt2 = it_params2['dt']
    model.set_iter_params(**it_params2)

    dfu2_du1 = model.assem_df1_du0_adj()
    dfu2_dv1 = model.assem_df1_dv0_adj()
    dfu2_da1 = model.assem_df1_da0_adj()

    dfv2_du1 = 0 - newmark_v_du0(dt2)
    dfv2_dv1 = 0 - newmark_v_dv0(dt2)
    dfv2_da1 = 0 - newmark_v_da0(dt2)

    dfa2_du1 = 0 - newmark_a_du0(dt2)
    dfa2_dv1 = 0 - newmark_a_dv0(dt2)
    dfa2_da1 = 0 - newmark_a_da0(dt2)

    ## Do the matrix vector multiplication that gets the RHS for the adjoint equations
    # Allocate a vector the for fluid side mat-vec multiplication
    adj_u1_rhs = dcost_du1 - (dfu2_du1*adj_u2 + dfv2_du1*adj_v2 + dfa2_du1*adj_a2)
    adj_v1_rhs = dcost_dv1 - (dfu2_dv1*adj_u2 + dfv2_dv1*adj_v2 + dfa2_dv1*adj_a2)
    adj_a1_rhs = dcost_da1 - (dfu2_da1*adj_u2 + dfv2_da1*adj_v2 + dfa2_da1*adj_a2)
    adj_q1_rhs = dcost_dq1 - 0
    adj_p1_rhs = dcost_dp1 - 0

    return adj_u1_rhs, adj_v1_rhs, adj_a1_rhs, adj_q1_rhs, adj_p1_rhs

def solve_adj_exp(model, adj_rhs, it_params, out=None):
    """
    Solve for adjoint states given the RHS source vector

    Parameters
    ----------

    Returns
    -------
    """
    ## Assemble sensitivity matrices
    model.set_iter_params(**it_params)
    dt = it_params['dt']

    dfu2_du2 = model.assem_df1_du1_adj()
    dfv2_du2 = 0 - newmark_v_du1(dt)
    dfa2_du2 = 0 - newmark_a_du1(dt)

    # model.set_ini_state(it_params['u1'], 0, 0)
    dq_du, dp_du = model.solve_dqp1_du1_solid(adjoint=True)
    dfq2_du2 = 0 - dq_du
    dfp2_du2 = 0 - dp_du

    ## Do the linear algebra that solves for the adjoint states
    if out is None:
        out = tuple([vec.copy() for vec in adj_rhs])
    adj_u, adj_v, adj_a, adj_q, adj_p = out

    adj_u_rhs, adj_v_rhs, adj_a_rhs, adj_q_rhs, adj_p_rhs = adj_rhs

    model.solid.bc_base.apply(adj_a_rhs)
    adj_a[:] = adj_a_rhs

    model.solid.bc_base.apply(adj_v_rhs)
    adj_v[:] = adj_v_rhs

    # TODO: Think of how to apply fluid boundary conditions in a generic way.
    # There are no boundary conditions for the Bernoulli case because of the way it's coded but
    # this will be needed for different models
    adj_q[:] = adj_q_rhs

    adj_p = dfp2_du2.getVecRight()
    adj_p[:] = adj_p_rhs

    adj_u_rhs -= dfv2_du2*adj_v + dfa2_du2*adj_a + dfq2_du2*adj_q + dfn.PETScVector(dfp2_du2*adj_p)
    model.solid.bc_base.apply(dfu2_du2, adj_u_rhs)
    dfn.solve(dfu2_du2, adj_u, adj_u_rhs)

    return adj_u, adj_v, adj_a, adj_q, adj_p

def solve_adj_rhs_exp(model, adj_state2, dcost_dstate1, it_params2, out=None):
    """
    Solves the adjoint recurrence relations to return the rhs

    ## Set form coefficients to represent f^{n+2} aka f2(uva1, uva2) -> uva2

    Parameters
    ----------

    Returns
    -------
    """
    adj_u2, adj_v2, adj_a2, adj_q2, adj_p2 = adj_state2
    dcost_du1, dcost_dv1, dcost_da1, dcost_dq1, dcost_dp1 = dcost_dstate1

    ## Assemble sensitivity matrices
    dt2 = it_params2['dt']
    model.set_iter_params(**it_params2)

    dfu2_du1 = model.assem_df1_du0_adj()
    dfu2_dv1 = model.assem_df1_dv0_adj()
    dfu2_da1 = model.assem_df1_da0_adj()
    # df1_dp1 is assembled because the explicit coupling is achieved through passing the previous
    # pressure as the current pressure rather than changing the actualy governing equations to use
    # the previous pressure
    dfu2_dp1 = dfn.assemble(model.forms['form.bi.df1_dp1_adj'])

    dfv2_du1 = 0 - newmark_v_du0(dt2)
    dfv2_dv1 = 0 - newmark_v_dv0(dt2)
    dfv2_da1 = 0 - newmark_v_da0(dt2)

    dfa2_du1 = 0 - newmark_a_du0(dt2)
    dfa2_dv1 = 0 - newmark_a_dv0(dt2)
    dfa2_da1 = 0 - newmark_a_da0(dt2)

    solid_dofs, fluid_dofs = model.get_fsi_scalar_dofs()
    dfu2_dp1 = dfn.as_backend_type(dfu2_dp1).mat()
    dfu2_dp1 = linalg.reorder_mat_rows(dfu2_dp1, solid_dofs, fluid_dofs, fluid_dofs.size)
    matvec_adj_p_rhs = dfu2_dp1*dfn.as_backend_type(adj_u2).vec()

    adj_u1_rhs = dcost_du1 - (dfu2_du1*adj_u2 + dfv2_du1*adj_v2 + dfa2_du1*adj_a2)
    adj_v1_rhs = dcost_dv1 - (dfu2_dv1*adj_u2 + dfv2_dv1*adj_v2 + dfa2_dv1*adj_a2)
    adj_a1_rhs = dcost_da1 - (dfu2_da1*adj_u2 + dfv2_da1*adj_v2 + dfa2_da1*adj_a2)
    adj_q1_rhs = dcost_dq1 - 0
    adj_p1_rhs = dcost_dp1 - matvec_adj_p_rhs

    return adj_u1_rhs, adj_v1_rhs, adj_a1_rhs, adj_q1_rhs, adj_p1_rhs

def get_df1_dsolid_forms(solid):
    df1_dsolid = {}
    for key in solid.PROPERTY_TYPES:
        try:
            df1_dsolid[key] = dfn.adjoint(ufl.derivative(solid.f1, solid.forms[f'coeff.prop.{key}'],
                                                         solid.scalar_trial))
        except RuntimeError:
            df1_dsolid[key] = None

        if df1_dsolid[key] is not None:
            try:
                dfn.assemble(df1_dsolid[key])
            except RuntimeError:
                df1_dsolid[key] = None
    return df1_dsolid
