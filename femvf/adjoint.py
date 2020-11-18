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

    # Make adjoint forms for sensitivity of parameters
    solid = model.solid

    ## Allocate space for the adjoints of all the parameters
    adj_dt = []
    adj_solid = model.solid.get_properties_vec(set_default=False)
    adj_solid.set(0.0)
    adj_fluid = model.fluid.get_properties_vec(set_default=False)
    adj_fluid.set(0.0)

    ## Load states/parameters
    N = f.size
    times = f.get_times()

    ## Initialize the adj rhs
    dcost_duva1 = functional.duva(f, N-1)
    dcost_dqp1 = functional.dqp(f, N-1)
    adj_state1_rhs = linalg.concatenate(dcost_duva1, dcost_dqp1)

    ## Loop through states for adjoint computation
    # Note that ii corresponds to the time index of the adjoint state we are solving for.
    # In a given loop, adj^{ii+1} is known, and the iteration of the loop finds adj^{ii}
    for ii in range(N-1, 0, -1):
        # Properties at index 2 through 1 were loaded during initialization, so we only need to read
        # index 0
        state0, state1 = f.get_state(ii-1), f.get_state(ii)
        if variable_controls:
            control0, control1 = f.get_control(ii-1), f.get_control(ii)

        dt1 = times[ii] - times[ii-1]

        # All the calculations are based on the state of the model at iter_params1, so you only have
        # to set it once here
        model.set_ini_state(state0)
        model.set_fin_state(state1)
        model.set_control(control1)
        model.dt = dt1

        # adj_state1 = solve_adj(model, adj_state1_rhs, iter_params1)
        adj_state1 = model.solve_dres_dstate1_adj(adj_state1_rhs)

        # Update gradients wrt parameters using the adjoint
        # _adj_solid = solve_grad_solid(model, adj_state1, iter_params1, adj_solid.copy(), df1_dsolid_form_adj)
        adj_p = _solve_grad_solid(model, adj_state1, adj_solid) 
        adj_solid = adj_p[:len(adj_solid.size)]

        adj_dt1 = solve_grad_dt(model, adj_state1) + functional.ddt(f, ii)
        adj_dt.insert(0, adj_dt1)

        # Find the RHS for the next iteration
        dcost_duva0 = functional.duva(f, ii-1)
        dcost_dqp0 = functional.dqp(f, ii-1)
        dcost_dstate0 = linalg.concatenate(dcost_duva0, dcost_dqp0)

        # adj_state0_rhs = solve_adj_rhs(model, adj_state1, dcost_dstate0, iter_params1)
        adj_state0_rhs = _solve_adj_rhs(model, adj_state1, dcost_dstate0)

        # Set initial states to the previous states for the start of the next iteration
        adj_state1_rhs = adj_state0_rhs

    ## Calculate gradients
    grad_state = adj_state1_rhs

    # Finally, if the functional is sensitive to the parameters, you have to add their sensitivity
    # components once
    dfunc_dsolid = functional.dsolid(f)
    grad_solid = adj_solid + dfunc_dsolid

    dfunc_dfluid = functional.dfluid(f)
    grad_fluid = adj_fluid + dfunc_dfluid

    grad_props = linalg.concatenate(grad_solid, grad_fluid)

    ## Calculate gradients
    grad_controls = None

    # Calculate sensitivities w.r.t integration times
    grad_dt = np.array(adj_dt)

    grad_times = np.zeros(N)
    # the conversion below is becase dt = t1 - t0
    grad_times[1:] = grad_dt
    grad_times[:-1] -= grad_dt

    return functional_value, grad_state, grad_controls, grad_props, grad_times

def _solve_grad_solid(model, adj_state1, grad_solid):
    bsize = len(model.solid.get_properties_vec().size)
    adj_solid = model.apply_dres_dp_adj(adj_state1)[:bsize]
    return grad_solid - adj_solid 

def solve_grad_dt(model, adj_state1):
    """
    Calculate the gradietn wrt dt
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

# def solve_adj_exp(model, adj_rhs, it_params, out=None):
#     """
#     Solve for adjoint states given the RHS source vector

#     Parameters
#     ----------

#     Returns
#     -------
#     """
#     ## Assemble sensitivity matrices
#     # model.set_iter_params(**it_params)
#     dt = it_params['dt']

#     dfu2_du2 = model.solid.cached_form_assemblers['bilin.df1_du1_adj'].assemble()
#     dfv2_du2 = 0 - newmark_v_du1(dt)
#     dfa2_du2 = 0 - newmark_a_du1(dt)

#     # model.set_ini_state((it_params['u1'], 0, 0))
#     dq_du, dp_du = model.solve_dqp1_du1_solid(adjoint=True)
#     dfq2_du2 = 0 - dq_du
#     dfp2_du2 = 0 - dp_du

#     ## Do the linear algebra that solves for the adjoint states
#     adj_uva = model.solid.get_state_vec()
#     adj_qp = model.fluid.get_state_vec()

#     adj_u_rhs, adj_v_rhs, adj_a_rhs, adj_q_rhs, adj_p_rhs = adj_rhs

#     model.solid.bc_base.apply(adj_a_rhs)
#     adj_uva['a'][:] = adj_a_rhs

#     model.solid.bc_base.apply(adj_v_rhs)
#     adj_uva['v'][:] = adj_v_rhs

#     # TODO: Think of how to apply fluid boundary conditions in a generic way.
#     # There are no boundary conditions for the Bernoulli case because of the way it's coded but
#     # this will be needed for different models
#     adj_qp['q'][:] = adj_q_rhs
#     adj_qp['p'][:] = adj_p_rhs

#     _adj_p = dfp2_du2.getVecRight()
#     _adj_p[:] = adj_qp['p']

#     adj_u_rhs = adj_u_rhs - (
#         dfv2_du2*adj_uva['v'] + dfa2_du2*adj_uva['a'] + dfq2_du2*adj_qp['q'] 
#         + dfn.PETScVector(dfp2_du2*_adj_p))
#     model.solid.bc_base.apply(dfu2_du2, adj_u_rhs)
#     dfn.solve(dfu2_du2, adj_uva['u'], adj_u_rhs, 'petsc')

#     return linalg.concatenate(adj_uva, adj_qp)

def _solve_adj(model, adj_rhs, out=None):
    adj = model.solve_dres_dstate1_adj(adj_rhs)
    return adj

# def solve_adj_rhs_exp(model, adj_state2, dcost_dstate1, it_params2, out=None):
#     """
#     Solves the adjoint recurrence relations to return the rhs

#     ## Set form coefficients to represent f^{n+2} aka f2(uva1, uva2) -> uva2

#     Parameters
#     ----------

#     Returns
#     -------
#     """
#     adj_u2, adj_v2, adj_a2, adj_q2, adj_p2 = adj_state2
#     # adj_uva2, adj_qp2 = adj_state2[:3], adj_state2[3:]
#     dcost_du1, dcost_dv1, dcost_da1, dcost_dq1, dcost_dp1 = dcost_dstate1

#     ## Assemble sensitivity matrices
#     dt2 = it_params2['dt']
#     # model.set_iter_params(**it_params2)

#     dfu2_du1 = model.solid.cached_form_assemblers['bilin.df1_du0_adj'].assemble()
#     dfu2_dv1 = model.solid.cached_form_assemblers['bilin.df1_dv0_adj'].assemble()
#     dfu2_da1 = model.solid.cached_form_assemblers['bilin.df1_da0_adj'].assemble()
#     # df1_dp1 is assembled because the explicit coupling is achieved through passing the previous
#     # pressure as the current pressure rather than changing the actualy governing equations to use
#     # the previous pressure
#     dfu2_dp1 = dfn.assemble(model.solid.forms['form.bi.df1_dp1_adj'])

#     dfv2_du1 = 0 - newmark_v_du0(dt2)
#     dfv2_dv1 = 0 - newmark_v_dv0(dt2)
#     dfv2_da1 = 0 - newmark_v_da0(dt2)

#     dfa2_du1 = 0 - newmark_a_du0(dt2)
#     dfa2_dv1 = 0 - newmark_a_dv0(dt2)
#     dfa2_da1 = 0 - newmark_a_da0(dt2)

#     solid_dofs, fluid_dofs = model.get_fsi_scalar_dofs()
#     dfu2_dp1 = dfn.as_backend_type(dfu2_dp1).mat()
#     dfu2_dp1 = linalg.reorder_mat_rows(dfu2_dp1, solid_dofs, fluid_dofs, fluid_dofs.size)
#     matvec_adj_p_rhs = dfu2_dp1*dfn.as_backend_type(adj_u2).vec()

#     adj_u1_rhs = dcost_du1 - (dfu2_du1*adj_u2 + dfv2_du1*adj_v2 + dfa2_du1*adj_a2)
#     adj_v1_rhs = dcost_dv1 - (dfu2_dv1*adj_u2 + dfv2_dv1*adj_v2 + dfa2_dv1*adj_a2)
#     adj_a1_rhs = dcost_da1 - (dfu2_da1*adj_u2 + dfv2_da1*adj_v2 + dfa2_da1*adj_a2)
#     adj_q1_rhs = dcost_dq1 - 0
#     adj_p1_rhs = dcost_dp1 - matvec_adj_p_rhs

#     keys = adj_state2.keys
#     vecs = [adj_u1_rhs, adj_v1_rhs, adj_a1_rhs, adj_q1_rhs, adj_p1_rhs]
#     return linalg.BlockVec(vecs, keys)

def _solve_adj_rhs(model, adj_state2, dcost_dstate1, out=None):
    b = model.apply_dres_dstate0_adj(adj_state2)
    return dcost_dstate1 - b

# def get_df1_dsolid_forms(solid):
#     """
#     Return a dictionary of forms of derivatives of f1 with respect to the various solid parameters
#     """
#     df1_dsolid = {}
#     for key in solid.PROPERTY_TYPES:
#         try:
#             df1_dsolid[key] = dfn.adjoint(ufl.derivative(solid.f1, solid.forms[f'coeff.prop.{key}'],
#                                                          solid.scalar_trial))
#         except RuntimeError:
#             df1_dsolid[key] = None

#         if df1_dsolid[key] is not None:
#             try:
#                 dfn.assemble(df1_dsolid[key])
#             except RuntimeError:
#                 df1_dsolid[key] = None
#     return df1_dsolid
