"""
Forward model

Uses CGS (cm-g-s) units unless otherwise stated
"""

from functools import partial

import numpy as np
import dolfin as dfn
from petsc4py import PETSc

from . import solids
from . import statefile as sf
from . import linalg

DEFAULT_NEWTON_SOLVER_PRM = {
    'linear_solver': 'petsc',
    'absolute_tolerance': 1e-7,
    'relative_tolerance': 1e-9,
    'max_num_it': 5}

FIXEDPOINT_SOLVER_PRM = {
    'absolute_tolerance': 1e-8,
    'relative_tolerance': 1e-11}

def integrate(model, ini_state, controls, props, times, idx_meas=None,
              h5file='tmp.h5', h5group='/', newton_solver_prm=None):
    """
    Integrate the model over each time in `times` for the specified parameters

    Parameters
    ----------
    model : model.ForwardModel
    uva : tuple of dfn.Vector
        Initial solid state (displacement, velocity, acceleration)
    solid_props, fluid_props : femvf.Properties
        Solid / fluid parameter vectors
    times : array of float
        Array of discrete integration times. Each time point is one integration point so the time
        between successive time points is a time step.
    idx_meas : array of int
        Array marking which integration points correspond to something (usu. measurements)
    h5file : str
        Path to h5 file where states should be saved

    Returns
    -------
    info : dict
        Info about the run
    """
    if idx_meas is None:
        idx_meas = np.array([])

    # Get the solution times
    if times[-1] < times[0]:
        raise ValueError("The final time point must be greater than the initial one."
                         f"The input intial/final times were {times[0]}/{times[-1]}")
    if times.size <= 1:
        raise ValueError("There must be at least 2 time integration points.")

    # increment_forward = None
    # if coupling == 'implicit':
    #     increment_forward = partial(implicit_increment, method=coupling_method)
    # elif coupling == 'explicit':
    #     increment_forward = explicit_increment
    # else:
    #     raise ValueError("`coupling` must be one of 'explicit' of 'implicit'")
    
    solid_props = props[:len(model.solid.properties.size)]
    fluid_props = props[len(model.solid.properties.size):]
    model.set_fluid_props(fluid_props)
    model.set_solid_props(solid_props)

    # Allocate functions to store states
    uva = ini_state[:3]
    uva0 = model.solid.get_state_vec()
    uva0.vecs[0][:] = uva[0]
    uva0.vecs[1][:] = uva[1]
    uva0.vecs[2][:] = uva[2]
    
    qp0 = model.fluid.get_state_vec()
    qp0['q'][:] = ini_state['q']
    qp0['p'][:] = ini_state['p']

    # uva0.set(0.0)
    model.set_ini_solid_state(uva0.vecs)
    _, info = model.fluid.solve_qp0()

    ## Record things of interest
    # TODO: This should be removed. If you want to calculate a functional to record
    # during the solution, a parameter should be made available in the function for that
    glottal_width = []
    glottal_width.append(info['a_min'])

    ## Initialize datasets to save in h5 file
    with sf.StateFile(model, h5file, group=h5group, mode='a') as f:
        f.init_layout(uva0=uva0.vecs, qp0=qp0, fluid_props=fluid_props, solid_props=solid_props)
        f.append_time(times[0])
        if 0 in idx_meas:
            f.append_meas_index(0)

    ## Loop through solution times and write solution variables to the h5file.
    # TODO: Hardcoded the calculation of glottal width here, but it should be an option you
    # can pass in along with other functionals of interest
    with sf.StateFile(model, h5file, group=h5group, mode='a') as f:
        for n in range(times.size-1):
            dt = times[n+1] - times[n]
            # Increment the state

            model.set_iter_params(uva0=uva0, qp0=qp0, dt=dt)
            # breakpoint()
            state1, step_info = model.solve_state1(linalg.concatenate(uva0, qp0))
            uva1 = state1[:3]
            qp1 = state1[3:]

            glottal_width.append(step_info['fluid_info']['a_min'])

            ## Write the solution outputs to a file
            f.append_state(uva1)
            f.append_fluid_state(qp1)
            f.append_time(times[n+1])
            if n+1 in idx_meas:
                f.append_meas_index(n+1)

            ## Update initial conditions for the next time step
            uva0.vecs[0][:] = uva1[0]
            uva0.vecs[1][:] = uva1[1]
            uva0.vecs[2][:] = uva1[2]
            qp0.vecs[0][()] = qp1[0]
            qp0.vecs[1][:] = qp1[1]

        # Write out the quantities of interest to the h5file
        f.file[f'{h5group}/gw'] = np.array(glottal_width)
        info['glottal_width'] = np.array(glottal_width)

    return info

# def explicit_increment(model, uva0, qp0, dt, newton_solver_prm=None):
#     """
#     Return the state at the end of `dt` `uva1 = (u1, v1, a1)`.

#     Parameters
#     ----------
#     model : model.ForwardModel
#     uva0, qp0 : BlockVec
#         Initial states (u0, v0, a0) for the forward model
#     dt : float
#         The time step to increment over

#     Returns
#     -------
#     BlockVec
#         The next state (u1, v1, a1) of the forward model
#     BlockVec
#         The next fluid state (q1, p1) of the forward model
#     fluid_info : dict
#         A dictionary containing information on the fluid solution. These include the flow rate,
#         surface pressure, etc.
#     """
#     solid = model.solid
#     uva1 = model.solid.get_state_vec()
#     u0, v0, a0 = uva0.vecs

#     # Update form coefficients and initial guess
#     model.set_iter_params(uva0=uva0.vecs, dt=dt, uva1=(u0, 0.0, 0.0), qp1=qp0)

#     # TODO: You could implement this to use the non-linear solver only when collision is happening
#     if newton_solver_prm is None:
#         newton_solver_prm = DEFAULT_NEWTON_SOLVER_PRM

#     dfn.solve(solid.f1 == 0, solid.u1, bcs=solid.bc_base, J=solid.df1_du1,
#               solver_parameters={"newton_solver": newton_solver_prm})

#     res = dfn.assemble(model.solid.forms['form.un.f1'])
#     model.solid.bc_base.apply(res)

#     uva1.vecs[0][:] = solid.u1.vector()
#     uva1.vecs[1][:] = solids.newmark_v(uva1.vecs[0], u0, v0, a0, dt)
#     uva1.vecs[2][:] = solids.newmark_a(uva1.vecs[0], u0, v0, a0, dt)

#     model.set_fin_solid_state(uva1.vecs)
#     qp1, fluid_info = model.fluid.solve_qp1()

#     step_info = {'fluid_info': fluid_info}

#     return uva1, qp1, step_info

# def implicit_increment(model, uva0, qp0, dt, newton_solver_prm=None, max_nit=5, method='fp'):
#     """
#     Return the state at the end of `dt` `uva1 = (u1, v1, a1)`.

#     Parameters
#     ----------
#     model : model.ForwardModel
#     uva0, qp0 : BlockVec
#         Initial states (u0, v0, a0) for the forward model
#     dt : float
#         The time step to increment over

#     Returns
#     -------
#     tuple of dfn.Function
#         The next state (u1, v1, a1) of the forward model
#     tuple of (float, array_like)
#         The next fluid state (q1, p1) of the forward model
#     fluid_info : dict
#         A dictionary containing information on the fluid solution. These include the flow rate,
#         surface pressure, etc.
#     """
#     if method == 'newton':
#         return implicit_increment_newton(model, uva0, qp0, dt, newton_solver_prm=None, max_nit=5)
#     elif method == 'fp':
#         return implicit_increment_fp(model, uva0, qp0, dt, newton_solver_prm=None, max_nit=5)
#     else:
#         raise ValueError("'method' must be one of 'newton' or 'fp'")

# def implicit_increment_fp(model, uva0, qp0, dt, newton_solver_prm=None, max_nit=5):
#     """
#     Return the state at the end of `dt` `uva1 = (u1, v1, a1)`.

#     Uses a fixed point type iterative method.
#     """
#     solid = model.solid
#     u0, v0, a0 = uva0.vecs

#     # Set initial guesses for the states at the next time
#     uva1 = solid.get_state_vec()
#     uva1.vecs[0][:] = u0
#     qp1 = qp0

#     # Solve the coupled problem using fixed point iterations between the fluid and solid
#     if newton_solver_prm is None:
#         newton_solver_prm = DEFAULT_NEWTON_SOLVER_PRM

#     # Calculate the initial residual
#     model.set_iter_params(uva0=uva0.vecs, qp0=qp0, dt=dt, uva1=uva1.vecs, qp1=qp1)
#     res0 = dfn.assemble(model.solid.f1)
#     model.solid.bc_base.apply(res0)

#     # Set tolerances for the fixed point iterations
#     nit = 0
#     abs_tol, rel_tol = newton_solver_prm['absolute_tolerance'], newton_solver_prm['relative_tolerance']
#     abs_err0, abs_err, rel_err = res0.norm('l2'), np.inf, np.inf
#     while abs_err > abs_tol and rel_err > rel_tol and nit < max_nit:
#         model.set_iter_params(uva0=uva0.vecs, qp0=qp0, dt=dt, uva1=(uva1.vecs[0], 0, 0), qp1=qp1)
#         dfn.solve(solid.f1 == 0, solid.u1, bcs=solid.bc_base, J=solid.df1_du1,
#                   solver_parameters={"newton_solver": newton_solver_prm})

#         uva1.vecs[0][:] = solid.u1.vector()
#         uva1.vecs[1][:] = solids.newmark_v(uva1.vecs[0], u0, v0, a0, dt)
#         uva1.vecs[2][:] = solids.newmark_a(uva1.vecs[0], u0, v0, a0, dt)

#         model.set_fin_solid_state(uva1.vecs)
#         qp1, fluid_info = model.fluid.solve_qp1()

#         # Set the state to calculate the pressure, but you have to set it back after
#         # model.set_ini_solid_state((u1, v1, a1))
#         # q1, p1, fluid_info = model.get_pressure()

#         # Calculate the error in the solid residual with the updated pressures
#         model.set_iter_params(uva0=uva0.vecs, dt=dt, qp1=qp1)
#         res = dfn.assemble(solid.f1)
#         solid.bc_base.apply(res)

#         abs_err = res.norm('l2')
#         rel_err = abs_err/abs_err0

#         nit += 1

#     model.set_iter_params(uva0=uva0.vecs, dt=dt, qp1=qp1)
#     res = dfn.assemble(model.solid.forms['form.un.f1'])
#     model.solid.bc_base.apply(res)

#     step_info = {'fluid_info': fluid_info,
#                  'nit': nit, 'abs_err': abs_err, 'rel_err': rel_err}

#     return uva1.vecs, qp1, step_info

# def implicit_increment_newton(model, uva0, qp0, dt, newton_solver_prm=None, max_nit=5):
#     """
#     Solve for the state variables at the end of the time step using a Newton method
#     """
#     # Configure the Newton solver
#     if newton_solver_prm is None:
#         newton_solver_prm = DEFAULT_NEWTON_SOLVER_PRM

#     abs_tol = newton_solver_prm['absolute_tolerance']
#     rel_tol = newton_solver_prm['relative_tolerance']

#     solid = model.solid

#     # Set initial guesses for the states at the next time
#     u0, v0, a0 = uva0.vecs
#     uva1 = solid.get_state_vec()
#     uva1.vecs[0][:] = u0

#     qp1 = qp0

#     # Calculate the initial residual
#     model.set_iter_params(uva0=uva0.vecs, qp0=qp0, dt=dt, uva1=(uva1.vecs[0], 0.0, 0.0), qp1=qp1)
#     res_u = dfn.assemble(model.solid.f1)
#     model.solid.bc_base.apply(res_u)

#     res_p = qp1[1] - model.fluid.solve_qp1()[0][1]

#     # Set tolerances for the fixed point iterations
#     nit = 0
#     abs_err, rel_err = np.inf, np.inf
#     abs_err0 = res_u.norm('l2')
#     while abs_err > abs_tol and rel_err > rel_tol and nit < max_nit:
#         # assemble blocks of the residual jacobian
#         dfu_du = dfn.assemble(model.solid.forms['form.bi.df1_du1'])
#         dfu_dpsolid = dfn.assemble(model.solid.forms['form.bi.df1_dp1'])

#         solid_dofs, fluid_dofs = model.get_fsi_scalar_dofs()
#         dfu_dpsolid = dfn.as_backend_type(dfu_dpsolid).mat()
#         dfu_dp = linalg.reorder_mat_cols(dfu_dpsolid, solid_dofs, fluid_dofs, model.fluid.p1.size)

#         _, dp_du = model.solve_dqp1_du1_solid(adjoint=False)
#         dfp_du = 0 - dp_du

#         model.solid.bc_base.apply(dfu_du, res_u)
#         bc_dofs = np.array(list(model.solid.bc_base.get_boundary_values().keys()), dtype=np.int32)
#         dfu_dp.zeroRows(bc_dofs, diag=0.0)

#         # form the block matrix/vector representing the residual jacobian/vector
#         dfup_dup = linalg.form_block_matrix(
#             [[dfn.as_backend_type(dfu_du).mat(), dfu_dp],
#              [                           dfp_du,    1.0]])
#         res_up, dup = dfup_dup.getVecLeft(), dfup_dup.getVecRight()
#         res_up[:res_u.size()] = res_u
#         res_up[res_u.size():] = res_p

#         # solve for the increment in the solution
#         ksp = PETSc.KSP().create()
#         ksp.setType(ksp.Type.PREONLY)

#         pc = ksp.getPC()
#         pc.setType(pc.Type.LU)

#         ksp.setOperators(dfup_dup)
#         ksp.solve(-res_up, dup)

#         # increment the solution to start the next iteration
#         uva1.vecs[0][:] += dup[:uva1.vecs[0].size()]
#         uva1.vecs[1][:] = solids.newmark_v(uva1.vecs[0], u0, v0, a0, dt)
#         uva1.vecs[2][:] = solids.newmark_a(uva1.vecs[0], u0, v0, a0, dt)

#         qp1[1][:] += dup[uva1.vecs[0].size():]

#         # Calculate the new solid/fluid residuals with the updated disp/pressure
#         model.set_iter_params(uva0=uva0.vecs, dt=dt, uva1=uva1.vecs, qp1=qp1)
#         res_u = dfn.assemble(solid.f1)
#         solid.bc_base.apply(res_u)

#         res_p = qp1[1] - model.fluid.solve_qp1()[0][1]

#         abs_err = res_u.norm('l2')
#         rel_err = abs_err/abs_err0

#         nit += 1

#     model.set_iter_params(uva0=uva0.vecs, dt=dt, qp1=qp1)
#     res = dfn.assemble(model.solid.forms['form.un.f1'])
#     model.solid.bc_base.apply(res)

#     qp1, fluid_info = model.fluid.solve_qp1()

#     step_info = {'fluid_info': fluid_info,
#                  'nit': nit, 'abs_err': abs_err, 'rel_err': rel_err}

#     return uva1.vecs, qp1, step_info

# def _increment_forward(model, state, newton_solver_prm=None):
#     if newton_solver_prm is None:
#         newton_solver_prm = {}
#     omega = newton_solver_prm.get('relaxation', 1.0)
#     # linear_solver = newton_solver_prm.get('linear_solver', 'petsc')
#     abs_tol = newton_solver_prm.get('abs_tol', 1e-8)
#     rel_tol = newton_solver_prm.get('rel_tol', 1e-6)
#     maxiter = newton_solver_prm.get('maxiter', 25)

#     ii = 0
#     state_n = state.copy()

#     # breakpoint()
#     model.set_fin_state(state_n)
#     res = model.res()
#     print(res['u'].norm('l2'))
#     res_norm_prev = 1.0
#     res_norm = res['u'].norm('l2')

#     abs_err = res_norm
#     rel_err = 1.0

#     while abs_err > abs_tol and rel_err > rel_tol and ii < maxiter:
#         ## Perform a newton update
#         state_n = state_n + omega*model.solve_dres_dstate1(-res)
        
#         ## Compute the new residual and error
#         model.set_fin_state(state_n)
#         res = model.res()
#         res_norm_prev = res_norm
#         res_norm = res['u'].norm('l2')

#         rel_err = abs((res_norm - res_norm_prev)/res_norm_prev)
#         abs_err = res_norm
#         ii += 1

#     info = {'niter': ii, 'abs_err': abs_err, 'rel_err': rel_err}
#     return state_n, info

# def increment_forward(model, state, newton_solver_prm=None):
#     return model.solve_state1(state, newton_solver_prm)

def newton_solve(u, du, jac, res, bcs, **kwargs):
    """
    Solves the system using a newton method.

    Parameters
    ----------
    u : dfn.cpp.la.Vector
        The initial guess of the solution.
    du : dfn.cpp.la.Vector
        A vector for storing increments in the solution.
    res : callable(dfn.GenericVector) -> dfn.cpp.la.Vector
    jac : callable(dfn.GenericVector) -> dfn.cpp.la.Matrix
    bcs : list of dfn.DirichletBC

    Returns
    -------
    u1 : dfn.Function
    """
    omega = kwargs.get('relaxation', 1.0)
    linear_solver = kwargs.get('linear_solver', 'petsc')
    abs_tol = kwargs.get('abs_tol', 1e-8)
    rel_tol = kwargs.get('rel_tol', 1e-6)
    maxiter = kwargs.get('maxiter', 25)

    abs_err = 1.0
    rel_err = 1.0

    _res = res(u)
    for bc in bcs:
        bc.apply(_res)
    res_norm0 = _res.norm('l2')
    res_norm1 = 1.0

    ii = 0
    while abs_err > abs_tol and rel_err > rel_tol and ii < maxiter:
        _jac = jac(u)
        for bc in bcs:
            bc.apply(_jac, _res)

        dfn.solve(_jac, du, _res, linear_solver)

        u[:] = u - omega*du

        _res = res(u)
        for bc in bcs:
            bc.apply(_res)
        res_norm1 = _res.norm('l2')

        rel_err = abs((res_norm1 - res_norm0)/res_norm0)
        abs_err = res_norm1

        ii += 1

    info = {'niter': ii, 'abs_err': abs_err, 'rel_err': rel_err}
    return u, info

def newmark_error_estimate(a1, a0, dt, beta=1/4):
    """
    Return an estimate of the truncation error in `u` over the step.

    Error is esimated using eq (18) in [1]. Note that their paper defines $\beta2$ as twice $\beta$
    in the original newmark notation (used here). Therefore the beta term is multiplied by 2.

    [1] A simple error estimator and adaptive time stepping procedure for dynamic analysis.
    O. C. Zienkiewicz and Y. M. Xie. Earthquake Engineering and Structural Dynamics, 20:871-887
    (1991).

    Parameters
    ----------
    a1 : dfn.Vector()
        The newmark prediction of acceleration at :math:`n+1`
    a0 : dfn.Vector()
        The newmark prediction of acceleration at :math:`n`
    dt : float
        The time step integrated over
    beta : float
        The newmark-beta method :math:`beta` parameter

    Returns
    -------
    dfn.Vector()
        An estimate of the error in :math:`u_{n+1}`
    """
    return 0.5*dt**2*(2*beta - 1/3)*(a1-a0)
