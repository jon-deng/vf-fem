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
    'relative_tolerance': 1e-9}

FIXEDPOINT_SOLVER_PRM = {
    'absolute_tolerance': 1e-8,
    'relative_tolerance': 1e-11}

def integrate(model, uva, solid_props, fluid_props, times, idx_meas=None,
              h5file='tmp.h5', h5group='/', newton_solver_prm=None,
              coupling='implicit', coupling_method='newton'):
    """
    Integrate the model over each time in `times` for the specified parameters
    """
    if idx_meas is None:
        idx_meas = np.array([])

    increment_forward = None
    if coupling == 'implicit':
        increment_forward = partial(implicit_increment, method=coupling_method)
    elif coupling == 'explicit':
        increment_forward = explicit_increment
    else:
        raise ValueError("`coupling` must be one of 'explicit' of 'implicit'")

    model.set_fluid_props(fluid_props)
    model.set_solid_props(solid_props)

    # Allocate functions to store states
    u0 = dfn.Function(model.solid.vector_fspace).vector()
    v0 = dfn.Function(model.solid.vector_fspace).vector()
    a0 = dfn.Function(model.solid.vector_fspace).vector()
    u0[:], v0[:], a0[:] = uva

    model.set_ini_solid_state(u0, v0, a0)
    q0, p0, info = model.solve_qp0()

    ## Record things of interest
    # TODO: This should be removed. If you want to calculate a functional to record
    # during the solution, a parameter should be made available in the function for that
    idx_separation = []
    idx_min_area = []
    glottal_width = []
    flow_rate = []
    pressure = []
    glottal_width.append(info['a_min'])
    flow_rate.append(info['flow_rate'])

    # Get the solution times
    if times[-1] < times[0]:
        raise ValueError("The final time point must be greater than the initial one."
                         f"The input intial/final times were {times[0]}/{times[-1]}")
    if times.size <= 1:
        raise ValueError("There must be atleast 2 time integration points.")

    ## Initialize datasets to save in h5 file
    with sf.StateFile(model, h5file, group=h5group, mode='a') as f:
        f.init_layout(uva0=(u0, v0, a0), qp0=(q0, p0), fluid_props=fluid_props, solid_props=solid_props)
        f.append_time(times[0])
        if 0 in idx_meas:
            f.append_meas_index(0)

    ## Loop through solution times and write solution variables to the h5file.
    # TODO: Hardcoded the calculation of glottal width here, but it should be an option you
    # can pass in along with other functionals of interest
    with sf.StateFile(model, h5file, group=h5group, mode='a') as f:
        for n in range(times.size-1):
            dt = times[n+1] - times[n]
            uva0 = (u0, v0, a0)
            qp0 = (q0, p0)

            # Increment the state
            uva1, qp1, step_info = None, None, None

            uva1, qp1, step_info = increment_forward(model, uva0, qp0, dt,
                                                     newton_solver_prm=newton_solver_prm)

            idx_separation.append(step_info['fluid_info']['idx_sep'])
            idx_min_area.append(step_info['fluid_info']['idx_min'])
            glottal_width.append(step_info['fluid_info']['a_min'])
            flow_rate.append(step_info['fluid_info']['flow_rate'])
            pressure.append(step_info['fluid_info']['pressure'])

            ## Write the solution outputs to a file
            f.append_state(uva1)
            f.append_fluid_state(qp1)
            f.append_time(times[n+1])
            if n+1 in idx_meas:
                f.append_meas_index(n+1)

            ## Update initial conditions for the next time step
            u0[:] = uva1[0]
            v0[:] = uva1[1]
            a0[:] = uva1[2]
            q0 = qp1[0]
            p0 = qp1[1]

        # Write out the quantities fo interest to the h5file
        f.file[f'{h5group}/gaw'] = np.array(glottal_width)

        info['meas_ind'] = f.get_meas_indices()
        info['time'] = f.get_times()
        info['glottal_width'] = np.array(glottal_width)
        info['flow_rate'] = np.array(flow_rate)
        info['idx_separation'] = np.array(idx_separation)
        info['idx_min_area'] = np.array(idx_min_area)
        info['pressure'] = np.array(pressure)
        info['h5file'] = h5file
        info['h5group'] = h5group

    return info

def explicit_increment(model, uva0, qp0, dt, newton_solver_prm=None):
    """
    Return the state at the end of `dt` `uva1 = (u1, v1, a1)`.

    Parameters
    ----------
    model : model.ForwardModel
    uva0 : tuple of dfn.Function
        Initial states (u0, v0, a0) for the forward model
    dt : float
        The time step to increment over
    solid_props : properties.SolidProperties, optional
        A dictionary of solid properties
    fluid_props : properties.FluidProperties, optional
        A dictionary of fluid properties.

    Returns
    -------
    tuple of dfn.Function
        The next state (u1, v1, a1) of the forward model
    tuple of (float, array_like)
        The next fluid state (q1, p1) of the forward model
    fluid_info : dict
        A dictionary containing information on the fluid solution. These include the flow rate,
        surface pressure, etc.
    """
    solid = model.solid
    u0, v0, a0 = uva0

    u1 = dfn.Function(solid.vector_fspace).vector()
    v1 = dfn.Function(solid.vector_fspace).vector()
    a1 = dfn.Function(solid.vector_fspace).vector()

    # Update form coefficients and initial guess
    model.set_iter_params(uva0=uva0, dt=dt, uva1=(u0, 0.0, 0.0), qp1=qp0)

    # TODO: You could implement this to use the non-linear solver only when collision is happening
    if newton_solver_prm is None:
        newton_solver_prm = DEFAULT_NEWTON_SOLVER_PRM

    dfn.solve(solid.f1 == 0, solid.u1, bcs=solid.bc_base, J=solid.df1_du1,
              solver_parameters={"newton_solver": newton_solver_prm})

    res = dfn.assemble(model.solid.forms['form.un.f1'])
    model.solid.bc_base.apply(res)

    u1[:] = solid.u1.vector()
    v1[:] = solids.newmark_v(u1, u0, v0, a0, dt)
    a1[:] = solids.newmark_a(u1, u0, v0, a0, dt)

    model.set_fin_solid_state(u1, v1, a1)
    q1, p1, fluid_info = model.solve_qp1()

    step_info = {'fluid_info': fluid_info}

    return (u1, v1, a1), (q1, p1), step_info

def implicit_increment(model, uva0, qp0, dt, newton_solver_prm=None, max_nit=5, method='fp'):
    """
    Return the state at the end of `dt` `uva1 = (u1, v1, a1)`.

    Parameters
    ----------
    model : model.ForwardModel
    uva0 : tuple of dfn.Function
        Initial states (u0, v0, a0) for the forward model
    dt : float
        The time step to increment over
    solid_props : properties.SolidProperties, optional
        A dictionary of solid properties
    fluid_props : properties.FluidProperties, optional
        A dictionary of fluid properties.

    Returns
    -------
    tuple of dfn.Function
        The next state (u1, v1, a1) of the forward model
    tuple of (float, array_like)
        The next fluid state (q1, p1) of the forward model
    fluid_info : dict
        A dictionary containing information on the fluid solution. These include the flow rate,
        surface pressure, etc.
    """
    if method == 'newton':
        return implicit_increment_newton(model, uva0, qp0, dt, newton_solver_prm=None, max_nit=5)
    elif method == 'fp':
        return implicit_increment_fp(model, uva0, qp0, dt, newton_solver_prm=None, max_nit=5)
    else:
        raise ValueError("'method' must be one of 'newton' or 'fp'")

def implicit_increment_fp(model, uva0, qp0, dt, newton_solver_prm=None, max_nit=5):
    """
    Return the state at the end of `dt` `uva1 = (u1, v1, a1)`.

    Uses a fixed point type iterative method.
    """
    solid = model.solid
    u0, v0, a0 = uva0

    # Set initial guesses for the states at the next time
    u1 = dfn.Function(solid.vector_fspace).vector()
    v1 = dfn.Function(solid.vector_fspace).vector()
    a1 = dfn.Function(solid.vector_fspace).vector()

    u1[:] = u0
    q1, p1 = qp0

    # Solve the coupled problem using fixed point iterations between the fluid and solid
    if newton_solver_prm is None:
        newton_solver_prm = DEFAULT_NEWTON_SOLVER_PRM

    # Calculate the initial residual
    model.set_iter_params(uva0=uva0, qp0=qp0, dt=dt, uva1=(u1, 0, 0), qp1=(q1, p1))
    res0 = dfn.assemble(model.solid.f1)
    model.solid.bc_base.apply(res0)

    # Set tolerances for the fixed point iterations
    nit = 0
    abs_tol, rel_tol = newton_solver_prm['absolute_tolerance'], newton_solver_prm['relative_tolerance']
    abs_err0, abs_err, rel_err = res0.norm('l2'), np.inf, np.inf
    while abs_err > abs_tol and rel_err > rel_tol and nit < max_nit:
        model.set_iter_params(uva0=uva0, qp0=qp0, dt=dt, uva1=(u1, 0, 0), qp1=(q1, p1))
        dfn.solve(solid.f1 == 0, solid.u1, bcs=solid.bc_base, J=solid.df1_du1,
                  solver_parameters={"newton_solver": newton_solver_prm})

        u1[:] = solid.u1.vector()
        v1[:] = solids.newmark_v(u1, u0, v0, a0, dt)
        a1[:] = solids.newmark_a(u1, u0, v0, a0, dt)

        model.set_fin_solid_state(u1, v1, a1)
        q1, p1, fluid_info = model.solve_qp1()

        # Set the state to calculate the pressure, but you have to set it back after
        # model.set_ini_solid_state(u1, v1, a1)
        # q1, p1, fluid_info = model.get_pressure()

        # Calculate the error in the solid residual with the updated pressures
        model.set_iter_params(uva0=uva0, dt=dt, qp1=(q1, p1))
        res = dfn.assemble(solid.f1)
        solid.bc_base.apply(res)

        abs_err = res.norm('l2')
        rel_err = abs_err/abs_err0

        nit += 1

    model.set_iter_params(uva0=uva0, dt=dt, qp1=(q1, p1))
    res = dfn.assemble(model.solid.forms['form.un.f1'])
    model.solid.bc_base.apply(res)

    step_info = {'fluid_info': fluid_info,
                 'nit': nit, 'abs_err': abs_err, 'rel_err': rel_err}

    return (u1, v1, a1), (q1, p1), step_info

def implicit_increment_newton(model, uva0, qp0, dt, newton_solver_prm=None, max_nit=5):
    """
    Solve for the state variables at the end of the time step using a Newton method
    """
    # Configure the Newton solver
    if newton_solver_prm is None:
        newton_solver_prm = DEFAULT_NEWTON_SOLVER_PRM

    abs_tol = newton_solver_prm['absolute_tolerance']
    rel_tol = newton_solver_prm['relative_tolerance']


    solid = model.solid

    # Set initial guesses for the states at the next time
    u0, v0, a0 = uva0
    u1 = dfn.Function(solid.vector_fspace).vector()
    v1 = dfn.Function(solid.vector_fspace).vector()
    a1 = dfn.Function(solid.vector_fspace).vector()

    u1[:] = u0
    # v1[:] = v0
    # a1[:] = a0
    q1, p1 = qp0

    # Calculate the initial residual
    model.set_iter_params(uva0=uva0, qp0=qp0, dt=dt, uva1=(u1, 0.0, 0.0), qp1=(q1, p1))
    res_u = dfn.assemble(model.solid.f1)
    model.solid.bc_base.apply(res_u)

    res_p = p1 - model.fluid.solve_qp1()[1]

    # Set tolerances for the fixed point iterations
    nit = 0
    abs_err, rel_err = np.inf, np.inf
    abs_err0 = res_u.norm('l2')
    while abs_err > abs_tol and rel_err > rel_tol and nit < max_nit:
        # assemble blocks of the residual jacobian
        dfu_du = model.assem_df1_du1()
        dfu_dpsolid = dfn.assemble(model.solid.forms['form.bi.df1_dp1'])

        solid_dofs, fluid_dofs = model.get_fsi_scalar_dofs()
        dfu_dpsolid = dfn.as_backend_type(dfu_dpsolid).mat()
        dfu_dp = linalg.reorder_mat_cols(dfu_dpsolid, solid_dofs, fluid_dofs, model.fluid.p1.size)

        _, dp_du = model.solve_dqp1_du1_solid(adjoint=False)
        dfp_du = 0 - dp_du

        model.solid.bc_base.apply(dfu_du, res_u)
        bc_dofs = np.array(list(model.solid.bc_base.get_boundary_values().keys()), dtype=np.int32)
        dfu_dp.zeroRows(bc_dofs, diag=0.0)

        # form the block matrix/vector representing the residual jacobian/vector
        dfup_dup = linalg.form_block_matrix(
            [[dfn.as_backend_type(dfu_du).mat(), dfu_dp],
             [                           dfp_du,    1.0]])
        res_up, dup = dfup_dup.getVecLeft(), dfup_dup.getVecRight()
        res_up[:res_u.size()] = res_u
        res_up[res_u.size():] = res_p

        # solve for the increment in the solution
        ksp = PETSc.KSP().create()
        ksp.setType(ksp.Type.PREONLY)

        pc = ksp.getPC()
        pc.setType(pc.Type.LU)

        ksp.setOperators(dfup_dup)
        ksp.solve(-res_up, dup)

        # increment the solution to start the next iteration
        u1[:] += dup[:u1.size()]
        v1[:] = solids.newmark_v(u1, u0, v0, a0, dt)
        a1[:] = solids.newmark_a(u1, u0, v0, a0, dt)

        p1[:] += dup[u1.size():]

        # Calculate the new solid/fluid residuals with the updated disp/pressure
        model.set_iter_params(uva0=uva0, dt=dt, uva1=(u1, v1, a1), qp1=(q1, p1))
        res_u = dfn.assemble(solid.f1)
        solid.bc_base.apply(res_u)

        res_p = p1 - model.fluid.solve_qp1()[1]

        abs_err = res_u.norm('l2')
        rel_err = abs_err/abs_err0

        nit += 1

    model.set_iter_params(uva0=uva0, dt=dt, qp1=(q1, p1))
    res = dfn.assemble(model.solid.forms['form.un.f1'])
    model.solid.bc_base.apply(res)

    q1, p1, fluid_info = model.fluid.solve_qp1()

    step_info = {'fluid_info': fluid_info,
                 'nit': nit, 'abs_err': abs_err, 'rel_err': rel_err}

    return (u1, v1, a1), (q1, p1), step_info


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
