"""
This module contains code that solves for static equilibium of models
"""

import dolfin as dfn
from petsc4py import PETSc
import numpy as np

from .models.transient import solid as slmodel, coupled as comodel
from .solverconst import DEFAULT_NEWTON_SOLVER_PRM

from blockarray import subops as gops
from blockarray.blockvec import BlockVector, concatenate_vec, convert_subtype_to_petsc
from blockarray.blockmat import BlockMatrix
from blockarray.subops import zero_mat, ident_mat

from nonlineq import newton_solve

# import warnings
# warnings.filterwarnings("error")

dfn.set_log_level(50)

def static_configuration(solid: slmodel.Solid):
    # Set the initial guess u=0 and constants (v, a) = (0, 0)
    state = solid.get_state_vec()
    state.set(0.0)
    solid.set_fin_state(state)
    solid.set_ini_state(state)

    # Set initial pressure as 0 for the static problem
    control = solid.get_control_vec()
    control['p'][:] = 0.0

    jac = dfn.derivative(solid.forms['form.un.f1uva'], solid.forms['coeff.state.u1'])
    dfn.solve(solid.forms['form.un.f1uva'] == 0.0, solid.forms['coeff.state.u1'],
              bcs=[solid.bc_base], J=jac, solver_parameters={"newton_solver": DEFAULT_NEWTON_SOLVER_PRM})

    u = solid.state1['u']
    return u

def set_coupled_model_substate(model, xsub):
    """
    Set the coupled model state
    """
    _state = model.get_state_vec()
    for key in xsub.labels[0]:
        gops.set_vec(_state[key], xsub[key])
        # _state[key][:] = xsub[key].array
    # Set both initial and final states to ensure that the fluid pressure
    # is set for the final state; for explicit models only the initial fluid
    # state is passed as a force on the final solid residual
    model.set_ini_state(_state)
    model.set_fin_state(_state)

# TODO: This one has a strange bug where residual seem to oscillate wildly
def static_configuration_coupled_picard(model: comodel.FSIModel):
    solid = model.solid
    fluid = model.fluid
    def norm(x):
        return x.norm('l2')

    def picard(x_n):
        """
        Returns a new guess x_n+1

        x_n = [u, q, p]
        """
        set_coupled_model_substate(model, x_n)

        # solve for the solid deformation under the guessed fluid load
        dfn.solve(solid.forms['form.un.f1uva'] == 0.0, solid.forms['coeff.state.u1'],
                  bcs=[solid.bc_base], J=solid.forms['form.bi.df1uva_du1'], solver_parameters={"newton_solver": DEFAULT_NEWTON_SOLVER_PRM})
        u = BlockVector([solid.state1['u'].copy()], labels=[['u']]) # the vector corresponding to solid.forms['coeff.state.u1']

        # update the fluid load for the new solid deformation
        x_n['u'][:] = u[0]
        set_coupled_model_substate(model, x_n)
        qp, _ = fluid.solve_state1(x_n[['q', 'p']])
        return concatenate_vec([u, qp.copy()])

    # Set the initial state
    x_n = model.get_state_vec()[['u', 'q', 'p']]
    x_n.set(0)
    # set_coupled_model_substate(model, x_n)
    # x_n[['q', 'p']] = fluid.solve_state1(x_n[['q', 'p']])[0]

    abs_errs = []
    rel_errs = []

    n = 0
    ABS_TOL = 1e-8
    REL_TOL = 1e-8
    NMAX = 15
    while True:
        # Compute the picard iteration
        # print("Before picard ||x_n||=", x_n['u'].norm('l2'))
        x_n = picard(x_n)
        # print("After picard ||x_n||=", x_n['u'].norm('l2'))

        # Compute the error in the iteration
        set_coupled_model_substate(model, x_n)

        res = dfn.assemble(solid.forms['form.un.f1uva'])
        solid.bc_base.apply(res)

        abs_errs.append(norm(res))
        rel_errs.append(abs_errs[-1]/abs_errs[0])

        if abs_errs[-1] < ABS_TOL or rel_errs[-1] < REL_TOL:
            break
        elif n > NMAX:
            break
        else:
            n += 1

    info = {
        'abs_errs': np.array(abs_errs),
        'rel_errs': np.array(rel_errs)
    }

    return x_n, info

# TODO: This one has a strange bug where Newton convergence is very slow
# I'm not sure if the answer it return is correct or not
def static_configuration_coupled_newton(model: comodel.FSIModel):
    """
    Return the static equilibrium state for a coupled model
    """
    solid = model.solid
    fluid = model.fluid

    def make_linear_subproblem(x_0):
        """
        Linear subproblem to be solved in a Newton solution strategy
        """
        ### Set the state to linearize around
        set_coupled_model_substate(model, x_0)

        ### Debugging
        # state = model.state1
        # print("In linear subproblem")
        # print("State: ", [f"{key}: {gops.norm_vec(vec):.2e}" for key, vec in zip(state.labels[0], state)])

        ### Form the residual
        def assem_res():
            fu = dfn.assemble(solid.forms['form.un.f1uva'], tensor=dfn.PETScVector())
            solid.bc_base.apply(fu)
            fu = BlockVector([fu], labels=[['u']])
            fqp = fluid.res()

            return concatenate_vec([fu, fqp])

        ### Form the jacobian
        def solve_jac(res):
            x_n = x_0.copy()

            # dfu/du, dfu/dq, dfu/dp
            dfu_du = dfn.assemble(solid.forms['form.bi.df1uva_du1'], tensor=dfn.PETScMatrix())
            solid.bc_base.apply(dfu_du)
            dfu_du = dfu_du.mat()

            solid_dofs, fluid_dofs = model.get_fsi_scalar_dofs()
            dfu_dp = dfn.assemble(solid.forms['form.bi.df1uva_dp1'], tensor=dfn.PETScMatrix())
            solid.bc_base.zero(dfu_dp)
            dfu_dp = dfu_dp.mat()
            dfu_dp = mat.reorder_mat_cols(dfu_dp, solid_dofs, fluid_dofs, x_n['p'].size)

            dfu_dq = zero_mat(x_n['u'].size, x_n['q'].size)

            dq_du, dp_du = fluid.solve_dqp1_du1_solid(model, adjoint=False)
            # dfq/du, dfq/dq, dfq/dp
            dfq_du = gops.convert_vec_to_rowmat(0 - dq_du.vec())
            dfq_dq = ident_mat(x_n['q'].size)
            dfq_dp = zero_mat(x_n['q'].size, x_n['p'].size)

            # dfp/du dfp/dq, dfp/dp
            dfp_du = 0 - dp_du
            dfp_dq = zero_mat(x_n['p'].size, x_n['q'].size)
            dfp_dp = ident_mat(x_n['p'].size)

            mats = [
                [dfu_du, dfu_dq, dfu_dp],
                [dfq_du, dfq_dq, dfq_dp],
                [dfp_du, dfp_dq, dfp_dp]]
            jac = BlockMatrix(mats, labels=(('u', 'q', 'p'), ('u', 'q', 'p')))

            ### Solve with KSP
            _jac = jac.to_petsc()
            _res = res.to_petsc()
            _x_n = _jac.getVecRight()

            ksp = PETSc.KSP().create()
            ksp.setOperators(_jac)
            ksp.setType(PETSc.KSP.Type.PREONLY)
            ksp.getPC().setType(PETSc.PC.Type.LU)
            ksp.setUp()
            ksp.solve(_res, _x_n)

            x_n.set_vec(_x_n)
            return x_n

        return assem_res, solve_jac

    ### Initial guess
    x_0 = convert_subtype_to_petsc(
        concatenate_vec([solid.get_state_vec()[['u']], fluid.get_state_vec()[['q', 'p']]])
        )

    # x_0['p'].array[:] = 500000000000.0

    set_coupled_model_substate(model, x_0)
    # qp, _ = fluid.solve_state1(x_0[['q', 'p']])
    # x_0[['q', 'p']] = qp

    # x_n = x_0.copy()
    # for ii in range(15):
    #     print(f"Iteration {ii}")
    #     assem_res, solve_jac = make_linear_subproblem(x_n)
    #     res = assem_res()
    #     dx_n = solve_jac(res)
    #     breakpoint()
    #     x_n = x_n - dx_n

    x_n, info = newton_solve(x_0, make_linear_subproblem, step_size=1.0)
    # breakpoint()
    return x_n, info

    # print(info)