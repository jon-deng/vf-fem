"""
This module contains code that solves static problems

Solving static problems requires some minor changes as the solid model equations
are formulated over a time step and involve states before and after the time
step ((u, v, a)_0 and (u, v, a)_1 for the displacement, velocity and
acceleration). The residual then involves solving three equations
    Fu(u1, u0, v0, a0; ...) = 0
    Fv = v1 - vnewmark(u1, u0, v0, a0) = 0 * explicit solve once you know u1
    Fa = a1 - anewmark(u1, u0, v0, a0) = 0 * explicit solve once you know u1

where v1 and a1 are further functions of the initial state and time step.

To recover the static problem, you have to ensure v1=0, a1=0 in the solution.
To do this, you can use a very large time step dt.

Another strategy is to manually ensure that u0=u1 and v0=a0=0. I think this
approach is a little trickier as the jacobian dFu/du1 would change since u0 must
be linked to u1.
"""

from typing import Tuple, Mapping

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

def static_solid_configuration(
        solid: slmodel.Solid,
        control: BlockVector,
        props: BlockVector
    ) -> Tuple[BlockVector, Mapping]:
    """
    Return the static state of a solid model

    Parameters
    ----------
    solid :
        The solid model to solve for a static state
    control :
        The control parameters of the solid model for the static state
    props :
        The properties of the solid model
    """
    # Set the initial guess u=0 and constants (v, a) = (0, 0)
    state_n = solid.state0.copy()
    state_n[:] = 0.0
    solid.set_fin_state(state_n)
    solid.set_ini_state(state_n)

    solid.set_control(control)
    solid.set_props(props)

    jac = dfn.derivative(
        solid.forms['form.un.f1uva'],
        solid.forms['coeff.state.u1']
    )
    dfn.solve(
        solid.forms['form.un.f1uva'] == 0.0,
        solid.forms['coeff.state.u1'],
        bcs=[solid.forms['bc.dirichlet']],
        J=jac,
        solver_parameters={"newton_solver": DEFAULT_NEWTON_SOLVER_PRM}
    )

    state_n['u'] = solid.state1['u']

    info = {}
    return state_n, info

def set_coupled_model_substate(model, xsub):
    """
    Set the coupled model state
    """
    _state = model.state0.copy()
    _labels = list(xsub.labels[0])
    _state[_labels] = xsub
        # _state[key][:] = xsub[key].array
    # Set both initial and final states to ensure that the fluid pressure
    # is set for the final state; for explicit models only the initial fluid
    # state is passed as a force on the final solid residual
    model.set_ini_state(_state)
    model.set_fin_state(_state)

def static_coupled_configuration_picard(
        model: comodel.FSIModel,
        control: BlockVector,
        props: BlockVector,
    ) -> Tuple[BlockVector, Mapping]:
    """
    Solve for static equilibrium for a coupled model
    """
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
        dfn.solve(
            solid.forms['form.un.f1uva'] == 0.0,
            solid.forms['coeff.state.u1'],
            bcs=[solid.forms['bc.dirichlet']],
            J=solid.forms['form.bi.df1uva_du1'],
            solver_parameters={"newton_solver": DEFAULT_NEWTON_SOLVER_PRM}
        )
        u = BlockVector([solid.state1['u'].copy()], labels=[['u']]) # the vector corresponding to solid.forms['coeff.state.u1']

        # update the fluid load for the new solid deformation
        x_n['u'][:] = u[0]
        set_coupled_model_substate(model, x_n)
        qp, _ = fluid.solve_state1(x_n[['q', 'p']])
        return concatenate_vec([u, qp.copy()])

    # Set the initial state
    x_n = model.state0.copy()[['u', 'q', 'p']]
    x_n[:] = 0

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
        solid.forms['bc.dirichlet'].apply(res)

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
# I'm not sure if the answer it returns is correct or not
def static_coupled_configuration_newton(
        model: comodel.FSIModel,
        control: BlockVector,
        props: BlockVector,
        dt: float=1e6
    ) -> Tuple[BlockVector, Mapping]:
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
        model.dt = dt
        set_coupled_model_substate(model, x_0)

        ### Form the residual
        def assem_res():
            return model.assem_res()

        ### Form the jacobian
        def solve_jac(res):
            return model.solve_dres_dstate1(res)

        return assem_res, solve_jac

    ### Initial guess
    x_0 = model.state0.copy()
    x_0[:] = 0.0

    x_n, info = newton_solve(x_0, make_linear_subproblem, step_size=1.0)
    return x_n, info
