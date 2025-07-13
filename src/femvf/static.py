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

from typing import Any

import dolfin as dfn
import ufl
import numpy as np

from blockarray import blockmat as bm, blockvec as bv
import nonlineq

from .models import dynamical
from .models import transient
from .solverconst import DEFAULT_NEWTON_SOLVER_PRM

# import warnings
# warnings.filterwarnings("error")

dfn.set_log_level(50)

Info = dict[str, Any]


def _add_static_docstring(func):
    docs = """
    Solve for the static state of a coupled model

    Parameters
    ----------
    model :
        The model
    control :
        The control parameters for the model
    prop :
        The properties of the model

    Returns
    -------
    bv.BlockVector
        The static state
    Info
        A dictionary of info about the solve
    """
    func.__doc__ = func.__doc__ + docs
    return func


@_add_static_docstring
def static_solid_configuration(
    model: transient.FenicsModel,
    control: bv.BlockVector,
    prop: bv.BlockVector,
    state=None,
    solver: str = 'manual',
) -> tuple[bv.BlockVector, Info]:
    """
    Return the static state for a solid model

    """
    if isinstance(model, transient.BaseTransientModel):
        is_tra_model = True
    elif isinstance(model, dynamical.BaseDynamicalModel):
        is_tra_model = False
    else:
        raise TypeError(f"Unknown `model` type {type(model)}")

    # Create a variable to store the solution
    if is_tra_model:
        state_n = model.state0.copy()
    else:
        state_n = model.state.copy()

    # Create a zero state (useful for initial guesses)
    zero_state = state_n.copy()
    zero_state[:] = 0

    # Set the initial guess (u, v, a = 0) if one isn't provided
    if state is None:
        state_n[:] = 0.0
    else:
        state_n[:] = state

    model.set_control(control)
    model.set_prop(prop)

    if solver == 'manual':
        # Here use the non-linear governing residual to solve
        # For a transient model, the residual contains initial condition effects
        # and to get rid of these effects we can create a new residual where
        # 'state/u0' always matches 'state/u1'
        form = model.residual.form
        if is_tra_model:
            # Zero final/initial states to solve for a transient
            zero_state = model.state1.copy()
            zero_state[:] = 0
            model.set_fin_state(zero_state)
            model.set_ini_state(zero_state)

            res_form = ufl.replace(
                form.ufl_forms, {form['state/u0']: form['state/u1']}
            )
        else:
            res_form = form.ufl_forms

        jac = dfn.derivative(res_form, form['state/u1'])

        def iterative_subproblem(x_n):
            form['state/u1'].vector()[:] = x_n
            dx = form['state/u1'].vector()

            def assem_res():
                res = dfn.assemble(res_form)
                for bc in model.residual.dirichlet_bcs['state/u1']:
                    bc.apply(res)
                return res

            def solve_res(res):
                A = dfn.assemble(jac)
                for bc in model.residual.dirichlet_bcs['state/u1']:
                    bc.apply(A)
                dfn.solve(A, dx, res)
                return dx

            return assem_res, solve_res

        def norm(res_n):
            return res_n.norm('l2')

        u_0 = model.residual.form['state/u1'].vector().copy()
        u_0 = state_n.sub['u']
        u, info = nonlineq.newton_solve(u_0, iterative_subproblem, norm=norm)
        state_n['u'] = u
    elif solver == 'automatic':
        jac = dfn.derivative(
            model.residual.form.ufl_forms, model.residual.form['state/u1']
        )
        dfn.solve(
            model.residual.form.ufl_forms == 0.0,
            model.residual.form['state/u1'],
            bcs=model.residual.dirichlet_bcs['state/u1'],
            J=jac,
            solver_parameters={"newton_solver": DEFAULT_NEWTON_SOLVER_PRM},
        )
        state_n['u'] = model.state1['u']
        info = {}
    else:
        raise ValueError(f"Unknown `solver`: '{solver}'")

    return state_n, info


# TODO: Refactor this to simply set appropriate blocks to a vector from value
def _set_coupled_model_substate(
    model: transient.BaseTransientFSIModel, xsub: bv.BlockVector
):
    """
    Set a subset of blocks in `model.state` from a given block vector

    Parameters
    ----------
    model:
        The model
    xsub:
        The block vector to set values from. Blocks with labels in `xsub` are
        used to set corresponding blocks of `model.state`
    """
    _state = model.state0.copy()
    _labels = list(xsub.labels[0])
    _state[_labels] = xsub

    # Set both initial and final states to ensure that the fluid pressure
    # is set for the final state; for explicit models only the initial fluid
    # state is passed as a force on the final solid residual
    model.set_ini_state(_state)
    model.set_fin_state(_state)


@_add_static_docstring
def static_coupled_configuration_picard(
    model: transient.BaseTransientFSIModel,
    control: bv.BlockVector,
    prop: bv.BlockVector,
) -> tuple[bv.BlockVector, Info]:
    """
    Solve for the static state of a coupled model

    """
    solid = model.solid
    fluid = model.fluid

    # This solves the static state only with (displacement, flow rate, pressure)
    # i.e. this ignores the velocity and acceleration of the solid
    labels = ['u', 'q', 'p']

    def iterative_subproblem(x_n):
        _set_coupled_model_substate(model, x_n)

        def assem_res():
            return model.assem_res()[labels]

        def solve(res):
            """
            Returns a new guess x_n+1

            x_n = [u, q, p]
            """
            # Solve for the solid deformation under the guessed fluid load
            dfn.solve(
                solid.residual.form.ufl_forms == 0.0,
                solid.residual.form['state/u1'],
                bcs=solid.residual.dirichlet_bcs['state/u1'],
                # J= ... ()
                solver_parameters={"newton_solver": DEFAULT_NEWTON_SOLVER_PRM},
            )
            # the vector corresponding to solid.residual.form['state/u1']
            u = bv.BlockVector([solid.state1['u'].copy()], labels=[['u']])

            # update the fluid load for the new solid deformation
            x_n['u'][:] = u[0]
            _set_coupled_model_substate(model, x_n)
            qp, _ = fluid.solve_state1(x_n[['q', 'p']])
            return bv.concatenate_vec([u, qp.copy()])

        return assem_res, solve

    # Set the initial state
    _x_n = model.state0.copy()[labels]
    _x_n[:] = 0
    _x_n, info = nonlineq.iterative_solve(_x_n, iterative_subproblem)

    # Assign the reduced state back into the full state
    x_n = model.state0.copy()
    x_n[:] = 0
    x_n[labels] = _x_n
    return x_n, info


# TODO: This one has a strange bug where Newton convergence is very slow
# I'm not sure if the answer it returns is correct or not
@_add_static_docstring
def static_coupled_configuration_newton(
    model: transient.BaseTransientFSIModel,
    control: bv.BlockVector,
    prop: bv.BlockVector,
    dt: float = 1e6,
) -> tuple[bv.BlockVector, Info]:
    """
    Return the static equilibrium state for a coupled model

    """

    def newton_subproblem(x_0):
        """
        Linear subproblem to be solved in a Newton solution strategy
        """
        ### Set the state to linearize around
        model.dt = dt
        # TODO: This might be the source of the bug since the newton strategy
        # would require only the final state to be updated; the function
        # belows updates both initial and final states
        _set_coupled_model_substate(model, x_0)

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
    return nonlineq.newton_solve(x_0, newton_subproblem, step_size=1.0)
