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

from typing import Tuple, Mapping, Any

import dolfin as dfn
import numpy as np

from blockarray import blockmat as bm, blockvec as bv
import nonlineq

from .models.transient import solid as slmodel, coupled as comodel
from .solverconst import DEFAULT_NEWTON_SOLVER_PRM

# import warnings
# warnings.filterwarnings("error")

dfn.set_log_level(50)

Info = Mapping[str, Any]

def _add_static_docstring(func):
    docs = \
    """
    Solve for the static state of a coupled model

    Parameters
    ----------
    model :
        The model
    control :
        The control parameters for the model
    props :
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
        model: slmodel.Solid,
        control: bv.BlockVector,
        props: bv.BlockVector
    ) -> Tuple[bv.BlockVector, Info]:
    """
    Return the static state for a solid model

    """
    # Set the initial guess u=0 and constants (v, a) = (0, 0)
    state_n = model.state0.copy()
    state_n[:] = 0.0
    model.set_fin_state(state_n)
    model.set_ini_state(state_n)

    model.set_control(control)
    model.set_props(props)

    jac = dfn.derivative(
        model.forms['form.un.f1uva'],
        model.forms['coeff.state.u1']
    )
    dfn.solve(
        model.forms['form.un.f1uva'] == 0.0,
        model.forms['coeff.state.u1'],
        bcs=[model.forms['bc.dirichlet']],
        J=jac,
        solver_parameters={"newton_solver": DEFAULT_NEWTON_SOLVER_PRM}
    )

    state_n['u'] = model.state1['u']

    info = {}
    return state_n, info

def _set_coupled_model_substate(model: comodel.FSIModel, xsub: bv.BlockVector):
    """
    Set a subset of blocks in `model.state` from a given block vector

    Parameters
    ----------
    model:
        The model
    xsub:
        The block vector to set values from. Block with labels in `xsub` are
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
        model: comodel.FSIModel,
        control: bv.BlockVector,
        props: bv.BlockVector,
    ) -> Tuple[bv.BlockVector, Info]:
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
                solid.forms['form.un.f1uva'] == 0.0,
                solid.forms['coeff.state.u1'],
                bcs=[solid.forms['bc.dirichlet']],
                J=solid.forms['form.bi.df1uva_du1'],
                solver_parameters={"newton_solver": DEFAULT_NEWTON_SOLVER_PRM}
            )
            # the vector corresponding to solid.forms['coeff.state.u1']
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
        model: comodel.FSIModel,
        control: bv.BlockVector,
        props: bv.BlockVector,
        dt: float=1e6
    ) -> Tuple[bv.BlockVector, Info]:
    """
    Return the static equilibrium state for a coupled model

    """
    def newton_subproblem(x_0):
        """
        Linear subproblem to be solved in a Newton solution strategy
        """
        ### Set the state to linearize around
        model.dt = dt
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
