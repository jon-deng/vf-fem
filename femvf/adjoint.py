"""
Adjoint model.

Makes a vocal fold go elggiw elggiw.

I'm using CGS : cm-g-s units
"""

import numpy as np

from matplotlib import tri
from matplotlib import pyplot as plt

import dolfin as dfn
import ufl


# @profile
def decrement_adjoint(model, adj_x2, iter_params1, iter_params2, dcost_du1):
    """
    Returns the adjoint at the previous time step.

    Each adjoint step is based on an indexing scheme where the postfix on a variable represents that
    variable at time index n + postfix. For example, variables x0, x1, and x2 correspond to states
    at n, n+1, and n+2.

    This is done because the adjoint calculation to solve for :math:`lambda_{n+1}` given
    :math:`lambda_{n+2}` requires the forward equations :math:`f^{n+2}=0`, and :math:`f^{n+1}=0`,
    which in turn requires states :math:`x^{n}`, :math:`x^{n+1}`, and :math:`x^{n+2}` to be defined.

    Note that :math:`f^{n+1} = f^{n+1}([u, v, a]^{n+1}, [u, v, a]^{n}) = 0` involves the FEM
    approximation and time stepping scheme that defines the state :math`x^{n+1} = (u, v, a)^{n+1}`
    implicitly, which could be linear or non-linear.

    Parameters
    ----------
    model : forms.ForwardModel
    adj_x2 : tuple of dfn.Function
        A tuple (adj_u2, adj_v2, adj_a2) of 'initial' (time index 2) states for the adjoint model.
    iter_params1, iter_params2 : tuple
        iter_params1 is a tuple of:
        (x^{n}, t^{n+1}-t^{n}, solid_props^{n}, fluid_props^{n}, u^{n+1})
        where x is itself a tuple of (u, v, a)
    h5path : string
        Path to an hdf5 file containing states from a forward run of the model.
    h5group : string
        The group where states are stored in the hdf5 file.

    Returns
    -------
    adj_x1 : tuple of dfn.Function
        The 'next' state (adj_u1, adj_v1, adj_a1) of the adjoint model.
    info : dict
        Additional info computed during the solve that might be useful.
    """
    adj_u2, adj_v2, adj_a2 = adj_x2

    ## Set form coefficients to represent f^{n+2} aka f2(x1, x2) -> x2
    dt1 = iter_params1[1]
    model.set_iter_params(*iter_params2)

    # Assemble needed forms
    # breakpoint()
    df2_du1 = model.assem_df1_du0_adj()
    df2_dv1 = model.assem_df1_dv0_adj()
    df2_da1 = model.assem_df1_da0_adj()

    # Correct df2_du1 since pressure depends on u1 for explicit FSI forcing
    df2_dpressure = dfn.assemble(model.df1_dp_adj)
    dpressure_du1 = dfn.PETScMatrix(model.get_flow_sensitivity()[0])

    ## Set form coefficients to represent f^{n+1} aka f1(x0, x1) -> x1
    # _x0 = (x0[0].vector(), x0[1].vector(), x0[2].vector())
    dt2 = iter_params2[1]
    model.set_iter_params(*iter_params1)

    # Assemble needed forms
    df1_du1 = model.assem_df1_du1_adj()

    ## Adjoint recurrence relations
    # Allocate adjoint states
    adj_u1 = dfn.Function(model.vector_function_space)
    adj_v1 = dfn.Function(model.vector_function_space)
    adj_a1 = dfn.Function(model.vector_function_space)

    # You probably don't need to set Dirichlet BCs on adj_a or adj_v because they are always zeroed
    # out when solving for adj_u (but I'm gonna do it anyway)
    gamma, beta = model.gamma.values()[0], model.beta.values()[0]

    # These are manually implemented matrix multiplications representing adjoint recurrence
    # relations
    # TODO: you can probably take advantage of autodiff capabilities of fenics of the newmark
    # schemes so you don't have to do the differentiation manually like you did here.
    adj_a1.vector()[:] = -1 * (df2_da1 * adj_u2.vector()
                               + dt2 * (gamma/2/beta-1) * adj_v2.vector()
                               + (1/2/beta-1) * adj_a2.vector())
    model.bc_base_adj.apply(adj_a1.vector())

    adj_v1.vector()[:] = -1 * (df2_dv1 * adj_u2.vector()
                               + (gamma/beta-1) * adj_v2.vector()
                               + 1/beta/dt2 * adj_a2.vector())
    model.bc_base_adj.apply(adj_v1.vector())

    # import ipdb; ipdb.set_trace()
    df2_du1_correction = dfn.Function(model.vector_function_space)
    dpressure_du1.transpmult(df2_dpressure * adj_u2.vector(), df2_du1_correction.vector())
    adj_u1_lhs = dcost_du1 \
                 + gamma/beta/dt1 * adj_v1.vector() + 1/beta/dt1**2 * adj_a1.vector() \
                 - (df2_du1 * adj_u2.vector() + df2_du1_correction.vector()
                    + gamma/beta/dt2 * adj_v2.vector()
                    + 1/beta/dt2**2 * adj_a2.vector())
    model.bc_base_adj.apply(df1_du1, adj_u1_lhs)
    dfn.solve(df1_du1, adj_u1.vector(), adj_u1_lhs, 'petsc')

    return (adj_u1, adj_v1, adj_a1)

# @profile
def adjoint(model, f, Functional, functional_kwargs, show_figure=False):
    """
    Returns the gradient of the cost function w.r.t elastic modulus using the adjoint model.

    Parameters
    ----------
    model : forms.ForwardModel
    f : statefile.StateFile
    Functional : class functionals.GenericFunctional
    functional_kwargs : dict
        Options to pass to the functional
    show_figures : bool
        Whether to display a figure showing the solution or not.

    Returns
    -------
    np.array of float
        The sensitivity of the functional wrt parameters.
    """
    # x_vecs = 3*[None,]

    # Initialize the functional instance and run it once to initialize any cached values
    functional = Functional(model, f, **functional_kwargs)
    functional_value = functional()

    df1_dparam_form_adj = dfn.adjoint(ufl.derivative(model.f1, model.emod, model.scalar_trial))

    ## Preallocating vector
    # Adjoint states
    vfunc_space = model.vector_function_space
    adj_u1 = dfn.Function(vfunc_space)
    adj_v1 = dfn.Function(vfunc_space)
    adj_a1 = dfn.Function(vfunc_space)

    adj_u2 = dfn.Function(vfunc_space)
    adj_v2 = dfn.Function(vfunc_space)
    adj_a2 = dfn.Function(vfunc_space)

    # Model states
    x0 = (dfn.Function(vfunc_space), dfn.Function(vfunc_space), dfn.Function(vfunc_space))
    x1 = (dfn.Function(vfunc_space), dfn.Function(vfunc_space), dfn.Function(vfunc_space))
    x2 = (dfn.Function(vfunc_space), dfn.Function(vfunc_space), dfn.Function(vfunc_space))

    # Allocate space for the gradient
    gradient = dfn.Function(model.scalar_function_space).vector() # np.zeros(model.emod.vector().size())

    ## Initialize Adjoint calculation
    #

    # Set form coefficients to represent f^{N-1} (the final forward increment model that solves
    # for the final state)
    # To initialize, we need to solve for \lambda^{N-1} i.e. `adj_u2`, `adj_v2`, `adj_a2` etc.
    N = f.size
    times = f.get_solution_times()

    f.get_state(N-1, out=x2)
    f.get_state(N-2, out=x1)
    fluid_props2 = f.get_fluid_props(N-1)
    fluid_props1 = f.get_fluid_props(N-2)
    solid_props = f.get_solid_props()
    dt2 = times[N-1]-times[N-2]

    _x1 = [comp.vector() for comp in x1]
    _x2 = [comp.vector() for comp in x2]

    # Cache mass and stiffness matrices
    model.set_solid_props(solid_props)
    model.reset_cache()
    model.reset_adj_cache()

    iter_params2 = (_x1, dt2, solid_props, fluid_props1, _x2[0])
    iter_params3 = (_x2, 0.0, solid_props, fluid_props2, None)

    dcost_du2 = functional.du(N-1, iter_params2, iter_params3)

    model.set_iter_params(*iter_params2)
    df2_du2 = model.assem_df1_du1_adj()

    model.bc_base_adj.apply(df2_du2, dcost_du2)
    dfn.solve(df2_du2, adj_u2.vector(), dcost_du2)
    adj_v2.vector()[:] = 0
    adj_a2.vector()[:] = 0

    df2_dparam = dfn.assemble(df1_dparam_form_adj)
    gradient -= df2_dparam*adj_u2.vector()

    ## Loop through states for adjoint computation
    # Note that ii corresponds to the time index of the adjoint state we are solving for.
    # In a given loop, adj^{ii+1} is known, and the iteration of the loop finds adj^{ii}
    for ii in range(N-2, 0, -1):
        # Properties at index 2 through 1 were loaded during initialization, so we only need to
        # read index 0
        x0 = f.get_state(ii-1, out=x0)
        fluid_props0 = f.get_fluid_props(ii-1)

        dt1 = times[ii] - times[ii-1]

        _x0 = [comp.vector() for comp in x0]
        _x1 = [comp.vector() for comp in x1]
        _x2 = [comp.vector() for comp in x2]

        # breakpoint()
        iter_params1 = (_x0, dt1, solid_props, fluid_props0, _x1[0])
        iter_params2 = (_x1, dt2, solid_props, fluid_props1, _x2[0])

        dcost_du1 = functional.du(ii, iter_params1, iter_params2)

        (adj_u1, adj_v1, adj_a1) = decrement_adjoint(
            model, (adj_u2, adj_v2, adj_a2), iter_params1, iter_params2, dcost_du1)

        # Update gradient using the adjoint state
        # TODO: Functionals are assumed to not depend on velocity or acceleration, so we only
        # multiply by adj_u1. In the future you might have to use adj_v1 and adj_a1 too.

        # Assemble needed forms
        _x0 = [comp.vector() for comp in x0]
        model.set_iter_params(_x0, dt1, solid_props, fluid_props0, u1=x1[0].vector())
        df1_dparam = dfn.assemble(df1_dparam_form_adj)

        gradient -= df1_dparam*adj_u1.vector()

        if functional.dparam() is not None:
            gradient += functional.dparam()

        # Update properties for the next iteration
        for comp1, comp2 in zip(x1, x2):
            comp2.assign(comp1)

        for comp0, comp1 in zip(x0, x1):
            comp1.assign(comp0)

        dt2 = dt1
        fluid_props1 = fluid_props0

        # Update adjoint recurrence relations for the next iteration
        adj_a2.assign(adj_a1)
        adj_v2.assign(adj_v1)
        adj_u2.assign(adj_u1)

    return functional_value, gradient
