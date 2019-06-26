"""
Adjoint model.

Makes a vocal fold go elggiw elggiw.

I'm using CGS : cm-g-s units
"""

from time import perf_counter

import numpy as np

from matplotlib import tri
from matplotlib import pyplot as plt

import h5py

import dolfin as dfn
import ufl

#import petsc4py
#petsc4py.init()
from petsc4py import PETSc

import forms as frm
import linalg
import statefileutils as sfu
import constants
import functionals

from misc import get_dynamic_fluid_props

# dfn.parameters['form_compiler']['optimize'] = True
# dfn.parameters['form_compiler']['cpp_optimize'] = True

dfu_dparam_form_adjoint = dfn.adjoint(ufl.derivative(frm.f1, frm.emod, frm.scalar_trial))

def decrement_adjoint(adj_x1, adj_x0, gradient, ii, solid_props, fluid_props, h5path, h5group='/'):
    """
    Returns the adjoint at the previous time step adj_x0 (adj_u0, adj_v0, adj_a0) of the forward model

    Parameters
    ----------
    adj_x1 : tuple (adj_u1, adj_v1, adj_a1) of dfn.Function
        'Initial' states for the adjoint model
    adj_x0 : tuple (adj_u0, adj_v0, adj_a0) of dfn.Function
        Decremented states for the adjoint model.
    solid_props : dict
        Should use this style of call in the future?
    fluid_props : dict
        A dictionary storing fluid properties.
    gradient : np.ndarray?
        The gradient
    h5path : string
        Path to an hdf5 file containing states from a forward run of the model.
    h5group : string
        The group where states are stored in the hdf5 file.

    Returns
    -------
    adj_x0 : tuple (adj_u0, adj_v0, adj_a0) of dfn.Function
        The next state of the forward model
    info : dict
        Additional info computed during the solve that might be useful.
    """
    adj_u1, adj_v1, adj_a1 = adj_x1
    adj_u0, adj_v0, adj_a0 = adj_x0

    ## Set form coefficients to represent f^{i+1}
    sfu.set_states(ii+1, h5path, group=h5group, u0=frm.u0, v0=frm.v0, a0=frm.a0, u1=frm.u1)
    frm.set_pressure(fluid_props)
    dpressure_du0, dq_du0 = frm.set_flow_sensitivity(fluid_props)
    frm.emod.vector()[:] = solid_props['elastic_modulus']

    # Assemble needed forms
    df1_du0 = dfn.assemble(frm.df1_du0_adjoint)
    df1_dv0 = dfn.assemble(frm.df1_dv0_adjoint)
    df1_da0 = dfn.assemble(frm.df1_da0_adjoint)

    # Correct df1_du0 since pressure depends on u0
    df1_dpressure = dfn.as_backend_type(dfn.assemble(frm.df1_dp_adjoint)).mat()
    df1_du0 = df1_du0 + dfn.Matrix(dfn.PETScMatrix(dpressure_du0.transposeMatMult(df1_dpressure)))

    ## Set form coefficients to represent f^{i}
    sfu.set_states(ii, h5path, group=h5group, u0=frm.u0, v0=frm.v0, a0=frm.a0, u1=frm.u1)
    frm.set_pressure(fluid_props)
    frm.emod.vector()[:] = solid_props['elastic_modulus']

    # Assemble needed forms
    df0_du0 = dfn.assemble(frm.df1_du1_adjoint)
    df0_dparam = dfn.assemble(dfu_dparam_form_adjoint)

    ## Adjoint recurrence relations
    # Probably don't need to set Dirichlet BCs on adj_a or adj_v because they are always zeroed
    # out when solving for adj_u (but doing it anyway)
    gamma, beta = frm.gamma.values()[0], frm.beta.values()[0]
    dt = frm.dt.values()[0]

    # These are manually implemented matrix multiplications
    adj_a0.vector()[:] = -1 * (df1_da0 * adj_u1.vector()
                               + dt*(gamma/2/beta-1) * adj_v1.vector()
                               + (1/2/beta-1) * adj_a1.vector())
    frm.bc_base_adjoint.apply(adj_a0.vector())

    adj_v0.vector()[:] = -1 * (df1_dv0 * adj_u1.vector()
                               + (gamma/beta-1) * adj_v1.vector()
                               + 1/beta/dt * adj_a1.vector())
    frm.bc_base_adjoint.apply(adj_v0.vector())

    # dcost_du0 = functionals.dfluidwork_du(ii, states_path, fluid_properties, dpressure_du0, dq_du0)
    dcost_du0 = functionals.dvocaleff_du(ii, h5path, fluid_props, h5group=h5group)
    adj_u0_lhs = dcost_du0 \
                 + gamma/beta/dt * adj_v0.vector() + 1/beta/dt**2 * adj_a0.vector() \
                 - (df1_du0 * adj_u1.vector()
                    + gamma/beta/dt * adj_v1.vector()
                    + 1/beta/dt**2 * adj_a1.vector())
    frm.bc_base_adjoint.apply(df0_du0, adj_u0_lhs)
    dfn.solve(df0_du0, adj_u0.vector(), adj_u0_lhs)

    ## Update gradient
    gradient = gradient - 1*df0_dparam*adj_u0.vector()

    return (adj_u0, adj_v0, adj_a0), gradient

def adjoint(solid_props, fluid_props, h5path, h5group='/', show_figure=False):
    """
    Returns the gradient of the cost function w.r.t elastic modulus using the adjoint model.

    Parameters
    ----------
    solid_props : dict
        Should use this style of call in the future?
    fluid_props : dict
        A dictionary storing fluid properties.
    states_path : string
        The path of the file containing states from the forward model run.
    h5path : string
        Path to an hdf5 file containing states from a forward run of the model.
    h5group : string
        The group where states are stored in the hdf5 file.
    show_figures : bool
        Whether to display a figure showing the solution or not.
    """
    ## Allocate adjoint states
    adj_u0 = dfn.Function(frm.vector_function_space)
    adj_v0 = dfn.Function(frm.vector_function_space)
    adj_a0 = dfn.Function(frm.vector_function_space)

    adj_u1 = dfn.Function(frm.vector_function_space)
    adj_v1 = dfn.Function(frm.vector_function_space)
    adj_a1 = dfn.Function(frm.vector_function_space)

    ## Allocate space for the gradient
    gradient = np.zeros(frm.emod.vector().size())

    ## Loop through states for adjoint computation
    num_states = sfu.get_num_states(h5path, group=h5group)
    time = sfu.get_time(h5path, group=h5group)
    print(time)

    ## Set form coefficients to represent f^{N-1}
    fluid_props_ = get_dynamic_fluid_props(fluid_props, time[-2])
    frm.emod.vector()[:] = solid_props['elastic_modulus']
    sfu.set_states(num_states-1, h5path, group=h5group, u0=frm.u0, v0=frm.v0, a0=frm.a0, u1=frm.u1)
    frm.set_pressure(fluid_props_)

    df1_du1 = dfn.assemble(frm.df1_du1_adjoint)

    ## Initialize adjoint state
    # dcost_du1 = functionals.dfluidwork_du(num_states-1, states_path, fluid_properties, dpressure_du0, dq_du0)
    dcost_du1 = functionals.dvocaleff_du(num_states-1, h5path, fluid_props_, h5group=h5group)

    frm.bc_base_adjoint.apply(df1_du1, dcost_du1)
    dfn.solve(df1_du1, adj_u1.vector(), dcost_du1)
    adj_v1.vector()[:] = 0
    adj_a1.vector()[:] = 0

    df1_dparam = dfn.assemble(dfu_dparam_form_adjoint)
    gradient += -1*df1_dparam*adj_u1.vector() + 0

    for ii in range(num_states-2, 0, -1):
        # You need to modify this in some way because there are potentially two fluid property states, one at time 0 and one at time 1 that are needed.
        fluid_props_ = get_dynamic_fluid_props(fluid_props, time[ii-5])
        (adj_u0, adj_v0, adj_a0), gradient = decrement_adjoint(
            (adj_u1, adj_v1, adj_a1), (adj_u0, adj_v0, adj_a0), gradient, ii, solid_props,
            fluid_props_, h5path, h5group=h5group)

        # Update adjoint recurrence relations for the next iteration
        adj_a1.assign(adj_a0)
        adj_v1.assign(adj_v0)
        adj_u1.assign(adj_u0)

    ## Plot the gradient
    if show_figure:
        fig, ax = plt.subplots(1, 1)

        ax.set_aspect('equal')
        coords = frm.mesh.coordinates()[...]
        triangulation = tri.Triangulation(coords[:, 0], coords[:, 1], triangles=frm.mesh.cells())

        ax.set_xlim(-0.1, 0.7, auto=False)
        ax.set_ylim(-0.1, 0.7)

        ax.axhline(y=frm.y_midline, ls='-.')
        ax.axhline(y=frm.y_midline-frm.collision_eps, ls='-.', lw=0.5)

        ax.set_title('Gradient')

        mappable = ax.tripcolor(triangulation, gradient[frm.vert_to_sdof], edgecolors='k', shading='flat')
        coords_fixed = frm.mesh.coordinates()[frm.fixed_vertices]
        ax.plot(coords_fixed[:, 0], coords_fixed[:, 1], color='C1')

        fig.colorbar(mappable, ax=ax)

        plt.show()

    return gradient

if __name__ == '__main__':
    input_path = 'out/test.h5'

    fluid_props = constants.DEFAULT_FLUID_PROPERTIES
    solid_props = {'elastic_modulus': frm.emod.vector()}

    runtime_start = perf_counter()
    gradient = adjoint(solid_props, fluid_props, input_path)
    runtime_end = perf_counter()
    print(f"Runtime {runtime_end-runtime_start:.2f} seconds")

    with h5py.File('out/adjoint.h5', mode='w') as f:
        f.create_dataset('gradient', data=gradient)
