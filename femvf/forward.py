"""
Forward model

Uses CGS (cm-g-s) units unless otherwise stated
"""
from time import perf_counter
import os
from os.path import join

import h5py

import numpy as np
from math import isclose, ceil, floor, remainder

from matplotlib import tri
from matplotlib import pyplot as plt
from matplotlib import animation

import dolfin as dfn
# import ufl

from . import forms as frm
from . import fluids
from . import constants
from . import functionals
from . import visualization as vis

# from .collision import detect_collision
from .misc import get_dynamic_fluid_props

def solution_times(measured_times, dt):
    """
    Returns an array of solution times spaced at dt starting from the first measurement time.
    
    Measurement times are interlaced between starting from tmeas[0] with measurement times 
    interlaced between if they are not divisible.
    """
    times = np.array([measured_times[0]])

    for n in range(measured_times.size-1):
        n_start = None
        n_stop = None

        tgap = measured_times[n]-measured_times[0]
        if isclose(remainder(tgap, dt), 0, rel_tol=1e-10, abs_tol=2**-52):
            n_start = int(tgap/dt) + 1
        else:
            n_start = ceil(tgap/dt)

        tgap = measured_times[n+1]-measured_times[0]
        if isclose(remainder(tgap, dt), 0, rel_tol=1e-10, abs_tol=2**-52):
            n_stop = int(tgap/dt) - 1
        else:
            n_stop = floor(tgap/dt)
        between_times = times[0] + dt*np.arange(n_start, n_stop+1)
        times = np.concatenate((times, between_times, [measured_times[n+1]]), axis=0)

    return times
        
def init_figure(fluid_props):
    """
    Returns a figure and tuple of axes to plot the solution into.
    """
    gridspec_kw = {'height_ratios': [4, 2, 2]}
    fig, axs = plt.subplots(3, 1, gridspec_kw=gridspec_kw, figsize=(6, 8))
    axs[0].set_aspect('equal', adjustable='datalim')

    x = np.arange(frm.surface_vertices.shape[0])
    y = np.arange(frm.surface_vertices.shape[0])

    axs[1].plot(x, y, marker='o')

    # Initialize lines for plotting flow rate and the rate of flow rate
    axs[2].plot([0], [0])
    # axs[3].plot([0], [0])
    # axs[3].plot([0], [0])

    axs[0].set_xlim(-0.1, frm.thickness_bottom+0.1, auto=False)
    axs[0].set_ylim(0.0, 0.7, auto=False)

    axs[1].set_xlim(-0.1, frm.thickness_bottom+0.1, auto=False)
    p_sub = fluid_props['p_sub'] / constants.PASCAL_TO_CGS
    axs[1].set_ylim(-0.25*p_sub, 1.1*p_sub, auto=False)

    axs[1].set_ylabel("Surface pressure [Pa]")

    axs[2].set_ylabel("Glottal flow rate [cm^2/s]")

    # axs[3].set_xlabel("Time [s]")
    # axs[3].set_ylabel("Glottal flow rate rate [cm^2/s^2]")
    return fig, axs

def update_figure(fig, axs, t, x, fluid_info, fluid_props):
    """
    Plots the FEM solution into a figure.

    Parameters
    ----------
    fig : matplotlib.Figure
    axs : tuple of matplotlib.Axes
    x : tuple of dfn.Function
        Kinematic states (u, v, a)
    fluid_props : dict
        Fluid properties at time t

    Returns
    -------
    fig, axs
    """
    axs[0].clear()

    delta_xy = x[0].vector()[frm.vert_to_vdof.reshape(-1)].reshape(-1, 2)
    xy_current = frm.mesh.coordinates() + delta_xy
    triangulation = tri.Triangulation(xy_current[:, 0], xy_current[:, 1],
                                      triangles=frm.mesh.cells())
    axs[0].triplot(triangulation)

    xy_surface = xy_current[frm.surface_vertices]

    xy_min, xy_sep = fluid_info['xy_min'], fluid_info['xy_sep']
    axs[0].plot(*xy_min, marker='o', mfc='none', color='C0')
    axs[0].plot(*xy_sep, marker='o', mfc='none', color='C1')
    axs[0].plot(xy_surface[:, 0], xy_surface[:, 1], color='C3')

    axs[0].set_title(f'Time: {1e3*t:>5.1f} ms')

    axs[0].axhline(y=frm.y_midline, ls='-.', lw=0.5)
    axs[0].axhline(y=frm.y_midline-frm.collision_eps, ls='-.', lw=0.5)

    pressure_profile = axs[1].lines[0]
    pressure_profile.set_data(xy_surface[:, 0], fluid_info['pressure']/constants.PASCAL_TO_CGS)

    line = axs[2].lines[0]
    xdata = np.concatenate((line.get_xdata(), [t]), axis=0)
    ydata = np.concatenate((line.get_ydata(), [fluid_info['flow_rate']]), axis=0)
    line.set_data(xdata, ydata)

    # line = axs[3].lines[1]
    # fd = (ydata[1:]-ydata[:-1])/(xdata[1:]-xdata[:-1])
    # line.set_data(xdata[:-1], fd)

    # line = axs[3].lines[0]
    # ydata = np.concatenate((line.get_ydata(), [fluid_info['dt_flow_rate']]), axis=0)
    # line.set_data(xdata, ydata)

    axs[2].set_xlim(0, 1.2*t)
    axs[2].set_ylim(0, 200)

    # axs[3].set_xlim(0, 1.2*t)
    # axs[3].set_ylim(np.min(ydata)*1.1, np.max(ydata)*1.1)

    return fig, axs

def increment_forward(x0, dt, solid_props, fluid_props):
    """
    Returns the states at the next time, x1 = (u1, v1, a1).

    Parameters
    ----------
    x0 : tuple of dfn.Function
        Initial states (u0, v0, a0) for the forward model
    dt : float
        The time step to increment over
    solid_props : dict
        A dictionary of solid properties
    fluid_props : dict
        A dictionary storing fluid properties.

    Returns
    -------
    tuple of dfn.Function
        The next state (u1, v1, a1) of the forward model
    fluid_info : dict
        A dictionary containing information on the fluid solution. These include the flow rate,
        surface pressure, etc.
    """
    u0, v0, a0 = x0

    u1 = dfn.Function(frm.vector_function_space)
    v1 = dfn.Function(frm.vector_function_space)
    a1 = dfn.Function(frm.vector_function_space)

    ## Update form coefficients
    frm.emod.vector()[:] = solid_props['elastic_modulus']
    frm.u0.assign(u0)
    frm.v0.assign(v0)
    frm.a0.assign(a0)
    frm.dt.assign(dt)
    fluid_info = frm.set_pressure(fluid_props)

    # # check for collision
    # verts_coll = detect_collision(frm.mesh, frm.vertex_marker, frm.omega_contact, frm.u0, frm.v0, frm.a0,
    #                               frm.vert_to_vdof)
    # # Collision boundary conditions
    # if verts_coll.size > 0:
    #     print('Woah, collision!')

    ## Solve the thing
    frm.u1.assign(u0)
    dfn.solve(frm.fu_nonlin == 0, frm.u1, bcs=frm.bc_base, J=frm.jac_fu_nonlin)
    u1.assign(frm.u1)

    v1.vector()[:] = frm.newmark_v(u1.vector(), u0.vector(), v0.vector(), a0.vector(), frm.dt)
    a1.vector()[:] = frm.newmark_a(u1.vector(), u0.vector(), v0.vector(), a0.vector(), frm.dt)

    return (u1, v1, a1), fluid_info

def forward(tmeas, dt, solid_props, fluid_props, h5file='tmp.h5', h5group='/', show_figure=False, 
            figure_path=None):
    """
    Solves the forward model over a time interval.

    Parameters
    ----------
    tmeas : array_like of float
        Specific times at which the model should be solved. There should be a minimum of two 
        entries. The first entry is the starting time while the final entry is the final time. A 
        common way to set this would be to set [0, tfinal] to record the first step and final step 
        + all time steps in between.
    dt : float
        The time step in seconds.
    solid_props : dict
        A dictionary of solid properties.
    fluid_props : dict
        A dictionary of fluid properties.
    h5file : string
        Path to an hdf5 file where states will be appended.
    group : string
        An h5 group to save under
    show_figure : bool
        Determines whether to display figures of the solution or not.
    figure_path : string
        A path to save figures to. The figures will have a postfix of the iteration number and
        extension added.
    """
    # Allocate functions for states
    u0 = dfn.Function(frm.vector_function_space)
    v0 = dfn.Function(frm.vector_function_space)
    a0 = dfn.Function(frm.vector_function_space)

    u1 = dfn.Function(frm.vector_function_space)
    v1 = dfn.Function(frm.vector_function_space)
    a1 = dfn.Function(frm.vector_function_space)

    # Set solid material properties
    elastic_modulus = solid_props['elastic_modulus']
    frm.emod.vector()[:] = elastic_modulus

    ## Allocate a figure for plotting
    fig, axs = None, None
    if show_figure:
        fig, axs = init_figure(fluid_props)

    # Get the solution times
    tmeas = np.array(tmeas)
    assert tmeas.size >= 2
    assert tmeas[-1] > tmeas[0]

    times = solution_times(tmeas, dt)

    ## Initialize datasets to save in h5 file
    with h5py.File(h5file, mode='a') as f:
        # Kinematic states
        for data, dataset_name in zip([u0, v0, a0], ['u', 'v', 'a']):
            f.create_dataset(join(h5group, dataset_name), shape=(times.size, data.vector()[:].size),
                             dtype=np.float64)
            f[join(h5group, dataset_name)][0] = data.vector()

        f.create_dataset(join(h5group, 'time'), data=times)

        # Fluid properties
        for label in fluids.FLUID_PROP_LABELS:
            f.create_dataset(join(h5group, 'fluid_properties', label), shape=(times.size-1,))

        # Solid properties
        # TODO: Assuming only one time constant solid property here but there may be more.
        f.create_dataset(join(h5group, 'solid_properties', 'elastic_modulus'),
                         data=frm.emod.vector()[:])

    ## Loop through solution times and write solution variables to h5file.
    with h5py.File(h5file, mode='a') as f:
        for ii, t in enumerate(times[:-1]):
            ## Increment the state
            fluid_props_ii = get_dynamic_fluid_props(fluid_props, t)
            dt_ = times[ii+1] - times[ii]
            (u1, v1, a1), info = increment_forward([u0, v0, a0], dt_, solid_props, fluid_props_ii)

            ## Write the solution outputs to a file
            # State variables
            for label, value in zip(['u', 'v', 'a'], [u1, v1, a1]):
                f[join(h5group, label)][ii+1] = value.vector()[:]

            # Fluid properties
            for label in fluids.FLUID_PROP_LABELS:
                f[join(h5group, 'fluid_properties', label)][ii] = fluid_props_ii[label]

            ## Update initial conditions for the next time step
            u0.assign(u1)
            v0.assign(v1)
            a0.assign(a1)

            ## Plot the solution
            if show_figure:
                fig, axs = update_figure(fig, axs, t, (u0, v0, a0), info, fluid_props_ii)
                plt.pause(0.001)

                if figure_path is not None:
                    ext = '.png'
                    fig.savefig(f'{figure_path}_{ii}{ext}')

if __name__ == '__main__':
    dfn.set_log_level(30)
    emod = None
    with h5py.File('out/opt-nlopt/ElasticModuli.h5') as f:
        emod = f['elastic_modulus'][0, :]
    # emod = constants.DEFAULT_SOLID_PROPERTIES['elastic_modulus']

    solid_props = {'elastic_modulus': emod}
    fluid_props = constants.DEFAULT_FLUID_PROPERTIES
    # fluid_props['p_sub'] = [1800 * constants.PASCAL_TO_CGS, 1800 * constants.PASCAL_TO_CGS, 1, 1]
    # fluid_props['p_sub_time'] = [0, 2.5e-3, 2.5e-3, 0.02]

    save_path = f"out/test.h5"
    try:
        os.remove(save_path)
    except FileNotFoundError:
        pass

    dt = 1e-4
    runtime_start = perf_counter()
    forward([0, 0.05], dt, solid_props, fluid_props, save_path, show_figure=True,
            figure_path='out/opt-nlopt/anim_start/frame')
    runtime_end = perf_counter()

    print(f"Runtime {runtime_end-runtime_start:.2f} seconds")
    plt.show()
