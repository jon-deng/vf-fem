"""
Functions for visualizing solutions.
"""

import numpy as np
from matplotlib import tri
import matplotlib.pyplot as plt

from . import constants

def triangulation(model):
    """
    Returns a triangulation for a mesh.

    Parameters
    ----------
    model : ForwardModel

    Returns
    -------
    matplotlib.triangulation
    """
    xy = model.get_ref_config()

    triangu = tri.Triangulation(xy[..., 0], xy[..., 1], triangles=model.mesh.cells())

    return triangu

def init_figure(model, fluid_props):
    """
    Returns a figure and tuple of axes to plot the solution into.
    """
    gridspec_kw = {'height_ratios': [4, 2, 2], 'width_ratios': [10, 0.5]}
    fig, axs = plt.subplots(3, 2, gridspec_kw=gridspec_kw, figsize=(6, 8))
    axs[0, 0].set_aspect('equal', adjustable='datalim')

    thickness_bottom = np.amax(model.mesh.coordinates()[..., 0])

    x = np.arange(model.surface_vertices.shape[0])
    y = np.arange(model.surface_vertices.shape[0])

    axs[1, 0].plot(x, y, marker='o')

    # Initialize lines for plotting flow rate and the rate of flow rate
    axs[2, 0].plot([0], [0])

    axs[1, 0].set_xlim(-0.2, 1.4, auto=False)
    p_sub = fluid_props['p_sub'] / constants.PASCAL_TO_CGS
    axs[1, 0].set_ylim(-0.25*p_sub, 1.1*p_sub, auto=False)

    axs[1, 0].set_ylabel("Surface pressure [Pa]")

    axs[2, 0].set_ylim(-0.01, 0.1)
    axs[2, 0].set_ylabel("Glottal width [cm]")

    for ax in axs[1:, -1]:
        ax.set_axis_off()

    return fig, axs

def update_figure(fig, axs, model, t, x, fluid_info, solid_props, fluid_props):
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
    axs[0, 0].clear()
    delta_xy = x[0].vector()[model.solid.vert_to_vdof.reshape(-1)].reshape(-1, 2)
    xy_current = model.mesh.coordinates() + delta_xy
    triangulation = tri.Triangulation(xy_current[:, 0], xy_current[:, 1],
                                      triangles=model.mesh.cells())
    mappable = axs[0, 0].tripcolor(triangulation,
                                   solid_props['elastic_modulus'][model.vert_to_sdof],
                                   edgecolors='k', shading='flat')
    fig.colorbar(mappable, cax=axs[0, 1])
    axs[0, 1].set_ylabel('[kPa]')

    xy_surface = xy_current[model.surface_vertices]

    xy_min, xy_sep = fluid_info['xy_min'], fluid_info['xy_sep']
    axs[0, 0].plot(*xy_min, marker='o', mfc='none', color='C0')
    axs[0, 0].plot(*xy_sep, marker='o', mfc='none', color='C1')
    axs[0, 0].plot(xy_surface[:, 0], xy_surface[:, 1], color='C3')

    axs[0, 0].set_title(f'Time: {1e3*t:>5.1f} ms')
    axs[0, 0].set_xlim(-0.2, 1.8, auto=False)
    axs[0, 0].set_ylim(0.0, 1.0, auto=False)

    axs[0, 0].axhline(y=fluid_props['y_midline'], ls='-.', lw=0.5)
    axs[0, 0].axhline(y=model.y_collision.values()[0], ls='-.', lw=0.5)

    pressure_profile = axs[1, 0].lines[0]
    pressure_profile.set_data(xy_surface[:, 0], fluid_info['pressure']/constants.PASCAL_TO_CGS)
    axs[1, 0].set_xlim(-0.2, 1.8, auto=False)

    gw = fluid_props['y_midline'] - np.amax(xy_current[:, 1])
    line = axs[2, 0].lines[0]
    xdata = np.concatenate((line.get_xdata(), [t]), axis=0)
    ydata = np.concatenate((line.get_ydata(), [gw]), axis=0)
    line.set_data(xdata, ydata)

    axs[2, 0].set_xlim(0, np.maximum(1.2*t, 0.01))
    axs[2, 0].set_ylim(0, 0.03)

    return fig, axs

def plot_grad(model):
    ## Plot the gradient
    fig, ax = plt.subplots(1, 1)

    ax.set_aspect('equal')
    coords = model.get_ref_config()
    triangulation = tri.Triangulation(coords[:, 0], coords[:, 1], triangles=model.mesh.cells())

    ax.set_xlim(-0.1, 0.7, auto=False)
    ax.set_ylim(-0.1, 0.7)

    ax.axhline(y=model.y_midline, ls='-.')
    ax.axhline(y=model.y_midline-model.collision_eps, ls='-.', lw=0.5)

    ax.set_title('Gradient')

    mappable = ax.tripcolor(triangulation, gradient[model.vert_to_sdof],
                            edgecolors='k', shading='flat')
    coords_fixed = coords[model.fixed_vertices]
    ax.plot(coords_fixed[:, 0], coords_fixed[:, 1], color='C1')

    fig.colorbar(mappable, ax=ax)

    plt.show()