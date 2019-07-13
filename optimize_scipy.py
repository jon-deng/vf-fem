"""
Find optimal elastic modulus distribution to maximize vocal efficiency.
"""

from os import path

from time import perf_counter
import dolfin as dfn
import numpy as np
from scipy import optimize

from matplotlib import pyplot as plt
from matplotlib import tri

import h5py

from forward import forward
from adjoint import adjoint
import functionals

import forms as frm
import constants

np.seterr(all='raise')

def plot_elastic_modulus(elastic_modulus):
    """
    Plots the elastic modulus

    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal')

    coords = frm.mesh.coordinates()[...]
    triangulation = tri.Triangulation(coords[:, 0], coords[:, 1], triangles=frm.mesh.cells())

    ax.set_xlim(-0.1, frm.thickness_bottom+0.1, auto=False)
    ax.set_ylim(0.0, frm.depth+0.1)

    ax.axhline(y=frm.y_midline, ls='-.')
    ax.axhline(y=frm.y_midline-frm.collision_eps, ls='-.', lw=0.5)

    ax.set_title('Elastic modulus')

    mappable = ax.tripcolor(triangulation, elastic_modulus[frm.vert_to_sdof], edgecolors='k',
                            shading='flat')
    coords_fixed = frm.mesh.coordinates()[frm.fixed_vertices]
    ax.plot(coords_fixed[:, 0], coords_fixed[:, 1], color='C1')

    fig.colorbar(mappable, ax=ax)

    return fig, ax

def objective(scaled_elastic_modulus, scale):
    """
    Returns the objective function and the scaled gradient

    Parameters
    ----------
    scaled_elastic_modulus : array_like
        The elastic modulus, E, divided by a scale factor. Note that:
        scaled_elastic_modulus = elastic_modulus/scale
    scale : float
        The scale factor by which to scale the elastic modulus
    """
    solid_props = {'elastic_modulus': scaled_elastic_modulus*scale}
    fluid_props = constants.DEFAULT_FLUID_PROPERTIES

    # Create an empty file to write to
    with h5py.File('temp.h5', 'w'):
        pass

    dt = 1e-4
    forward([0, 0.1], dt, solid_props, fluid_props, h5file='temp.h5', show_figure=False)
    plt.close()

    totalfluidwork = None
    totalinputwork = None
    with h5py.File('temp.h5', mode='r') as f:
        totalfluidwork = functionals.totalfluidwork(0, f)
        totalinputwork = functionals.totalinputwork(0, f)
    fkwargs = {'cache_totalfluidwork': totalfluidwork, 'cache_totalinputwork': totalinputwork}
    gradient = adjoint(solid_props, 'temp.h5', functional_kwargs=fkwargs)
    _objective = totalfluidwork/totalinputwork

    return 1 - _objective, -1 * gradient * scale


if __name__ == '__main__':
    save_dir = 'out/opt-scipy'
    save_path = path.join(save_dir, 'ElasticModuli.h5')
    dfn.set_log_level(30)

    ##
    SCALE = 10 * 10e3 * constants.PASCAL_TO_CGS
    ii = 0
    def decorated_objective(scaled_elastic_modulus):
        """
        Modifies the gradient inplace and returns the objective function value + other useful stuff

        Parameters
        ----------
        scaled_elastic_modulus :

        Returns
        -------
        float
            The objective function value
        np.ndarray
            The gradient of the objective function
        """
        global ii
        print("Running objective function")

        # Save elastic modulus to disk
        with h5py.File(save_path, mode='a') as f:
            f['elastic_modulus'].resize(ii+1, axis=0)
            f['elastic_modulus'][ii] = scaled_elastic_modulus*SCALE

        fig, ax = plot_elastic_modulus(scaled_elastic_modulus*SCALE)

        fig.savefig(path.join(save_dir, f"iteration{ii}.png"))
        plt.close(fig)

        tstart = perf_counter()
        _objective, _grad = objective(scaled_elastic_modulus, SCALE)
        tend = perf_counter()
        print(f"Iteration {ii}: {tend-tstart:.2f} s")

        print(f"Vocal inefficiency: {_objective*100:.2f}%")
        print(f"Gradient norm: {np.linalg.norm(_grad)*SCALE}")

        ii += 1
        return _objective, _grad


    scaled_emod0 = frm.emod.vector()[:].copy()/SCALE
    NVERTS = scaled_emod0.size

    with h5py.File(save_path, mode='w') as f:
        f.create_dataset('elastic_modulus', (1, NVERTS), maxshape=(None, NVERTS))

    bounds = optimize.Bounds(np.zeros(NVERTS), np.full(NVERTS, np.inf))

    general_opts = {'method': 'L-BFGS-B', 'jac': True, 'bounds': bounds}
    solver_opts = {'disp': None, 'ftol': 1e-6}
    optimum_emod = optimize.minimize(decorated_objective, scaled_emod0, **general_opts,
                                     options=solver_opts)

    plt.show()
