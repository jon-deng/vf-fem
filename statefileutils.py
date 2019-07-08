"""
Module to work with state values from a forward pass stored in an hdf5 file.

The hdf5 file is organized as:

Information concerning a run is stored under a containing group:
/.../container_group

States are stored under labels:
./u : (N_STATES, N_DOFS)
./v : (N_STATES, N_DOFS)
./a : (N_STATES, N_DOFS)

Fluid properties are stored under labels:
./fluid_properties/p_sub : (N_STATES-1,)
./fluid_properties/p_sup : (N_STATES-1,)
./fluid_properties/rho : (N_STATES-1,)
./fluid_properties/y_midline : (N_STATES-1,)

Solid properties are stored under labels:
./solid_properties/elastic_modulus : (N_VERTICES,)
"""

import h5py

from os.path import join

def get_time(h5file, group='/'):
    """
    Returns the time vector.
    """
    return h5file[join(group, 'time')][:]

def get_num_states(h5file, group='/'):
    """
    Returns the number of states in the solution
    """
    return h5file[join(group, 'u')].shape[0]

def get_state(n, h5file, group='/'):
    """
    Returns form coefficient vectors for states (u, v, a) at index n.

    Parameters
    ----------
    n : int
        Index to set the functions for.
    path : string
        The path of the hdf5 file containing states.
    group : string
        The group where states are stored.
    """
    u = h5file[join(group, 'u')][n, ...]
    v = h5file[join(group, 'v')][n, ...]
    a = h5file[join(group, 'a')][n, ...]

    return (u, v, a)

def get_fluid_properties(n, h5file, group='/'):
    """
    Returns the fluid properties dictionary at index n.
    """
    fluid_props = {}
    for label in ('p_sub', 'p_sup', 'rho', 'y_midline'):
        fluid_props[label] = h5file[join(group, 'fluid_properties', label)][n]

    return fluid_props

def set_states(n, h5file, group='/', u0=None, v0=None, a0=None, u1=None):
    """
    Sets form coefficient vectors for states u_n-1, v_n-1, a_n-1, u_n at index n.

    Parameters
    ----------
    n : int
        Index to set the functions for.
    group : string
        The group where states are stored.
    path : string
        The path of the hdf5 file containing states.
    """
    if u0 is not None:
        u0.vector()[:] = h5file[join(group, 'u')][n-1]
    if v0 is not None:
        v0.vector()[:] = h5file[join(group, 'v')][n-1]
    if a0 is not None:
        a0.vector()[:] = h5file[join(group, 'a')][n-1]
    if u1 is not None:
        u1.vector()[:] = h5file[join(group, 'u')][n]
