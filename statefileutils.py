"""
Module to work with state values from a forward pass stored in an hdf5 file.
"""

import h5py

from os.path import join

def get_time(path, group='/'):
    """
    Returns the time vector.
    """
    time = None
    with h5py.File(path, mode='r') as f:
        time = f[join(group, 'time')][:]

    return time

def get_num_states(path, group='/'):
    """
    Returns the number of stored states
    """
    num = None
    with h5py.File(path, mode='r') as f:
        num = f[join(group, 'u')].shape[0]

    return num

def set_states(n, path, group='/', u0=None, v0=None, a0=None, u1=None):
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
    with h5py.File(path, mode='r') as f:
        if u0 is not None:
            u0.vector()[:] = f[join(group, 'u')][n-1]
        if v0 is not None:
            v0.vector()[:] = f[join(group, 'v')][n-1]
        if a0 is not None:
            a0.vector()[:] = f[join(group, 'a')][n-1]
        if u1 is not None:
            u1.vector()[:] = f[join(group, 'u')][n]