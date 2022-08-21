"""
Module to work with state values from a forward pass stored in an hdf5 file.
"""

# TODO: Note that the solid and fluid parameters are assumed to be stationary so some codes are
# hard-coded around this (which is not good). The current code doesn't really handle this explicitly
# so you may have to fix bugs that are associated with this if you want to use time-varying
# fluid/solid parameters

from typing import Union, Tuple, Optional, Mapping
from collections import OrderedDict

import h5py
import dolfin as dfn
import numpy as np

from .models.transient.base import BaseTransientModel

def get_from_cache(cache_name):
    """
    Return a decorator that gives a functions caching behaviour to the specified cache

    Parameters
    ----------
    cache_key : str
        key to the cache in the `StateFile` objects `cache` attribute that the function should use
        for storing cached values into
    """
    def decorator(func):
        """
        Parameters
        ----------
        func : callable(key) -> value
        """
        def wrapper(self, key):
            cache = self.cache[cache_name]
            if cache.get(key) is None:
                val = func(self, key)
                cache.put(key, val)

            return cache.get(key)

        return wrapper

    return decorator

class StateFile:
    r"""
    An HDF5 file containing the history of states in a transient model simulation

    # TODO: Add mesh information and vertex/cell/face region information etc...

    State information is stored in the hdf5 file under a containing group:
    /.../group

    The remaining information is stored as:
    /mesh/solid/coordinates
    /mesh/solid/connectivity

    /dofmap/scalar
    /dofmap/vector

    /state/state_name : dataset, (None, N_VECTOR_DOF)

    Controls are stored under labels:
    /control/control_name

    /properties/property_name : dataset

    /meas_indices : dataset, (None,)
    /time : dataset, (None,)


    Parameters
    ----------
    fname :
        Path to the HDF5 file or an `h5py.Group` instance
    mode :
        Mode to open the HDF5 file in (applicable if `fname` is a path)
    NCHUNK : int
        Number of chunks along the time dimension used to store data
    kwargs :
        Keyword arguments for `h5py.File` (applicable if `fname` is a path)
    """

    file: h5py.Group

    def __init__(
            self,
            model: BaseTransientModel,
            fname: Union[str, h5py.Group],
            mode: str='r',
            NCHUNK: int=100,
            **kwargs
        ):
        self.model = model
        if isinstance(fname, str):
            self.file = h5py.File(fname, mode=mode, **kwargs)
        elif isinstance(fname, h5py.Group):
            self.file = fname
        else:
            raise TypeError(
                f"`fname` must be `str` or `h5py.Group` not {type(fname)}"
            )
        self.NCHUNK = NCHUNK

        # Create the root group and initilizae the data layout
        # group = self.file.name
        # if (mode == 'w' or mode == 'a') and group not in self.file:
        #     self.file.require_group(group)
        self.init_layout()

        # TODO: This is probably buggy
        self.dset_chunk_cache = {}
        if mode == 'r' or 'a':
            ## Create caches for reading states and controls, since these vary in time
            # h5py is supposed to do this caching but I found that each dataset read call in h5py
            # has a lot of overhead so I made this cache instead
            for name in model.state0.keys():
                self.dset_chunk_cache[f'state/{name}'] = DatasetChunkCache(self.file[f'state/{name}'])

            for name in model.control.keys():
                # breakpoint()
                self.dset_chunk_cache[f'control/{name}'] = DatasetChunkCache(self.file[f'control/{name}'])

    ## Functions mimicking the `h5py.File` interface
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.file.close()

    def keys(self):
        return self.file.keys()

    def __getitem__(self, name):
        return self.file[name]

    def __setitem__(self, name, value):
        self.file[name] = value

    def __len__(self):
        return self.size

    def close(self):
        """
        Close the file.
        """
        self.file.close()

    ## Convenience functions
    @property
    def size(self):
        """
        Return the number of states in the file.

        Note the 'size' of the dataset is based on the number of time indices
        that have been written.
        """
        if 'time' in self.file:
            return self.file['time'].shape[0]
        else:
            return 0

    @property
    def variable_controls(self):
        if self.num_controls > 1:
            return True
        else:
            return False

    @property
    def num_controls(self):
        num = 1
        control_group = self.file['control']
        for key in self.model.control.keys():
            num = max(control_group[key].shape[0], num)

        return num

    ## Functions for initializing layout when writing
    def init_layout(self):
        r"""
        Initializes the layout of the state file.
        """
        self.file.require_dataset('time', (self.size,), maxshape=(None,), chunks=(self.NCHUNK,),
                                        dtype=np.float64, exact=False)

        if 'meas_indices' not in self.file:
            self.file.create_dataset(
                'meas_indices', (0,),
                maxshape=(None,), chunks=(self.NCHUNK,), dtype=np.intp
            )
        self.init_mesh()

        self.init_state()
        self.init_control()
        self.init_properties()

        self.init_solver_info()

    def init_mesh(self):
        """
        Writes the mesh information to the h5 file
        """
        solid = self.model.solid
        coords = solid.mesh.coordinates()
        cells = solid.mesh.cells()
        self.file.require_dataset(
            'mesh/solid/coordinates', coords.shape,
            data=coords, dtype=np.float64
        )
        self.file.require_dataset(
            'mesh/solid/connectivity', cells.shape,
            data=cells, dtype=np.intp
        )

        # TODO: Write mesh function information + label: int region indentifiers
        # self.file.require_dataset(
        #     'mesh/solid/facet_func', data=np.inf, dtype=np.intp
        # )
        # self.file.require_dataset(
        #     'mesh/solid/cell_func', data=np.inf, dtype=np.intp
        # )

    def init_state(self):
        state_group = self.file.require_group('state')
        for name, vec in self.model.state0.items():
            NDOF = len(vec)
            state_group.require_dataset(
                name, (self.size, NDOF), maxshape=(None, NDOF),
                chunks=(self.NCHUNK, NDOF), dtype=np.float64
            )

    def init_control(self):
        control_group = self.file.require_group('control')

        for name, vec in self.model.control.items():
            NDOF = len(vec)
            control_group.require_dataset(name, (self.size, NDOF), maxshape=(None, NDOF),
                                         chunks=(self.NCHUNK, NDOF), dtype=np.float64)

    def init_properties(self):
        properties_group = self.file.require_group('properties')

        for name, value in self.model.props.items():
            size = None
            try:
                size = len(value)
            except TypeError:
                size = value.size
            properties_group.require_dataset(name, (size,), dtype=np.float64)

    def init_solver_info(self):
        solver_info_group = self.file.require_group('solver_info')
        for key in ['num_iter', 'rel_err', 'abs_err']:
            solver_info_group.require_dataset(
                key, (self.size,), dtype=np.float64, maxshape=(None,),
                chunks=(self.NCHUNK,)
            )

    ## Functions for writing by appending
    def append_state(self, state):
        """
        Append state to the file.

        Parameters
        ----------
        """
        state_group = self.file['state']
        for name, value in state.items():
            dset = state_group[name]
            dset.resize(dset.shape[0]+1, axis=0)
            dset[-1, :] = value

    def append_control(self, control):
        control_group = self.file['control']
        for name, value in control.items():
            dset = control_group[name]
            dset.resize(dset.shape[0]+1, axis=0)
            dset[-1] = value

    def append_properties(self, properties):
        """
        Append properties vector to the file.

        Parameters
        ----------
        """
        properties_group = self.file['properties']

        for name, value in properties.items():
            dset = properties_group[name]
            # dset.resize(dset.shape[0]+1, axis=0)
            dset[:] = value

    def append_time(self, time):
        """
        Append times to the file.

        Parameters
        ----------
        time : float
            Time to append
        """
        dset = self.file['time']
        dset.resize(dset.shape[0]+1, axis=0)
        dset[-1] = time

    def append_meas_index(self, index):
        """
        Append measured indices to the file.

        Parameters
        ----------
        index : int
        """
        dset = self.file['meas_indices']
        dset.resize(dset.shape[0]+1, axis=0)
        dset[-1] = index

    def append_solver_info(self, solver_info):
        solver_info_group = self.file['solver_info']
        for key, dset in solver_info_group.items():
            dset.resize(dset.shape[0]+1, axis=0)
            if key in solver_info:
                dset[-1] = solver_info[key]
            else:
                dset[-1] = np.nan

    ## Functions for reading specific indices
    def get_time(self, n):
        """
        Returns the time at state n.
        """
        return self.file['time'][n]

    def get_times(self):
        """
        Returns the time vector.
        """
        return self.file['time'][:]

    def get_meas_indices(self):
        """
        Returns the measured indices.
        """
        return self.file['meas_indices'][:]

    def get_state(self, n):
        """
        Return form coefficient vectors for states (u, v, a) at index n.

        Parameters
        ----------
        n : int
            Index to set the functions for.
        out : tuple of 3 dfn.Function
            A set of functions to set vector values for.
        """
        state = self.model.state0.copy()
        for key, vec in state.items():
            value = self.dset_chunk_cache[f'state/{key}'].get(n)
            try:
                vec[:] = value
            except IndexError:
                vec[()] = value

        return state

    def get_control(self, n):
        """
        Return form coefficient vectors for states (u, v, a) at index n.

        Parameters
        ----------
        n : int
            Index to set the functions for.
        out : tuple of 3 dfn.Function
            A set of functions to set vector values for.
        """
        control = self.model.control.copy()
        num_controls = self.file[f'control/{control.keys()[0]}'].size
        if n > num_controls-1:
            n = num_controls-1

        for key, vec in control.items():
            value = self.dset_chunk_cache[f'control/{key}'].get(n)
            try:
                vec[:] = value
            except IndexError:
                vec[()] = value

        return control

    def get_props(self):
        properties = self.model.props.copy()

        for name, vec in zip(properties.keys(), properties.blocks):
            dset = self.file[f'properties/{name}']
            try:
                vec[:] = dset[:]
            except IndexError as e:
                vec[()] = dset[()]
        return properties

    def get_solver_info(self, n):
        solver_info_group = self.file['solver_info']
        solver_info = {key: solver_info_group[key][n] for key in solver_info_group.keys()}
        return solver_info

    ## Functions for writing/modifying specific indices
    def set_state(self, n, state):
        """
        Set form coefficient vectors for states `uva=(u, v, a)` at index n.

        Parameters
        ----------
        n : int
            Index to set the functions for.
        uva : tuple of 3 array_like
            A set of vectors to assign.
        """
        for label, value in zip(state.keys(), state.vecs):
            self.file[label][n] = value

class DatasetChunkCache:
    """
    Cache of chunked datasets

    The dataset must contain full chunks along the last dimensions

    Reading values from this class will load entire chunks at a time. Reading slices that are
    already loaded in a chunk will use the data from the cached chunk instead of reading again.
    """

    def __init__(self, dset, num_chunks=1):
        # Don't allow scalar datasets
        assert len(dset.shape) > 0

        # Check that dataset is chunked properly
        assert dset.shape[1:] == dset.chunks[1:]

        self.dset = dset

        self.chunk_size = dset.chunks[0]
        self.M = dset.shape[0]

        self.num_chunks = num_chunks

        self.cache = OrderedDict()

    def get(self, m):
        """
        Parameters
        ----------
        m : int
            Index along the chunk dimension
        """
        m_chunk = m//self.chunk_size

        if m_chunk in self.cache:
            self.cache.move_to_end(m_chunk, last=False)
        else:
            self.load(m)

        m_local = m - m_chunk*self.chunk_size
        return self.cache[m_chunk][m_local, ...].copy()

    def load(self, m):
        """
        Loads the chunk containing index m

        Parameters
        ----------
        m : int
            Index along the chunk dimension
        """
        m_chunk = m//self.chunk_size

        if len(self.cache) == self.num_chunks:
            self.cache.popitem(last=True)

        m_start = m_chunk*self.chunk_size
        m_end = min((m_chunk+1)*self.chunk_size, self.dset.shape[0])
        self.cache[m_chunk] = self.dset[m_start:m_end, ...]

        assert len(self.cache) <= self.num_chunks
