"""
Module to work with state values from a forward pass stored in an hdf5 file.
"""

# TODO: Note that the solid and fluid parameters are assumed to be stationary so some codes are
# hard-coded around this (which is not good). The current code doesn't really handle this explicitly
# so you may have to fix bugs that are associated with this if you want to use time-varying
# fluid/solid parameters

from os.path import join
from collections import OrderedDict

import h5py
import dolfin as dfn
import numpy as np

from .models.solid import Solid
from . import constants

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
    Represents a state file.

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
    fname : str
        Path to the hdf5 file.
    group : str
        Group path where states are stored in the hdf5 file.
    """

    def __init__(self, model, fname, group='/', mode='r', NCHUNK=100,
                 **kwargs):
        self.model = model
        self.file = h5py.File(fname, mode=mode, **kwargs)

        # Create the root group if applicable
        if (mode == 'w' or mode == 'a') and group not in self.file:
            self.file.create_group(group)

        self.root_group_name = group

        self.NCHUNK = NCHUNK

        # TODO: This is probably buggy
        self.dset_chunk_cache = {}
        if mode == 'r':
            ## Create caches for reading states and controls, since these vary in time
            for name in model.state0.keys:
                self.dset_chunk_cache[f'state/{name}'] = DatasetChunkCache(self.root_group[f'state/{name}'])

            for name in model.control.keys:
                # breakpoint()
                self.dset_chunk_cache[f'control/{name}'] = DatasetChunkCache(self.root_group[f'control/{name}'])

    ## Implement an h5 group interface to the underlying root group
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.file.close()

    def keys(self):
        return self.root_group.keys()

    def __getitem__(self, name):
        return self.root_group[name]

    def __setitem__(self, name, value):
        self.root_group[name] = value

    def __len__(self):
        return self.size

    def close(self):
        """
        Close the file.
        """
        self.file.close()

    @property
    def size(self):
        """
        Return the number of states in the file.
        """
        return self.root_group['state/u'].shape[0]

    @property
    def variable_controls(self):
        num_controls = 1
        control_group = self.root_group['control']
        for key in self.model.control.keys:
            num_controls = max(control_group[key].shape[0], num_controls)

        if num_controls > 1:
            return True
        else: 
            return False

    ## Statefile properties related to the root group where things are stored
    @property
    def root_group_name(self):
        """
        Return the group path the states are stored under
        """
        return self._root_group_name

    @root_group_name.setter
    def root_group_name(self, name):
        """
        Set the root group path and creates the group if it doesn't exist
        """
        self._root_group_name = name

    @property
    def root_group(self):
        """
        Return the `h5py.Group` object where states are stored
        """
        return self.file[self.root_group_name]

    ## Functions for initializing layout when writing
    def init_layout(self):
        r"""
        Initializes the layout of the state file.
        """
        self.root_group.create_dataset('time', (0,), maxshape=(None,), chunks=(self.NCHUNK,),
                                       dtype=np.float64)
        self.root_group.create_dataset('meas_indices', (0,), maxshape=(None,),
                                       chunks=(self.NCHUNK,), dtype=np.intp)

        self.init_mesh()
        self.init_dofmap()

        self.init_state()
        self.init_control()
        self.init_properties()

    def init_dofmap(self):
        """
        Writes the dofmaps for scalar and vector data
        """
        solid = self.model.solid
        scalar_dofmap_processor = solid.scalar_fspace.dofmap()
        vector_dofmap_processor = solid.vector_fspace.dofmap()

        scalar_dofmap = np.array([scalar_dofmap_processor.cell_dofs(cell.index())
                                  for cell in dfn.cells(solid.mesh)])
        vector_dofmap = np.array([vector_dofmap_processor.cell_dofs(cell.index())
                                  for cell in dfn.cells(solid.mesh)])
        self.root_group.create_dataset('dofmap/vector', data=vector_dofmap,
                                       dtype=np.intp)
        self.root_group.create_dataset('dofmap/scalar', data=scalar_dofmap,
                                       dtype=np.intp)

    def init_mesh(self):
        """
        Writes the mesh information to the h5 file
        """
        solid = self.model.solid
        self.root_group.create_dataset('mesh/solid/coordinates', data=solid.mesh.coordinates(),
                                       dtype=np.float64)
        self.root_group.create_dataset('mesh/solid/connectivity', data=solid.mesh.cells(),
                                       dtype=np.intp)

        # TODO: Write facet/cell labels, mapping string identifiers to the integer mesh functions
        self.root_group.create_dataset('mesh/solid/facet_func', data=np.inf,
                                       dtype=np.intp)
        self.root_group.create_dataset('mesh/solid/cell_func', data=np.inf,
                                       dtype=np.intp)

    def init_state(self):
        state_group = self.root_group.create_group('state')
        for name, vec in zip(self.model.state0.keys, self.model.state0.vecs):
            NDOF = len(vec)
            state_group.create_dataset(name, (0, NDOF), maxshape=(None, NDOF),
                                       chunks=(self.NCHUNK, NDOF), dtype=np.float64)

    def init_control(self):
        control_group = self.root_group.create_group('control')

        for name, vec in zip(self.model.control.keys, self.model.control.vecs):
            NDOF = len(vec)
            control_group.create_dataset(name, (0, NDOF), maxshape=(None, NDOF), 
                                         chunks=(self.NCHUNK, NDOF), dtype=np.float64)
        pass

    def init_properties(self):
        properties_group = self.root_group.create_group('properties')

        for name, value in zip(self.model.properties.keys, self.model.properties.vecs):
            size = None
            try:
                size = len(value)
            except TypeError:
                size = value.size
            properties_group.create_dataset(name, (size,), dtype=np.float64)

    ## Functions for writing by appending
    def append_state(self, state):
        """
        Append state to the file.

        Parameters
        ----------
        """
        state_group = self.root_group['state']
        for name, value in zip(state.keys, state.vecs):
            dset = state_group[name]
            dset.resize(dset.shape[0]+1, axis=0)

            if isinstance(value, dfn.GenericVector):
                # This converts dolfin vector types to numpy arrays which 
                # allows the h5py to write them much faster
                value = value[:]
            dset[-1, :] = value

    def append_control(self, control):
        control_group = self.root_group['control']
        for name, value in zip(control.keys, control.vecs):
            dset = control_group[name]
            dset.resize(dset.shape[0]+1, axis=0)
            dset[-1] = value
        pass

    def append_properties(self, properties):
        """
        Append properties vector to the file.

        Parameters
        ----------
        """
        properties_group = self.root_group['properties']

        for name, value in zip(properties.keys, properties.vecs):
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
        dset = self.root_group['time']
        dset.resize(dset.shape[0]+1, axis=0)
        dset[-1] = time

    def append_meas_index(self, index):
        """
        Append measured indices to the file.

        Parameters
        ----------
        index : int
        """
        dset = self.root_group['meas_indices']
        dset.resize(dset.shape[0]+1, axis=0)
        dset[-1] = index

    ## Functions for reading specific indices
    def get_time(self, n):
        """
        Returns the time at state n.
        """
        return self.root_group['time'][n]

    def get_times(self):
        """
        Returns the time vector.
        """
        return self.root_group['time'][:]

    def get_meas_indices(self):
        """
        Returns the measured indices.
        """
        return self.root_group['meas_indices'][:]

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
        state = self.model.get_state_vec()
        for vec, key in zip(state.vecs, state.keys):
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
        control = self.model.get_control_vec()
        for vec, key in zip(control.vecs, control.keys):
            value = self.dset_chunk_cache[f'control/{key}'].get(n)
            try:
                vec[:] = value
            except IndexError:
                vec[()] = value

        return control

    def get_properties(self):
        properties = self.model.get_properties_vec()

        for name, vec in zip(properties.keys, properties.vecs):
            dset = self.root_group[f'properties/{name}']
            try: 
                vec[:] = dset[:]
            except IndexError as e:
                vec[()] = dset[()]
        return properties

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
        for label, value in zip(state.keys, state.vecs):
            self.root_group[label][n] = value

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
