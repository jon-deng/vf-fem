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

from . import constants
from .parameters.properties import FluidProperties

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
    """
    Represents a state file.

    # TODO: Add mesh information and vertex/cell/face region information etc...

    State information is stored in the hdf5 file under a containing group:
    /.../group

    States are stored under labels:
    ./u : (N_TIME, N_VECTOR_DOF)
    ./v : (N_TIME, N_VECTOR_DOF)
    ./a : (N_TIME, N_VECTOR_DOF)

    Fluid properties are stored under labels:
    ./fluid_properties/label : (N_TIME,)

    Solid properties are stored under labels:
    ./solid_properties/label : (N_SCALAR_DOF,) or ()

    Parameters
    ----------
    name : str
        Path to the hdf5 file.
    group : str
        Group path where states are stored in the hdf5 file.
    """

    def __init__(self, model, name, group='/', mode='r', NCHUNK=100,
                 **kwargs):
        self.model = model
        self.file = h5py.File(name, mode=mode, **kwargs)

        # Create the root group if applicable
        if (mode == 'w' or mode == 'a') and group not in self.file:
            self.file.create_group(group)

        self.root_group_name = group

        self.NCHUNK = NCHUNK

        # create caches to store read values
        self.cache = {}
        for name in ['uva', 'qp', 'solid', 'fluid']:
            self.cache[name] = Cache(5)

        # TODO: This is probably buggy
        self.dset_chunk_cache = {}
        if mode == 'r':
            for name in ['u', 'v', 'a', 'q', 'p']:
                self.dset_chunk_cache[name] = DatasetChunkCache(self.root_group[name])

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
        return self.get_num_states()

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
    def init_layout(self, uva0=None, qp0=None, fluid_props=None, solid_props=None):
        r"""
        Initializes the layout of the state file.

        This creates groups and/or datasets in the hdf5 file:
        \mesh\solid\coordinates
        \mesh\solid\connectivity
        \dofmap\scalar
        \dofmap\vector
        \u : dataset, (None, N_VECTOR_DOF)
        \v : dataset, (None, N_VECTOR_DOF)
        \a : dataset, (None, N_VECTOR_DOF)
        \q : dataset, (None, N_VECTOR_DOF)
        \p : dataset, (None, N_VECTOR_DOF)
        \solid_properties
            \solid_property_label : dataset, ()
        \fluid_properties
            \fluid_property_label : dataset, (None,)
        \meas_indices : dataset, (None,)
        \time : dataset, (None,)

        Parameters
        ----------
        u0, v0, a0 : dfn.Function
            The initial velocity, displacement and acceleration respectively.
        solid_props : dict
            A dictionary of solid properties
        fluid_props : dict
            A dictionary of fluid properties
        """
        self.root_group.create_dataset('time', (0,), maxshape=(None,), chunks=(self.NCHUNK,),
                                       dtype=np.float64)
        self.root_group.create_dataset('meas_indices', (0,), maxshape=(None,),
                                       chunks=(self.NCHUNK,), dtype=np.intp)

        self.init_mesh()
        self.init_dofmap()
        self.init_state(uva0=uva0)
        self.init_fluid_state(qp0=qp0)

        # Fluid properties
        if 'fluid_properties' not in self.root_group:
            self.init_fluid_props(fluid_props=fluid_props)

        # Solid properties (assumed to not be time-varying)
        if 'solid_properties' not in self.root_group:
            self.init_solid_props(solid_props=solid_props)

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

    def init_state(self, uva0=None):
        """
        Initializes the states layout of the file.

        Parameters
        ----------
        model : femvf.ForwardModel
        """
        # Kinematic states
        NDOF = self.model.solid.vector_fspace.dim()
        for dataset_name in ['u', 'v', 'a']:
            self.root_group.create_dataset(dataset_name, (0, NDOF), maxshape=(None, NDOF),
                                           chunks=(self.NCHUNK, NDOF), dtype=np.float64)

        if uva0 is not None:
            self.append_state(uva0)

    def init_fluid_state(self, qp0=None):
        """
        Initializes the states layout of the file.

        Parameters
        ----------
        model : femvf.ForwardModel
        """
        # For Bernoulli, there is only 1 flow rate/flow velocity vector
        NQ = 1
        self.root_group.create_dataset('q', (0, NQ), maxshape=(None, 1),
                                       chunks=(self.NCHUNK, NQ), dtype=np.float64)

        # For Bernoulli, you only have to store pressure at each of the vertices
        NDOF = self.model.surface_vertices.size
        self.root_group.create_dataset('p', (0, NDOF), maxshape=(None, NDOF),
                                       chunks=(self.NCHUNK, NDOF), dtype=np.float64)

        if qp0 is not None:
            self.append_fluid_state(qp0)

    def init_fluid_props(self, fluid_props=None):
        """
        Initializes the fluid properties layout of the file.

        Parameters
        ----------
        model : femvf.model.ForwardModel
            Not really needed for this one but left the arg here since it's in solid properties init
        """
        group_fluid = self.root_group.create_group('fluid_properties')
        for key, prop_desc in self.model.fluid.PROPERTY_TYPES.items():
            shape = solid_property_shape(prop_desc, self.model.solid)
            group_fluid.create_dataset(key, shape=(0,)+shape, chunks=(self.NCHUNK,)+shape,
                                       maxshape=(None,)+shape, dtype=np.float64)

        if fluid_props is not None:
            self.append_fluid_props(fluid_props)

    def init_solid_props(self, solid_props=None):
        """
        Initializes the solid properties layout of the file.

        Parameters
        ----------
        """
        solid_group = self.root_group.create_group('solid_properties')
        for key, prop_desc in self.model.solid.PROPERTY_TYPES.items():
            shape = solid_property_shape(prop_desc, self.model.solid)
            solid_group.create_dataset(key, shape, dtype=np.float64)

        if solid_props is not None:
            self.append_solid_props(solid_props)

    ## Functions for writing by appending
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

    def append_state(self, uva):
        """
        Append state to the file.

        Parameters
        ----------
        uva : tuple of dfn.Function
            (u, v, a) states to append
        """
        for dset_name, value in zip(['u', 'v', 'a'], uva):
            dset = self.root_group[dset_name]
            dset.resize(dset.shape[0]+1, axis=0)
            dset[-1] = value[:]

    def append_fluid_state(self, qp):
        """
        Append state to the file.

        Parameters
        ----------
        qp : tuple of dfn.Function or array_like and float
            (q, p) states to append
        """
        for dset_name, value in zip(['q', 'p'], qp):
            dset = self.root_group[dset_name]
            dset.resize(dset.shape[0]+1, axis=0)
            dset[-1, :] = value

    def append_fluid_props(self, fluid_props):
        """
        Append fluid properties to the file.

        Parameters
        ----------
        fluid_props : dict
            Dictionary of fluid properties to append
        """
        fluid_group = self.root_group['fluid_properties']

        for label in self.model.fluid.PROPERTY_TYPES:
            dset = fluid_group[label]
            dset.resize(dset.shape[0]+1, axis=0)
            dset[-1] = fluid_props[label]

    def append_solid_props(self, solid_props):
        """
        Append solid properties to the file.

        It doesn't actually append, it just rewrites the previous values if they exist.

        Parameters
        ----------
        solid_props : dict
            Dictionary of solid properties to append
        """
        solid_group = self.root_group['solid_properties']

        for label, shape in self.model.solid.PROPERTY_TYPES.items():
            dset = solid_group[label]

            if shape[0] == 'field':
                dset[:] = solid_props[label]
            else:
                dset[()] = solid_props[label]

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

    ## Functions for reading and writing to specific indices
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

    def get_num_states(self):
        """
        Returns the number of states in the solution
        """
        return self.root_group['u'].shape[0]

    def set_state(self, n, uva):
        """
        Set form coefficient vectors for states `uva=(u, v, a)` at index n.

        Parameters
        ----------
        n : int
            Index to set the functions for.
        uva : tuple of 3 array_like
            A set of vectors to assign.
        """
        for label, value in zip(('u', 'v', 'a'), uva):
            self.root_group[label][n] = value

    def set_fluid_state(self, n, qp):
        """
        Set form coefficient vectors for states `qp=(q, p)` at index n.

        Parameters
        ----------
        n : int
            Index to set the functions for.
        q : float
            the flow rates
        p : array_like
            pressures
        """
        self.root_group['q'][n, 0] = qp[0]
        self.root_group['p'][n, :] = qp[1]

    def get_iter_params(self, n):
        """
        Return parameter defining iteration `n`

        Parameters
        ----------
        n : int
            Index of the iteration.
        """

        uva0 = self.get_state(n-1)
        dt = self.get_time(n) - self.get_time(n-1)
        solid_props = self.get_solid_props(n-1)
        fluid_props = self.get_fluid_props(n-1)
        u1 = self.get_u(n)

        return {'uva0': uva0, 'dt': dt, 'u1': u1,
                'solid_props': solid_props, 'fluid_props': fluid_props}

    # these read functions are cached for performance reasons
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
        labels = ('u', 'v', 'a')
        uva = self.model.solid.get_state()
        for ii, label in enumerate(labels):
            uva[ii][:] = self.dset_chunk_cache[label].get(n)

        return uva

    def get_fluid_state(self, n):
        """
        Return fluid states q, p at index n.

        Parameters
        ----------
        n : int
            Index to set the functions for.
        out : tuple of 3 dfn.Function
            A set of functions to set vector values for.
        """
        qp = self.model.fluid.get_state()

        qp[0][:] = self.dset_chunk_cache['q'].get(n)[0]
        qp[1][:] = self.dset_chunk_cache['p'].get(n)

        return qp

    @get_from_cache('fluid')
    def get_fluid_props(self, n):
        """
        Return the fluid properties dictionary at index n.
        """
        # There can only be stationary properties right now
        assert n == 0

        fluid_props = self.model.fluid.get_properties_vec()
        fluid_group = self.root_group['fluid_properties']

        # Correct for constant fluid properties in time
        # TODO: Refactor how constant fluid/solid properties are defined.
        m = n
        if self.root_group['fluid_properties/p_sub'].size == 1:
            m = 0

        for label in fluid_props.labels:
            if fluid_props[label].shape == ():
                fluid_props[label][()] = fluid_group[label][m]
            else:
                fluid_props[label][:] = fluid_group[label][m]

        return fluid_props

    @get_from_cache('solid')
    def get_solid_props(self, n):
        """
        Returns the solid properties
        """
        # There can only be stationary properties right now
        assert n == 0

        solid_props = self.model.solid.get_properties()
        solid_group = self.root_group['solid_properties']
        for label, shape in self.model.solid.PROPERTY_TYPES.items():
            data = solid_group[label]

            if solid_props[label].shape == ():
                # have to index differently for scalar datasets
                solid_props[label][()] = data[()]
            else:
                solid_props[label][:] = data[:]

        return solid_props

def solid_property_shape(property_desc, solid):
    const_or_field = property_desc[0]
    data_shape = property_desc[1]

    shape = None
    if const_or_field == 'field':
        shape = (solid.mesh.num_vertices(),) + data_shape
    else:
        shape = data_shape

    return shape

class Cache:
    """
    Represents a cache of arbitrary items
    """
    def __init__(self, cache_size):
        self.N = cache_size
        self.data = OrderedDict()

    def __len__(self):
        return len(self.data)

    def put(self, key, val):
        """
        Add `key`: `val` to the cache

        Parameters
        ----------
        key : hashable
        val : object
        """
        if key in self.data:
            self.data.move_to_end(key, last=False)
        else:
            self.data[key] = val

            # remove the oldest item in the cache if the cache exceeds the cache size
            if len(self.data) > self.N:
                self.data.popitem(last=False)

    def get(self, key):
        if key in self.data:
            self.data.move_to_end(key, last=False)
            return self.data[key]
        else:
            return None

class DatasetChunkCache:
    """
    Cache of chunked datasets

    The dataset must contain full chunks along the last dimensions

    Reading values from this class will load entire chunks at a time. Reading slices that are
    already loaded in a chunk will use the cached chunk
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
