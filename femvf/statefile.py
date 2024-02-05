"""
Module to work with state values from a forward pass stored in an hdf5 file.
"""

# TODO: Note that the solid and fluid parameters are assumed to be stationary so some codes are
# hard-coded around this (which is not good). The current code doesn't really handle this explicitly
# so you may have to fix bugs that are associated with this if you want to use time-varying
# fluid/solid parameters

from typing import Union, Tuple, Optional, Mapping, Any
from collections import OrderedDict

import h5py
import dolfin as dfn
import numpy as np
from blockarray import blockvec as bv

from .models.transient.base import BaseTransientModel

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

    # NOTE: Should you refactor the init to use more basic variables instead of
    # a `BaseTransientModel` object?
    # - when a `StateFile` is being created (mode='w' or 'a') you need a
    # mesh + prototoype state, control, prop vectors (for the shape information)
    # - when a `StateFile` is being read (mode='r' or 'a') you don't need
    # any of those, although it is convenient to have them
    def __init__(
            self,
            model: BaseTransientModel,
            fname: Union[str, h5py.Group],
            mode: str='r',
            NCHUNK: int=100,
            **kwargs
        ):
        self.model: BaseTransientModel = model
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
        self.file.require_dataset(
            'time', (self.size,), maxshape=(None,), chunks=(self.NCHUNK,),
            dtype=np.float64, exact=False
        )

        if 'meas_indices' not in self.file:
            self.file.create_dataset(
                'meas_indices', (0,),
                maxshape=(None,), chunks=(self.NCHUNK,), dtype=np.intp
            )
        self.init_mesh()

        self.init_state()
        self.init_control()
        self.init_prop()

        self.init_solver_info()

    def init_mesh(self):
        """
        Writes the mesh information to the h5 file
        """
        solid = self.model.solid
        coords = solid.residual.mesh().coordinates()
        cells = solid.residual.mesh().cells()
        self.file.require_dataset(
            'mesh/solid/coordinates', coords.shape,
            data=coords, dtype=np.float64
        )
        self.file.require_dataset(
            'mesh/solid/connectivity', cells.shape,
            data=cells, dtype=np.intp
        )
        self.file.require_dataset(
            'mesh/solid/dim', (),
            data=solid.residual.mesh().topology().dim(), dtype=np.intp
        )

        dofmaps = [solid.residual.form['coeff.state.u0'].function_space().dofmap()]
        for dofmap in dofmaps:
            dofmap_array = np.array([
                dofmap.cell_dofs(idx_cell) for idx_cell in range(cells.shape[0])
            ])
            self.file.require_dataset(
                'dofmap/CG1', dofmap_array.shape, data=dofmap_array, dtype=np.intp
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
        bvec = self.model.state0
        for name, ndof in zip(bvec.labels[0], bvec.bshape[0]):
            state_group.require_dataset(
                name, (self.size, ndof),
                maxshape=(None, ndof), chunks=(self.NCHUNK, ndof),
                dtype=np.float64
            )

    def init_control(self):
        control_group = self.file.require_group('control')

        bvec = self.model.control
        for name, ndof in zip(bvec.labels[0], bvec.bshape[0]):
            control_group.require_dataset(
                name, (self.size, ndof),
                maxshape=(None, ndof), chunks=(self.NCHUNK, ndof),
                dtype=np.float64
            )

    def init_prop(self):
        properties_group = self.file.require_group('properties')

        bvec = self.model.prop
        for name, ndof in zip(bvec.labels[0], bvec.bshape[0]):
            properties_group.require_dataset(name, (ndof,), dtype=np.float64)

    def init_solver_info(self):
        solver_info_group = self.file.require_group('solver_info')
        for key in ['num_iter', 'rel_err', 'abs_err']:
            solver_info_group.require_dataset(
                key, (self.size,),
                dtype=np.float64, maxshape=(None,), chunks=(self.NCHUNK,)
            )

    ## Functions for writing by appending
    def append_state(self, state: bv.BlockVector):
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

    def append_control(self, control: bv.BlockVector):
        control_group = self.file['control']
        for name, value in control.items():
            dset = control_group[name]
            dset.resize(dset.shape[0]+1, axis=0)
            dset[-1] = value

    def append_prop(self, properties: bv.BlockVector):
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

    def append_time(self, time: float):
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

    def append_meas_index(self, index: int):
        """
        Append measured indices to the file.

        Parameters
        ----------
        index : int
        """
        dset = self.file['meas_indices']
        dset.resize(dset.shape[0]+1, axis=0)
        dset[-1] = index

    def append_solver_info(self, solver_info: Mapping[str, Any]):
        solver_info_group = self.file['solver_info']
        for key, dset in solver_info_group.items():
            dset.resize(dset.shape[0]+1, axis=0)
            if key in solver_info:
                dset[-1] = solver_info[key]
            else:
                dset[-1] = np.nan

    ## Functions for reading specific indices
    def get_time(self, n: int) -> float:
        """
        Returns the time at state n.
        """
        return self.file['time'][n]

    def get_times(self) -> np.ndarray:
        """
        Returns the time vector.
        """
        return self.file['time'][:]

    def get_meas_indices(self) -> np.ndarray:
        """
        Returns the measured indices.
        """
        return self.file['meas_indices'][:]

    def get_state(self, n: int) -> bv.BlockVector[np.ndarray]:
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

    def get_control(self, n: int) -> bv.BlockVector[np.ndarray]:
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

    def get_prop(self) -> bv.BlockVector[np.ndarray]:
        properties = self.model.prop.copy()

        for name, vec in zip(properties.keys(), properties.blocks):
            dset = self.file[f'properties/{name}']
            try:
                vec[:] = dset[:]
            except IndexError as e:
                vec[()] = dset[()]
        return properties

    def get_solver_info(self, n) -> Mapping[str, np.ndarray]:
        solver_info_group = self.file['solver_info']
        solver_info = {key: solver_info_group[key][n] for key in solver_info_group.keys()}
        return solver_info

    ## Functions for writing/modifying specific indices
    def set_state(self, n: int, state: bv.BlockVector):
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

# TODO: Test whether this cache improves performance or not; you made this when
# you didn't know how exactly to test `h5py` performace
# also `h5py` is supposed to cache datasets by chunks already anyway
class DatasetChunkCache:
    """
    Cache for chunked datasets to improve read performance

    The dataset must contain full chunks along the last dimensions

    Reading values from this class will load entire chunks at a time. Reading slices that are
    already loaded in a chunk will use the data from the cached chunk instead of reading again.
    """

    def __init__(self, dset: h5py.Dataset, num_chunks: int=1):
        # Don't allow scalar datasets
        assert len(dset.shape) > 0

        # Check that dataset is chunked properly
        assert dset.shape[1:] == dset.chunks[1:]

        self.dset = dset

        self.chunk_size = dset.chunks[0]

        self.num_chunks = num_chunks

        self.cache = OrderedDict()

    @property
    def size(self):
        return self.dset.shape[0]

    def _convert_neg_to_pos_index(self, m: int):
        if m < 0:
            return self.size + m
        else:
            return m

    def _dec_handle_neg_index(func):
        def wrapped(self, m: int):
            m = self._convert_neg_to_pos_index(m)
            return func(self, m)
        return wrapped

    @_dec_handle_neg_index
    def get(self, m: int):
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

    @_dec_handle_neg_index
    def load(self, m: int):
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
