"""
Module to work with state values from a forward pass stored in an hdf5 file.
"""

from os.path import join

import h5py
import dolfin as dfn
import numpy as np

from . import constants
from .properties import SolidProperties, FluidProperties

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

    def __init__(self, model, name, group='/', mode='r', **kwargs):
        self.model = model
        self.file = h5py.File(name, mode=mode, **kwargs)
        self.root_group_name = group

        # self.data = self.file[self.group]

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.file.close()

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

        if name not in self.file:
            self.file.create_group(name)

    @property
    def root_group(self):
        """
        Return the `h5py.Group` object where states are stored
        """
        return self.file[self.root_group_name]

    def init_layout(self, model, x0=None, fluid_props=None, solid_props=None):
        r"""
        Initializes the layout of the state file.

        This creates groups and/or datasets in the hdf5 file:
        \u : dataset, (None, N_VECTOR_DOF)
        \v : dataset, (None, N_VECTOR_DOF)
        \a : dataset, (None, N_VECTOR_DOF)
        \fluid_properties
            \fluid_property_label : dataset, (None,)
        \solid_properties
            \solid_property_label : dataset, ()
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
        solution_times : array_like
            Times at which the model will be solved
        """
        self.root_group.create_dataset('time', (0,), maxshape=(None,), dtype=np.float)

        self.root_group.create_dataset('meas_indices', (0,), maxshape=(None,), dtype=np.intp)

        self.init_state(model, x0=x0)

        # Fluid properties
        if 'fluid_properties' not in self.root_group:
            self.init_fluid_props(model, fluid_props=fluid_props)

        # Solid properties (assumed to not be time-varying)
        if 'solid_properties' not in self.root_group:
            self.init_solid_props(model, solid_props=solid_props)

    def init_state(self, model, x0=None):
        """
        Initializes the states layout of the file.

        Parameters
        ----------
        model : femvf.ForwardModel
        """
        # Kinematic states
        NDOF = model.vector_function_space.dim()
        for dataset_name in ['u', 'v', 'a']:
            self.root_group.create_dataset(dataset_name, (0, NDOF),
                                      maxshape=(None, NDOF), chunks=(1, NDOF), dtype='f8')

        if x0 is not None:
            self.append_state(x0)

    def init_fluid_props(self, model, fluid_props=None):
        """
        Initializes the fluid properties layout of the file.

        Parameters
        ----------
        model : femvf.forms.ForwardModel
            Not really needed for this one but left the arg here since it's in solid properties init
        """
        group_fluid = self.root_group.create_group('fluid_properties')
        for key, prop_desc in FluidProperties.TYPES.items():
            shape = _property_shape(prop_desc, model)
            group_fluid.create_dataset(key, shape=(0,)+shape, maxshape=(None,)+shape)

        if fluid_props is not None:
            self.append_fluid_props(fluid_props)

    def init_solid_props(self, model, solid_props=None):
        """
        Initializes the solid properties layout of the file.

        Parameters
        ----------
        model :

        """
        solid_group = self.root_group.create_group('solid_properties')
        for key, prop_desc in SolidProperties.TYPES.items():
            shape = _property_shape(prop_desc, model)
            solid_group.create_dataset(key, shape)

        if solid_props is not None:
            self.append_solid_props(solid_props)

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

    def append_state(self, x):
        """
        Append state to the file.

        Parameters
        ----------
        x : tuple of dfn.Function
            (u, v, a) states to append
        """
        for dset_name, value in zip(['u', 'v', 'a'], x):
            dset = self.root_group[dset_name]
            dset.resize(dset.shape[0]+1, axis=0)
            dset[-1] = value.vector()[:]

    def append_fluid_props(self, fluid_props):
        """
        Append fluid properties to the file.

        Parameters
        ----------
        fluid_props : dict
            Dictionary of fluid properties to append
        """
        fluid_group = self.root_group['fluid_properties']

        for label in FluidProperties.TYPES:
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
        for label in SolidProperties.TYPES:
            dset = solid_group[label]

            if label == 'elastic_modulus':
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


    def get_time(self, n):
        """
        Returns the time at state n.
        """
        return self.root_group['time'][n]

    def get_solution_times(self):
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

    def get_u(self, n, out=None):
        """
        Returns displacement at index `n`.
        """
        ret = None
        dset = self.root_group['u']
        if out is None:
            ret = dset[n]
        else:
            out[:] = dset[n]
            ret = out

        return ret

    # def get_v(self, n):

    # def get_a(self, n):

    def get_state(self, n, out=None):
        """
        Returns form coefficient vectors for states (u, v, a) at index n.

        Parameters
        ----------
        n : int
            Index to set the functions for.
        out : tuple of 3 dfn.Function
            A set of functions to set vector values for.
        """

        labels = ('u', 'v', 'a')
        ret = []
        if out is None:
            for label in labels:
                dset = self.root_group[label]
                ret.append(dset[n])
        else:
            for function, label in zip(out, labels):
                dset = self.root_group[label]
                function[:] = dset[n]

            ret = out

        return tuple(ret)

    def set_state(self, n, x):
        """
        Set form coefficient vectors for states `x=(u, v, a)` at index n.

        Parameters
        ----------
        n : int
            Index to set the functions for.
        x : tuple of 3 array_like
            A set of vectors to assign.
        """
        for label, value in zip(('u', 'v', 'a'), x):
            self.root_group[label][n] = value

    def get_fluid_props(self, n):
        """
        Returns the fluid properties dictionary at index n.
        """
        fluid_props = FluidProperties(self.model)
        fluid_group = self.root_group['fluid_properties']

        # Correct for constant fluid properties in time
        # TODO: Refactor how constant fluid/solid properties are defined.
        m = n
        if self.root_group['fluid_properties/p_sub'].size == 1:
            m = 0

        for label in FluidProperties.TYPES:
            fluid_props[label] = fluid_group[label][m]

        return fluid_props

    def get_solid_props(self):
        """
        Returns the solid properties
        """
        solid_props = SolidProperties(self.model)
        solid_group = self.root_group['solid_properties']
        for label in SolidProperties.TYPES:
            data = solid_group[label]

            if label == 'elastic_modulus':
                solid_props[label] = data[:]
            else:
                # have to index differently for scalar datasets
                solid_props[label] = data[()]

        return solid_props

    def get_iter_params(self, n):
        """
        Return parameter defining iteration `n`

        Parameters
        ----------
        n : int
            Index of the iteration.
        """

        x0 = self.get_state(n-1)
        dt = self.get_time(n) - self.get_time(n-1)
        solid_props = self.get_solid_props()
        fluid_props = self.get_fluid_props(n-1)
        u1 = self.get_u(n)

        return {'x0': x0, 'dt': dt, 'u1': u1,
                'solid_props': solid_props, 'fluid_props': fluid_props}

def _property_shape(property_desc, model):
    const_or_field = property_desc[0]
    data_shape = property_desc[1]

    shape = None
    if const_or_field == 'field':
        shape = (model.mesh.num_vertices(),) + data_shape
    else:
        shape = data_shape

    return shape