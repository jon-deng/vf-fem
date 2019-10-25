"""
Module to work with state values from a forward pass stored in an hdf5 file.
"""

from os.path import join

import h5py
import dolfin as dfn
import numpy as np

from . import constants

class StateFile:
    """
    Represents a state file.

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

    def __init__(self, name, group='/', **kwargs):
        self.file = h5py.File(name, **kwargs)
        self.group = group

        # self.data = self.file[self.group]

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.file.close()

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
        self.file.create_dataset(join(self.group, 'time'), (0,), maxshape=(None,),
                                 dtype=np.float)

        self.file.create_dataset(join(self.group, 'meas_indices'), (0,), maxshape=(None,),
                                 dtype=np.intp)

        self.init_state(model, x0=x0)

        # Fluid properties
        if 'fluid_properties' not in self.file[self.group]:
            self.init_fluid_props(model, fluid_props=fluid_props)

        # Solid properties (assumed to not be time-varying)
        if 'solid_properties' not in self.file[self.group]:
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
            self.file.create_dataset(join(self.group, dataset_name), (0, NDOF),
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
        group = self.file.create_group(join(self.group, 'fluid_properties'))
        for label in constants.FLUID_PROPERTY_LABELS:
            group.create_dataset(label, shape=(0,), maxshape=(None,))

        if fluid_props is not None:
            self.append_fluid_props(fluid_props)

    def init_solid_props(self, model, solid_props=None):
        """
        Initializes the solid properties layout of the file.

        Parameters
        ----------
        model :

        """
        group = self.file.create_group(join(self.group, 'solid_properties'))
        for label in constants.SOLID_PROPERTY_LABELS:
            # Only elastic modulus is a vector, other solid properties are currently scalars
            shape = None
            if label == 'elastic_modulus':
                shape = (model.scalar_function_space.dim(),)
            else:
                shape = ()

            group.create_dataset(label, shape)

        if solid_props is not None:
            self.append_solid_props(solid_props)

    def append_meas_index(self, index):
        """
        Append measured indices to the file.

        Parameters
        ----------
        index : int
        """
        dset = self.file[join(self.group, 'meas_indices')]
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
            dset = self.file[join(self.group, dset_name)]
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
        for label in constants.FLUID_PROPERTY_LABELS:
            dset = self.file[join(self.group, 'fluid_properties', label)]
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
        for label in constants.SOLID_PROPERTY_LABELS:
            dset = self.file[join(self.group, 'solid_properties', label)]

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
        dset = self.file[join(self.group, 'time')]
        dset.resize(dset.shape[0]+1, axis=0)
        dset[-1] = time


    def get_time(self, n):
        """
        Returns the time at state n.
        """
        return self.file[join(self.group, 'time')][n]

    def get_solution_times(self):
        """
        Returns the time vector.
        """
        return self.file[join(self.group, 'time')][:]

    def get_meas_indices(self):
        """
        Returns the measured indices.
        """
        return self.file[join(self.group, 'meas_indices')][:]

    def get_num_states(self):
        """
        Returns the number of states in the solution
        """
        return self.file[join(self.group, 'u')].shape[0]

    def get_u(self, n, out=None):
        """
        Returns displacement at index `n`.
        """
        ret = None
        dset = self.file[join(self.group, 'u')]
        if out is None:
            ret = dset[n]
        else:
            out.vector()[:] = dset[n]
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
            A set of functions to assign into.
        """

        labels = ('u', 'v', 'a')
        ret = []
        if out is None:
            for label in labels:
                dset = self.file[join(self.group, label)]
                ret.append(dset[n])
        else:
            for function, label in zip(out, labels):
                dset = self.file[join(self.group, label)]
                function.vector()[:] = dset[n]

            ret = out

        return tuple(ret)

    def get_fluid_props(self, n):
        """
        Returns the fluid properties dictionary at index n.
        """
        fluid_props = {}
        for label in constants.FLUID_PROPERTY_LABELS:
            fluid_props[label] = self.file[join(self.group, 'fluid_properties', label)][n]

        return fluid_props

    def get_solid_props(self):
        """
        Returns the solid properties
        """
        solid_props = {}
        for label in constants.SOLID_PROPERTY_LABELS:
            data = self.file[join(self.group, 'solid_properties', label)]

            if label == 'elastic_modulus':
                solid_props[label] = data[:]
            else:
                # have to index differently for scalar datasets
                solid_props[label] = data[()]

        return solid_props
