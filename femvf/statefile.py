"""
Module to work with state values from a forward pass stored in an hdf5 file.
"""

from os.path import join

import h5py
import dolfin as dfn

from . import constants
from . import fluids

class StateFile:
    """
    Represents a state file.

    State information is stored in the hdf5 file under a containing group:
    /.../group

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

    def initialize_layout(self, model, x0=None, fluid_props=None, solid_props=None):
        """
        Initializes the layout of the state file.

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
        self.file.create_dataset(join(self.group, 'time'), (0,), maxshape=(None,))

        self._initialize_state(model, x0=x0)

        # Fluid properties
        self._initialize_fluid_props(model, fluid_props=fluid_props)

        # Solid properties (assumed to not be time-varying)
        self._initialize_solid_props(model, solid_props=solid_props)

    def _initialize_state(self, model, x0=None):
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

    def _initialize_fluid_props(self, model, fluid_props=None):
        """
        Initializes the fluid properties layout of the file.

        Parameters
        ----------
        model :
            Not really needed for this one but left the arg here since it's in solid properties init
        """
        for label in constants.FLUID_PROPERTY_LABELS:
            self.file.create_dataset(join(self.group, 'fluid_properties', label), shape=(0,),
                                     maxshape=(None,))

        if fluid_props is not None:
            self.append_fluid_props(fluid_props)

    def _initialize_solid_props(self, model, solid_props=None):
        """
        Initializes the solid properties layout of the file.

        Parameters
        ----------
        model :

        """
        for label in constants.SOLID_PROPERTY_LABELS:
            # Only elastic modulus is a vector, other solid properties are currently scalars
            shape = None
            if label == 'elastic_modulus':
                shape = (model.scalar_function_space.dim(),)
            else:
                shape = ()

            self.file.create_dataset(join(self.group, 'solid_properties', label), shape)

        if solid_props is not None:
            self.append_solid_props(solid_props)

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

    def append_time(self, time):
        """
        Append times to the file.

        Parameters
        ----------
        time : array_like
            Times to append
        """
        dset = self.file[join(self.group, 'time')]
        dset.resize(dset.shape[0]+time.size, axis=0)
        dset[-time.size:] = time

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

    def write_state(self, x, n):
        """
        Writes the n'th state.

        Parameters
        ----------
        x : tuple of dfn.Function
        n : int
            State index to write to.
        """
        for function, label in zip(x, ('u', 'v', 'a')):
            self.file[join(self.group, label)][n] = function.vector()

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

    def get_num_states(self):
        """
        Returns the number of states in the solution
        """
        return self.file[join(self.group, 'u')].shape[0]

    def get_u(self, n, function_space=None):
        """
        Returns displacement at index `n`.
        """
        ret = None
        _ret = self.file[join(self.group, 'u')][n, ...]
        if function_space is None:
            ret = _ret
        else:
            ret = dfn.Function(function_space)
            ret.vector()[:] = _ret

        return ret

    # def get_v(self, n):

    # def get_a(self, n):

    def get_state(self, n, function_space=None):
        """
        Returns form coefficient vectors for states (u, v, a) at index n.

        Parameters
        ----------
        n : int
            Index to set the functions for.
        function_space : dfn.FunctionSpace
            If a function space is supplied, an instance of `dfn.Function` is returned.
        """
        labels = ('u', 'v', 'a')

        ret = None
        _ret = [self.file[join(self.group, label)][n, ...] for label in labels]

        if function_space is None:
            ret = _ret
        else:
            ret = []
            for ii, label in enumerate(labels):
                function = dfn.Function(function_space)

                function.vector()[:] = _ret[ii]
                ret.append(function)

        return tuple(ret)

    def set_state(self, n, x):
        _x = self.get_state(n)

        for function, vec in zip(x, _x):
            function.vector()[:] = vec

        return x

    def get_fluid_properties(self, n):
        """
        Returns the fluid properties dictionary at index n.
        """
        fluid_props = {}
        for label in constants.FLUID_PROPERTY_LABELS:
            fluid_props[label] = self.file[join(self.group, 'fluid_properties', label)][n]

        return fluid_props

    def get_solid_properties(self):
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

    def set_iteration_states(self, n, u0=None, v0=None, a0=None, u1=None):
        """
        Sets form coefficient vectors for states u_n-1, v_n-1, a_n-1, u_n at index n.

        Parameters
        ----------
        n : int
            Index to set the functions for.
        """
        if u0 is not None:
            u0.vector()[:] = self.file[join(self.group, 'u')][n-1]
        if v0 is not None:
            v0.vector()[:] = self.file[join(self.group, 'v')][n-1]
        if a0 is not None:
            a0.vector()[:] = self.file[join(self.group, 'a')][n-1]
        if u1 is not None:
            u1.vector()[:] = self.file[join(self.group, 'u')][n]

    def set_time_step(self, n, dt=None):
        if dt is not None:
            tspan = self.file[join(self.group, 'time')][n-1:n+1]
            dt.assign(tspan[1]-tspan[0])
