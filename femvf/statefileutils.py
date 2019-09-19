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

    def initialize(self, u0, v0, a0, solid_props, fluid_props, solution_times):
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
        N = solution_times.size

        # Kinematic states
        for data, dataset_name in zip([u0, v0, a0], ['u', 'v', 'a']):
            self.file.create_dataset(join(self.group, dataset_name),
                                     shape=(N, data.vector()[:].size),
                                     dtype='f8')
            self.file[join(self.group, dataset_name)][0] = data.vector()

        self.file.create_dataset(join(self.group, 'time'), data=solution_times)

        # Fluid properties
        for label in constants.FLUID_PROPERTY_LABELS:
            self.file.create_dataset(join(self.group, 'fluid_properties', label), shape=(N,))

        # Solid properties (Assumed to not be time-varying)
        for label in constants.SOLID_PROPERTY_LABELS:
            self.file.create_dataset(join(self.group, 'solid_properties', label),
                                     data=solid_props[label])

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
        # TODO: You might want to have time variable properties in the future
        for label in constants.SOLID_PROPERTY_LABELS:
            data = self.file[join(self.group, 'solid_properties', label)]

            if not data.shape:
                # If `data.shape` is an empty tuple then we have to index differently
                solid_props[label] = data[()]
            else:
                solid_props[label] = data[:]

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
