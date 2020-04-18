"""
This module contains base class definitions for parameters
"""

import math
# from functools import reduce
from collections import OrderedDict
from . import constants

import dolfin as dfn
from petsc4py import PETSc

import numpy as np

class KeyIndexedVector:
    """
    A vector where slices within the vector can be accesed by a key through a dict-like interface

    Parameters
    ----------
    shape : OrderedDict
        A dictionary containing `key: tuple` pairs that indicate the shape of each subvector

    Attributes
    ----------
    key_to_slice : dict
        The mapping of keys to slices
    vector : np.ndarray
        The vector
    """
    def __init__(self, shapes, data=None):
        # Calculate the shape of the array for each labeled parameter and its offset in the vector
        self._SHAPES = OrderedDict()
        self._OFFSETS = OrderedDict()
        offset = 0
        for key, shape in shapes:
            self._SHAPES[key] = shape
            self._OFFSETS[key] = offset
            offset += np.prod(shape, dtype=int, initial=1)

        # Initialize the size of the containing vector (the final calculated offset)
        self._vector = np.zeros(offset, dtype=float)

        if data is None:
            data = {}

        for key in data:
            offset = self._OFFSETS[key]
            size = np.prod(self._SHAPES[key])

            if key in self:
                self._vector[offset:offset+size] = data[key]

    def __contains__(self, key):
        return key in self.PARAM_TYPES

    def __getitem__(self, key):
        """
        Returns the slice corresponding to the labelled parameter

        Parameters
        ----------
        key : str
            A parameter label
        """
        label = key

        if label not in self:
            raise KeyError(f"`{label}` is not a valid parameter label")
        else:
            # Get the parameter label from the key

            offset = self._OFFSETS[label]
            shape = self._SHAPES[label]
            size = np.prod(shape, dtype=int, initial=1)

            return self.vector[offset:offset+size].reshape(shape)

    def __iter__(self):
        """
        Copy dictionary iter behaviour
        """
        return self.PARAM_TYPES.__iter__()

    def __str__(self):
        return self._SHAPES.__str__()

    def __repr__(self):
        return f"{type(self)}({self._SHAPES})"

    def copy(self):
        out = type(self)(self._SHAPES)
        out.vector[:] = self.vector
        return out

    @property
    def vector(self):
        """
        Return the underlying vector
        """
        return self._vector

    @property
    def size(self):
        """
        Return the size of the parameter vector
        """
        return self.vector.size

class Parameterization(KeyIndexedVector):
    """
    A parameterization is a mapping from a set of parameters to the set of basic parameters for the 
    forward model.

    Parameter values are stored in a single array. The slice corresponding to a specific parameter
    can be accessed through a label based index i.e. `self[param_label]`

    Each parameterization has to have `convert` and `dconvert` methods. `convert` transforms the 
    parameterization to a standard parameterization for the forward model and `dconvert` transforms
    gradients wrt. standard parameters, to gradients wrt. the parameterization.

    Parameters
    ----------
    model : model.ForwardModel
    constants : dict
        A dictionary of labelled constants mapping labels to constant values
        used in the parameterization.
    parameters : dict, optional
        A mapping of labeled parameters to values to initialize the parameterization
    kwargs : optional
        Additional keyword arguments needed to specify the parameterization.
        These will vary depending on the specific parameterization.

    Attributes
    ----------
    model : femvf.model.ForwardModel
    constants : dict({str: value})
        A dictionary of labeled constants to values
    vector : np.ndarray
        The parameter vector
    PARAM_TYPES : OrderedDict(tuple( 'field'|'const' , tuple), ...)
        A dictionary storing the shape of each labeled parameter in the parameterization
    """

    PARAM_TYPES = OrderedDict(
        {'abstract_parameters': ('field', ())}
        )
    CONSTANT_LABELS = {'foo': 2, 'bar': 99}

    def __new__(cls, model, constants, parameters=None):
        # Raise an error if one of the required constant labels is missing, then proceed with
        # initializing
        for label in cls.CONSTANT_LABELS:
            if label not in constants:
                raise ValueError(f"Could not find constant `{label}` in supplied constants")

        return super().__new__(cls)

    def __init__(self, model, constants, parameters=None):
        self._constants = constants
        self.model = model

        # Calculate the array shape for each labeled parameter
        shapes = OrderedDict()
        N_DOF = model.solid.scalar_fspace.dim()
        for key, param_type in self.PARAM_TYPES.items():
            shape = None
            if param_type[0] == 'field':
                shape = (N_DOF, *param_type[1])
            elif param_type[0] == 'const':
                shape = (*param_type[1], )
            else:
                raise ValueError("Parameter type must be one of 'field' or 'const'")
            shapes[key] = shape

        super().__init__(shapes)

    def __str__(self):
        return self.PARAM_TYPES.__str__()

    def __repr__(self):
        return f"{type(self).__name__}(model, {self.constants})"

    def copy(self):
        out = type(self)(self.model, self.constants)
        out.vector[:] = self.vector
        return out

    @property
    def constants(self):
        """
        Return constant values associated with the parameterization
        """
        return self._constants

    def convert(self):
        """
        Return the solid/fluid properties for the forward model.

        Returns
        -------
        uva : tuple
            Initial state
        solid_props : properties.SolidProperties
            A collection of solid properties
        fluid_props : properties.FluidProperties
            A collection of fluid properties
        timing_props :
        """
        return NotImplementedError

    def dconvert(self, demod):
        """
        Return the sensitivity of the solid/fluid properties to the parameter vector.

        Parameters
        ----------
        demod : array_like
            The sensitivity of a functional wrt. the elastic moduli
        dg_solid_props : dict
            The sensitivity of a functional with respect each property in solid_props
        dg_fluid_props:
            The sensitivity of a functional with respect each property in fluid_props

        Returns
        -------
        array_like
            The sensitvity of the functional wrt. the parameterization
        """
        return NotImplementedError
