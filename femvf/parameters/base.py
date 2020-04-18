"""
This module contains base class definitions for parameters
"""

import math
# from functools import reduce
from collections import OrderedDict
from .. import constants

import dolfin as dfn
from petsc4py import PETSc

import numpy as np

class KeyIndexedArray:
    """
    An array where ranges within the array can be accesed by a label via a dict-like interface

    Parameters
    ----------
    shapes : OrderedDict
        A dictionary containing `key: tuple` pairs that indicate the shape of each slice in the
        array corresponding to `key`
    data : dict-like, optional
        A dictionary of `key: array` pairs to initialize values in the array

    Attributes
    ----------
    key_to_slice : dict
        The mapping of keys to slices
    vector : np.ndarray
        The vector
    """
    def __init__(self, shapes, data=None):
        # Calculate the shape and offset of each subarray
        self._shapes = OrderedDict()
        self._offsets = OrderedDict()
        offset = 0
        for key, shape in shapes.items():
            self._shapes[key] = shape
            self._offsets[key] = offset
            offset += np.prod(shape, dtype=int, initial=1)

        # Initialize the size of the containing vector (the final calculated offset)
        self._vector = np.zeros(offset, dtype=float)

        # Initialize the values of the vector if `data` is supplied
        if data is None:
            data = {}

        for key, value in data.items():
            offset = self.OFFSETS[key]
            shape = self.SHAPES[key]

            if key in self:
                if shape == ():
                    self[key][()] = value
                else:
                    self[key][:] = value

    def __str__(self):
        return self.SHAPES.__str__()

    def __repr__(self):
        return f"{type(self)}({self.SHAPES})"

    def copy(self):
        out = type(self)(self.SHAPES)
        out.vector[:] = self.vector
        return out

    ## Implement a dict-like interface
    def __contains__(self, key):
        return key in self.SHAPES

    def __getitem__(self, key):
        """
        Returns the slice corresponding to the labelled parameter

        A slice references the memory of the original array, so the returned slice can be used to
        set values.

        Parameters
        ----------
        key : str
            A parameter label
        """
        if key not in self:
            raise KeyError(f"`{key}` is not a valid parameter label")
        else:
            # Get the parameter label from the key

            offset = self.OFFSETS[key]
            shape = self.SHAPES[key]
            size = np.prod(shape, dtype=int, initial=1)

            return self.vector[offset:offset+size].reshape(shape)

    def __iter__(self):
        """
        Copy dictionary iter behaviour
        """
        return self.SHAPES.__iter__()

    def keys(self):
        return self.SHAPES.keys()

    ## Implement an array-like interface
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


    @property
    def SHAPES(self):
        return self._shapes

    @property
    def OFFSETS(self):
        return self._offsets

