"""
Classes for definining property values
"""
from collections import OrderedDict

import numpy as np

from .base import KeyIndexedArray
from ..constants import PASCAL_TO_CGS, SI_DENSITY_TO_CGS

class Properties:
    """
    Represents a collection of properties defining a particular `model` (fluid or solid)

    The `model' must define class variables `PROPERTY_TYPES` and `PROPERTY_DEFAULTS`. This class
    builds an object containing values for each of the defined properties in `PROPERTY_TYPES` with
    the correct shapes.

    Currently this is made more for the solid models since the dimensions of parameters can
    be found from the function spaces/mesh. The 1D bernoulli fluid model doesn't have a mesh yet.
    """
    def __init__(self, model):
        # TODO: Move this check this to the actual model class?
        for key in model.PROPERTY_TYPES:
            if key not in model.PROPERTY_DEFAULTS:
                raise KeyError(f"Property `{key}` does not have a default value")
        
        self._model = model

        # Calculate shapes of each parameter
        shapes = OrderedDict()
        for key, property_type in model.PROPERTY_TYPES.items():
            field_or_const, data_shape = property_type

            shape = None
            if field_or_const == 'field':
                shape = (model.scalar_fspace.dim(), *data_shape)
            elif field_or_const == 'const':
                shape = (*data_shape, )
            else:
                raise ValueError("uh oh")

            shapes[key] = shape

        self._data = KeyIndexedArray(shapes, data=model.PROPERTY_DEFAULTS)

    def __str__(self):
        return self.data.__str__()

    def __repr__(self):
        return f"{type(self).__name__}({self.model.__repr__()})"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            try:
                return np.all(self.vector == other.vector)
            except:
                raise
        else:
            raise TypeError(f"Cannot compare type {type(self)} <-- {type(other)}")

    def __ne__(self, other):
        return not self.__eq__(other)

    def copy(self):
        out = type(self)(self.model)
        out.vector[:] = self.vector[:]
        return out

    @property
    def model(self):
        return self._model
    
    ## Implement the dict-like interface coming from the KeyIndexedArray
    @property
    def data(self):
        """
        Returns the `KeyIndexedArray` storing the parameter values
        """
        return self._data

    def __contains__(self, key):
        return key in self.data

    def __getitem__(self, key):
        """
        Gives dictionary like behaviour.

        Raises an errors if the key does not exist.
        """
        return self.data[key]
    
    def __iter__(self):
        return self.data.__iter__()

    def keys(self):
        """
        Return underlying property dict's keys
        """
        return self.data.keys()
    
    ## Implement the array-like interface coming from the KeyIndexedArray
    @property
    def vector(self):
        return self.data.vector

    @property
    def size(self):
        return self.vector.size


    @property
    def TYPES(self):
        return self.Model.PROPERTY_TYPES

    @property
    def DEFAULTS(self):
        return self.Model.PROPERTY_DEFAULTS

class SolidProperties(Properties):
    """
    Represents a collection of properties defining a particular Model

    The `Model' must define class variables PROPERTY_TYPES and PROPERTY_DEFAULTS
    """

class FluidProperties(Properties):
    """
    Represents a collection of 1D potential flow fluid properties

    Parameters
    ----------
    p_sub, p_sup :
        Subglottal and supraglottal pressure
    a_sub, a_sup :
        Subglottal and supraglottal area
    rho :
        Density of air
    y_midline :
        The y-coordinate of the midline for the fluid
    alpha, k, sigma :
        Smoothing parameters that control the smoothness of approximations used in separation.
    """

# def timing_props():

#     class TimingProperties(Properties):
#         """
#         A class storing timing parameters for a forward simulation.

#         Parameters
#         ----------
#         dt_max :
#             The maximum time step to integrate over
#         tmeas :
#             Times at which to record the solution
#         t0 :
#             Starting time of simulation
#         """
#         TYPES = {'dt_max': ('const', ()),
#                 'tmeas': ('const', (None,)),
#                 't0': ('const', ())}

#         DEFAULTS = {'dt_max': 1e-4,
#                     'tmeas': np.array([0, 1e-4]),
#                     't0': 0.0}
