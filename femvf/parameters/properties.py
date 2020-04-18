"""
Classes for definining property values
"""

from ..constants import PASCAL_TO_CGS, SI_DENSITY_TO_CGS
import numpy as np

class Properties:
    """
    Represents a collection of properties defining a particular Model

    The `Model' must define class variables PROPERTY_TYPES and PROPERTY_DEFAULTS
    """
    def __init__(self, Model, data_dict=None):
        for key in Model.PROPERTY_TYPES:
            if key not in Model.PROPERTY_DEFAULTS:
                raise KeyError(f"Property `{key}` does not have a default value")
        
        self.Model = Model
        self.data = dict()

        if data_dict is None:
            data_dict = dict()

        # Initialize the data
        for key in self.TYPES.keys():
            data_type, data_shape = self.TYPES[key]

            vector_shape = None
            if data_type == 'field':
                vector_shape = (Model.scalar_fspace.dim(), *data_shape)
            else:
                vector_shape = (*data_shape, )

            # Store scalar data directly as a float
            if vector_shape == ():
                self.data[key] = data_dict.get(key, self.DEFAULTS[key])
            else:
                self.data[key] = np.zeros(vector_shape)
                self.data[key][:] = data_dict.get(key, self.DEFAULTS[key])

    def __getitem__(self, key):
        """
        Gives dictionary like behaviour.

        Raises an errors if the key does not exist.
        """
        if key not in self.TYPES:
            raise KeyError(f"{key} is not a valid property")
        else:
            return self.data[key]

    def __setitem__(self, key, value):
        """
        Gives dictionary like behaviour.

        Raises an errors if the key does not exist.
        """
        if key not in self.TYPES:
            raise KeyError(f"{key} is not a valid property")
        else:
            self.data[key] = value

    def __iter__(self):
        return self.data.__iter__()

    def __contains__(self, key):
        return key in self.TYPES

    def __str__(self):
        return self.data.__str__()

    def __repr__(self):
        return self.data.__repr__()

    def get(self, key, default):
        if key in self:
            return self[key]
        else:
            return default

    def update(self, new_dict):
        for key in self.TYPES.keys():
            if key in new_dict:
                self[key] = new_dict[key]

    def items(self):
        """
        Return underlying property dict's iterms
        """
        return self.data.items()

    def keys(self):
        """
        Return underlying property dict's keys
        """
        return self.data.keys()

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
