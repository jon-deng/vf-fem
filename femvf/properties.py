"""
Classes for definining property values
"""

from .constants import PASCAL_TO_CGS, SI_DENSITY_TO_CGS
import numpy as np

class Properties:
    """
    Represents a collection of properties
    """

    TYPES = {'foo': ('field', ()),
             'bar': ('const', (3,))}

    DEFAULTS = {'foo': 1.0,
                'bar': -2.0}

    def __new__(cls, data_dict=None, **kwargs):
        # Check that there is a default value for each type
        for key in cls.TYPES:
            if key not in cls.DEFAULTS:
                raise KeyError(f"Property key `{key}`` does not have a default value")

        return super().__new__(cls)

    def __init__(self, data_dict=None, **kwargs):
        self.data = dict()

        if data_dict is not None:
            for key in self.TYPES.keys():
                self.data[key] = np.array(data_dict.get(key, self.DEFAULTS[key]))
        else:
            for key in self.TYPES.keys():
                self.data[key] = np.array(kwargs.get(key, self.DEFAULTS[key]))

    def __getitem__(self, key):
        """
        Gives dictionary like behaviour.

        Raises an errors if the key does not exist.
        """
        if key not in self.TYPES:
            raise KeyError(f"`{key}` is not a valid property")
        else:
            return self.data[key]

    def __setitem__(self, key, value):
        """
        Gives dictionary like behaviour.

        Raises an errors if the key does not exist.
        """
        if key not in self.TYPES:
            raise KeyError(f"`{key}` is not a valid property")
        else:
            self.data[key] = value

    def __contains__(self, key):
        return key in self.TYPES

    def __str__(self):
        return self.data.__str__()

    def __repr__(self):
        return self.data.__repr__()

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

class SolidProperties(Properties):
    """
    Represents a collection of linear-elastic solid properties

    Parameters
    ----------
    elastic_modulus :
    poissons_ratio :
    density :
    rayleigh_m, rayleigh_k :
        Rayleigh damping parameters for the mass and stiffness matrix terms
    y_collision :
        The y-coordinate of the collision plane
    k_collision :
        The stiffness of the collision penalty spring
    """
    # `types` indicates if each property is either a field or constant variable
    # and its shape in a tuple. A shape of `None` indicates a scalar value

    TYPES = {'elastic_modulus': ('field', ()),
             'poissons_ratio': ('const', ()),
             'density': ('const', ()),
             'rayleigh_m': ('const', ()),
             'rayleigh_k': ('const', ()),
             'y_collision': ('const', ()),
             'k_collision': ('const', ())}

    # LABELS = tuple(TYPES.keys())

    DEFAULTS = {'elastic_modulus': 10e3 * PASCAL_TO_CGS,
                'poissons_ratio': 0.49,
                'density': 1000 * SI_DENSITY_TO_CGS,
                'rayleigh_m': 10,
                'rayleigh_k': 1e-3,
                'y_collision': 0.61-0.001,
                'k_collision': 1e11}

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

    TYPES = {'p_sub': ('const', ()),
             'p_sup': ('const', ()),
             'a_sub': ('const', ()),
             'a_sup': ('const', ()),
             'rho': ('const', ()),
             'y_midline': ('const', ()),
             'alpha': ('const', ()),
             'k': ('const', ()),
             'sigma': ('const', ())}

    # LABELS = tuple(TYPES.keys())

    DEFAULTS = {'p_sub': 800 * PASCAL_TO_CGS,
                'p_sup': 0 * PASCAL_TO_CGS,
                'a_sub': 100000,
                'a_sup': 0.6,
                'rho': 1.1225 * SI_DENSITY_TO_CGS,
                'y_midline': 0.61,
                'alpha': -3000,
                'k': 50,
                'sigma': 0.002}

class TimingProperties(Properties):
    """
    A class storing timing parameters for a forward simulation.

    Parameters
    ----------
    dt_max :
        The maximum time step to integrate over
    tmeas :
        Times at which to record the solution
    t0 :
        Starting time of simulation
    """
    TYPES = {'dt_max': ('const', ()),
             'tmeas': ('const', (None,)),
             't0': ('const', ())}

    DEFAULTS = {'dt_max': 1e-4,
                'tmeas': np.array([0, 1e-4]),
                't0': 0.0}
