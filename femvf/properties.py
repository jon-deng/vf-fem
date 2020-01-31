"""
Classes for definining property values
"""

from .constants import PASCAL_TO_CGS, SI_DENSITY_TO_CGS

class Properties:
    """
    Represents a collection of properties
    """

    TYPES = {'foo': ('field', ()),
             'bar': ('const', (3,))}

    DEFAULTS = {'foo': 1.0,
                'bar': -2.0}

    def __init__(self, *args, **kwargs):
        self.data = dict()

        for key in self.TYPES.keys():
            self.data[key] = kwargs.get(key, self.DEFAULTS[key])

    def __getitem__(self, key):
        """
        Gives dictionary like behaviour.

        Raises an errors if the key does not exist.
        """
        if key not in self.TYPES:
            raise ValueError(f"{key} is not a valid property")
        else:
            return self.data[key]

    def __setitem__(self, key, value):
        """
        Gives dictionary like behaviour.

        Raises an errors if the key does not exist.

        TODO: You might want to raise an error if you set the property wrong. For example and error
        should be raised if you try to set a field of values to a constant property.
        """
        if key not in self.TYPES:
            raise ValueError(f"{key} is not a valid property")
        else:
            self.data[key] = value

    def __contains__(self, key):
        return key in self.TYPES

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

    TODO: Remove subglottal pressure as a property. It's more like a boundary condition?

    alpha, k and sigma are smoothing parameters that control the smoothness of approximations used in
    separation
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
