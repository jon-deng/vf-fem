"""
Classes for definining property values
"""

class SolidProperties:
    """
    Represents a collection of linear-elastic solid properties
    """

    SOLID_PROPERTY_LABELS = ('elastic_modulus', 'poissons_ratio', 'density',
                             'rayleigh_m', 'rayleigh_k', 'y_collision')
    def __init__(self):
        self.elastic_modulus
        self.rayleigh_m
        self.rayleigh_k
        self.nu
        self.y_collision

class FluidProperties:
    """
    Represents a collection of 1D potential flow fluid properties
    """
