"""
Module containing some unit conversions and other constants
"""

## Unit conversion constants
PASCAL_TO_CGS = 1000/100
SI_DENSITY_TO_CGS = 1/1000
SI_VISCOSITY_TO_CGS = PASCAL_TO_CGS
SI_POWER_TO_CGS = 1e7

## Default properties you can use if you can't think of anything else
FLUID_PROPERTY_LABELS = ('p_sub', 'p_sup', 'a_sub', 'a_sup', 'rho', 'y_midline')
DEFAULT_FLUID_PROPERTIES = {'p_sub': 800 * PASCAL_TO_CGS,
                            'p_sup': 0 * PASCAL_TO_CGS,
                            'a_sub': 100000,
                            'a_sup': 0.6,
                            'rho': 1.1225 * SI_DENSITY_TO_CGS,
                            'y_midline': 0.61}
# FLUID_PROPERTY_LABELS = tuple(DEFAULT_FLUID_PROPERTIES.keys())

# Rayleigh damping parameters are roughly based on
# A three-dimensional model of vocal fold abduction/adduction
SOLID_PROPERTY_LABELS = ('elastic_modulus', 'poissons_ratio', 'density', 'rayleigh_m', 'rayleigh_k',
                         'y_collision')
DEFAULT_SOLID_PROPERTIES = {'elastic_modulus': 10e3 * PASCAL_TO_CGS,
                            'poissons_ratio': 0.49,
                            'density': 1000 * SI_DENSITY_TO_CGS,
                            'rayleigh_m': 10,
                            'rayleigh_k': 1e-3,
                            'y_collision': 0.61-0.001,
                            'k_collision': 1e11}
# SOLID_PROPERTY_LABELS = tuple(DEFAULT_FLUID_PROPERTIES.keys())
