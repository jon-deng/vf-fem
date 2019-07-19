"""
Module containing some unit conversions and other constants
"""

## Unit conversion constants
PASCAL_TO_CGS = 1000/100
SI_DENSITY_TO_CGS = 1/1000
SI_VISCOSITY_TO_CGS = PASCAL_TO_CGS

## Default properties you can use if you can't think of anything else
DEFAULT_FLUID_PROPERTIES = {'p_sub': 800*PASCAL_TO_CGS,
                            'p_sup': 0*PASCAL_TO_CGS,
                            'a_sub': 100000,
                            'a_sup': 0.6,
                            'rho': 1.1225*SI_DENSITY_TO_CGS,
                            'y_midline': 0.61}

DEFAULT_SOLID_PROPERTIES = {'elastic_modulus': 10e3*PASCAL_TO_CGS,
                            'nu': 0.49,
                            'density': 1000 * SI_DENSITY_TO_CGS}
