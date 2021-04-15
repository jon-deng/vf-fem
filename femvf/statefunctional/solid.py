"""
This module contains definitions of functionals over the solid state.
"""

import numpy as np
# import matplotlib.pyplot as plt
# import scipy.signal as sig

import dolfin as dfn
import ufl

from .. import linalg
from .base import AbstractFunctional
from ..models.solid import form_inf_strain, Solid

def glottal_width_smooth(model, state, control, props):
    solid = model.solid
    XREF = model.solid.scalar_fspace.tabulate_all_coordinates(solid.mesh)

    xcur = XREF + state['u']
    widths = props['y_collision'] - xcur[1::2]
    gw = np.min(widths)
    return gw

def glottal_width_sharp(model, state, control, props):
    solid = model.solid
    XREF = model.solid.scalar_fspace.tabulate_all_coordinates(solid.mesh)

    xcur = XREF + state['u']
    widths = props['y_collision'] - xcur[1::2]
    gw = np.min(widths)
    return gw

def peak_collision_pressure(model, state, control, props):
    pass
