"""
This module contains definitions of functionals over the fluid state.
"""

import numpy as np

from blocklinalg import linalg
# from .base import AbstractFunctional

def separation_point(model, state, control, props):
    model.set_fin_state(state)
    model.set_control(control)

    _, info = model.fluid.solve_state1(model.fluid.state0)
    return info['s_sep']
