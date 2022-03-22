"""
This module contains definitions of functionals over the fluid state.
"""

# import numpy as np

# from blocktensor import linalg
# from .base import AbstractFunctional
from .decorator import transform_to_proc_signals

def make_separation_point(model):
    def separation_point(model, state, control, props):
        model.set_fin_state(state)
        model.set_control(control)

        _, info = model.fluid.solve_state1(model.fluid.state0)
        return info['s_sep']
    return separation_point
proc_separation_point = transform_to_proc_signals(make_separation_point)
