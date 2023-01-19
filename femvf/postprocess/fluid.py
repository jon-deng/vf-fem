"""
This module contains definitions of functionals over the fluid state.
"""

from .base import transform_to_make_signals, BaseStateMeasure

class SeparationPoint(BaseStateMeasure):

    def __init_measure_context__(self, *args, **kwargs):
        pass

    def __call__(self, state, control, prop):
        model = self.model
        model.set_fin_state(state)
        model.set_control(control)

        _, info = model.fluid.solve_state1(model.fluid.state1)
        return info['s_sep']