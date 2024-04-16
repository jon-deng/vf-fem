"""

"""

import numpy as np


def make_residuals(model):
    MODEL_ATTR_NAMES = ('solid', 'fluid', 'acoustic')
    SUB_MODELS = tuple(
        [getattr(model, name) for name in MODEL_ATTR_NAMES if hasattr(model, name)]
    )

    def residuals(ini_state, fin_state, control, dt, prop):
        model.set_ini_state(ini_state)
        model.set_fin_state(fin_state)
        model.set_control(control)
        model.dt = dt

        return np.array([model.assem_res().norm() for model in SUB_MODELS])

    return residuals
