"""
Functions returning a scalar given the model state at a time

Each should have the format:
callback(model, state, control, properties, time)
"""

import numpy as np

from .models.fluid import (smoothlb, expweight, wavg)

def safe_glottal_width(model, state, control, props, time):
    model.set_fin_state(state)
    model.set_control(control)

    usurf, *_ = model.fluid.control.vecs

    y = usurf[1::2]
    a = 2 * (props['y_midline'] - y)
    s = model.fluid.s_vertices

    asafe = smoothlb(a, 2*props['ygap_lb'], props['zeta_lb'])

    wmin = expweight(asafe, props['zeta_amin'])
    amin = wavg(s, asafe, wmin)

    return amin

def glottal_width(model, state, control, props, time):
    ymidline = props['y_midline']
    model.set_fin_state(state)
    x_cur_surf = model.get_cur_config()[model.fsi_verts]
    return ymidline - np.max(x_cur_surf[:, 1])

def glottal_flow_rate(model, state, control, props, time):
    if 'q' in state:
        return state['q'][0]
    else:
        return np.nan

