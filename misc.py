"""
Don't know what to do with this yet.
"""

import numpy as np

def get_dynamic_fluid_props(fluid_props, time):
    """
    Returns fluid properties at specified time given a time varying set of properties.

    Parameters
    ----------
    fluid_props : dict
        Time varying set of fluid properties. If a property called 'key' is time varying, there
        should be a key called 'key_time' that stores time points at which the property changes.
        fluid_props[]
    """
    out = {}

    dynamic_properties = ['p_sub', 'p_sup']
    for key in dynamic_properties:
        if f'{key}_time' in fluid_props:
            out[key] = np.interp(time, fluid_props[f'{key}_time'], fluid_props[key])
        else:
            out[key] = fluid_props[key]

    for key, value in fluid_props.items():
        if key not in dynamic_properties:
            out[key] = value

    return out
    