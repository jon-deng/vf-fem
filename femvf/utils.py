"""
Utility functions that don't have another place to go
"""
import os
from os import path
from time import perf_counter
import pickle

from .forward import integrate
from . import statefile as sf

import numpy as np


def line_search(hs, model, 
                ini_state, controls, props, times,
                dstate, dcontrols, dprops, dtimes,
                filepath='temp.h5'):
    if path.exists(filepath):
        os.remove(filepath)

    for n, h in enumerate(hs):
        ## Increment the inputs for the provided step
        state_n = ini_state + h*dstate

        controls_n = []
        for control, dcontrol in zip(controls, dcontrols):
            controls_n.append(control + h*dcontrol)

        props_n = props + h*dprops

        times_n = np.array(times)
        times_n += h*dtimes

        ## Run simulations at the step
        runtime_start = perf_counter()
        info = integrate(model, state_n, controls_n, props_n, times_n,
                         h5file=filepath, h5group=f'{n}')
        runtime_end = perf_counter()

        print(f"Run duration {runtime_end-runtime_start} s")

        ## Save the run info to a pickled file
        if h == 0:
            with open(path.splitext(filepath)[0] + ".pickle", 'wb') as f:
                pickle.dump(info, f)

    return filepath

def line_search_p(hs, model, p, dp, coupling='explicit', filepath='temp.h5'):
    """
    Returns a parameterized line search for parameterization `p` in direction `dp`.
    """
    if os.path.exists(filepath):
        os.remove(filepath)

    p_n = p.copy()
    for n, h in enumerate(hs):
        # Increment all the properties along the search direction
        p_n.vector[:] = p.vector + h*dp.vector

        uva_n, solid_props_n, fluid_props_n, times_n = p_n.convert()
        # breakpoint()
        # print(uva_n[0].norm('l2'), uva_n[1].norm('l2'), uva_n[2].norm('l2'))

        runtime_start = perf_counter()
        info = integrate(model, uva_n, solid_props_n, fluid_props_n, times_n,
                         h5file=filepath, h5group=f'{n}')
        runtime_end = perf_counter()

        print(f"Run duration {runtime_end-runtime_start} s")

        # Save the run info to a pickled file
        if h == 0:
            with open(path.splitext(filepath)[0] + ".pickle", 'wb') as f:
                pickle.dump(info, f)

    return filepath

def functionals_on_line_search(hs, functional, model, filepath):
    functionals = list()
    for n, h in enumerate(hs):
        with sf.StateFile(model, filepath, group=f'{n}', mode='r') as f:
            val = functional(f)
            functionals.append(val)

    return np.array(functionals)