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


def line_search(hs, model, uva, solid_props, fluid_props, times,
                duva=(0, 0, 0), dsolid_props=None, dfluid_props=None, dtimes=None,
                coupling='explicit', filepath='temp.h5'):
    if path.exists(filepath):
        os.remove(filepath)

    for n, h in enumerate(hs):
        # Increment all the properties along the search direction
        uva_n = tuple([uva[i] + h*duva[i] for i in range(3)])

        solid_props_n = solid_props.copy()
        if dsolid_props is not None:
            solid_props_n.vector[:] += h*dsolid_props.vector

        fluid_props_n = fluid_props.copy()
        if dfluid_props is not None:
            fluid_props_n.vector[:] += h*dfluid_props.vector

        times_n = np.array(times)
        if dtimes is not None:
            times_n += h*dtimes
        # timing_props_n = {'t0': times_n[0], 'tmeas': times_n, 'dt_max': np.inf}

        runtime_start = perf_counter()
        info = integrate(model, uva_n, solid_props_n, fluid_props_n, times_n,
                         coupling=coupling, h5file=filepath, h5group=f'{n}')
        runtime_end = perf_counter()

        print(f"Run duration {runtime_end-runtime_start} s")

        # Save the run info to a pickled file
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
                         coupling=coupling, h5file=filepath, h5group=f'{n}')
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