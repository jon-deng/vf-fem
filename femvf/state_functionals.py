"""
This module contains definitions of functionals that operate on a single state i.e. at an instant in time.

Each functional has a general signature

functional(model, h5file, h5group='/', **kwargs) -> float, dict

, where h5file is an hdf5 file reference that contains the states history, h5group is the group path
in the file where the states are stored, and **kwargs are and specific keyword arguments. The
functional returns it's value as the first argument and a dictionary of additional info as a second
argument.

For computing the sensitivity of the functional through the discrete adjoint method, you also need
the sensitivity of the functional with respect to the n'th state. This function has the signature

dfunctional_du(model, n, statefile, **kwargs) -> float, dict

, where the parameters have the same uses as defined previously.
"""

import os.path as path

import numpy as np
import dolfin as dfn
import ufl

from petsc4py import PETSc

# from . import statefileutils as sfu
from . import forms

## Functionals defined for a single state index
def fluidwork(model, n, statefile):
    """
    Returns the fluid work from n-1 to n.

    Parameters
    ----------
    n : int
        State index
    h5path : str
        Path to the file containing states from the forward run
    """
    # Set form coefficients to represent the fluidwork at index n
    statefile.set_iteration_states(n, u0=model.u0, v0=model.v0, a0=model.a0, u1=model.u1)
    fluid_props = statefile.get_fluid_properties(n)
    model.set_pressure(fluid_props)

    return dfn.assemble(model.fluid_work)

def vocaleff(model, n, statefile):
    """
    Returns the vocal efficiency over the timestep from n to n+1.

    h5path : str
        Path to the file containing states from the forward run
    """
    # Set form coefficients to represent the fluidwork at index n
    statefile.set_iteration_states(n, u0=model.u0, v0=model.v0, a0=model.a0, u1=model.u1)
    fluid_props = statefile.get_fluid_properties(n)
    info = model.set_pressure(fluid_props)

    dt = float(model.dt)
    inputwork = dt*info['flow_rate']*fluid_props['p_sub']
    cost = dfn.assemble(model.fluid_work)/inputwork
    return cost
