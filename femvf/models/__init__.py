"""
This package contains definitions of various "models", systems that can be integrated over time.

Each model represents the residual defined over a time step, n, as
.. math:: F^n{u^n, u^{n-1}, g^n, p^n, dt},
where u is the state vector, g is a control vector, p is a vector of constant properties, 
and dt is the time step.

The model object should have attributes storing the current time step inputs
    ini_state/fin_state : previous and current state vectors
    control : current control vectors
    properties : properties (constant in time)
    dt : time step

In addition to the residual, first order sensitivities of the residual with respect to it's inputs
should also be defined.
"""

from . import (acoustic, fluid, solid, coupled)

# from .load import *
from .acoustic import *
from .solid import *
from .fluid import *
from .coupled import *
