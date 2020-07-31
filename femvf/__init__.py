"""
A package for simulating vocal folds.
"""

# Make load_fsi_model available at the package level because it's commonly used
from .model import load_fsi_model

# Make common modules available
from . import forward
from . import adjoint

from . import statefile
from . import fluids
from . import solids

from . import linalg

# Make common packages available
from . import functionals as funcs
from . import parameters as params
from . import vis
