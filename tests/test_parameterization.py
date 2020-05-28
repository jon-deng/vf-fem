import sys
import os
from os import path
from time import perf_counter
import pickle

import unittest

# from math import round
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import dolfin as dfn

sys.path.append('../')
from femvf import meshutils, statefile as sf
from femvf.forward import integrate
from femvf.adjoint import adjoint
from femvf.model import ForwardModel
from femvf.solids import Rayleigh, KelvinVoigt
from femvf.fluids import Bernoulli
from femvf.constants import PASCAL_TO_CGS
from femvf.parameters.properties import SolidProperties, FluidProperties
from femvf.functionals import basic as funcs

# TODO: To test the parameterization adjoint method, you have to implement functionals that act on a
# single input state, i.e. (uva0, qp0, solid_props, fluid_props, t)
# This functional should then have methods that return sensitivities w.r.t the appropriate
# parameters.
# A special case is the time variable, since parameterization produce a list of times, rather
# than a single time instance, which will require some further though on what should be done there.