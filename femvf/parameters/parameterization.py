"""
Contains definitions of parametrizations. These objects should provide a mapping from their specific parameters to standardized parameters of the forward model, as well as the derivative of the map.
"""

import math

from collections import OrderedDict
import jax

from blockarray import blockvec as bv
from . import properties as props
from .. import constants, linalg
from ..solverconst import DEFAULT_NEWTON_SOLVER_PRM

import dolfin as dfn
import ufl
from petsc4py import PETSc

import numpy as np

class Parameterization:
    """
    A parameterization is a mapping from one set of parameters to the basic parameters for the
    forward model.

    Each parameterization has `convert` and `dconvert` methods tha perform the mapping and
    calculate it's sensitivity. `convert` transforms the parameterization to a standard
    parameterization for the forward model and `dconvert` transforms
    gradients wrt. standard parameters, to gradients wrt. the parameterization.

    Parameters
    ----------
    model : model.ForwardModel
    kwargs : optional
        Additional keyword arguments needed to specify the parameterization.
        These will vary depending on the specific parameterization.

    Attributes
    ----------
    model : femvf.model.ForwardModel
    constants : tuple
        A dictionary of labeled constants to values
    bvector : np.ndarray
        The BlockVector representation of the parameterization
    """

    def __init__(self, model, *args, **kwargs):
        self.model = model
        self._out_labels = model.props.labels
        # self._out_bshape = model.props.bshape

    def apply(self, in_vec: bv.BlockVector) -> bv.BlockVector:
        """
        Return the solid/fluid properties for the forward model.

        Returns
        -------
        uva : tuple
            Initial state
        solid_props : BlockVector
            A collection of solid properties
        fluid_props : BlockVector
            A collection of fluid properties
        timing_props :
        """
        in_dict = {label: subvec for label, subvec in in_vec.items()}
        out_dict = self.map(in_dict)
        out_subvecs = [out_dict[label] for label in self._out_labels]
        return bv.BlockVector(out_subvecs, labels=self._out_labels)

    def apply_vjp(self, in_vec: bv.BlockVector, din_vec: bv.BlockVector) -> bv.BlockVector:
        """
        """
        in_dict = {label: subvec for label, subvec in in_vec.items()}
        din_dict = {label: subvec for label, subvec in in_vec.items()}
        out_dict = jax.vjp(self.map, in_dict, din_dict)
        out_subvecs = [out_dict[label] for label in self._out_labels]
        return bv.BlockVector(out_subvecs, labels=self._out_labels)

    def apply_jvp(self, in_vec: bv.BlockVector, din_vec: bv.BlockVector) -> bv.BlockVector:
        """
        """
        in_dict = {label: subvec for label, subvec in in_vec.items()}
        din_dict = {label: subvec for label, subvec in in_vec.items()}
        out_dict = jax.jvp(self.map, in_dict, din_dict)
        out_subvecs = [out_dict[label] for label in self._out_labels]
        return bv.BlockVector(out_subvecs, labels=self._out_labels)

