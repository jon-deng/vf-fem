"""
Contains definitions of parametrizations. These objects should provide a mapping from their specific parameters to standardized parameters of the forward model, as well as the derivative of the map.
"""

from typing import Mapping, Union, Optional, Tuple

import numpy as np
from jax import numpy as jnp
import jax
import dolfin as dfn
import ufl
from femvf.models.transient.base import BaseTransientModel as TranModel
from femvf.models.dynamical.base import BaseDynamicalModel as DynModel
from femvf import meshutils
from petsc4py import PETSc

from blockarray import blockvec as bv
from blockarray import typing

class BaseParameterization:
    """
    Map a parameterization to a model's `.props` parameter set

    Parameters
    ----------
    model :
        The model to convert parameters to
    out_default :
        A vector of constants/defaults for `model.props`
    kwargs : optional
        Additional keyword arguments needed to specify the parameterization.
        These will vary depending on the specific parameterization.

    Attributes
    ----------
    model : femvf.model.ForwardModel
    x, y : bv.BlockVector
        Prototype `BlockVectors` of the input (parameterization) and output
        (`model.props`) attribute, respectively
    """

    def __init__(
            self,
            model: Union[DynModel, TranModel],
            out_default: bv.BlockVector,
            *args,
            **kwargs
        ):
        self.model = model

        self._y_vec = out_default
        assert out_default.labels == model.props.labels

    @property
    def x(self):
        """
        Return a prototype input vector for the parameterization
        """
        raise NotImplementedError()
        # return self._x_vec

    @property
    def y(self):
        """
        Return a prototype/default output vector for the parameterization
        """
        return self._y_vec

    def apply(self, x: bv.BlockVector) -> bv.BlockVector:
        """
        Map the parameterization `x` to a `model.props` vector

        Parameters
        ----------
        x : bv.BlockVector
            The input parameterization

        Returns
        -------
        y : bv.BlockVector
            The output `model.props` vector
        """
        raise NotImplementedError()

    def apply_vjp(
            self,
            x: bv.BlockVector,
            hy: bv.BlockVector
        ) -> bv.BlockVector:
        """
        Map a dual vector `hy` (in `model.props` space) to a dual vector `hx`

        Parameters
        ----------
        hy : bv.BlockVector
            The input dual vector in `model.props` space

        Returns
        -------
        hx : bv.BlockVector
            The output dual vector in `x` space
        """
        raise NotImplementedError()

    def apply_jvp(
            self,
            x: bv.BlockVector,
            dx: bv.BlockVector
        ) -> bv.BlockVector:
        """
        Map a differential primal vector `dx` to a `dy` (`model.props`)

        Parameters
        ----------
        dx : bv.BlockVector
            The input parameterization

        Returns
        -------
        dy : bv.BlockVector
            The output primal vector in `model.props` space
        """
        raise NotImplementedError()

class BaseJaxParameterization(BaseParameterization):
    """
    Map an alternative parameterization to a model's `props` parameters

    The map here must be automatically defined through a `make_map` function
    which uses `jax` to create output vector
    """

    def __init__(
            self,
            model: Union[DynModel, TranModel],
            out_default: bv.BlockVector,
            *args,
            **kwargs
        ):
        super().__init__(model, out_default, *args, **kwargs)

        self._x_vec, self.map = self.make_map()
        # self._x_labels = self._x_vec.labels

    @property
    def x(self):
        """
        Return a prototype input vector for the parameterization
        """
        return self._x_vec

    @property
    def y(self):
        """
        Return a prototype/default output vector for the parameterization
        """
        return self._y_vec

    def make_map(self):
        """
        Return a prototype input vector and `jax` function that performs the map
        """
        raise NotImplementedError()

    def apply(self, x: bv.BlockVector) -> bv.BlockVector:
        """
        Return the corresponding `self.model.props` vector
        """
        x_dict = bvec_to_dict(x)
        y_dict = self.map(x_dict)
        return dict_to_bvec(y_dict, self.y.labels)

    def apply_vjp(self, x: bv.BlockVector, hy: bv.BlockVector) -> bv.BlockVector:
        """
        """
        x_dict = bvec_to_dict(x)
        hy_dict = bvec_to_dict(hy)
        hx_dict = jax.vjp(self.map, x_dict, hy_dict)
        return dict_to_bvec(hx_dict, self.x.labels)

    def apply_jvp(self, x: bv.BlockVector, dx: bv.BlockVector) -> bv.BlockVector:
        """
        """
        x_dict = bvec_to_dict(x)
        dx_dict = bvec_to_dict(dx)
        y_dict = jax.jvp(self.map, x_dict, dx_dict)
        return dict_to_bvec(y_dict, self.y.labels)

class Identity(BaseJaxParameterization):

    def make_map(self):
        def map(x):
            return x
        return self.y, map

class LayerModuli(BaseJaxParameterization):

    def make_map(self):
        ## Get the mapping from labelled cell regions to DOFs
        E = self.model.solid.forms['coeff.prop.emod']
        cell_label_to_dofs = meshutils.process_celllabel_to_dofs_from_forms(
            self.model.solid.forms,
            E.function_space()
        )

        _y_dict = bvec_to_dict(self.y)
        def map(x):
            y_dict = _y_dict.copy()
            new_emod = jnp.array(y_dict['emod'], copy=True)
            for label, value in x.items():
                dofs = cell_label_to_dofs[label]
                new_emod = new_emod.at[dofs][:] = value

            y_dict['emod'] = new_emod
            return y_dict

        ## Define the input vector
        labels = (tuple(cell_label_to_dofs.keys()),)
        subvecs = [np.zeros(1) for _ in labels[0]]
        in_vec = bv.BlockVector(subvecs, labels=labels)

        return in_vec, map

def bvec_to_dict(x: bv.BlockVector) -> Mapping[str, np.ndarray]:
    return {label: subvec for label, subvec in x.items()}

def dict_to_bvec(
        y: Mapping[str, np.ndarray],
        labels: Optional[typing.MultiLabels]=None
    ) -> bv.BlockVector:
    if labels is None:
        labels = (tuple(y.keys()), )
    subvecs = [y[label] for label in labels[0]]
    return bv.BlockVector(subvecs, labels=labels)