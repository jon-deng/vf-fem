"""
Contains definitions of parametrizations.

These objects should mappings from a given parameter set to the standard
properties of the forward model.
"""

from typing import Mapping, Union, Optional, Callable, Tuple
from numpy.typing import NDArray

from functools import reduce

import numpy as np
from jax import numpy as jnp
import jax
import dolfin as dfn
import ufl
from femvf.models.transient import BaseTransientModel
from femvf.models.dynamical.base import BaseDynamicalModel
from femvf import meshutils

from blockarray import blockvec as bv, blockarray as ba
from blockarray import typing


Model = Union[BaseDynamicalModel, BaseTransientModel]
BlockVectorDict = Mapping[str, NDArray]


class Transform:
    """
    Map `BlockVector`s between two spaces

    Letting the spaces be denoted by 'X' (input space) and 'Y' (output space),
    instances of this class map `BlockVector` objects from 'X' to 'Y' and also
    linearizations.

    Attributes
    ----------
    x, y : bv.BlockVector
        Prototype `BlockVector` of the input and output spaces
        (`model.prop`) attribute, respectively
    """

    _x: bv.BlockVector
    _y: bv.BlockVector

    @property
    def x(self):
        """
        Return a prototype input vector for the parameterization
        """
        # raise NotImplementedError()
        return self._x

    @property
    def y(self):
        """
        Return a prototype/default output vector for the parameterization
        """
        return self._y

    def apply(self, x: bv.BlockVector) -> bv.BlockVector:
        """
        Map an input `x` to an output `y`

        Parameters
        ----------
        x : bv.BlockVector
            The input vector

        Returns
        -------
        y : bv.BlockVector
            The output vector
        """
        raise NotImplementedError()

    def apply_vjp(self, x: bv.BlockVector, hy: bv.BlockVector) -> bv.BlockVector:
        """
        Map a dual vector `hy` to a dual vector `hx` at the point `x`

        Parameters
        ----------
        hy : bv.BlockVector
            The output dual vector

        Returns
        -------
        hx : bv.BlockVector
            The input dual vector
        """
        raise NotImplementedError()

    def apply_jvp(self, x: bv.BlockVector, dx: bv.BlockVector) -> bv.BlockVector:
        """
        Map a differential input vector `dx` to an output vector `dy` at the point `x`

        Parameters
        ----------
        dx : bv.BlockVector
            The input vector

        Returns
        -------
        dy : bv.BlockVector
            The output vector
        """
        raise NotImplementedError()

    def __mul__(self, other):
        return TransformComposition(self, other)

    def __rmul__(self, other):
        return TransformComposition(other, self)


class TransformComposition(Transform):
    """
    A composition of two `Transform` instances

    Attributes
    ----------
    transforms
    """

    def __init__(self, transform_1: Transform, transform_2: Transform):
        self._transforms = (transform_1, transform_2)

        self._x = transform_1.x
        self._y = transform_2.y

    def apply(self, x: bv.BlockVector) -> bv.BlockVector:
        trans1, trans2 = self._transforms
        return trans2.apply(trans1.apply(x))

    def apply_vjp(self, x: bv.BlockVector, hy: bv.BlockVector) -> bv.BlockVector:
        trans1, trans2 = self._transforms

        x1 = x
        hy2 = hy

        y1 = trans1.apply(x1)

        x2 = y1
        hx2 = trans2.apply_vjp(x2, hy2)

        hy1 = hx2
        hx1 = trans1.apply_vjp(x1, hy1)

        return hx1

    def apply_jvp(self, x: bv.BlockVector, dx: bv.BlockVector) -> bv.BlockVector:
        trans1, trans2 = self._transforms

        x1 = x
        dx1 = dx

        y1 = trans1.apply(x1)
        dy1 = trans1.apply_jvp(x1, dx1)

        x2 = y1
        dx2 = dy1
        dy2 = trans2.apply_jvp(x2, dx2)

        return dy2


class TransformFromModel(Transform):
    """
    Map `BlockVector`s from an input space to a `model.prop` space

    This is used by defining subclasses that implement the input space vector,
    and the appropriate transformations.

    Note that subclasses have to supply a `self._x` attribute
    """

    def __init__(self, model: Model):
        self.model = model

        # NOTE: Subclasses have to supply a `self._x` attribute
        _y_vec = ba.zeros(model.prop.bshape)
        self._y = bv.BlockVector(_y_vec.sub_blocks, labels=model.prop.labels)


class TractionShape(TransformFromModel):
    """
    Map a surface traction to mesh displacement
    """

    def __init__(self, model: Model, lame_lambda=1.0, lame_mu=1.0, dirichlet_bcs=None):
        super().__init__(model)

        # The input vector simply renames the mesh displacement vector
        _x_vec = ba.zeros(model.prop.f_bshape)
        x_subvecs = _x_vec.sub_blocks
        x_labels = list(model.prop.labels[0])
        try:
            ii = x_labels.index('umesh')
        except ValueError as err:
            raise ValueError("model properties does not contain a shape") from err

        x_labels[ii] = 'tmesh'
        self._x = bv.BlockVector(x_subvecs, labels=(tuple(x_labels),))

        ## Define the stiffness matrix and traction sensitivity matrices:
        # The two maps are
        # dF/du : sensitivity of residual to mesh displacement
        # dF/dt : sensitivity of residual to medial surface tractions
        residual = model.solid.residual
        fspace = residual.form['coeff.prop.umesh'].function_space()
        dx = residual.measure('dx')
        ds = residual.measure('ds')

        facet_label_to_id = residual.mesh_function_label_to_value('facet')
        # ds_traction_surfaces = [
        #     ds(int(facet_label_to_id[facet_label]))
        #     for facet_label in residual.fsi_facet_labels
        # ]
        # ds_traction = reduce(lambda x, y: x + y, ds_traction_surfaces)

        tmesh = dfn.Function(fspace)
        trial = dfn.TrialFunction(fspace)
        test = dfn.TestFunction(fspace)

        lmbda, mu = dfn.Constant(lame_lambda), dfn.Constant(lame_mu)
        trial_strain = 1 / 2 * (ufl.grad(trial) + ufl.grad(trial).T)
        test_strain = 1 / 2 * (ufl.grad(test) + ufl.grad(test).T)
        dim = trial_strain.ufl_shape[0]
        form_dF_du = (
            ufl.inner(
                2 * mu * trial_strain
                + lmbda * ufl.tr(trial_strain) * ufl.Identity(dim),
                test_strain,
            )
            * dx
        )
        mat_dF_du = dfn.assemble(
            form_dF_du, tensor=dfn.PETScMatrix(), keep_diagonal=True
        )

        form_dF_dt = ufl.inner(trial, test) * ds
        mat_dF_dt = dfn.assemble(
            form_dF_dt, tensor=dfn.PETScMatrix(), keep_diagonal=True
        )

        if dirichlet_bcs == None:
            dirichlet_bcs = residual.dirichlet_bcs
        for bc_dir in dirichlet_bcs:
            bc_dir.apply(mat_dF_du)
            bc_dir.zero_columns(mat_dF_du, tmesh.vector(), diagonal_value=1.0)
            bc_dir.apply(mat_dF_dt)
            bc_dir.zero_columns(mat_dF_dt, tmesh.vector(), diagonal_value=0.0)

        self.mat_dF_du = mat_dF_du
        self.mat_dF_dt = mat_dF_dt
        self.umesh = dfn.Function(fspace)
        self.tmesh = tmesh

    def _set_y_from_x(self, x: bv.BlockVector, y: bv.BlockVector):
        """
        Set all `y` subvectors except for 'umesh' from `x`
        """
        x_dict = bvec_to_dict(x)
        y_dict = bvec_to_dict(y)

        for key, val in x_dict.items():
            if key in y_dict:
                y_dict[key][:] = val

        return x_dict, y_dict

    def _set_y_from_x_linear(self, x: bv.BlockVector, y: bv.BlockVector):
        """
        Set all linearized `y` subvectors except for 'umesh' from `x`
        """
        x_dict = bvec_to_dict(x)
        y_dict = bvec_to_dict(y)

        for key, val in x_dict.items():
            if key in y_dict:
                y_dict[key][:] = val

        return x_dict, y_dict

    def apply(self, x: bv.BlockVector) -> bv.BlockVector:
        self.x[:] = x
        x_dict, y_dict = self._set_y_from_x(self.x, self.y)

        # Assemble the RHS for the given medial surface traction
        self.tmesh.vector()[:] = x_dict['tmesh']
        rhs = self.mat_dF_dt * self.tmesh.vector()

        # Solve for the mesh displacement
        dfn.solve(self.mat_dF_du, self.umesh.vector(), rhs, 'lu')
        y_dict['umesh'][:] = self.umesh.vector()[:]
        return dict_to_bvec(y_dict, self.y.labels)

    def apply_vjp(self, x: bv.BlockVector, hy: bv.BlockVector) -> bv.BlockVector:
        """
        Return the corresponding `self.model.prop` vector
        """
        self.x[:] = x
        # self.y[:] = hy
        hy_dict, hx_dict = self._set_y_from_x_linear(hy, self.x)

        # Assemble the RHS for the given medial surface traction
        humesh = self.umesh.vector()
        humesh[:] = hy_dict['umesh']

        hF = self.tmesh.vector()
        dfn.solve(self.mat_dF_du, hF, humesh, 'lu')
        hx_dict['tmesh'][:] = self.mat_dF_dt * hF
        return dict_to_bvec(hx_dict, self.x.labels)

    def apply_jvp(self, x: bv.BlockVector, dx: bv.BlockVector) -> bv.BlockVector:
        """
        Return the corresponding `self.model.prop` vector
        """
        self.x[:] = x
        dx_dict, dy_dict = self._set_y_from_x_linear(dx, self.y)

        # Assemble the RHS for the given medial surface traction
        dtmesh = self.tmesh.vector()
        dtmesh[:] = dx_dict['tmesh']
        dF = self.mat_dF_dt * dtmesh

        # Solve for the mesh displacement
        dumesh = self.umesh.vector()
        dfn.solve(self.mat_dF_du, dumesh, dF, 'lu')
        dy_dict['umesh'][:] = dumesh[:]
        return dict_to_bvec(dy_dict, self.y.labels)


class JaxTransform(Transform):
    """
    Map `BlockVector`s between two spaces

    Letting the spaces be denoted by 'X' (input space) and 'Y' (output space),
    instances of this class map `BlockVector` objects from 'X' to 'Y' and also
    linearizations.

    The map here must be automatically defined through a `make_map` function
    which uses `jax` to create the output vector and mapping.

    Attributes
    ----------
    x, y : bv.BlockVector
        Prototype `BlockVector` of the input and output spaces
        (`model.prop`) attribute, respectively
    """

    def __init__(
        self,
        x_y_map: Tuple[
            bv.BlockVector, bv.BlockVector, Callable[[BlockVectorDict], BlockVectorDict]
        ],
    ):
        x, y, map = x_y_map

        self._x = bv.convert_subtype_to_numpy(x)
        self._y = bv.convert_subtype_to_numpy(y)

        self._map = map

    @property
    def map(self):
        return self._map

    def apply(self, x: bv.BlockVector) -> bv.BlockVector:
        x_dict = bvec_to_dict(x)

        # Convert any JAX arrays into numpy arrays
        y_dict = self.map(x_dict)
        y_dict = jax_to_numpy_dict(y_dict)
        return dict_to_bvec(y_dict, self.y.labels)

    def apply_vjp(self, x: bv.BlockVector, hy: bv.BlockVector) -> bv.BlockVector:
        x_dict = bvec_to_dict(x)
        _hy = self.y.copy()
        _hy[:] = hy
        hy_dict = bvec_to_dict(_hy)

        _, vjp_fun = jax.vjp(self.map, x_dict)
        (hx_dict,) = vjp_fun(hy_dict)
        hx_dict = jax_to_numpy_dict(hx_dict)
        return dict_to_bvec(hx_dict, self.x.labels)

    def apply_jvp(self, x: bv.BlockVector, dx: bv.BlockVector) -> bv.BlockVector:
        x_dict = bvec_to_dict(x)
        dx_dict = bvec_to_dict(dx)
        _, dy_dict = jax.jvp(self.map, (x_dict,), (dx_dict,))
        dy_dict = jax_to_numpy_dict(dy_dict)
        return dict_to_bvec(dy_dict, self.y.labels)


class JaxTransformFromModel(JaxTransform):
    """
    Map an alternative parameterization to a model's `prop` parameters

    The map here must be automatically defined through a `make_map` function
    which uses `jax` to create output vector
    """

    def __init__(self, model: Model, **kwargs):
        super().__init__()
        x_y_map = self.make_x_y_map(model, **kwargs)
        super().__init__(x_y_map)

    @staticmethod
    def make_x_y_map(model: Model, **kwargs):
        """
        Return a prototype input vector and `jax` function that performs the map
        """
        raise NotImplementedError()


class LayerModuli(JaxTransformFromModel):

    @staticmethod
    def make_x_y_map(model):
        ## Get the mapping from labelled cell regions to DOFs
        cell_label_to_dofs = meshutils.process_celllabel_to_dofs_from_residual(
            model.solid.residual,
            model.solid.residual.form['coeff.prop.emod'].function_space(),
        )

        y_dict = bvec_to_dict(model.prop)

        def map(x):

            emod = jnp.zeros(y_dict['emod'].size, copy=True)
            weight = jnp.zeros(y_dict['emod'].size, copy=True)
            for label, layer_stiffness in x.items():
                dofs = cell_label_to_dofs[label]

                weight_region = jnp.zeros(emod.shape)
                weight_region = weight_region.at[dofs][:] = layer_stiffness
                emod_region = layer_stiffness * weight_region

                emod = emod + emod_region
                weight = weight + weight_region

            new_y_dict = y_dict.copy()
            new_y_dict['emod'] = emod
            return new_y_dict

        ## Define the input vector
        labels = (tuple(cell_label_to_dofs.keys()),)
        subvecs = [np.zeros(1) for _ in labels[0]]
        in_vec = bv.BlockVector(subvecs, labels=labels)

        return (in_vec, model.prop.copy(), map)


class JaxTransformFromX(JaxTransform):
    """
    Map `BlockVector`s between two spaces
    """

    def __init__(self, x: bv.BlockVector, **kwargs):
        y, map = self.make_y_map(x, **kwargs)
        super().__init__((x, y, map))

    @staticmethod
    def make_y_map(x: bv.BlockVector, **kwargs):
        """
        Return a `jax` function that performs the map
        """
        raise NotImplementedError()


class Identity(JaxTransformFromX):

    @staticmethod
    def make_y_map(x: bv.BlockVector, **kwargs):
        def map(input):
            return input

        y = x

        return y, map


class ConstantSubset(JaxTransformFromX):

    def __init__(self, x: bv.BlockVector, const_vals=None):
        super().__init__(x, const_vals=const_vals)

    @staticmethod
    def make_y_map(x: bv.BlockVector, const_vals=None):

        if const_vals is None:
            const_vals = {}

        def map(x):
            """
            Return the x->y mapping
            """
            y = {
                key: (
                    const_vals[key] * np.ones(value.shape)
                    if key in const_vals
                    else value
                )
                for key, value in x.items()
            }
            return y

        y = x.copy()
        return y, map


class Scale(JaxTransformFromX):

    def __init__(self, x: bv.BlockVector, scale=None):
        super().__init__(x, scale=scale)

    @staticmethod
    def make_y_map(x: bv.BlockVector, scale=None):
        _scale = {key: 1.0 for key in x.labels[0]}
        if scale is not None:
            _scale.update(scale)
        scale = _scale

        def map(x):
            """
            Return the x->y mapping
            """
            y = {key: scale[key] * x_sub for key, x_sub in x.items()}
            return y

        y = x.copy()

        return y, map


class JaxTransformFromY(JaxTransform):
    """
    Map `BlockVector`s between two spaces
    """

    def __init__(self, y: bv.BlockVector, **kwargs):
        x, map = self.make_x_map(y, **kwargs)
        super().__init__((x, y, map))

    @staticmethod
    def make_x_map(y: bv.BlockVector, **kwargs):
        """
        Return a `jax` function that performs the map
        """
        raise NotImplementedError()


class ExtractSubset(JaxTransformFromY):

    def __init__(self, y: bv.BlockVector, keys_to_extract=None):

        _const_vals = bvec_to_dict(y.copy())
        super().__init__(y, keys_to_extract=keys_to_extract, const_vals=_const_vals)

    @staticmethod
    def make_x_map(y: bv.BlockVector, keys_to_extract=None, const_vals=None):
        """
        Return a `jax` function that performs the map
        """
        if keys_to_extract is None:
            keys_to_extract = list(self._const.keys())

        x = bv.BlockVector(
            [np.array(y[key]) for key in keys_to_extract],
            shape=(len(keys_to_extract),),
            labels=(keys_to_extract,),
        )

        def map(x):
            y = {
                key: x[key] if key in x else value for key, value in const_vals.items()
            }
            return y

        return x, map


def bvec_to_dict(x: bv.BlockVector) -> Mapping[str, np.ndarray]:
    return {label: subvec for label, subvec in x.sub_items()}


def dict_to_bvec(
    y: Mapping[str, np.ndarray], labels: Optional[typing.MultiLabels] = None
) -> bv.BlockVector:
    if labels is None:
        labels = (tuple(y.keys()),)
    subvecs = [y[label] for label in labels[0]]
    return bv.BlockVector(subvecs, labels=labels)


def jax_to_numpy_dict(dict: Mapping[str, NDArray]):
    return {key: np.array(value) for key, value in dict.items()}
