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
from femvf.models.transient.base import BaseTransientModel as TranModel
from femvf.models.dynamical.base import BaseDynamicalModel as DynModel
from femvf.models.assemblyutils import CachedFormAssembler
from femvf import meshutils

from blockarray import blockvec as bv, blockarray as ba
from blockarray import typing

class BaseTransform:
    """
    Map `BlockVector`s between spaces X and Y

    Attributes
    ----------
    x, y : bv.BlockVector
        Prototype `BlockVectors` of the input (parameterization) and output
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
        Map the parameterization `x` to a `model.prop` vector

        Parameters
        ----------
        x : bv.BlockVector
            The input parameterization

        Returns
        -------
        y : bv.BlockVector
            The output `model.prop` vector
        """
        raise NotImplementedError()

    def apply_vjp(
            self,
            x: bv.BlockVector,
            hy: bv.BlockVector
        ) -> bv.BlockVector:
        """
        Map a dual vector `hy` (in `model.prop` space) to a dual vector `hx`

        Parameters
        ----------
        hy : bv.BlockVector
            The input dual vector in `model.prop` space

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
        Map a differential primal vector `dx` to a `dy` (`model.prop`)

        Parameters
        ----------
        dx : bv.BlockVector
            The input parameterization

        Returns
        -------
        dy : bv.BlockVector
            The output primal vector in `model.prop` space
        """
        raise NotImplementedError()

    def apply_inv(self, y: bv.BlockVector) -> bv.BlockVector:
        """
        Map the parameterization `x` to a `model.prop` vector

        Parameters
        ----------
        x : bv.BlockVector
            The input parameterization

        Returns
        -------
        y : bv.BlockVector
            The output `model.prop` vector
        """
        raise NotImplementedError()

    def apply_inv_vjp(
            self,
            y: bv.BlockVector,
            hx: bv.BlockVector
        ) -> bv.BlockVector:
        """
        Map a dual vector `hx` (in `model.prop` space) to a dual vector `hy`

        Parameters
        ----------
        hx : bv.BlockVector
            The input dual vector in `model.prop` space

        Returns
        -------
        yx : bv.BlockVector
            The output dual vector in `x` space
        """
        raise NotImplementedError()

    def apply_inv_jvp(
            self,
            y: bv.BlockVector,
            dy: bv.BlockVector
        ) -> bv.BlockVector:
        """
        Map a differential primal vector `dy` to `dx`

        Parameters
        ----------
        dy : bv.BlockVector
            The input parameterization

        Returns
        -------
        dx : bv.BlockVector
            The output primal vector
        """
        raise NotImplementedError()


class BaseDolfinParameterization(BaseTransform):
    def __init__(
            self,
            model: Union[DynModel, TranModel]
        ):
        self.model = model

        _y_vec = ba.zeros(model.prop.bshape)
        self._y = bv.BlockVector(
            _y_vec.sub_blocks, labels=model.prop.labels
        )

        # NOTE: Subclasses have to supply a `self._x_vec` attribute

class TractionShape(BaseDolfinParameterization):
    def __init__(
            self,
            model: Union[DynModel, TranModel],
            lame_lambda=1.0, lame_mu=1.0,
            const_vals=None
        ):
        super().__init__(model)

        if const_vals is None:
            self.const_vals = {}
        else:
            self.const_vals = const_vals

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

        ## Define the stiffness matrix and traction sensitivity matrices that
        # The two maps are
        # dF/du : sensitivity of residual to mesh displacement
        # dF/dt : sensitivity of residual to medial surface tractions
        residual = model.solid.residual
        fspace = residual.form['coeff.prop.umesh'].function_space()
        dirichlet_bcs = residual.dirichlet_bcs
        dx = residual.measure('dx')
        ds = residual.measure('ds')

        facet_label_to_id = residual.mesh_function_label_to_value('facet')
        ds_traction_surfaces = [
            ds(int(facet_label_to_id[facet_label]))
            for facet_label in residual.fsi_facet_labels
        ]
        ds_traction = reduce(lambda x, y: x+y, ds_traction_surfaces)

        tmesh = dfn.Function(fspace)
        trial = dfn.TrialFunction(fspace)
        test = dfn.TestFunction(fspace)

        lmbda, mu = dfn.Constant(lame_lambda), dfn.Constant(lame_mu)
        trial_strain = 1/2*(ufl.grad(trial) + ufl.grad(trial).T)
        test_strain = 1/2*(ufl.grad(test) + ufl.grad(test).T)
        dim = trial_strain.ufl_shape[0]
        form_dF_du = ufl.inner(
            2*mu*trial_strain + lmbda*ufl.tr(trial_strain)*ufl.Identity(dim),
            test_strain
        ) * dx
        mat_dF_du = dfn.assemble(
            form_dF_du,
            tensor=dfn.PETScMatrix(),
            keep_diagonal=True
        )

        form_dF_dt = ufl.inner(trial, test)*ds_traction
        mat_dF_dt = dfn.assemble(
            form_dF_dt,
            tensor=dfn.PETScMatrix(),
            keep_diagonal=True
        )

        for bc_dir in dirichlet_bcs:
            bc_dir.apply(mat_dF_du)
            bc_dir.zero_columns(mat_dF_du, tmesh.vector(), diagonal_value=1.0)
            bc_dir.apply(mat_dF_dt)
            bc_dir.zero_columns(mat_dF_dt, tmesh.vector(), diagonal_value=0.0)

        self.mat_dF_du = mat_dF_du
        self.mat_dF_dt = mat_dF_dt
        self.umesh = dfn.Function(fspace)
        self.tmesh = tmesh

    def _set_y_defaults_from_x(self, x: bv.BlockVector, y: bv.BlockVector):
        x_dict = bvec_to_dict(x)
        y_dict = bvec_to_dict(y)

        for key, val in x_dict.items():
            if key in y_dict:
                if key in self.const_vals:
                    y_dict[key][:] = self.const_vals[key]*np.ones(val.shape)
                else:
                    y_dict[key][:] = val

        return x_dict, y_dict

    def _set_y_defaults_from_x_linear(self, x: bv.BlockVector, y: bv.BlockVector):
        x_dict = bvec_to_dict(x)
        y_dict = bvec_to_dict(y)

        for key, val in x_dict.items():
            if key in y_dict:
                if key in self.const_vals:
                    y_dict[key][:] = np.zeros(val.shape)
                else:
                    y_dict[key][:] = val

        return x_dict, y_dict

    def apply(self, x: bv.BlockVector) -> bv.BlockVector:
        """
        Return the corresponding `self.model.prop` vector
        """
        self.x[:] = x
        x_dict, y_dict = self._set_y_defaults_from_x(self.x, self.y)

        # Assemble the RHS for the given medial surface traction
        self.tmesh.vector()[:] = x_dict['tmesh']
        rhs = self.mat_dF_dt * self.tmesh.vector()

        # Solve for the mesh displacement
        dfn.solve(self.mat_dF_du, self.umesh.vector(), rhs, 'lu')
        y_dict['umesh'][:] = self.umesh.vector()[:]
        return dict_to_bvec(y_dict, self.y.labels)

    def apply_vjp(self, x, hy):
        """
        Return the corresponding `self.model.prop` vector
        """
        self.y[:] = hy
        hy_dict, hx_dict = self._set_y_defaults_from_x_linear(self.y, self.x)

        # Assemble the RHS for the given medial surface traction
        self.umesh.vector()[:] = hy_dict['umesh']

        # Solve for the mesh displacement
        dfn.solve(self.mat_dF_du, self.tmesh.vector(), self.umesh.vector(), 'lu')
        hx_dict['tmesh'][:] = self.mat_dF_dt*self.tmesh.vector()
        return dict_to_bvec(hx_dict, self.x.labels)

    def apply_jvp(self, x, dx) -> bv.BlockVector:
        """
        Return the corresponding `self.model.prop` vector
        """
        self.x[:] = dx
        dx_dict, dy_dict = self._set_y_defaults_from_x_linear(self.x, self.y)

        # Assemble the RHS for the given medial surface traction
        self.tmesh.vector()[:] = dx_dict['tmesh']
        rhs = self.mat_dF_dt * self.tmesh.vector()

        # Solve for the mesh displacement
        dfn.solve(self.mat_dF_du, self.umesh.vector(), rhs, 'lu')
        dy_dict['umesh'][:] = self.umesh.vector()[:]
        return dict_to_bvec(dy_dict, self.y.labels)

BlockVectorDict = Mapping[str, NDArray]

class JaxParameterization(BaseTransform):
    """
    Map an alternative parameterization to a model's `prop` parameters

    The map here must be automatically defined through a `make_map` function
    which uses `jax` to create output vector
    """

    def __init__(
            self,
            map_x_to_y: Tuple[bv.BlockVector, bv.BlockVector, Callable[[BlockVectorDict], BlockVectorDict]],
            map_y_to_x: Tuple[bv.BlockVector, bv.BlockVector, Callable[[BlockVectorDict], BlockVectorDict]]
        ):
        _x, _y, map = map_x_to_y
        y, x, map_inv = map_y_to_x

        self._x = bv.convert_subtype_to_numpy(x)
        self._y = bv.convert_subtype_to_numpy(y)

        self._map = map
        self._map_inv = map_inv

    @property
    def map(self):
        return self._map

    @property
    def map_inv(self):
        return self._map_inv

    def apply(self, x: bv.BlockVector) -> bv.BlockVector:
        """
        Return the corresponding `self.model.prop` vector
        """
        x_dict = bvec_to_dict(x)
        y_dict = self.map(x_dict)
        return dict_to_bvec(y_dict, self.y.labels)

    def apply_vjp(self, x: bv.BlockVector, hy: bv.BlockVector) -> bv.BlockVector:
        """
        """
        x_dict = bvec_to_dict(x)
        _hy = self.y.copy()
        _hy[:] = hy
        hy_dict = bvec_to_dict(_hy)
        _, vjp_fun = jax.vjp(self.map, x_dict)
        hx_dict = vjp_fun(hy_dict)[0]
        return dict_to_bvec(hx_dict, self.x.labels)

    def apply_jvp(self, x: bv.BlockVector, dx: bv.BlockVector) -> bv.BlockVector:
        """
        """
        x_dict = bvec_to_dict(x)
        dx_dict = bvec_to_dict(dx)
        _, dy_dict = jax.jvp(self.map, (x_dict,), (dx_dict,))
        return dict_to_bvec(dy_dict, self.y.labels)

class PredefJaxParametrization(JaxParameterization):
    """
    Map an alternative parameterization to a model's `prop` parameters

    The map here must be automatically defined through a `make_map` function
    which uses `jax` to create output vector
    """

    def __init__(
            self,
            model: Union[DynModel, TranModel],
            **kwargs
        ):
        map_x_to_y, map_y_to_x = self.make_map(model, **kwargs)
        super().__init__(map_x_to_y, map_y_to_x)

    @staticmethod
    def make_map(model, **kwargs):
        """
        Return a prototype input vector and `jax` function that performs the map
        """
        raise NotImplementedError()

    def apply(self, x: bv.BlockVector) -> bv.BlockVector:
        """
        Return the corresponding `self.model.prop` vector
        """
        x_dict = bvec_to_dict(x)
        y_dict = self.map(x_dict)
        return dict_to_bvec(y_dict, self.y.labels)

    def apply_vjp(self, x: bv.BlockVector, hy: bv.BlockVector) -> bv.BlockVector:
        """
        """
        x_dict = bvec_to_dict(x)
        _hy = self.y.copy()
        _hy[:] = hy
        hy_dict = bvec_to_dict(_hy)
        _, vjp_fun = jax.vjp(self.map, x_dict)
        hx_dict = vjp_fun(hy_dict)[0]
        return dict_to_bvec(hx_dict, self.x.labels)

    def apply_jvp(self, x: bv.BlockVector, dx: bv.BlockVector) -> bv.BlockVector:
        """
        """
        x_dict = bvec_to_dict(x)
        dx_dict = bvec_to_dict(dx)
        _, dy_dict = jax.jvp(self.map, (x_dict,), (dx_dict,))
        return dict_to_bvec(dy_dict, self.y.labels)

class Identity(PredefJaxParametrization):

    def __init__(
            self,
            model: Union[DynModel, TranModel]
        ):
        super().__init__(model)

    @staticmethod
    def make_map(model):
        def map(x):
            return x

        y = model.prop
        x = model.prop.copy()
        return (x, y, map), (y, x, map)

class LayerModuli(PredefJaxParametrization):

    def __init__(
            self,
            model: Union[DynModel, TranModel]
        ):
        super().__init__(self, model)

    @staticmethod
    def make_map(model):
        ## Get the mapping from labelled cell regions to DOFs
        cell_label_to_dofs = meshutils.process_celllabel_to_dofs_from_residual(
            model.solid.residual,
            model.solid.residual.form['coeff.prop.emod'].function_space()
        )

        y_dict = bvec_to_dict(model.prop)
        def map(x):

            emod = jnp.zeros(y_dict['emod'].size, copy=True)
            weight = jnp.zeros(y_dict['emod'].size, copy=True)
            for label, layer_stiffness in x.items():
                dofs = cell_label_to_dofs[label]

                weight_region = jnp.zeros(emod.shape)
                weight_region = weight_region.at[dofs][:] = layer_stiffness
                emod_region = layer_stiffness*weight_region

                emod = emod + emod_region
                weight = weight + weight_region

            new_y_dict = y_dict.copy()
            new_y_dict['emod'] = emod
            return new_y_dict

        ## Define the input vector
        labels = (tuple(cell_label_to_dofs.keys()),)
        subvecs = [np.zeros(1) for _ in labels[0]]
        in_vec = bv.BlockVector(subvecs, labels=labels)

        return (in_vec, model.prop.copy(), map), (None, None, None)

class ConstantSubset(PredefJaxParametrization):

    def __init__(
            self,
            model: Union[DynModel, TranModel],
            const_vals=None,
            scale=None
        ):
        super().__init__(model, const_vals=const_vals, scale=scale)

    @staticmethod
    def make_map(model, const_vals=None, scale=None):
        """"""
        x = model.prop.copy()
        y = x.copy()

        _scale = {key: 1.0 for key in x.labels[0]}
        if scale is not None:
            _scale.update(scale)
        scale = _scale

        if const_vals is None:
            const_vals = {}

        def map(x):
            """
            Return the x->y mapping
            """
            y = {}
            for key in x:
                if key in const_vals:
                    y[key] = const_vals[key]*np.ones(x[key].shape)
                else:
                    y[key] = scale[key]*x[key]
            return y

        def map_inv(y):
            """
            Return the y->x mapping
            """
            x = {}
            for key in y:
                if key in const_vals:
                    x[key] = const_vals[key]*np.ones(x[key].shape)
                else:
                    x[key] = 1/scale[key]*y[key]
            return x

        return (x, y, map), (y, x, map_inv)

def bvec_to_dict(x: bv.BlockVector) -> Mapping[str, np.ndarray]:
    return {label: subvec for label, subvec in x.sub_items()}

def dict_to_bvec(
        y: Mapping[str, np.ndarray],
        labels: Optional[typing.MultiLabels]=None
    ) -> bv.BlockVector:
    if labels is None:
        labels = (tuple(y.keys()), )
    subvecs = [y[label] for label in labels[0]]
    return bv.BlockVector(subvecs, labels=labels)