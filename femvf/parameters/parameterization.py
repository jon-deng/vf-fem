"""
Contains definitions of parametrizations.

These objects should mappings from a given parameter set to the standard
properties of the forward model.
"""

from typing import Mapping, Union, Optional, Tuple

import numpy as np
from jax import numpy as jnp
import jax
import dolfin as dfn
import ufl
from femvf.models.transient.base import BaseTransientModel as TranModel
from femvf.models.dynamical.base import BaseDynamicalModel as DynModel
from femvf.models.assemblyutils import CachedFormAssembler
from femvf import meshutils
from petsc4py import PETSc

from blockarray import blockvec as bv, blockarray as ba
from blockarray import typing

class BaseParameterization:
    """
    Map a parameterization to a model's `.props` parameter set

    Parameters
    ----------
    model :
        The model to convert parameters to
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

    @property
    def x(self):
        """
        Return a prototype input vector for the parameterization
        """
        # raise NotImplementedError()
        return self._x_vec

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


class BaseDolfinParameterization(BaseParameterization):
    def __init__(
            self,
            model: Union[DynModel, TranModel]
        ):
        self.model = model

        _y_vec = ba.zeros(model.props.bshape)
        self._y_vec = bv.BlockVector(
            _y_vec.sub_blocks, labels=model.props.labels
        )

        # NOTE: Subclasses have to supply a `self._x_vec` attribute

class TractionShape(BaseDolfinParameterization):
    def __init__(
            self,
            model: Union[DynModel, TranModel],
            lame_lambda=1.0, lame_mu=1.0
        ):
        super().__init__(model)

        # The input vector simply renames the mesh displacement vector
        _x_vec = ba.zeros(model.props.f_bshape)
        x_subvecs = _x_vec.sub_blocks
        x_labels = list(model.props.labels[0])
        try:
            ii = x_labels.index('umesh')
        except ValueError as err:
            raise ValueError("model properties does not contain a shape") from err

        x_labels[ii] = 'tmesh'
        self._x_vec = bv.BlockVector(x_subvecs, labels=(tuple(x_labels),))

        ## Define the stiffness matrix and traction sensitivity matrices that
        # The two maps are
        # dF/du : sensitivity of residual to mesh displacement
        # dF/dt : sensitivity of residual to medial surface tractions
        fspace = model.solid.forms['fspace.vector']
        bc_dir = model.solid.forms['bc.dirichlet']
        dx = model.solid.forms['measure.dx']
        ds = model.solid.forms['measure.ds_traction']

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

        form_dF_dt = ufl.inner(trial, test)*ds
        mat_dF_dt = dfn.assemble(
            form_dF_dt,
            tensor=dfn.PETScMatrix(),
            keep_diagonal=True
        )

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
                y_dict[key][:] = val

        return x_dict, y_dict

    def apply(self, x: bv.BlockVector) -> bv.BlockVector:
        """
        Return the corresponding `self.model.props` vector
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
        Return the corresponding `self.model.props` vector
        """
        self.y[:] = hy
        hy_dict, hx_dict = self._set_y_defaults_from_x(self.y, self.x)

        # Assemble the RHS for the given medial surface traction
        self.umesh.vector()[:] = hy_dict['umesh']

        # Solve for the mesh displacement
        dfn.solve(self.mat_dF_du, self.tmesh.vector(), self.umesh.vector(), 'lu')
        hx_dict['tmesh'][:] = self.mat_dF_dt*self.tmesh.vector()
        return dict_to_bvec(hx_dict, self.x.labels)

    def apply_jvp(self, x, dx) -> bv.BlockVector:
        """
        Return the corresponding `self.model.props` vector
        """
        self.x[:] = dx
        dx_dict, dy_dict = self._set_y_defaults_from_x(self.x, self.y)

        # Assemble the RHS for the given medial surface traction
        self.tmesh.vector()[:] = dx_dict['tmesh']
        rhs = self.mat_dF_dt * self.tmesh.vector()

        # Solve for the mesh displacement
        dfn.solve(self.mat_dF_du, self.umesh.vector(), rhs, 'lu')
        dy_dict['umesh'][:] = self.umesh.vector()[:]
        return dict_to_bvec(dy_dict, self.y.labels)


class BaseJaxParameterization(BaseParameterization):
    """
    Map an alternative parameterization to a model's `props` parameters

    The map here must be automatically defined through a `make_map` function
    which uses `jax` to create output vector
    """

    def __init__(
            self,
            model: Union[DynModel, TranModel],
            **kwargs
        ):
        _x_vec, self.map = self.make_map(model, **kwargs)
        x_subvecs = ba.zeros(_x_vec.f_bshape).sub_blocks
        x_labels = _x_vec.labels
        self._x_vec = bv.BlockVector(x_subvecs, labels=x_labels)
        # self._x_labels = self._x_vec.labels

        _y_vec = ba.zeros(model.props.bshape)
        self._y_vec = bv.BlockVector(
            _y_vec.sub_blocks, labels=model.props.labels
        )

    @staticmethod
    def make_map(model, **kwargs):
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

class Identity(BaseJaxParameterization):

    @staticmethod
    def make_map(model, **kwargs):
        def map(x):
            return x

        y = model.props
        return y, map

class LayerModuli(BaseJaxParameterization):

    @staticmethod
    def make_map(model, **kwargs):
        ## Get the mapping from labelled cell regions to DOFs
        E = model.solid.forms['coeff.prop.emod']
        cell_label_to_dofs = meshutils.process_celllabel_to_dofs_from_forms(
            model.solid.forms,
            E.function_space()
        )

        _y_dict = bvec_to_dict(model.props)
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

class ConstantSubset(BaseJaxParameterization):
    def __init__(
            self,
            model: Union[DynModel, TranModel],
            const_vals=None,
            scale=None
        ):
        super().__init__(model, const_vals=const_vals, scale=scale)

    @staticmethod
    def make_map(model, const_vals=None, scale=None):
        if scale is None:
            scale = {key: 1 for key in model.props.labels[0]}

        if const_vals is None:
            const_vals = {}

        def map(x):
            y = {}
            for key in x:
                if key in const_vals:
                    y[key] = const_vals[key]*np.ones(x[key].shape)
                else:
                    y[key] = scale[key]*x[key]
            return y

        y = model.props
        return y, map

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