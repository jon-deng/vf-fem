"""
Contains definitions of parametrizations. These objects should provide a mapping from their specific parameters to standardized parameters of the forward model, as well as the derivative of the map.
"""

import numpy as np
from jax import numpy as jnp
import jax
import dolfin as dfn
import ufl
from femvf import meshutils
from petsc4py import PETSc

from blockarray import blockvec as bv

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

    def __init__(self, model, out_default, *args, **kwargs):
        self.model = model
        self._out_labels = model.props.labels
        self._out_default = out_default
        self.in_vec, self.map = self.make_map()

    def make_map(self):
        raise NotImplementedError()

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

class LayerModuli(Parameterization):

    def make_map(self):
        ## Get the mapping from labelled cell regions to DOFs
        # mesh = self.model.solid.forms['mesh.mesh']
        # cell_func = self.model.solid.forms['mesh.cell_function']
        # cell_label_to_id = self.model.solid.forms['mesh.cell_label_to_id']
        E = self.model.solid.forms['coeff.prop.emod']
        cell_label_to_dofs = meshutils.process_celllabel_to_dofs_from_forms(
            self.model.solid.forms,
            E.function_space()
        )

        out_labels = self._out_labels
        out_subvecs = [self._out_default[label] for label in out_labels]
        out_dict = {label: subvec for label, subvec in zip(out_subvecs, out_labels)}
        def map(x):

            new_emod = jnp.array(out_dict['emod'], copy=True)
            for label, value in x.items():
                dofs = cell_label_to_dofs[label]
                new_emod.at[dofs] = value

            out_dict['emod'] = new_emod
            return out_dict

        ## Define the input vector
        labels = list(cell_label_to_dofs.keys())
        subvecs = [np.zeros(1) for _ in labels]
        in_vec = bv.BlockVector(subvecs, labels=labels)

        return in_vec, map

