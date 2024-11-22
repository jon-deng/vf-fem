"""
Base residual class definitions
"""

from typing import Callable, Tuple, Mapping, Union, Optional, Any
from numpy.typing import NDArray

import dolfin as dfn
import numpy as np

from femvf.equations.form import UFLForm
from femvf.meshutils import mesh_element_type_dim


class BaseResidual:
    pass
    # _res: Any

# A `DirichletBCTuple` consists of:
# (BC value, mesh element to apply over, subdomain str)
DirichletBCTuple = tuple[Any, str, str]

class FenicsResidual(BaseResidual):
    """
    Representation of a (non-linear) residual in `Fenics`

    This take a pure UFL form, then adds mesh and boundary condition information so that
    numerical integration can be used to evaluate a residual vector.
    """

    def __init__(
        self,
        form: UFLForm,
        mesh: dfn.Mesh,
        mesh_functions: list[dfn.MeshFunction],
        mesh_subdomains: list[Mapping[str, int]],
        dirichlet_bcs: Optional[dict[str, list[DirichletBCTuple]]] = None
    ):

        self._mesh = mesh
        self._ref_mesh_coords = np.array(mesh.coordinates())
        self._form = form

        self._mesh_functions = mesh_functions
        self._mesh_subdomains = mesh_subdomains

        zero_value = dfn.Constant(mesh.topology().dim() * [0.0])
        if dirichlet_bcs is None:
            dirichlet_bcs = {
                'coeff.state.u1': [(zero_value, 'facet', 'fixed')]
            }

        self._dirichlet_bcs = {
            coeff_key: tuple(
                dfn.DirichletBC(
                    form[coeff_key].function_space(),
                    value,
                    self.mesh_function(element_type),
                    self.mesh_subdomain(element_type)[subdomain]
                )
                for (value, element_type, subdomain) in dirichlet_bc_tuples
            )
            for coeff_key, dirichlet_bc_tuples in dirichlet_bcs.items()
        }

    @property
    def form(self) -> UFLForm:
        return self._form

    def mesh(self) -> dfn.Mesh:
        return self._mesh

    @property
    def ref_mesh_coords(self) -> NDArray[float]:
        """
        Return the original/reference mesh coordinates

        These are the mesh coordinates for zero mesh motion
        """
        return self._ref_mesh_coords

    def mesh_function(self, mesh_element_type: Union[str, int]) -> dfn.MeshFunction:
        idx = mesh_element_type_dim(mesh_element_type)
        return self._mesh_functions[idx]

    def mesh_subdomain(
        self, mesh_element_type: Union[str, int]
    ) -> Mapping[str, int]:
        idx = mesh_element_type_dim(mesh_element_type)
        return self._mesh_subdomains[idx]

    def measure(self, integral_type: str):
        if integral_type == 'dx':
            mf = self.mesh_function('cell')
        elif integral_type == 'ds':
            mf = self.mesh_function('facet')
        else:
            raise ValueError("Unknown `integral_type` '{integral_type}'")
        return dfn.Measure(integral_type, self.mesh(), subdomain_data=mf)

    @property
    def dirichlet_bcs(self):
        return self._dirichlet_bcs

    @property
    def fsi_facet_labels(self):
        return self._fsi_facet_labels

    @property
    def fixed_facet_labels(self):
        return self._fixed_facet_labels


# TODO: Formalize the jax residual more with argument size/shape definitions?
# The Jax residual is kind of fragile since you just have to create arrays
# and arguments etc. in the right sizes and shapes based on how you coded the
# `res` function

ResArgs = Tuple[Mapping[str, NDArray], ...]
ResReturn = Mapping[str, NDArray]

class JaxResidual(BaseResidual):
    """
    Representation of a (non-linear) residual in `JAX`
    """

    # TODO: Document/refactor the format of res and res_args?
    def __init__(self, res: Callable[[ResArgs], ResReturn], res_args: ResArgs):

        self._res = res
        self._res_args = res_args

    @property
    def res(self):
        return self._res

    @property
    def res_args(self):
        return self._res_args
