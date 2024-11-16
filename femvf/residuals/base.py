"""
Base residual class definitions
"""

from typing import Callable, Tuple, Mapping, Union, Optional
from numpy.typing import NDArray

import dolfin as dfn
import numpy as np

from femvf.equations.form import FenicsForm


class BaseResidual:
    pass
    # _res: Any


class FenicsResidual(BaseResidual):
    """
    Representation of a (non-linear) residual in `Fenics`

    This includes additional information defining the residual such as the mesh,
    mesh functions, Dirichlet boundary conditions, etc.
    """

    def __init__(
        self,
        form: FenicsForm,
        mesh: dfn.Mesh,
        # TODO: Remove the below 4 (mesh_functions, ..., fixed_facet labels)
        # These are redundant since you should be able to totally evaluate
        # the residual given just the FenicsForm, a mesh, and dirichlet boundary
        # conditions!
        mesh_functions: list[dfn.MeshFunction],
        mesh_functions_label_to_value: list[Mapping[str, int]],
        fsi_facet_labels: list[str],
        fixed_facet_labels: list[str],
        dirichlet_bcs: Optional[dict[str, dfn.DirichletBC]]=None
    ):

        self._mesh = mesh
        self._ref_mesh_coords = np.array(mesh.coordinates())
        self._form = form

        self._mesh_functions = mesh_functions
        self._mesh_functions_label_to_value = mesh_functions_label_to_value

        fixed_subdomain_idxs = [
            self.mesh_function_label_to_value('facet')[facet_label]
            for facet_label in fixed_facet_labels
        ]
        # TODO: Refactor `FenicsResidual` without this super-specific key requirement!
        # The class should work in general for any input form
        fun_space = self.form['coeff.state.u1'].function_space()
        fixed_dis = dfn.Constant(mesh.topology().dim() * [0.0])
        if dirichlet_bcs is None:
            dirichlet_bcs = {
                'coeff.state.u1': tuple(
                    dfn.DirichletBC(
                        fun_space, fixed_dis, self.mesh_function('facet'), fixed_subdomain_idx
                    )
                    for fixed_subdomain_idx in fixed_subdomain_idxs
                )
            }
        self._dirichlet_bcs = dirichlet_bcs

        self._fsi_facet_labels = fsi_facet_labels
        self._fixed_facet_labels = fixed_facet_labels

    @property
    def form(self) -> FenicsForm:
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

    @staticmethod
    def _mesh_element_type_to_idx(mesh_element_type: Union[str, int]) -> int:
        if isinstance(mesh_element_type, str):
            if mesh_element_type == 'vertex':
                return 0
            elif mesh_element_type == 'facet':
                return -2
            elif mesh_element_type == 'cell':
                return -1
        elif isinstance(mesh_element_type, int):
            return mesh_element_type
        else:
            raise TypeError(
                f"`mesh_element_type` must be `str` or `int`, not `{type(mesh_element_type)}`"
            )

    def mesh_function(self, mesh_element_type: Union[str, int]) -> dfn.MeshFunction:
        idx = self._mesh_element_type_to_idx(mesh_element_type)
        return self._mesh_functions[idx]

    def mesh_function_label_to_value(
        self, mesh_element_type: Union[str, int]
    ) -> Mapping[str, int]:
        idx = self._mesh_element_type_to_idx(mesh_element_type)
        return self._mesh_functions_label_to_value[idx]

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


class PredefinedFenicsResidual(FenicsResidual):
    """
    Class representing a pre-defined residual
    """

    def __init__(
        self,
        mesh: dfn.Mesh,
        mesh_functions: list[dfn.MeshFunction],
        mesh_functions_label_to_value: list[Mapping[str, int]],
        fsi_facet_labels: list[str],
        fixed_facet_labels: list[str],
    ):

        functional = self._make_functional(
            mesh,
            mesh_functions,
            mesh_functions_label_to_value,
            fsi_facet_labels,
            fixed_facet_labels,
        )
        super().__init__(
            functional,
            mesh,
            mesh_functions,
            mesh_functions_label_to_value,
            fsi_facet_labels,
            fixed_facet_labels,
        )

    def _make_functional(
        self,
        mesh: dfn.Mesh,
        mesh_functions: list[dfn.MeshFunction],
        mesh_functions_label_to_value: list[Mapping[str, int]],
        fsi_facet_labels: list[str],
        fixed_facet_labels: list[str],
    ) -> dfn.Form:
        raise NotImplementedError()


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


class PredefinedJaxResidual(JaxResidual):
    """
    Predefined `JaxResidual`
    """

    def __init__(self, mesh: NDArray, *args, **kwargs):
        res, res_args = self._make_residual(mesh, *args, **kwargs)
        super().__init__(res, res_args)

        self._mesh = mesh

    def mesh(self):
        return self._mesh

    def _make_residual(self, mesh, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

