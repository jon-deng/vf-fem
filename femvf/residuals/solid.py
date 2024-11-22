"""
Contains definitions of different solid model forms

References
----------
[Gou2016] Gou, K., Pence, T.J. Hyperelastic modeling of swelling in fibrous soft tissue with application to tracheal angioedema. J. Math. Biol. 72, 499-526 (2016). https://doi.org/10.1007/s00285-015-0893-0
"""

from typing import Mapping, Union

import operator
import warnings
from functools import reduce

import numpy as np
import dolfin as dfn
import ufl

from .base import FenicsResidual, PredefinedFenicsResidual

from femvf.equations import form as _form

DfnFunction = Union[ufl.Constant, dfn.Function]
FunctionLike = Union[ufl.Argument, dfn.Function, dfn.Constant]
FunctionSpace = Union[ufl.FunctionSpace, dfn.FunctionSpace]

CoefficientMapping = Mapping[str, DfnFunction]
FunctionSpaceMapping = Mapping[str, dfn.FunctionSpace]

## Utilities for handling Fenics functions

# These are used to treat `ufl.Constant` and `dfn.Function` uniformly

## Residual definitions


def _process_measures(
    mesh: dfn.Mesh,
    mesh_functions: list[dfn.MeshFunction],
    mesh_subdomains: list[Mapping[str, int]],
    fsi_facet_labels: list[str],
    fixed_facet_labels: list[str],
):
    if len(mesh_functions) == 3:
        vertex_func, facet_func, cell_func = mesh_functions
        vertex_label_to_id, facet_label_to_id, cell_label_to_id = (
            mesh_subdomains
        )
    elif len(mesh_functions) == 4:
        vertex_func, edge_func, facet_func, cell_func = mesh_functions
        vertex_label_to_id, edge_label_to_id, facet_label_to_id, cell_label_to_id = (
            mesh_subdomains
        )
    else:
        raise ValueError(f"`mesh_functions` has length {len(mesh_functions):d}")

    dx = dfn.Measure('dx', domain=mesh, subdomain_data=cell_func)
    ds = dfn.Measure('ds', domain=mesh, subdomain_data=facet_func)
    _traction_ds = [
        ds(int(facet_label_to_id[facet_label])) for facet_label in fsi_facet_labels
    ]
    traction_ds = reduce(operator.add, _traction_ds)
    return dx, ds, traction_ds


class Rayleigh(PredefinedFenicsResidual):

    def init_form(
        self,
        mesh: dfn.Mesh,
        mesh_functions: list[dfn.MeshFunction],
        mesh_subdomains: list[Mapping[str, int]]
    ):
        dx, ds, traction_ds = _process_measures(
            mesh,
            mesh_functions,
            mesh_subdomains
        )

        form = (
            _form.InertialForm({}, dx, mesh)
            + _form.IsotropicElasticForm({}, dx, mesh)
            + _form.RayleighDampingForm({}, dx, mesh)
            - _form.SurfacePressureForm({}, traction_ds, mesh)
            - _form.ManualSurfaceContactTractionForm({}, traction_ds, mesh)
        )
        return form


class KelvinVoigt(PredefinedFenicsResidual):

    def init_form(
        self,
        mesh: dfn.Mesh,
        mesh_functions: list[dfn.MeshFunction],
        mesh_subdomains: list[Mapping[str, int]]
    ):

        dx, ds, traction_ds = _process_measures(
            mesh,
            mesh_functions,
            mesh_subdomains
        )

        form = (
            _form.InertialForm({}, dx, mesh)
            + _form.KelvinVoigtForm({}, dx, mesh)
            + _form.IsotropicElasticForm({}, dx, mesh)
            - _form.SurfacePressureForm({}, traction_ds, mesh)
            - _form.ManualSurfaceContactTractionForm({}, traction_ds, mesh)
        )
        return form


class KelvinVoigtWShape(PredefinedFenicsResidual):

    def init_form(
        self,
        mesh: dfn.Mesh,
        mesh_functions: list[dfn.MeshFunction],
        mesh_subdomains: list[Mapping[str, int]]
    ):

        dx, ds, traction_ds = _process_measures(
            mesh,
            mesh_functions,
            mesh_subdomains
        )

        form = (
            _form.InertialForm({}, dx, mesh)
            + _form.IsotropicElasticForm({}, dx, mesh)
            + _form.KelvinVoigtForm({}, dx, mesh)
            - _form.SurfacePressureForm({}, traction_ds, mesh)
            - _form.ManualSurfaceContactTractionForm({}, traction_ds, mesh)
            - _form.ShapeForm({}, dx, mesh)
        )
        return form


class KelvinVoigtWEpithelium(PredefinedFenicsResidual):

    def init_form(
        self,
        mesh: dfn.Mesh,
        mesh_functions: list[dfn.MeshFunction],
        mesh_subdomains: list[Mapping[str, int]]
    ):
        dx, ds, traction_ds = _process_measures(
            mesh,
            mesh_functions,
            mesh_subdomains
        )

        form = (
            _form.InertialForm({}, dx, mesh)
            + _form.IsotropicMembraneForm({}, traction_ds, mesh)
            + _form.IsotropicElasticForm({}, dx, mesh)
            + _form.KelvinVoigtForm({}, dx, mesh)
            - _form.SurfacePressureForm({}, traction_ds, mesh)
            - _form.ManualSurfaceContactTractionForm({}, traction_ds, mesh)
        )
        return form


class IncompSwellingKelvinVoigt(PredefinedFenicsResidual):

    def init_form(
        self,
        mesh: dfn.Mesh,
        mesh_functions: list[dfn.MeshFunction],
        mesh_subdomains: list[Mapping[str, int]]
    ):
        dx, ds, traction_ds = _process_measures(
            mesh,
            mesh_functions,
            mesh_subdomains
        )

        form = (
            _form.InertialForm({}, dx, mesh)
            + _form.IsotropicIncompressibleElasticSwellingForm({}, dx, mesh)
            + _form.KelvinVoigtForm({}, dx, mesh)
            - _form.SurfacePressureForm({}, traction_ds, mesh)
            - _form.ManualSurfaceContactTractionForm({}, traction_ds, mesh)
        )
        return form


class SwellingKelvinVoigt(PredefinedFenicsResidual):

    def init_form(
        self,
        mesh: dfn.Mesh,
        mesh_functions: list[dfn.MeshFunction],
        mesh_subdomains: list[Mapping[str, int]]
    ):
        dx, ds, traction_ds = _process_measures(
            mesh,
            mesh_functions,
            mesh_subdomains
        )

        form = (
            _form.InertialForm({}, dx, mesh)
            + _form.IsotropicElasticSwellingForm({}, dx, mesh)
            + _form.KelvinVoigtForm({}, dx, mesh)
            - _form.SurfacePressureForm({}, traction_ds, mesh)
            - _form.ManualSurfaceContactTractionForm({}, traction_ds, mesh)
        )
        return form


class SwellingKelvinVoigtWEpithelium(PredefinedFenicsResidual):

    def init_form(
        self,
        mesh: dfn.Mesh,
        mesh_functions: list[dfn.MeshFunction],
        mesh_subdomains: list[Mapping[str, int]]
    ):
        dx, ds, traction_ds = _process_measures(
            mesh,
            mesh_functions,
            mesh_subdomains
        )

        form = (
            _form.InertialForm({}, dx, mesh)
            + _form.IsotropicMembraneForm({}, traction_ds, mesh)
            + _form.IsotropicElasticSwellingForm({}, dx, mesh)
            + _form.KelvinVoigtForm({}, dx, mesh)
            - _form.SurfacePressureForm({}, traction_ds, mesh)
            - _form.ManualSurfaceContactTractionForm({}, traction_ds, mesh)
        )
        return form


class SwellingKelvinVoigtWEpitheliumNoShape(PredefinedFenicsResidual):

    def init_form(
        self,
        mesh: dfn.Mesh,
        mesh_functions: list[dfn.MeshFunction],
        mesh_subdomains: list[Mapping[str, int]]
    ):
        dx, ds, traction_ds = _process_measures(
            mesh,
            mesh_functions,
            mesh_subdomains
        )

        form = (
            _form.InertialForm({}, dx, mesh)
            + _form.IsotropicMembraneForm({}, traction_ds, mesh)
            + _form.IsotropicElasticSwellingForm({}, dx, mesh)
            + _form.KelvinVoigtForm({}, dx, mesh)
            - _form.SurfacePressureForm({}, traction_ds, mesh)
            - _form.ManualSurfaceContactTractionForm({}, traction_ds, mesh)
        )
        return form


class SwellingPowerLawKelvinVoigtWEpitheliumNoShape(PredefinedFenicsResidual):

    def init_form(
        self,
        mesh: dfn.Mesh,
        mesh_functions: list[dfn.MeshFunction],
        mesh_subdomains: list[Mapping[str, int]]
    ):
        dx, ds, traction_ds = _process_measures(
            mesh,
            mesh_functions,
            mesh_subdomains
        )

        form = (
            _form.InertialForm({}, dx, mesh)
            + _form.IsotropicMembraneForm({}, traction_ds, mesh)
            + _form.IsotropicElasticSwellingPowerLawForm({}, dx, mesh)
            + _form.KelvinVoigtForm({}, dx, mesh)
            - _form.SurfacePressureForm({}, traction_ds, mesh)
            - _form.ManualSurfaceContactTractionForm({}, traction_ds, mesh)
        )
        return form


class Approximate3DKelvinVoigt(PredefinedFenicsResidual):

    def init_form(
        self,
        mesh: dfn.Mesh,
        mesh_functions: list[dfn.MeshFunction],
        mesh_subdomains: list[Mapping[str, int]]
    ):
        dx, ds, traction_ds = _process_measures(
            mesh,
            mesh_functions,
            mesh_subdomains
        )

        form = (
            _form.InertialForm({}, dx, mesh)
            + _form.IsotropicMembraneForm({}, traction_ds, mesh)
            + _form.IsotropicElasticForm({}, dx, mesh)
            - _form.APForceForm({}, dx, mesh)
            + _form.KelvinVoigtForm({}, dx, mesh)
            - _form.SurfacePressureForm({}, traction_ds, mesh)
            - _form.ManualSurfaceContactTractionForm({}, traction_ds, mesh)
        )
        return form
