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
from femvf.meshutils import mesh_element_type_dim

DfnFunction = Union[ufl.Constant, dfn.Function]
FunctionLike = Union[ufl.Argument, dfn.Function, dfn.Constant]
FunctionSpace = Union[ufl.FunctionSpace, dfn.FunctionSpace]

CoefficientMapping = Mapping[str, DfnFunction]
FunctionSpaceMapping = Mapping[str, dfn.FunctionSpace]

## Utilities for handling Fenics functions

# These are used to treat `ufl.Constant` and `dfn.Function` uniformly

## Residual definitions


def get_measure(
    mesh_element_type: str | int,
    mesh: dfn.Mesh,
    mesh_functions: list[dfn.MeshFunction]
):
    """
    Return a measure

    Parameters
    ----------
    mesh_element_type: str | int
        The element type

        This can be a dimension (int) or a string (vertex, facet, cell)
    mesh: dfn.Mesh
        The mesh
    mesh_function: list[dfn.MeshFunction]
        A list of mesh functions for each dimension
    """
    mesh_dim = mesh.topology().dim()
    # NOTE: legacy Fenics only supports facet and cell integrals
    # The two empty strings
    INTEGRAL_TYPES = (mesh_dim-2)*('',) + ('ds', 'dx')

    el_dim = mesh_element_type_dim(mesh_element_type)

    measure = dfn.Measure(
        INTEGRAL_TYPES[el_dim], domain=mesh, subdomain_data=mesh_functions[el_dim]
    )
    return measure

def get_subdomain(
    mesh_element_type: str | int,
    mesh_subdomains: list[Mapping[str, int]]
):
    """
    Return a mesh subdomain mapping

    Parameters
    ----------
    mesh_element_type: str | int
        The element type

        This can be a dimension (int) or a string (vertex, facet, cell)
    mesh_subdomains: list[Mapping[str, int]]
        Mesh subdomain info for each dimension
    """

    el_dim = mesh_element_type_dim(mesh_element_type)
    return mesh_subdomains[el_dim]

def subdomain_measure(
        measure: dfn.Measure,
        subdomain_to_marker: Mapping[str, int],
        subdomain_id: str = 'everywhere'
    ):
    """
    Return the measure over a subdomain of the mesh

    Parameters
    ----------
    measure: dfn.Measure
        The measure
    subdomain_to_marker: Mapping[str, int]
        The dictionary of subdomain keys to mesh markers
    subdomain_id: str
        The subdomain key
    """
    if subdomain_id == 'everywhere':
        return measure('everywhere')
    else:
        return measure(subdomain_to_marker[subdomain_id])

# NOTE: All the forms below apply a traction over a facet subdomain named
# 'traction'
class Rayleigh(PredefinedFenicsResidual):

    def init_form(
        self,
        mesh: dfn.Mesh,
        mesh_functions: list[dfn.MeshFunction],
        mesh_subdomains: list[Mapping[str, int]]
    ):
        dx = get_measure('cell', mesh, mesh_functions)
        ds = get_measure('facet', mesh, mesh_functions)

        ds_subdomain = get_subdomain('facet', mesh_subdomains)
        ds_traction = subdomain_measure(ds, ds_subdomain, 'traction')

        form = (
            _form.InertialForm({}, dx, mesh)
            + _form.IsotropicElasticForm({}, dx, mesh)
            + _form.RayleighDampingForm({}, dx, mesh)
            - _form.SurfacePressureForm({}, ds_traction, mesh)
            - _form.ManualSurfaceContactTractionForm({}, ds_traction, mesh)
        )
        return form


class KelvinVoigt(PredefinedFenicsResidual):

    def init_form(
        self,
        mesh: dfn.Mesh,
        mesh_functions: list[dfn.MeshFunction],
        mesh_subdomains: list[Mapping[str, int]]
    ):
        dx = get_measure('cell', mesh, mesh_functions)
        ds = get_measure('facet', mesh, mesh_functions)

        ds_subdomain = get_subdomain('facet', mesh_subdomains)
        ds_traction = subdomain_measure(ds, ds_subdomain, 'traction')

        form = (
            _form.InertialForm({}, dx, mesh)
            + _form.KelvinVoigtForm({}, dx, mesh)
            + _form.IsotropicElasticForm({}, dx, mesh)
            - _form.SurfacePressureForm({}, ds_traction, mesh)
            - _form.ManualSurfaceContactTractionForm({}, ds_traction, mesh)
        )
        return form


class KelvinVoigtWShape(PredefinedFenicsResidual):

    def init_form(
        self,
        mesh: dfn.Mesh,
        mesh_functions: list[dfn.MeshFunction],
        mesh_subdomains: list[Mapping[str, int]]
    ):

        dx = get_measure('cell', mesh, mesh_functions)
        ds = get_measure('facet', mesh, mesh_functions)

        ds_subdomain = get_subdomain('facet', mesh_subdomains)
        ds_traction = subdomain_measure(ds, ds_subdomain, 'traction')

        form = (
            _form.InertialForm({}, dx, mesh)
            + _form.IsotropicElasticForm({}, dx, mesh)
            + _form.KelvinVoigtForm({}, dx, mesh)
            - _form.SurfacePressureForm({}, ds_traction, mesh)
            - _form.ManualSurfaceContactTractionForm({}, ds_traction, mesh)
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
        dx = get_measure('cell', mesh, mesh_functions)
        ds = get_measure('facet', mesh, mesh_functions)

        ds_subdomain = get_subdomain('facet', mesh_subdomains)
        ds_traction = subdomain_measure(ds, ds_subdomain, 'traction')

        form = (
            _form.InertialForm({}, dx, mesh)
            + _form.IsotropicMembraneForm({}, ds_traction, mesh)
            + _form.IsotropicElasticForm({}, dx, mesh)
            + _form.KelvinVoigtForm({}, dx, mesh)
            - _form.SurfacePressureForm({}, ds_traction, mesh)
            - _form.ManualSurfaceContactTractionForm({}, ds_traction, mesh)
        )
        return form


class IncompSwellingKelvinVoigt(PredefinedFenicsResidual):

    def init_form(
        self,
        mesh: dfn.Mesh,
        mesh_functions: list[dfn.MeshFunction],
        mesh_subdomains: list[Mapping[str, int]]
    ):
        dx = get_measure('cell', mesh, mesh_functions)
        ds = get_measure('facet', mesh, mesh_functions)

        ds_subdomain = get_subdomain('facet', mesh_subdomains)
        ds_traction = subdomain_measure(ds, ds_subdomain, 'traction')

        form = (
            _form.InertialForm({}, dx, mesh)
            + _form.IsotropicIncompressibleElasticSwellingForm({}, dx, mesh)
            + _form.KelvinVoigtForm({}, dx, mesh)
            - _form.SurfacePressureForm({}, ds_traction, mesh)
            - _form.ManualSurfaceContactTractionForm({}, ds_traction, mesh)
        )
        return form


class SwellingKelvinVoigt(PredefinedFenicsResidual):

    def init_form(
        self,
        mesh: dfn.Mesh,
        mesh_functions: list[dfn.MeshFunction],
        mesh_subdomains: list[Mapping[str, int]]
    ):
        dx = get_measure('cell', mesh, mesh_functions)
        ds = get_measure('facet', mesh, mesh_functions)

        ds_subdomain = get_subdomain('facet', mesh_subdomains)
        ds_traction = subdomain_measure(ds, ds_subdomain, 'traction')

        form = (
            _form.InertialForm({}, dx, mesh)
            + _form.IsotropicElasticSwellingForm({}, dx, mesh)
            + _form.KelvinVoigtForm({}, dx, mesh)
            - _form.SurfacePressureForm({}, ds_traction, mesh)
            - _form.ManualSurfaceContactTractionForm({}, ds_traction, mesh)
        )
        return form


class SwellingKelvinVoigtWEpithelium(PredefinedFenicsResidual):

    def init_form(
        self,
        mesh: dfn.Mesh,
        mesh_functions: list[dfn.MeshFunction],
        mesh_subdomains: list[Mapping[str, int]]
    ):
        dx = get_measure('cell', mesh, mesh_functions)
        ds = get_measure('facet', mesh, mesh_functions)

        ds_subdomain = get_subdomain('facet', mesh_subdomains)
        ds_traction = subdomain_measure(ds, ds_subdomain, 'traction')

        form = (
            _form.InertialForm({}, dx, mesh)
            + _form.IsotropicMembraneForm({}, ds_traction, mesh)
            + _form.IsotropicElasticSwellingForm({}, dx, mesh)
            + _form.KelvinVoigtForm({}, dx, mesh)
            - _form.SurfacePressureForm({}, ds_traction, mesh)
            - _form.ManualSurfaceContactTractionForm({}, ds_traction, mesh)
        )
        return form


class SwellingKelvinVoigtWEpitheliumNoShape(PredefinedFenicsResidual):

    def init_form(
        self,
        mesh: dfn.Mesh,
        mesh_functions: list[dfn.MeshFunction],
        mesh_subdomains: list[Mapping[str, int]]
    ):
        dx = get_measure('cell', mesh, mesh_functions)
        ds = get_measure('facet', mesh, mesh_functions)

        ds_subdomain = get_subdomain('facet', mesh_subdomains)
        ds_traction = subdomain_measure(ds, ds_subdomain, 'traction')

        form = (
            _form.InertialForm({}, dx, mesh)
            + _form.IsotropicMembraneForm({}, ds_traction, mesh)
            + _form.IsotropicElasticSwellingForm({}, dx, mesh)
            + _form.KelvinVoigtForm({}, dx, mesh)
            - _form.SurfacePressureForm({}, ds_traction, mesh)
            - _form.ManualSurfaceContactTractionForm({}, ds_traction, mesh)
        )
        return form


class SwellingPowerLawKelvinVoigtWEpitheliumNoShape(PredefinedFenicsResidual):

    def init_form(
        self,
        mesh: dfn.Mesh,
        mesh_functions: list[dfn.MeshFunction],
        mesh_subdomains: list[Mapping[str, int]]
    ):
        dx = get_measure('cell', mesh, mesh_functions)
        ds = get_measure('facet', mesh, mesh_functions)

        ds_subdomain = get_subdomain('facet', mesh_subdomains)
        ds_traction = subdomain_measure(ds, ds_subdomain, 'traction')

        form = (
            _form.InertialForm({}, dx, mesh)
            + _form.IsotropicMembraneForm({}, ds_traction, mesh)
            + _form.IsotropicElasticSwellingPowerLawForm({}, dx, mesh)
            + _form.KelvinVoigtForm({}, dx, mesh)
            - _form.SurfacePressureForm({}, ds_traction, mesh)
            - _form.ManualSurfaceContactTractionForm({}, ds_traction, mesh)
        )
        return form


class Approximate3DKelvinVoigt(PredefinedFenicsResidual):

    def init_form(
        self,
        mesh: dfn.Mesh,
        mesh_functions: list[dfn.MeshFunction],
        mesh_subdomains: list[Mapping[str, int]]
    ):
        dx = get_measure('cell', mesh, mesh_functions)
        ds = get_measure('facet', mesh, mesh_functions)

        ds_subdomain = get_subdomain('facet', mesh_subdomains)
        ds_traction = subdomain_measure(ds, ds_subdomain, 'traction')

        form = (
            _form.InertialForm({}, dx, mesh)
            + _form.IsotropicMembraneForm({}, ds_traction, mesh)
            + _form.IsotropicElasticForm({}, dx, mesh)
            - _form.APForceForm({}, dx, mesh)
            + _form.KelvinVoigtForm({}, dx, mesh)
            - _form.SurfacePressureForm({}, ds_traction, mesh)
            - _form.ManualSurfaceContactTractionForm({}, ds_traction, mesh)
        )
        return form
