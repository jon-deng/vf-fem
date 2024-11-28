"""
Functionality for creating model objects from meshes, etc.
"""

from os import path
from typing import Union, Optional, Any
from numpy.typing import NDArray
import numpy as np
import dolfin as dfn

from . import meshutils
from femvf.residuals import solid as slr, fluid as flr
from .models import transient
from .models import dynamical

SolidModel = Union[transient.FenicsModel, dynamical.FenicsModel]
FluidModel = Union[transient.JaxModel, dynamical.JaxModel]
FluidClass = Union[type[transient.JaxModel], type[dynamical.JaxModel]]

Labels = list[str]


def load_fenics_model(
    mesh: str | tuple[dfn.Mesh, list[dfn.MeshFunction], list[dict[str, int]]],
    Residual: type[slr.PredefinedSolidResidual],
    model_type: str = 'transient',
    **kwargs: dict[str, Any]
) -> SolidModel:
    """
    Load a solid model

    Parameters
    ----------
    mesh: str | tuple[dfn.Mesh, list[dfn.MeshFunction], list[dict[str, int]]]
        A mesh file path or mesh information tuple

        Only GMSH meshes are currently supported
    Residual: slr.PredefinedSolidResidual
        A predefined Fenics residual class
    model_type: str
        Type of model to load ('transient', 'dynamical', 'linearized_dynamical')
    **kwargs:
        Additional keyword args for the residual
    """
    if isinstance(mesh, str):
        ext = path.splitext(mesh)[1]
        if ext.lower() == '.msh':
            mesh, mesh_funcs, mesh_subdomains = meshutils.load_fenics_gmsh(mesh)
        else:
            raise ValueError(f"Invalid mesh extension {ext}")
    elif isinstance(mesh, (tuple, list)):
        mesh, mesh_funcs, mesh_subdomains = mesh
    else:
        raise TypeError(f"Invalid `mesh` type {type(mesh)}")

    residual = Residual(mesh, mesh_funcs, mesh_subdomains, **kwargs)
    if model_type == 'transient':
        return transient.FenicsModel(residual)
    elif model_type == 'dynamical':
        return dynamical.FenicsModel(residual)
    elif model_type == 'linearized_dynamical':
        return dynamical.LinearizedFenicsModel(residual)
    else:
        raise ValueError(f"Invalid model type {model_type}")


def load_jax_model(
    mesh: NDArray,
    Residual: type[flr.PredefinedFluidResidual],
    model_type: str = 'transient',
    **kwargs
) -> FluidModel:
    """
    Load a JAX model of the specified type

    Parameters
    ----------
    mesh: NDArray
        An NDArray representing mesh coordinates
    Residual: flr.PredefinedFluidResidual
        A predefined JAX residual class
    model_type: str
        Type of model to load ('transient', 'dynamical', 'linearized_dynamical')
    **kwargs:
        Additional keyword args for the residual
    """
    residual = Residual(mesh, **kwargs)

    if model_type == 'transient':
        return transient.JaxModel(residual)
    elif model_type == 'dynamical':
        return dynamical.JaxModel(residual)
    elif model_type == 'linearized_dynamical':
        return dynamical.LinearizedJaxModel(residual)
    else:
        raise ValueError(f"Invalid model type {model_type}")


def load_fsi_model(
    solid_mesh: str | tuple[dfn.Mesh, list[dfn.MeshFunction], list[dict[str, int]]],
    SolidResidual: type[slr.PredefinedSolidResidual],
    FluidResidual: type[flr.PredefinedFluidResidual],
    solid_kwargs: dict[str, Any],
    fluid_kwargs: dict[str, Any],
    model_type: str = 'transient',
    coupling: str = 'explicit',
    zs: Optional[tuple[float]] = None,
) -> transient.BaseTransientFSIModel:
    """
    Load a coupled (fsi) model

    Parameters
    ----------
    solid_mesh : str
        Path to the solid mesh
    SolidResidual, FluidResidual:
        Classes of the solid and fluid models to load
    fsi_facet_labels, fixed_facet_labels:
        String identifiers for facets corresponding to traction/dirichlet
        conditions
    separation_vertex_label:
        A string corresponding to a labelled vertex where separation should
        occur. This is only relevant for quasi-static fluid models with a fixed
        separation point
    coupling: str
        One of 'explicit' or 'implicit' indicating the coupling strategy between
        fluid/solid domains
    """
    ## Load the solid
    solid = load_fenics_model(
        solid_mesh, SolidResidual, model_type=model_type, **solid_kwargs
    )

    # TODO: Refactor hard-coded keys ('traction' ...)!
    mesh = solid.residual.mesh()
    dim = mesh.topology().dim()
    facet_mesh_func = solid.residual.mesh_function('facet')
    filtering_edge_values = set(
        solid.residual.mesh_subdomain('facet')[name] for name in ['traction']
    )

    def filter_edges(edges, origin, normal):
        filtered_edges = meshutils.filter_mesh_entities_by_subdomain(
            edges, facet_mesh_func, filtering_edge_values
        )
        filtered_edges = meshutils.filter_mesh_entities_by_plane(
            filtered_edges, origin, normal
        )
        return filtered_edges

    if dim == 2:
        fsi_edges = [
            edge.index() for edge in filter_edges(
                dfn.edges(mesh), np.zeros(2), np.zeros(2)
            )
        ]
        s, fsi_verts = derive_1dfluidmesh_from_edges(mesh, fsi_edges)
    elif dim == 3:
        fsi_edges = [
            [
                edge.index() for edge in filter_edges(
                    dfn.edges(mesh), np.array([0, 0, z]), np.array([0, 0, 1])
                )
            ]
            for z in zs
        ]
        mesh_list = [derive_1dfluidmesh_from_edges(mesh, edges) for edges in fsi_edges]
        s = np.array([s for s, fsi_verts in mesh_list])
        fsi_verts = np.array([fsi_verts for s, fsi_verts in mesh_list], dtype=int)
    else:
        raise ValueError(f"Invalid mesh dimension {dim}")

    fluid = load_jax_model(s, FluidResidual, model_type=model_type, **fluid_kwargs)

    dofs_fsi_solid = dfn.vertex_to_dof_map(
        solid.residual.form['coeff.fsi.p1'].function_space()
    )[fsi_verts.flat]
    dofs_fsi_fluid = (
        np.ones(dofs_fsi_solid.shape[:-1], dtype=int)
        * np.arange(dofs_fsi_solid.shape[-1], dtype=int)
    ).reshape(-1)

    if model_type == 'transient' and coupling == 'explicit':
        FSIModel = transient.ExplicitFSIModel
    elif model_type == 'transient' and coupling == 'implicit':
        FSIModel = transient.ImplicitFSIModel
    elif model_type == 'dynamical':
        FSIModel = dynamical.FSIModel
    elif model_type == 'linearized_dynamical':
        FSIModel = dynamical.LinearizedFSIModel
    else:
        raise ValueError(
            f"Invalid `model_type` and `coupling` ({model_type}, {coupling})"
        )

    return FSIModel(solid, fluid, dofs_fsi_solid, dofs_fsi_fluid)


def derive_1dfluidmesh_from_edges(mesh, fsi_edges):

    # Load a fluid by computing a 1D fluid mesh from the solid's medial surface
    # TODO: The streamwise mesh can already be known from the fsi_coordinates
    # variable computed earlier
    x, y, fsi_verts = meshutils.sort_edge_vertices(mesh, fsi_edges)
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    s = np.concatenate([[0], np.cumsum(np.sqrt(dx**2 + dy**2))])

    return s, fsi_verts


def locate_separation_vertex(
    solid: slr.FenicsResidual, separation_vertex_label: str = 'separation'
):
    # If the fluid has a fixed-separation point, set the appropriate
    # separation point for the fluid
    vertex_label_to_id = solid.mesh_subdomain('vertex')
    vertex_mf = solid.mesh_function('vertex')
    if vertex_mf is None:
        raise ValueError(
            f"Couldn't find separation point label {separation_vertex_label}"
        )

    if separation_vertex_label in vertex_label_to_id:
        sep_mf_value = vertex_label_to_id[separation_vertex_label]
        # assert isinstance(sep_mf_value, int)
    else:
        raise ValueError(
            f"Couldn't find separation point label {separation_vertex_label}"
        )

    sep_vert = vertex_mf.where_equal(sep_mf_value)
    if len(sep_vert) == 1:
        sep_vert = sep_vert[0]
    else:
        raise ValueError(
            "A single separation point was expected but"
            f" {len(sep_vert):d} were supplied"
        )

    return sep_vert
