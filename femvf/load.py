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
    mesh: str,
    Residual: slr.PredefinedSolidResidual,
    model_type: str = 'transient',
    **kwargs: dict[str, Any]
) -> SolidModel:
    """
    Load a solid model

    Parameters
    ----------
    mesh: str
        A mesh file path

        Only GMSH meshes are currently supported
    Residual: slr.PredefinedSolidResidual
        A predefined Fenics residual class
    model_type: str
        Type of model to load ('transient' or 'dynamical')
    **kwargs:
        Additional keyword args for the residual
    """
    ext = path.splitext(mesh)[1]
    if ext.lower() == '.msh':
        mesh, mesh_funcs, mesh_subdomains = meshutils.load_fenics_gmsh(mesh)
    else:
        raise ValueError(f"Invalid mesh extension {ext}")

    residual = Residual(mesh, mesh_funcs, mesh_subdomains, **kwargs)
    if model_type == 'transient':
        return transient.FenicsModel(residual)
    elif model_type == 'dynamical':
        return dynamical.FenicsModel(residual)
    else:
        raise ValueError(f"Invalid model type {model_type}")


def load_jax_model(
    mesh: NDArray,
    Residual: flr.PredefinedFluidResidual,
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
        Type of model to load ('transient' or 'dynamical')
    **kwargs:
        Additional keyword args for the residual
    """
    residual = Residual(mesh, **kwargs)

    if model_type == 'transient':
        return transient.JaxModel(residual)
    elif model_type == 'dynamical':
        return dynamical.JaxModel(residual)
    else:
        raise ValueError(f"Invalid model type {model_type}")


# TODO: Combine transient and dynamical model loading functions?
def load_transient_fsi_model(
    solid_mesh: str,
    SolidResidual: slr.PredefinedSolidResidual,
    FluidResidual: flr.PredefinedFluidResidual,
    solid_kwargs: dict[str, Any],
    fluid_kwargs: dict[str, Any],
    coupling: str = 'explicit',
    zs: Optional[tuple[float]] = None,
) -> transient.BaseTransientFSIModel:
    """
    Load a transient coupled (fsi) model

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
    model_type = 'transient'
    ## Load the solid
    solid = load_fenics_model(
        solid_mesh, SolidResidual, model_type=model_type, **solid_kwargs
    )

    if zs is None:
        # TODO: Refactor hard-coded keys ('traction' ...)!
        fluid, fsi_verts = derive_1dfluid_from_2dsolid(
            solid,
            FluidResidual=FluidResidual,
            fsi_facet_labels=['traction'],
            separation_vertex_label='superior',
        )
    else:
        fluid, fsi_verts = derive_1dfluid_from_3dsolid(
            solid,
            FluidResidual=FluidResidual,
            fsi_facet_labels=['traction'],
            separation_vertex_label='superior',
            zs=zs,
        )

    dofs_fsi_solid = dfn.vertex_to_dof_map(
        solid.residual.form['coeff.fsi.p1'].function_space()
    )[fsi_verts.flat]
    dofs_fsi_fluid = (
        np.ones(dofs_fsi_solid.shape[:-1], dtype=int)
        * np.arange(dofs_fsi_solid.shape[-1], dtype=int)
    ).reshape(-1)

    if coupling == 'explicit':
        model = transient.ExplicitFSIModel(solid, fluid, dofs_fsi_solid, dofs_fsi_fluid)
    elif coupling == 'implicit':
        model = transient.ImplicitFSIModel(solid, fluid, dofs_fsi_solid, dofs_fsi_fluid)
    else:
        raise ValueError(
            f"'coupling' must be one of `['explicit', 'implicit']`, not `{coupling}`"
        )

    return model


def load_dynamical_fsi_model(
    solid_mesh: str,
    fluid_mesh: Any,
    SolidType: slr.PredefinedSolidResidual = slr.KelvinVoigt,
    FluidType: FluidClass = flr.BernoulliAreaRatioSep,
    fsi_facet_labels: Optional[Labels] = ('pressure',),
    fixed_facet_labels: Optional[Labels] = ('fixed',),
    separation_vertex_label: str = 'separation',
    zs: Optional[tuple[float]] = None,
) -> Union[dynamical.BaseDynamicalModel, dynamical.BaseLinearizedDynamicalModel]:
    """
    Load a transient coupled (fsi) model

    Parameters
    ----------
    solid_mesh : str
        Path to the solid mesh
    fluid_mesh : None or (future proofing for other fluid types)
        Currently this isn't used
    SolidType, FluidType:
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
    solid = load_fenics_model(
        solid_mesh, SolidType, fsi_facet_labels, fixed_facet_labels
    )
    if zs is None:
        fluid, fsi_verts = derive_1dfluid_from_2dsolid(
            solid,
            FluidResidual=FluidType,
            fsi_facet_labels=fsi_facet_labels,
            separation_vertex_label=separation_vertex_label,
        )
    else:
        fluid, fsi_verts = derive_1dfluid_from_3dsolid(
            solid,
            FluidResidual=FluidType,
            fsi_facet_labels=fsi_facet_labels,
            separation_vertex_label=separation_vertex_label,
            zs=zs,
        )

    # BUG: This FSI dof selection won't work for higher order elements!!
    dofs_fsi_solid = dfn.vertex_to_dof_map(
        solid.residual.form['coeff.fsi.p1'].function_space()
    )[fsi_verts.flat]
    dofs_fsi_fluid = (
        np.ones(dofs_fsi_solid.shape[:-1], dtype=int)
        * np.arange(dofs_fsi_solid.shape[-1], dtype=int)
    ).reshape(-1)

    if isinstance(solid, dynamical.LinearizedFenicsModel):
        return dynamical.LinearizedFSIModel(
            solid, fluid, dofs_fsi_solid, dofs_fsi_fluid
        )
    else:
        return dynamical.FSIModel(solid, fluid, dofs_fsi_solid, dofs_fsi_fluid)


# TODO: Refactor this function; currently does too many things
# the function should take a loaded solid model and derive a fluid mesh from it
def derive_1dfluid_from_2dsolid(
    solid: slr.FenicsResidual,
    FluidResidual: type[flr.PredefinedFluidResidual] = flr.BernoulliAreaRatioSep,
    fsi_facet_labels: Optional[Labels] = ('pressure',),
    separation_vertex_label: str = 'separation',
) -> tuple[flr.PredefinedFluidResidual, np.ndarray]:
    """
    Processes appropriate mappings between fluid/solid domains for FSI

    Parameters
    ----------
    solid_mesh : str
        Path to the solid mesh
    fluid_mesh : None or (future proofing for other fluid types)
        Currently this isn't used
    SolidType, FluidType:
        Classes of the solid and fluid models to load
    fsi_facet_labels, fixed_facet_labels:
        String identifiers for facets corresponding to traction/dirichlet
        conditions
    separation_vertex_label:
        A string corresponding to a labelled vertex where separation should
        occur. This is only relevant for quasi-static fluid models with a fixed
        separation point
    """
    ## Process the fsi surface vertices to set the coupling between solid and fluid
    # Find vertices corresponding to the fsi facets
    fsi_facet_ids = [
        solid.residual.mesh_subdomain('facet')[name]
        for name in fsi_facet_labels
    ]
    fsi_edges = np.array(
        [
            nedge
            for nedge, fedge in enumerate(solid.residual.mesh_function('facet').array())
            if fedge in set(fsi_facet_ids)
        ]
    )

    # Load a fluid by computing a 1D fluid mesh from the solid's medial surface
    mesh = solid.residual.mesh()
    s, fsi_verts = derive_1dfluidmesh_from_edges(mesh, fsi_edges)
    if issubclass(
        FluidResidual,
        (
            flr.BernoulliFixedSep,
            # flr.LinearizedBernoulliFixedSep,
            flr.BernoulliFlowFixedSep,
            # flr.LinearizedBernoulliFlowFixedSep,
            flr.BernoulliFixedSep,
        ),
    ):
        sep_vert = locate_separation_vertex(solid, separation_vertex_label)

        fsi_verts_fluid_ord = np.arange(fsi_verts.size)
        idx_sep = fsi_verts_fluid_ord[fsi_verts == sep_vert]
        if len(idx_sep) == 1:
            idx_sep = idx_sep[0]
        else:
            raise ValueError(
                "Expected to find single separation point on FSI surface"
                f" but found {len(idx_sep):d} instead"
            )
        fluid = FluidResidual(s, idx_sep=idx_sep)
    else:
        fluid = FluidResidual(s)

    return fluid, fsi_verts


def derive_1dfluid_from_3dsolid(
    solid: slr.FenicsResidual,
    FluidResidual: type[flr.PredefinedFluidResidual] = flr.BernoulliAreaRatioSep,
    fsi_facet_labels: Optional[Labels] = ('pressure',),
    separation_vertex_label: str = 'separation',
    zs: Optional[np.typing.NDArray[int]] = None,
) -> tuple[flr.PredefinedFluidResidual, np.ndarray]:
    """
    Processes appropriate mappings between fluid/solid domains for FSI

    Parameters
    ----------
    solid_mesh : str
        Path to the solid mesh
    fluid_mesh : None or (future proofing for other fluid types)
        Currently this isn't used
    SolidType, FluidType:
        Classes of the solid and fluid models to load
    fsi_facet_labels, fixed_facet_labels:
        String identifiers for facets corresponding to traction/dirichlet
        conditions
    separation_vertex_label:
        A string corresponding to a labelled vertex where separation should
        occur. This is only relevant for quasi-static fluid models with a fixed
        separation point
    """
    if zs is None:
        zs = np.array([0])

    ## Process the fsi surface vertices to set the coupling between solid and fluid
    # Find vertices corresponding to the fsi facets
    mesh = solid.residual.mesh()
    fsi_verts_list = []
    s_list = []
    for z in zs:
        facets = meshutils.extract_zplane_facets(mesh, z=z)

        fsi_facet_ids = [
            solid.residual.mesh_subdomain('facet')[name]
            for name in fsi_facet_labels
        ]
        fsi_edges = meshutils.extract_edges_from_facets(
            facets, solid.residual.mesh_function('facet'), fsi_facet_ids
        )
        fsi_edges = np.array([edge.index() for edge in fsi_edges])

        s, fsi_verts = derive_1dfluidmesh_from_edges(mesh, fsi_edges)
        s_list.append(s)
        fsi_verts_list.append(fsi_verts)
        if issubclass(
            FluidResidual,
            (
                flr.BernoulliFixedSep,
                # flr.LinearizedBernoulliFixedSep,
                flr.BernoulliFlowFixedSep,
                # flr.LinearizedBernoulliFlowFixedSep,
                flr.BernoulliFixedSep,
            ),
        ):
            # TODO: For this to work you should generalize a fixed separation point
            # to a fixed-separation line I guess
            # sep_vert = locate_separation_vertex(solid, separation_vertex_label)

            # fsi_verts_fluid_ord = np.arange(fsi_verts.size)
            # idx_sep = fsi_verts_fluid_ord[fsi_verts == sep_vert]
            # if len(idx_sep) == 1:
            #     idx_sep = idx_sep[0]
            # else:
            #     raise ValueError(
            #         "Expected to find single separation point on FSI surface"
            #         f" but found {len(idx_sep):d} instead"
            #     )
            # fluid = FluidType(s, idx_sep=idx_sep)
            raise ValueError("3D models can't handle fixed separation points yet")

    s = np.array(s_list)
    fluid = FluidResidual(s)
    return fluid, np.array(fsi_verts_list, dtype=int)


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
    solid: SolidModel, separation_vertex_label: str = 'separation'
):
    # If the fluid has a fixed-separation point, set the appropriate
    # separation point for the fluid
    vertex_label_to_id = solid.residual.mesh_subdomain('vertex')
    vertex_mf = solid.residual.mesh_function('vertex')
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
