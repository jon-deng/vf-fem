"""
Functionality for creating model objects from meshes, etc.
"""

from os import path
from typing import Union, Optional, Type, Any, Tuple
import numpy as np
import dolfin as dfn

from . import meshutils
from femvf.residuals import solid as slr, fluid as flr
from .models.transient import (
    solid as tsmd,
    fluid as tfmd,
    acoustic as tamd,
    coupled as tcmd,
)
from .models.dynamical import solid as dsmd, fluid as dfmd, coupled as dcmd

SolidModel = Union[tsmd.Model, dsmd.Model]
FluidModel = Union[tfmd.Model, dfmd.Model]
SolidClass = slr.PredefinedSolidResidual
FluidClass = Union[Type[tfmd.Model], Type[dfmd.Model]]

Labels = list[str]


def load_solid_model(
    mesh: str,
    SolidType: SolidClass,
    pressure_facet_labels: Optional[Labels] = ('pressure',),
    fixed_facet_labels: Optional[Labels] = ('fixed',),
) -> SolidModel:
    """
    Load a solid model of the specified type

    Parameters
    ----------
    mesh:
        A string indicating the path to a mesh file. This can be either in
         '.xml' or '.msh' format.
    SolidType:
        A class indicating the type of solid model to load.
    pressure_facet_labels, fixed_facet_labels:
        Lists of strings for labelled facets corresponding to the pressure
        loading and fixed boundaries.
    """
    if isinstance(mesh, str):
        ext = path.splitext(mesh)[1]
        # If no extension is supplied, assume it's a .msh file from gmsh
        if ext == '':
            ext = '.msh'

        # The solid model is loaded a bit differently depending on if
        # it uses the .xml (no longer supported) or newer gmsh interface
        if ext.lower() == '.xml':
            # The solid mesh is an xml file
            mesh, mesh_funcs, mesh_entities_label_to_value = meshutils.load_fenics_xml(
                mesh
            )
        elif ext.lower() == '.msh':
            # The solid mesh is an gmsh file
            mesh, mesh_funcs, mesh_entities_label_to_value = meshutils.load_fenics_gmsh(
                mesh
            )
        else:
            raise ValueError(f"Can't process mesh {mesh} with extension {ext}")
    else:
        raise TypeError(f"`solid_mesh` must be a path (`str`) not `{type(mesh)}`")

    return SolidType(
        mesh,
        mesh_funcs,
        mesh_entities_label_to_value,
        pressure_facet_labels,
        fixed_facet_labels,
    )


def load_fluid_model(
    mesh: str, FluidType: FluidClass, idx_sep: Optional[int] = 0
) -> FluidModel:
    """
    Load a solid model of the specified type

    Parameters
    ----------
    mesh:
        A string indicating the path to a mesh file. This can be either in
         '.xml' or '.msh' format.
    SolidType:
        A class indicating the type of solid model to load.
    pressure_facet_labels, fixed_facet_labels:
        Lists of strings for labelled facets corresponding to the pressure
        loading and fixed boundaries.
    """
    if issubclass(
        FluidType,
        (
            dfmd.BernoulliFixedSep,
            dfmd.LinearizedBernoulliFixedSep,
            dfmd.BernoulliFlowFixedSep,
            dfmd.LinearizedBernoulliFlowFixedSep,
            tfmd.BernoulliFixedSep,
        ),
    ):
        if len(idx_sep) == 1:
            idx_sep = idx_sep[0]
        else:
            raise ValueError(
                "Expected to find single separation point on FSI surface"
                f" but found {len(idx_sep):d} instead"
            )
        fluid = FluidType(mesh, idx_sep=idx_sep)
    else:
        fluid = FluidType(mesh)

    return fluid


# TODO: Combine transient and dynamical model loading functions?
def load_transient_fsi_model(
    solid_mesh: str,
    fluid_mesh: Any,
    SolidType: SolidClass = slr.KelvinVoigt,
    FluidType: FluidClass = flr.BernoulliAreaRatioSep,
    fsi_facet_labels: Optional[Labels] = ('pressure',),
    fixed_facet_labels: Optional[Labels] = ('fixed',),
    separation_vertex_label: str = 'separation',
    coupling: str = 'explicit',
    zs: Optional[Tuple[float]] = None,
) -> tcmd.BaseTransientFSIModel:
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
    ## Load the solid
    solid = load_solid_model(
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

    # Handle multiple fluid models by
    dofs_fsi_solid = dfn.vertex_to_dof_map(
        solid.residual.form['coeff.fsi.p1'].function_space()
    )[fsi_verts.flat]
    dofs_fsi_fluid = (
        np.ones(dofs_fsi_solid.shape[:-1], dtype=int)
        * np.arange(dofs_fsi_solid.shape[-1], dtype=int)
    ).reshape(-1)

    if coupling == 'explicit':
        model = tcmd.ExplicitFSIModel(solid, fluid, dofs_fsi_solid, dofs_fsi_fluid)
    elif coupling == 'implicit':
        model = tcmd.ImplicitFSIModel(solid, fluid, dofs_fsi_solid, dofs_fsi_fluid)
    else:
        raise ValueError(
            f"'coupling' must be one of `['explicit', 'implicit']`, not `{coupling}`"
        )

    return model


def load_dynamical_fsi_model(
    solid_mesh: str,
    fluid_mesh: Any,
    SolidType: SolidClass = slr.KelvinVoigt,
    FluidType: FluidClass = dfmd.BernoulliAreaRatioSep,
    fsi_facet_labels: Optional[Labels] = ('pressure',),
    fixed_facet_labels: Optional[Labels] = ('fixed',),
    separation_vertex_label: str = 'separation',
    zs: Optional[Tuple[float]] = None,
) -> Union[dcmd.BaseDynamicalModel, dcmd.BaseLinearizedDynamicalModel]:
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
    solid = load_solid_model(
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

    if isinstance(solid, dcmd.LinearizedModel):
        return dcmd.BaseLinearizedDynamicalFSIModel(
            solid, fluid, dofs_fsi_solid, dofs_fsi_fluid
        )
    else:
        return dcmd.BaseDynamicalFSIModel(solid, fluid, dofs_fsi_solid, dofs_fsi_fluid)


def load_transient_fsai_model(
    solid_mesh: str,
    fluid_mesh: Any,
    acoustic: tamd.Acoustic1D,
    SolidType: SolidClass = slr.KelvinVoigt,
    FluidType: FluidClass = flr.BernoulliAreaRatioSep,
    fsi_facet_labels: Optional[Labels] = ('pressure',),
    fixed_facet_labels: Optional[Labels] = ('fixed',),
    coupling: str = 'explicit',
):
    # TODO: I haven't updated the acoustic model in a while so it's likely this
    # doesn't work
    """
    Load a transient coupled (fsai) model

    Parameters
    ----------
    solid_mesh : str
        Path to the solid mesh
    fluid_mesh : None or (future proofing for other fluid types)
        Currently this isn't used
    acoustic :
        The acoustic model
    SolidType, FluidType:
        Classes of the solid and fluid models to load
    fsi_facet_labels, fixed_facet_labels:
        String identifiers for facets corresponding to traction/dirichlet
        conditions
    coupling: str
        One of 'explicit' or 'implicit' indicating the coupling strategy between
        fluid/solid domains
    """
    solid = load_solid_model(
        solid_mesh, SolidType, fsi_facet_labels, fixed_facet_labels
    )
    fluid, fsi_verts = derive_1dfluid_from_2dsolid(
        solid_mesh, FluidResidual=FluidType, fsi_facet_labels=fsi_facet_labels
    )

    dofs_fsi_solid = dfn.vertex_to_dof_map(solid.residual.form['fspace.scalar'])[
        fsi_verts
    ]
    dofs_fsi_fluid = np.arange(dofs_fsi_solid.size)

    return tcmd.FSAIModel(solid, fluid, acoustic, dofs_fsi_solid, dofs_fsi_fluid)


# TODO: Refactor this function; currently does too many things
# the function should take a loaded solid model and derive a fluid mesh from it
def derive_1dfluid_from_2dsolid(
    solid: slr.FenicsResidual,
    FluidResidual: type[flr.PredefinedJaxResidual] = flr.BernoulliAreaRatioSep,
    fsi_facet_labels: Optional[Labels] = ('pressure',),
    separation_vertex_label: str = 'separation',
) -> Tuple[flr.PredefinedJaxResidual, np.ndarray]:
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
            dfmd.BernoulliFixedSep,
            dfmd.LinearizedBernoulliFixedSep,
            dfmd.BernoulliFlowFixedSep,
            dfmd.LinearizedBernoulliFlowFixedSep,
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
    FluidResidual: type[flr.PredefinedJaxResidual] = flr.BernoulliAreaRatioSep,
    fsi_facet_labels: Optional[Labels] = ('pressure',),
    separation_vertex_label: str = 'separation',
    zs: Optional[np.typing.NDArray[int]] = None,
) -> Tuple[flr.PredefinedJaxResidual, np.ndarray]:
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
                dfmd.BernoulliFixedSep,
                dfmd.LinearizedBernoulliFixedSep,
                dfmd.BernoulliFlowFixedSep,
                dfmd.LinearizedBernoulliFlowFixedSep,
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
