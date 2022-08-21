"""
Functionality for creating model objects from meshes, etc.
"""
from lib2to3.pgen2.token import OP
from os import path
from typing import Union, Optional, List, Type, Any, Tuple
import numpy as np
import dolfin as dfn

from . import meshutils
from .models.transient import solid as tsmd, fluid as tfmd, acoustic as tamd, coupled as tcmd
from .models.dynamical import solid as dsmd, fluid as dfmd, coupled as dcmd

SolidModel = Union[tsmd.Solid, dsmd.BaseSolidDynamicalSystem]
FluidModel = Union[tfmd.QuasiSteady1DFluid, dfmd.BaseFluid1DDynamicalSystem]
SolidClass = Union[Type[tsmd.Solid], Type[dsmd.BaseSolidDynamicalSystem]]
FluidClass = Union[Type[tfmd.QuasiSteady1DFluid], Type[dfmd.BaseFluid1DDynamicalSystem]]

Labels = List[str]

def load_solid_model(
        solid_mesh: str,
        SolidType: SolidClass,
        pressure_facet_labels: Optional[Labels]=('pressure',),
        fixed_facet_labels: Optional[Labels]=('fixed',)
    ) -> SolidModel:
    """
    Load a solid model of the specified type

    Parameters
    ----------
    solid_mesh:
        A string indicating the path to a mesh file. This can be either in
         '.xml' or '.msh' format.
    SolidType:
        A class indicating the type of solid model to load.
    pressure_facet_labels, fixed_facet_labels:
        Lists of strings for labelled facets corresponding to the pressure
        loading and fixed boundaries.
    """
    if isinstance(solid_mesh, str):
        ext = path.splitext(solid_mesh)[1]
        # If no extension is supplied, assume it's a .msh file from gmsh
        if ext == '':
            ext = '.msh'

        if ext.lower() == '.xml':
            # The solid mesh is an xml file
            mesh, mesh_funcs, mesh_entities_label_to_value = meshutils.load_fenics_xml(solid_mesh)
        elif ext.lower() == '.msh':
            # The solid mesh is an gmsh file
            mesh, mesh_funcs, mesh_entities_label_to_value = meshutils.load_fenics_gmsh(solid_mesh)
        else:
            raise ValueError(f"Can't process mesh {solid_mesh} with extension {ext}")
    else:
        raise TypeError(f"`solid_mesh` must be a path (str) not {type(solid_mesh)}")

    return SolidType(mesh, mesh_funcs, mesh_entities_label_to_value, pressure_facet_labels, fixed_facet_labels)

# TODO: Can combine transient/dynamical model loading functions into single one
def load_transient_fsi_model(
        solid_mesh: str,
        fluid_mesh: Any,
        SolidType: SolidClass=tsmd.KelvinVoigt,
        FluidType: FluidClass=tfmd.BernoulliAreaRatioSep,
        fsi_facet_labels: Optional[Labels]=('pressure',),
        fixed_facet_labels: Optional[Labels]=('fixed',),
        separation_vertex_label: str='separation',
        coupling: str='explicit'
    ) -> tcmd.FSIModel:
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
    solid, fluid, fsi_verts = process_fsi(
        solid_mesh, fluid_mesh,
        SolidType=SolidType, FluidType=FluidType,
        fsi_facet_labels=fsi_facet_labels,
        fixed_facet_labels=fixed_facet_labels,
        separation_vertex_label=separation_vertex_label
    )

    # TODO: This FSI dof selection won't for higher order elements
    dofs_fsi_solid = dfn.vertex_to_dof_map(solid.forms['fspace.scalar'])[fsi_verts]
    dofs_fsi_fluid = np.arange(dofs_fsi_solid.size)

    if coupling == 'explicit':
        model = tcmd.ExplicitFSIModel(solid, fluid, dofs_fsi_solid, dofs_fsi_fluid)
    elif coupling == 'implicit':
        model = tcmd.ImplicitFSIModel(solid, fluid, dofs_fsi_solid, dofs_fsi_fluid)
    else:
        raise ValueError(
            f"'coupling' must be one of [explicit, implicit], not {coupling}"
        )

    return model

def load_dynamical_fsi_model(
        solid_mesh: str,
        fluid_mesh: Any,
        SolidType: SolidClass=dsmd.KelvinVoigt,
        FluidType: FluidClass=dfmd.BernoulliAreaRatioSep,
        fsi_facet_labels: Optional[Labels]=('pressure',),
        fixed_facet_labels: Optional[Labels]=('fixed',),
        separation_vertex_label: str='separation'
    ) -> dcmd.BaseDynamicalModel:
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
    solid, fluid, fsi_verts = process_fsi(
        solid_mesh, fluid_mesh,
        SolidType=SolidType, FluidType=FluidType,
        fsi_facet_labels=fsi_facet_labels,
        fixed_facet_labels=fixed_facet_labels,
        separation_vertex_label=separation_vertex_label
    )

    dofs_fsi_solid = dfn.vertex_to_dof_map(solid.forms['fspace.scalar'])[fsi_verts]
    dofs_fsi_fluid = np.arange(dofs_fsi_solid.size)

    return dcmd.FSIDynamicalSystem(solid, fluid, dofs_fsi_solid, dofs_fsi_fluid)

def load_transient_fsai_model(
        solid_mesh: str,
        fluid_mesh: Any,
        acoustic: tamd.Acoustic1D,
        SolidType: SolidClass=tsmd.KelvinVoigt,
        FluidType: FluidClass=tfmd.BernoulliAreaRatioSep,
        fsi_facet_labels: Optional[Labels]=('pressure',),
        fixed_facet_labels: Optional[Labels]=('fixed',),
        coupling: str='explicit'
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
    solid, fluid, fsi_verts = process_fsi(
        solid_mesh, fluid_mesh,
        SolidType=SolidType, FluidType=FluidType,
        fsi_facet_labels=fsi_facet_labels,
        fixed_facet_labels=fixed_facet_labels
        )

    dofs_fsi_solid = dfn.vertex_to_dof_map(solid.forms['fspace.scalar'])[fsi_verts]
    dofs_fsi_fluid = np.arange(dofs_fsi_solid.size)

    return tcmd.FSAIModel(solid, fluid, acoustic, dofs_fsi_solid, dofs_fsi_fluid)

# TODO: Refactor this function; currently does too many things
# the function should take a loaded solid model and derive a fluid mesh from it
def process_fsi(
        solid_mesh: str,
        fluid_mesh: Any,
        SolidType: SolidClass=tsmd.KelvinVoigt,
        FluidType: FluidClass=tfmd.BernoulliAreaRatioSep,
        fsi_facet_labels: Optional[Labels]=('pressure',),
        fixed_facet_labels: Optional[Labels]=('fixed',),
        separation_vertex_label: str='separation',
    ) -> Tuple[SolidModel, FluidModel, np.ndarray]:
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
    ## Load the solid
    solid = load_solid_model(solid_mesh, SolidType, fsi_facet_labels, fixed_facet_labels)

    ## Process the fsi surface vertices to set the coupling between solid and fluid
    # Find vertices corresponding to the fsi facets
    fsi_facet_ids = [solid.forms['mesh.facet_label_to_id'][name] for name in fsi_facet_labels]
    fsi_edges = np.array([nedge for nedge, fedge in enumerate(solid.forms['mesh.facet_function'].array())
                                if fedge in set(fsi_facet_ids)])
    fsi_verts = meshutils.vertices_from_edges(solid.forms['mesh.mesh'], fsi_edges)
    fsi_coordinates = solid.forms['mesh.mesh'].coordinates()[fsi_verts]

    # Sort the fsi vertices from inferior to superior
    # NOTE: This only works for a 1D fluid mesh and isn't guaranteed if the VF surface is shaped strangely
    idx_sort = meshutils.sort_vertices_by_nearest_neighbours(fsi_coordinates)
    fsi_verts = fsi_verts[idx_sort]
    fsi_coordinates = fsi_coordinates[idx_sort]

    # Load a fluid by computing a 1D fluid mesh from the solid's medial surface
    if fluid_mesh is None and issubclass(
            FluidType,
            (tfmd.QuasiSteady1DFluid, dfmd.BaseFluid1DDynamicalSystem)
        ):
        mesh = solid.forms['mesh.mesh']
        facet_func = solid.forms['mesh.facet_function']
        facet_labels = solid.forms['mesh.facet_label_to_id']
        # TODO: The streamwise mesh can already be known from the fsi_coordinates
        # variable computed earlier
        x, y = meshutils.streamwise1dmesh_from_edges(
            mesh, facet_func, [facet_labels[label] for label in fsi_facet_labels])
        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]
        s = np.concatenate([[0], np.cumsum(np.sqrt(dx**2 + dy**2))])
        if issubclass(FluidType, (dfmd.BaseBernoulliFixedSep, tfmd.BernoulliFixedSep)):
            # If the fluid has a fixed-separation point, set the appropriate
            # separation point for the fluid
            vertex_label_to_id = solid.forms['mesh.vertex_label_to_id']
            vertex_mf = solid.forms['mesh.vertex_function']
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

            fsi_verts_fluid_ord = np.arange(fsi_verts.size)
            idx_sep = fsi_verts_fluid_ord[fsi_verts == sep_vert]
            if len(idx_sep) == 1:
                idx_sep = idx_sep[0]
            else:
                raise ValueError(
                    "Expected to find single separation point on FSI surface"
                    f" but found {len(idx_sep):d} instead"
                )
            fluid = FluidType(s, idx_sep=idx_sep)
        else:
            fluid = FluidType(s)
    else:
        raise ValueError("This function does not yet support input fluid meshes.")
    return solid, fluid, fsi_verts
