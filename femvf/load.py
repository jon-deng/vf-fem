"""
Functionality for creating model objects from meshes, etc.
"""
from os import path
import numpy as np
import dolfin as dfn

from . import meshutils
from .models.transient import solid as tsmd, fluid as tfmd
from .models.transient.coupled import (FSAIModel, ExplicitFSIModel, ImplicitFSIModel)

from .models.dynamical import solid as dsmd, fluid as dfmd, coupled as dynfsimodel

# from .dynamicalmodels.coupled import (FSAIModel, ExplicitFSIModel, ImplicitFSIModel)

def load_solid_model(
        solid_mesh,
        SolidType,
        pressure_facet_labels=('pressure',),
        fixed_facet_labels=('fixed',)
    ):
    """
    Loads a solid model
    """
    # Process the mesh file(s)
    if isinstance(solid_mesh, str):
        ext = path.splitext(solid_mesh)[1]
        # if no extension is supplied, assume it's a .msh file from gmsh
        if ext == '':
            ext = '.xml'

        if ext.lower() == '.xml':
            # The solid mesh is an xml file
            mesh, mesh_funcs, mesh_entities_label_to_value = meshutils.load_fenics_xmlmesh(solid_mesh)
        elif ext.lower() == '.msh':
            # The solid mesh is an gmsh file
            mesh, mesh_funcs, mesh_entities_label_to_value = meshutils.load_fenics_gmsh(solid_mesh)
        else:
            raise ValueError(f"Can't process mesh {solid_mesh} with extension {ext}")
    else:
        raise TypeError(f"`solid_mesh` must be a path (str) not {type(solid_mesh)}")

    return SolidType(mesh, mesh_funcs, mesh_entities_label_to_value, pressure_facet_labels, fixed_facet_labels)

def load_transient_fsi_model(
        solid_mesh,
        fluid_mesh,
        SolidType=tsmd.KelvinVoigt,
        FluidType=tfmd.Bernoulli,
        fsi_facet_labels=('pressure',),
        fixed_facet_labels=('fixed',),
        separation_vertex_label='separation',
        coupling='explicit'
    ):
    """
    Factory function that loads a model based on input solid/fluid meshes.

    The function will delegate to specific loading routines depending on how the mesh
    is input etc.

    Parameters
    ----------
    solid_mesh : dolfin.Mesh
    fluid_mesh : None or (future proofing for other fluid types)
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

    if coupling == 'explicit':
        model = ExplicitFSIModel(solid, fluid, dofs_fsi_solid, dofs_fsi_fluid)
    elif coupling == 'implicit':
        model = ImplicitFSIModel(solid, fluid, dofs_fsi_solid, dofs_fsi_fluid)

    return model

def load_dynamical_fsi_model(
        solid_mesh,
        fluid_mesh,
        SolidType=dsmd.KelvinVoigt,
        FluidType=dfmd.BernoulliSmoothMinSep,
        fsi_facet_labels=('pressure',),
        fixed_facet_labels=('fixed',),
        separation_vertex_label='separation'
    ):
    """
    Factory function that loads a model based on input solid/fluid meshes.

    The function will delegate to specific loading routines depending on how the mesh
    is input etc.

    Parameters
    ----------
    solid_mesh : dolfin.Mesh
    fluid_mesh : None or (future proofing for other fluid types)
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

    return dynfsimodel.FSIDynamicalSystem(solid, fluid, dofs_fsi_solid, dofs_fsi_fluid)

def load_transient_fsai_model(
        solid_mesh,
        fluid_mesh,
        acoustic,
        SolidType=tsmd.KelvinVoigt,
        FluidType=tfmd.Bernoulli,
        fsi_facet_labels=('pressure',),
        fixed_facet_labels=('fixed',),
        coupling='explicit'
    ):
    """
    Loads a transient FSAI model
    """
    solid, fluid, fsi_verts = process_fsi(
        solid_mesh, fluid_mesh,
        SolidType=SolidType, FluidType=FluidType,
        fsi_facet_labels=fsi_facet_labels,
        fixed_facet_labels=fixed_facet_labels
        )

    dofs_fsi_solid = dfn.vertex_to_dof_map(solid.forms['fspace.scalar'])[fsi_verts]
    dofs_fsi_fluid = np.arange(dofs_fsi_solid.size)

    return FSAIModel(solid, fluid, acoustic, dofs_fsi_solid, dofs_fsi_fluid)

def process_fsi(
        solid_mesh,
        fluid_mesh,
        SolidType=tsmd.KelvinVoigt,
        FluidType=tfmd.Bernoulli,
        fsi_facet_labels=('pressure',),
        fixed_facet_labels=('fixed',),
        separation_vertex_label='separation',
    ):
    """
    Processes appropriate mappings between fluid/solid domains for FSI
    """
    ## Load the solid
    solid = load_solid_model(solid_mesh, SolidType, fsi_facet_labels, fixed_facet_labels)

    ## Process the fsi surface vertices to set the coupling between solid and fluid
    # Find vertices corresponding to the fsi facets
    fsi_facet_ids = [solid.forms['mesh.facet_label_to_id'][name] for name in fsi_facet_labels]
    fsi_edges = np.array([nedge for nedge, fedge in enumerate(solid.forms['mesh.facet_function'].array())
                                if fedge in set(fsi_facet_ids)])
    fsi_verts = meshutils.vertices_from_edges(fsi_edges, solid.forms['mesh.mesh'])
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
                raise ValueError("Loading a fixed separation point fluid model but solid mesh doesn't specify a separation point")

            if separation_vertex_label in vertex_label_to_id:
                sep_mf_value = vertex_label_to_id[separation_vertex_label]
                # assert isinstance(sep_mf_value, int)
            else:
                raise ValueError("Loading a fixed separation point fluid model but solid mesh doesn't specify a separation point")

            sep_vert = vertex_mf.array()[vertex_mf.array() == sep_mf_value]
            if len(sep_vert) == 1:
                sep_vert = sep_vert[0]
            else:
                raise ValueError(f"Single separation point expected but {len(sep_vert):d} were supplied")

            idx_sep = np.where(fsi_verts == sep_vert)
            if len(idx_sep) == 1:
                idx_sep = idx_sep[0]
            else:
                raise ValueError(f"Expected to find single separation point on FSI surface but found {len(idx_sep):d} instead")
            fluid = FluidType(s, idx_sep=idx_sep)
        else:
            fluid = FluidType(s)
    else:
        raise ValueError(f"This function does not yet support input fluid meshes.")
    return solid, fluid, fsi_verts
