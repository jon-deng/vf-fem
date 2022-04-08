"""
Functionality for creating model objects from meshes, etc.
"""
from os import path
import numpy as np
import dolfin as dfn

from . import meshutils
from .models.transient import solid as smd, fluid as fmd
from .models.transient.coupled import (FSAIModel, ExplicitFSIModel, ImplicitFSIModel)

from .models.dynamical import solid as dynsolidmodel, fluid as dynfluidmodel, coupled as dynfsimodel

# from .dynamicalmodels.coupled import (FSAIModel, ExplicitFSIModel, ImplicitFSIModel)

def load_solid_model(solid_mesh, SolidType,
    pressure_facet_labels=('pressure',), fixed_facet_labels=('fixed',)):
    # Process the mesh file(s)
    mesh, facet_func, cell_func, facet_labels, cell_labels = None, None, None, None, None
    if isinstance(solid_mesh, str):
        ext = path.splitext(solid_mesh)[1]
        # if no extension is supplied, assume it's a fenics xml mesh
        if ext == '':
            ext = '.xml'

        if ext.lower() == '.xml':
            # The solid mesh is an xml file
            mesh, facet_func, cell_func, (vertex_labels, facet_labels, cell_labels) = meshutils.load_fenics_xmlmesh(solid_mesh)
        else:
            raise ValueError(f"Can't process mesh {solid_mesh} with extension {ext}")
    else:
        raise ValueError("Not sure what to do with this")

    solid = SolidType(mesh, facet_func, facet_labels, cell_func, cell_labels,
                      pressure_facet_labels, fixed_facet_labels)
    return solid

def load_fsi_model(solid_mesh, fluid_mesh, SolidType=smd.KelvinVoigt, FluidType=fmd.Bernoulli,
                   fsi_facet_labels=('pressure',), fixed_facet_labels=('fixed',), coupling='explicit'):
    """
    Factory function that loads a model based on input solid/fluid meshes.

    The function will delegate to specific loading routines depending on how the mesh
    is input etc.

    Parameters
    ----------
    solid_mesh : dolfin.Mesh
    fluid_mesh : None or (future proofing for other fluid types)
    """
    solid, fluid, fsi_verts = process_fsi(solid_mesh, fluid_mesh,
        SolidType=SolidType, FluidType=FluidType, fsi_facet_labels=fsi_facet_labels, fixed_facet_labels=fixed_facet_labels)

    if coupling == 'explicit':
        model = ExplicitFSIModel(solid, fluid, fsi_verts)
    elif coupling == 'implicit':
        model = ImplicitFSIModel(solid, fluid, fsi_verts)

    return model

def load_dynamical_fsi_model(
    solid_mesh, fluid_mesh,
    SolidType=dynsolidmodel.KelvinVoigt, FluidType=dynfluidmodel.Bernoulli1DDynamicalSystem,
    fsi_facet_labels=('pressure',), fixed_facet_labels=('fixed',)):
    """
    Factory function that loads a model based on input solid/fluid meshes.

    The function will delegate to specific loading routines depending on how the mesh
    is input etc.

    Parameters
    ----------
    solid_mesh : dolfin.Mesh
    fluid_mesh : None or (future proofing for other fluid types)
    """
    solid, fluid, fsi_verts = process_fsi(solid_mesh, fluid_mesh,
        SolidType=SolidType, FluidType=FluidType, fsi_facet_labels=fsi_facet_labels, fixed_facet_labels=fixed_facet_labels)

    dofs_fsi_solid = dfn.vertex_to_dof_map(solid.forms['fspace.scalar'])[fsi_verts]
    dofs_fsi_fluid = np.arange(dofs_fsi_solid.size)

    return dynfsimodel.FSIDynamicalSystem(solid, fluid, dofs_fsi_solid, dofs_fsi_fluid)

def load_fsai_model(solid_mesh, fluid_mesh, acoustic, SolidType=smd.KelvinVoigt, FluidType=fmd.Bernoulli,
                    fsi_facet_labels=('pressure',), fixed_facet_labels=('fixed',), coupling='explicit'):
    solid, fluid, fsi_verts = process_fsi(solid_mesh, fluid_mesh,
        SolidType=SolidType, FluidType=FluidType, fsi_facet_labels=fsi_facet_labels, fixed_facet_labels=fixed_facet_labels)

    return FSAIModel(solid, fluid, acoustic, fsi_verts)

def process_fsi(solid_mesh, fluid_mesh, SolidType=smd.KelvinVoigt, FluidType=fmd.Bernoulli,
                fsi_facet_labels=('pressure',), fixed_facet_labels=('fixed',), coupling='explicit'):
    ## Load the solid
    mesh, facet_func, cell_func, facet_labels, cell_labels = None, None, None, None, None
    if isinstance(solid_mesh, str):
        ext = path.splitext(solid_mesh)[1]
        # if no extension is supplied, assume it's a fenics xml mesh
        if ext == '':
            ext = '.xml'

        if ext.lower() == '.xml':
            # The solid mesh is an xml file
            mesh, facet_func, cell_func, (vertex_labels, facet_labels, cell_labels) = meshutils.load_fenics_xmlmesh(solid_mesh)
        else:
            raise ValueError(f"Can't process mesh {solid_mesh} with extension {ext}")
    solid = SolidType(mesh, facet_func, facet_labels, cell_func, cell_labels,
                      fsi_facet_labels, fixed_facet_labels)

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

    # Load the fluid
    # There are separate cases for loading the transient fluid model and the dynamical system fluid
    # model because I changed the way the mesh is input between them
    if fluid_mesh is None and issubclass(FluidType, fmd.QuasiSteady1DFluid):
        x, y = meshutils.streamwise1dmesh_from_edges(
            mesh, facet_func, [facet_labels[label] for label in fsi_facet_labels])
        fluid = FluidType(x, y)
    elif fluid_mesh is None and issubclass(FluidType, dynfluidmodel.BaseFluid1DDynamicalSystem):
        x, y = meshutils.streamwise1dmesh_from_edges(
            mesh, facet_func, [facet_labels[label] for label in fsi_facet_labels])
        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]
        s = np.concatenate([[0], np.cumsum(np.sqrt(dx**2 + dy**2))])
        fluid = FluidType(s)
    else:
        raise ValueError(f"This function does not yet support input fluid meshes.")
    return solid, fluid, fsi_verts
