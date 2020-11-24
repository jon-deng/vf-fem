"""
Functionality for creating model objects from meshes, etc.
"""
from os import path
import numpy as np

from .. import meshutils
from .coupled import (FSAIModel, ExplicitFSIModel, ImplicitFSIModel, solid, fluid)

def load_fsi_model(solid_mesh, fluid_mesh, Solid=solid.KelvinVoigt, Fluid=fluid.Bernoulli, 
                   fsi_facet_labels=('pressure',), fixed_facet_labels=('fixed',), coupling='explicit'):
    """
    Factory function that loads a model based on input solid/fluid meshes.

    The function will delegate to specific loading routines depending on how the mesh
    is input etc.

    Parameters
    ----------
    """
    solid, fluid, fsi_verts = process_fsi(solid_mesh, fluid_mesh, 
        Solid=Solid, Fluid=Fluid, fsi_facet_labels=fsi_facet_labels, fixed_facet_labels=fixed_facet_labels)
    
    if coupling == 'explicit':
        model = ExplicitFSIModel(solid, fluid, fsi_verts)
    elif coupling == 'implicit':
        model = ImplicitFSIModel(solid, fluid, fsi_verts)

    return model

def load_fsai_model(solid_mesh, fluid_mesh, acoustic, Solid=solid.KelvinVoigt, Fluid=fluid.Bernoulli, 
                    fsi_facet_labels=('pressure',), fixed_facet_labels=('fixed',), coupling='explicit'):
    solid, fluid, fsi_verts = process_fsi(solid_mesh, fluid_mesh, 
        Solid=Solid, Fluid=Fluid, fsi_facet_labels=fsi_facet_labels, fixed_facet_labels=fixed_facet_labels)

    return FSAIModel(solid, fluid, acoustic, fsi_verts)

def process_fsi(solid_mesh, fluid_mesh, Solid=solid.KelvinVoigt, Fluid=fluid.Bernoulli, 
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
            mesh, facet_func, cell_func, facet_labels, cell_labels = meshutils.load_fenics_xmlmesh(solid_mesh)
        else:
            raise ValueError(f"Can't process mesh {solid_mesh} with extension {ext}")
    solid = Solid(mesh, facet_func, facet_labels, cell_func, cell_labels, 
                  fsi_facet_labels, fixed_facet_labels)

    ## Process the fsi surface vertices to set the coupling between solid and fluid
    # Find vertices corresponding to the fsi facets
    fsi_facet_ids = [solid.facet_labels[name] for name in solid.fsi_facet_labels]
    fsi_edges = np.array([nedge for nedge, fedge in enumerate(solid.facet_func.array()) 
                                if fedge in set(fsi_facet_ids)])
    fsi_verts = meshutils.vertices_from_edges(fsi_edges, solid.mesh)
    fsi_coordinates = solid.mesh.coordinates()[fsi_verts]

    # Sort the fsi vertices from inferior to superior
    # NOTE: This only works for a 1D fluid mesh and isn't guaranteed if the VF surface is shaped strangely
    idx_sort = meshutils.sort_vertices_by_nearest_neighbours(fsi_coordinates)
    fsi_verts = fsi_verts[idx_sort]

    # Load the fluid
    fluid_model = None
    if fluid_mesh is None and issubclass(Fluid, fluid.QuasiSteady1DFluid):
        xfluid, yfluid = meshutils.streamwise1dmesh_from_edges(mesh, facet_func, [facet_labels[label] for label in fsi_facet_labels])
        fluid_model = Fluid(xfluid, yfluid)
    elif fluid_mesh is None and not issubclass(Fluid, fluid.QuasiSteady1DFluid):
        raise ValueError(f"`fluid_mesh` cannot be `None` if the fluid is not 1D")
    else:
        raise ValueError(f"This function does not yet support input fluid meshes.")
    return solid, fluid_model, fsi_verts
