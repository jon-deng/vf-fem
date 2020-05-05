"""
Functionality for dealing with meshes
"""

from os import path

import numpy as np
import dolfin as dfn

def load_fenics_xmlmesh(mesh_path):
    """
    Return mesh and facet/cell info

    Parameters
    ----------
    mesh_path : str
        Path to the mesh .xml file
    facet_labels, cell_labels : dict(str: int)
        Dictionaries providing integer identifiers for given labels

    Returns
    -------
    mesh : dfn.Mesh
        The mesh object
    facet_function, cell_function : dfn.MeshFunction
        A mesh function marking facets with integers. Marked elements correspond to the label->int
        mapping given in facet_labels/cell_labels.
    """
    base_path, ext = path.splitext(mesh_path)
    facet_function_path = base_path +  '_facet_region.xml'
    cell_function_path = base_path + '_physical_region.xml'
    msh_function_path = base_path + '.msh'

    if ext == '':
        mesh_path = mesh_path + '.xml'

    mesh = dfn.Mesh(mesh_path)
    facet_function = dfn.MeshFunction('size_t', mesh, facet_function_path)
    cell_function = dfn.MeshFunction('size_t', mesh, cell_function_path)

    facet_labels, cell_labels = parse_2d_msh(msh_function_path)

    return mesh, facet_function, cell_function, facet_labels, cell_labels

def parse_2d_msh(mesh_path):
    """
    Parameters
    ----------
    mesh_path : str
        Path to a .msh file from gmsh. The gmsh file should be of format msh2.

    Returns
    -------
    facet_labels
    cell_labels
    """
    with open(mesh_path, 'r') as f:

        current_line = 'NONE'
        while '$PhysicalNames' not in current_line:
            current_line = f.readline()

        num_physical_regions = int(f.readline().rstrip())

        physical_regions = []
        for ii in range(num_physical_regions):
            current_line = f.readline()
            physical_regions.append(current_line.rstrip().split(' '))

        cell_labels = {}
        facet_labels = {}
        for physical_region in physical_regions:
            dim, val, name = physical_region

            # Convert the strings to int, int, and a string without surrounding quotes
            dim, val, name = int(dim), int(val), name[1:-1]

            if dim == 1:
                facet_labels[name] = val
            elif dim == 2:
                cell_labels[name] = val
    return facet_labels, cell_labels

def streamwise1dmesh_from_edges(mesh, edge_function, n):
    """
    Returns a list of x, y coordinates of the surface corresponding to edges numbered 'n'.

    It is assumed that the beginning of the stream is at the leftmost x-coordinate
    """
    edges = edge_function.where_equal(n)

    vertices = vertices_from_edges(edges, mesh)

    surface_coordinates = mesh.coordinates()[vertices]

    idx_sort = sort_vertices_by_nearest_neighbours(surface_coordinates)
    surface_coordinates = surface_coordinates[idx_sort]

    return surface_coordinates[:, 0], surface_coordinates[:, 1]

def vertices_from_edges(edge_indices, mesh):
    """
    Return vertices associates with a set of edges
    """
    edge_to_vertex = np.array([[vertex.index() for vertex in dfn.vertices(edge)]
                                for edge in dfn.edges(mesh)])

    vertices = np.unique(edge_to_vertex[edge_indices].reshape(-1))
    return vertices

def sort_vertices_by_nearest_neighbours(vertex_coordinates):
    """
    Return an index list sorting the vertices based on its nearest neighbours

    For the case of a collection of vertices along the surface of a mesh, this should sort them
    along an increasing or decreasing surface coordinate.

    This is mainly used to sort the inferior-superior direction is oriented along the positive x axis.

    Parameters
    ----------
    vertex_coordinates : (..., 2) array_like
        An array of surface coordinates, with x and y locations stored in the last dimension.
    """
    # Determine the very first coordinate
    idx_sort = [np.argmin(vertex_coordinates[..., 0])]

    while len(idx_sort) < vertex_coordinates.shape[0]:
        # Calculate array of distances to every other coordinate
        vector_distances = vertex_coordinates - vertex_coordinates[idx_sort[-1]]
        distances = np.sum(vector_distances**2, axis=-1)**0.5
        distances[idx_sort] = np.nan

        idx_sort.append(np.nanargmin(distances))

    return np.array(idx_sort)
