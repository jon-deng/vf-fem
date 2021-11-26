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

    vertexlabel_to_id, facetlabel_to_id, celllabel_to_id = parse_msh2_physical_groups(msh_function_path)

    return mesh, facet_function, cell_function, (vertexlabel_to_id, facetlabel_to_id, celllabel_to_id)

def parse_msh2_physical_groups(mesh_path):
    """
    Parameters
    ----------
    mesh_path : str
        Path to a .msh file from gmsh. The gmsh file should be of format msh2.

    Returns
    -------
    vertex_label_to_id
    facet_label_to_id
    cell_label_to_id
        Mappings from a string to numeric identifier for the physical group
    """
    with open(mesh_path, 'r') as f:

        current_line = ''
        while '$PhysicalNames' not in current_line:
            current_line = f.readline()

        num_physical_regions = int(f.readline().rstrip())

        physical_group_descrs = []
        for ii in range(num_physical_regions):
            current_line = f.readline()
            physical_group_descrs.append(current_line.rstrip().split(' '))

        vertexlabel_to_id = {}
        celllabel_to_id = {}
        facetlabel_to_id = {}
        for physical_group_descr in physical_group_descrs:
            topo_dim, val, name = physical_group_descr

            # Convert the strings to int, int, and a string without surrounding quotes
            topo_dim, val, name = int(topo_dim), int(val), name[1:-1]

            if topo_dim == 0:
                vertexlabel_to_id[name] = val
            if topo_dim == 1:
                facetlabel_to_id[name] = val
            elif topo_dim == 2:
                celllabel_to_id[name] = val
    return vertexlabel_to_id, facetlabel_to_id, celllabel_to_id

def streamwise1dmesh_from_edges(mesh, edge_function, f_edges):
    """
    Returns a list of x, y coordinates of the surface corresponding to edges numbered 'n'.

    f_edges: list or tuple of int
        edge function values to extract

    It is assumed that the beginning of the stream is at the leftmost x-coordinate
    """
    assert isinstance(f_edges, (list, tuple))
    edges = [n_edge for n_edge, f_edge in enumerate(edge_function.array()) if f_edge in set(f_edges)]
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

def sort_vertices_by_nearest_neighbours(vertex_coordinates, origin=None):
    """
    Return an index list sorting the vertices based on its nearest neighbours

    For the case of a collection of vertices along the surface of a mesh, this should sort them
    along an increasing or decreasing surface coordinate.

    This is mainly used to sort the inferior-superior direction is oriented along the positive x axis.

    Parameters
    ----------
    vertex_coordinates : (..., m) array_like
        An array of surface coordinates, with (x, y, ...) locations stored in the last dimension.
    origin : (m,) array_like
        The origin point used to determine which vertex should be first
    """
    origin = np.zeros(vertex_coordinates.shape[-1]) if origin is None else origin
    # Determine the very first coordinate
    idx_sort = [np.argmin(np.linalg.norm(vertex_coordinates-origin, axis=-1))]

    while len(idx_sort) < vertex_coordinates.shape[0]:
        # Calculate array of distances to every other coordinate
        vector_distances = vertex_coordinates - vertex_coordinates[idx_sort[-1]]
        distances = np.sum(vector_distances**2, axis=-1)**0.5
        distances[idx_sort] = np.nan

        idx_sort.append(np.nanargmin(distances))

    return np.array(idx_sort)

def verts_from_cell_function(mesh, cell_func, id):
    verts = []
    for n_cell, cell_verts in enumerate(mesh.cells()): 
        if cell_func[n_cell] == id:
            verts += cell_verts.tolist()

    return np.unique(verts)

def verts_from_facet_function(mesh, facet_func, id):
    verts = []
    for facet in dfn.facets(mesh):
        if facet_func[facet.index()] == id:
            verts += [vert.index() for vert in dfn.vertices(facet)]

    return np.unique(verts)

def dofs_from_cell_function(mesh, cell_func, cell_func_value, dofmap):
    dofs = []
    for cell in dfn.cells(mesh): 
        if cell_func[cell.index()] == cell_func_value:
            dofs += dofmap.cell_dofs(cell.index()).tolist()

    return np.unique(dofs)
