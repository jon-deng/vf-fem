"""
Functionality for dealing with meshes
"""

from os import path

import meshio as mio
import numpy as np
import dolfin as dfn

def split_meshio_cells(mesh):
    """
    Splits a `meshio.Mesh` into separate meshes with one cell block each

    Parameters
    ----------
    mesh: meshio.Mesh
    """
    meshes = [
        mio.Mesh(
            mesh.points,
            [cellblock],
            cell_data={
                key: [arrays[n]]
                for key, arrays in mesh.cell_data.items()
            }
        )
        for n, cellblock in enumerate(mesh.cells)
    ]
    return meshes

def dim_from_type(cell_type):
    """
    Return the topoligical dimension of a cellblock type
    """
    if cell_type in {'point', 'vertex', 'node'}:
        return 0
    elif cell_type in {'line'}:
        return 1
    elif cell_type in {'triangle'}:
        return 2
    elif cell_type in {'tetrahedron'}:
        return 3
    else:
        raise ValueError(f"Unknown cell type {cell_type}")

def load_fenics_gmsh(mesh_path):
    """
    Loads a fenics mesh from .msh file

    Parameters
    ----------
    mesh_path: str
        Path of the .msh file
    """
    mesh_dir, mesh_fname = path.split(mesh_path)
    mesh_name, mesh_ext = path.splitext(mesh_fname)

    mio_mesh = mio.read(mesh_path)

    # Check if the z-coordinate is uniformly zero. If it is then automatically
    # trim the z-coordinate to create a 2D mesh
    if mio_mesh.points.shape[1] == 3:
        if np.all(mio_mesh.points[:, 2] == 0):
            mio_mesh = mio.Mesh(
                mio_mesh.points[:, :2],
                mio_mesh.cells,
                cell_data=mio_mesh.cell_data,
                field_data=mio_mesh.field_data
            )

    # First load some basic information about the mesh and check compatibility
    # with fenics
    cell_types = [cell.type for cell in mio_mesh.cells]
    cell_dims = [dim_from_type(cell_type) for cell_type in cell_types]

    max_dim = max(cell_dims)
    for dim in range(max_dim+1):
        if cell_dims.count(max_dim) > 1:
            raise ValueError(
                f"The mesh contains multiple cell types of dimension {dim}"
                " which is not supported!"
            )

    # Split multiple cell blocks in mio_mesh to separate meshes
    split_meshes = split_meshio_cells(mio_mesh)
    split_mesh_names = [
        f'{mesh_name}_{n}_{cell_block_type}'
        for n, cell_block_type in enumerate(cell_types)
    ]
    split_mesh_paths = [
        path.join(mesh_dir, f'{split_mesh_name}.xdmf')
        for split_mesh_name in split_mesh_names
    ]
    for (split_path_n, mesh_n) in zip(split_mesh_paths, split_meshes):
        mio.write(split_path_n, mesh_n)

    # Read the highest-dimensional mesh as the base dolfin mesh,
    # which is assumed to be the last cell block
    dfn_mesh = dfn.Mesh()
    with dfn.XDMFFile(split_mesh_paths[-1]) as f:
        f.read(dfn_mesh)

    # Create `MeshValueCollection`s for each split cell block
    vcs = [
        dfn.MeshValueCollection('size_t', dfn_mesh, cell_dim)
        for cell_dim in cell_dims
    ]
    for vc, split_mesh_path in zip(vcs, split_mesh_paths):
        with dfn.XDMFFile(split_mesh_path) as f:
            f.read(vc, 'gmsh:physical')

    # Create a `MeshFunction` from each `MeshValueCollection` ordered by increasing
    # dimension. If entities of a certain dimensions have no mesh function, they
    # are None
    _mfs = [dfn.MeshFunction('size_t', dfn_mesh, vc) for vc in vcs]
    dim_to_mf = {mf.dim(): mf for mf in _mfs}
    mfs = [dim_to_mf[dim] if dim in dim_to_mf else None for dim in range(max_dim+1)]

    # Load mappings of 'field data' These associate labels to mesh function values
    entities_label_to_id = [
        {
            key: value
            for key, (value, entity_dim) in mio_mesh.field_data.items()
            if entity_dim == dim
        }
        for dim in range(max_dim+1)
    ]

    return dfn_mesh, tuple(mfs), tuple(entities_label_to_id)

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

    return mesh, (None, facet_function, cell_function), (vertexlabel_to_id, facetlabel_to_id, celllabel_to_id)

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

def dofs_from_mesh_func(mesh, mesh_func, mesh_func_value, dofmap):
    """
    Return all DOFs where an integer mesh function equals a value
    """
    mesh_ent_indices = [
        mesh_ent.index()
        for mesh_ent in dfn.cpp.mesh.entities(mesh, mesh_func.dim())
        if mesh_func[mesh_ent.index()] == mesh_func_value]

    dofs = dofmap.entity_closure_dofs(mesh, mesh_func.dim(), mesh_ent_indices)

    return np.array(list(set(dofs)))

def process_meshlabel_to_dofs(mesh, mesh_func, func_space, label_to_meshfunc):
    """
    Return a map from mesh region label(s) to DOFs associated with the region(s)
    """
    dofmap = func_space.dofmap()
    label_to_dofs = {
        region_label: dofs_from_mesh_func(mesh, mesh_func, region_value, dofmap)
        for region_label, region_value in label_to_meshfunc.items()}

    return label_to_dofs

def process_celllabel_to_dofs_from_forms(forms, func_space):
    """
    Return a map from cell regions to associated dofs
    """
    mesh = forms['mesh.mesh']
    cell_func = forms['mesh.cell_function']
    cell_label_to_id = forms['mesh.cell_label_to_id']
    return process_meshlabel_to_dofs(
        mesh, cell_func, func_space, cell_label_to_id)