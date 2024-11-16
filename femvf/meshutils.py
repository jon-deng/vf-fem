"""
Functionality for dealing with meshes
"""

from typing import List, Tuple, Mapping, Any
from os import path
import warnings

import meshio as mio
import numpy as np
import dolfin as dfn

from femvf.residuals import solid

## Functions for loading `dfn.Mesh` objects from other mesh formats

MeshFunctions = list[dfn.MeshFunction]
MeshFieldData = dict[str, int]
MeshFieldsData = list[MeshFieldData]

def load_fenics_gmsh(
    mesh_path: str
) -> Tuple[dfn.Mesh, MeshFunctions, MeshFieldsData]:
    """
    Return a `dfn.mesh` and mesh function info from a gmsh '.msh' file

    Parameters
    ----------
    mesh_path: str
        Path of the .msh file
    overwrite_xdmf: bool
        Whether to overwrite created '.xdmf' files. Note that '.xdmf' files are
        created by this function to interface with FEniCS.

    Returns
    -------
    dfn_mesh: dfn.Mesh
        The mesh
    mfs: MeshFunctions
        A `dfn.MeshFunction` for each type of mesh entity (vertex, line, triangle, tetrahedron)

        For example:
        - `mfs[0]` is a mesh function for vertices
        - `mfs[1]` is a mesh function for lines, etc.

        Each mesh function represents an integer for value for each mesh entity
    fields_data: MeshFieldsData
        A dictionary of tagged mesh values for each type of mesh entity (vertex, line, triangle, tetrahedron)

        For example:
        - `fields_data[0]` is a dictionary of tagged vertex values
        - `fields_data[1]` is a dictionary of tagged line values, etc.

        If `fields_data[0] == {'A': 3, 'B': 5, 'C': 10}` then:
        - 'A' references all vertices with value 3
        - 'PointCollectionB' represents all vertices with value 5, etc.
    """
    mesh_dir, mesh_fname = path.split(mesh_path)
    mesh_name, mesh_ext = path.splitext(mesh_fname)

    # This might print a blank line in some meshio versions
    mio_mesh = mio.read(mesh_path, file_format='gmsh')

    # Check if the z-coordinate is uniformly zero. If it is then automatically
    # trim the z-coordinate to create a 2D mesh
    if mio_mesh.points.shape[1] == 3:
        if np.all(mio_mesh.points[:, 2] == 0):
            mio_mesh = mio.Mesh(
                mio_mesh.points[:, :2],
                mio_mesh.cells,
                cell_data=mio_mesh.cell_data,
                field_data=mio_mesh.field_data,
            )

    # First load some basic information about the mesh and check compatibility
    # with fenics
    max_dim = max(set(cell_block.dim for cell_block in mio_mesh.cells))

    # Write out point, line, triangle, tetrahedron meshes separately
    submesh_cell_types = ('vertex', 'line', 'triangle', 'tetrahedron')[:max_dim]
    submesh_dims = (0, 1, 2, 3)[:max_dim]
    submesh_paths = tuple(
        path.join(mesh_dir, f'{mesh_name}_{cell_type}.xdmf')
        for cell_type in submesh_cell_types
    )
    cell_data_dict = mio_mesh.cell_data_dict
    for cell_type, submesh_path in zip(submesh_cell_types, submesh_paths):
        sub_cells = mio.CellBlock(cell_type, mio_mesh.get_cells_type(cell_type))
        sub_cell_data = {
            data_key: [cell_data.get(cell_type, np.array([]))]
            for data_key, cell_data in cell_data_dict.items()
        }
        _mesh = mio.Mesh(mio_mesh.points, [sub_cells], cell_data=sub_cell_data)
        mio.write(submesh_path, _mesh)

    # Read the highest-dimensional mesh as the base dolfin mesh (this is the last submesh)
    dfn_mesh = dfn.Mesh()
    with dfn.XDMFFile(submesh_paths[-1]) as f:
        f.read(dfn_mesh)

    # Create `MeshValueCollection`s for each split cell block
    vcs = [
        dfn.MeshValueCollection('size_t', dfn_mesh, cell_dim)
        for cell_dim in submesh_dims
    ]
    for vc, split_mesh_path in zip(vcs, submesh_paths):
        with dfn.XDMFFile(split_mesh_path) as f:
            f.read(vc, 'gmsh:physical')

    # Create a `MeshFunction` for each mesh entity type ('vertex', 'line', ...)
    mfs = [dfn.MeshFunction('size_t', dfn_mesh, vc) for vc in vcs]

    # Load mappings of 'field data' These associate labels to mesh function values
    fields_data = [
        {
            key: value
            for key, (value, entity_dim) in mio_mesh.field_data.items()
            if entity_dim == dim
        }
        for dim in range(max_dim + 1)
    ]

    return dfn_mesh, tuple(mfs), tuple(fields_data)


def _split_meshio_cells(mesh: mio.Mesh) -> List[mio.Mesh]:
    """
    Return each cell block of a `mio.Mesh` instance as a separate mesh

    Parameters
    ----------
    mesh: mio.Mesh
        The input mesh with multiple cellblocks

    Returns
    -------
    List[mio.Mesh]
        A list of `mio.Mesh` instances corresponding to each cell block
    """
    meshes = [
        mio.Mesh(
            mesh.points,
            [cellblock],
            cell_data={key: [arrays[n]] for key, arrays in mesh.cell_data.items()},
        )
        for n, cellblock in enumerate(mesh.cells)
    ]
    return meshes


def _dim_from_type(cell_type: str) -> int:
    """
    Return the topoligical dimension of a cellblock type

    This function applies for cellblock types as used in `meshio`.
    """
    if cell_type in {'point', 'vertex', 'node'}:
        return 0
    elif cell_type in {'line'}:
        return 1
    elif cell_type in {'triangle'}:
        return 2
    elif cell_type in {'tetrahedron', 'tetra'}:
        return 3
    else:
        raise ValueError(f"Unknown cell type {cell_type}")


# Load a mesh from the FEniCS .xml format
# This format is no longer updated/promoted by FEniCS
def load_fenics_xml(mesh_path: str) -> Tuple[dfn.Mesh, MeshFunctions, MeshFieldsData]:
    """
    Return a `dfn.mesh` and mesh function info from a FEniCS '.xml' file

    Note that the '.xml' format is not actively supported by FEniCS as of 2020.

    Parameters
    ----------
    mesh_path : str
        Path to the mesh .xml file

    Returns
    -------
    dfn.Mesh
        The mesh
    MeshFunctions
        A list of `dfn.MeshFunction` defined in gmsh (corresponding to
        physical groups)
    MeshFieldDatas
        A list of mappings from labelled mesh regions (physical groups) to
        corresponding integer values in the mesh function.
    """
    base_path, ext = path.splitext(mesh_path)
    facet_function_path = base_path + '_facet_region.xml'
    cell_function_path = base_path + '_physical_region.xml'
    msh_function_path = base_path + '.msh'

    if ext == '':
        mesh_path = mesh_path + '.xml'

    mesh = dfn.Mesh(mesh_path)
    facet_function = dfn.MeshFunction('size_t', mesh, facet_function_path)
    cell_function = dfn.MeshFunction('size_t', mesh, cell_function_path)

    vertexlabel_to_id, facetlabel_to_id, celllabel_to_id = _parse_msh2_physical_groups(
        msh_function_path
    )

    return (
        mesh,
        (None, facet_function, cell_function),
        (vertexlabel_to_id, facetlabel_to_id, celllabel_to_id),
    )


def _parse_msh2_physical_groups(mesh_path: str) -> MeshFieldsData:
    """
    Return mappings from physical group labels to integer ids

    Parameters
    ----------
    mesh_path : str
        Path to a .msh file from gmsh. The gmsh file should be of format msh2.

    Returns
    -------
    vertex_label_to_id, facet_label_to_id, cell_label_to_id
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


## Functions for getting z-slices from a 3D mesh


def extract_zplane_facets(mesh, z=0.0):
    """
    Return all facets on a z-normal plane
    """
    facets = [
        facet
        for facet in dfn.facets(mesh)
        if np.isclose(np.abs(np.dot(facet.normal().array(), [0, 0, 1])), 1.0)
        and np.isclose(facet.midpoint().array()[-1], z)
    ]

    return facets


def extract_edges_from_facets(facets, facet_function, facet_values):
    """
    Return all edges from facets that lie on a particular facet value
    """
    edges = [
        edge
        for facet in facets
        for edge in dfn.edges(facet)
        if any(
            [
                any(facet_function[facet] == x for x in facet_values)
                for facet in dfn.facets(edge)
            ]
        )
    ]
    return edges


## Functions for sorting medial surface coordinates in a stream-wise manner
# This is needed for getting 1D fluid model coordinates
def streamwise1dmesh_from_edges(mesh, edge_function, f_edges):
    """
    Returns a list of x, y coordinates of the surface corresponding to edges numbered 'n'.

    f_edges: list or tuple of int
        edge function values to extract

    It is assumed that the beginning of the stream is at the leftmost x-coordinate
    """
    assert isinstance(f_edges, (list, tuple))
    edges = [
        n_edge
        for n_edge, f_edge in enumerate(edge_function.array())
        if f_edge in set(f_edges)
    ]
    return sort_edge_vertices(mesh, edges)


def sort_edge_vertices(mesh, edge_indices):
    """
    Return sorted vertices associated with a set of 1D connected edges
    """
    vertices = vertices_from_edges(mesh, edge_indices)

    surface_coordinates = mesh.coordinates()[vertices]

    # TODO: You can get the vertices in order by using the topology of the edges
    # i.e. determine how the vertices are ordered by connections between edges
    idx_sort = sort_vertices_by_nearest_neighbours(surface_coordinates)
    surface_coordinates = surface_coordinates[idx_sort]

    return surface_coordinates[:, 0], surface_coordinates[:, 1], vertices[idx_sort]


def vertices_from_edges(mesh, edge_indices):
    """
    Return vertices associates with a set of edges
    """
    edge_to_vertex = np.array(
        [[vertex.index() for vertex in dfn.vertices(edge)] for edge in dfn.edges(mesh)]
    )

    vertices = np.unique(edge_to_vertex[edge_indices].reshape(-1))
    return vertices


def sort_vertices_by_nearest_neighbours(
    vertex_coordinates: np.ndarray, origin: np.ndarray = None
) -> np.ndarray:
    """
    Return a permutation that sorts point in succesive order starting from an origin

    For the case of a collection of vertices along the surface of a mesh, this
    should sort them along an increasing or decreasing surface coordinate.

    This is mainly used to sort the inferior-superior direction is oriented
    along the positive x axis.

    Parameters
    ----------
    vertex_coordinates : (..., m) array_like
        An array of surface coordinates, with (x, y, ...) locations stored in
        the last dimension. 'm' is the dimension of the mesh (for example, 3).
    origin : (m,) array_like
        The origin point used to determine which vertex should be first

    Returns
    -------
    np.ndarray
        The permutation array that sorts the points
    """
    # TODO: This function was really meant to extract the list of vertices along
    # a 1D edge chain
    origin = np.zeros(vertex_coordinates.shape[-1]) if origin is None else origin
    # Determine the very first coordinate
    idx_sort = [np.argmin(np.linalg.norm(vertex_coordinates - origin, axis=-1))]

    while len(idx_sort) < vertex_coordinates.shape[0]:
        # Calculate array of distances to every other coordinate
        vector_distances = vertex_coordinates - vertex_coordinates[idx_sort[-1]]
        distances = np.sum(vector_distances**2, axis=-1) ** 0.5
        distances[idx_sort] = np.nan

        idx_sort.append(np.nanargmin(distances))

    return np.array(idx_sort)


## Functions for getting vertices from mesh regions
def verts_from_mesh_func(
    mesh: dfn.Mesh, mesh_func: dfn.MeshFunction, mesh_func_value: int
) -> np.ndarray:
    """
    Return all vertices associated with a mesh region

    The mesh region is specified through a meshfunction and value.

    Parameters
    ----------
    mesh: dfn.Mesh
    mesh_func: dfn.MeshFunction
        The mesh function use to specify the mesh region
    mesh_func_value: int
        The mesh function value corresponding to the desired region

    Returns
    -------
    np.ndarray
        An array of vertex indices associated with the closure of the
        specfied mesh region
    """
    verts = []
    for mesh_entity in dfn.entities(mesh, mesh_func.dim()):
        if mesh_func[mesh_entity.index()] == mesh_func_value:
            # Append on all vertices (dimension 0) attached to the given mesh
            # entity
            verts += mesh_entity.entities(0).tolist()

    return np.unique(verts)


# TODO: Could add functions mimicking the `dofs_from_mesh_func` set of functions
# below, if you end up using this a lot


## Functions for getting DOFs from mesh regions
def dofs_from_mesh_func(
    mesh: dfn.Mesh,
    mesh_func: dfn.MeshFunction,
    mesh_func_value: int,
    dofmap: dfn.DofMap,
) -> np.ndarray:
    """
    Return all DOFs associated with a mesh region

    The mesh region is specified through a meshfunction and value.

    Parameters
    ----------
    mesh: dfn.Mesh
    mesh_func: dfn.MeshFunction
        The mesh function use to specify the mesh region
    mesh_func_value: int
        The mesh function value corresponding to the desired region
    dofmap:
        The Dofmap from the desired function space

    Returns
    -------
    np.ndarray
        An array of DOFs associated with the closure of the specfied mesh region
        and function space (through `dofmap`)
    """
    mesh_ent_indices = [
        mesh_ent.index()
        for mesh_ent in dfn.cpp.mesh.entities(mesh, mesh_func.dim())
        if mesh_func[mesh_ent.index()] == mesh_func_value
    ]

    dofs = dofmap.entity_closure_dofs(mesh, mesh_func.dim(), mesh_ent_indices)

    return np.unique(dofs)


def process_meshlabel_to_dofs(
    mesh: dfn.Mesh,
    mesh_func: dfn.MeshFunction,
    label_to_meshfunc: Mapping[str, int],
    dofmap: dfn.DofMap,
) -> Mapping[str, np.ndarray]:
    """
    Return a mapping from mesh region labels to associated DOFs

    Parameters
    ----------
    mesh: dfn.Mesh
    mesh_func: dfn.MeshFunction
        The mesh function use to specify the mesh region
    label_to_meshfunc: Mapping[str, int]
        A mapping from labels to integer ids corresponding to mesh regions
    dofmap:
        The Dofmap from the desired function space

    Returns
    -------
    np.ndarray
        An array of DOFs associated with the closure of the specfied mesh region
        and function space (through `dofmap`)
    """
    label_to_dofs = {
        label: dofs_from_mesh_func(mesh, mesh_func, value, dofmap)
        for label, value in label_to_meshfunc.items()
    }

    return label_to_dofs


def process_celllabel_to_dofs_from_residual(
    residual: solid.FenicsResidual, dofmap: dfn.DofMap
) -> Mapping[str, np.ndarray]:
    """
    Return a mapping from mesh region labels to associated DOFs

    Parameters
    ----------
    forms:
        A mapping (see `femvf.residuals.solid.solidforms` for examples)
    dofmap:
        The Dofmap from the desired function space

    Returns
    -------
    np.ndarray
        An array of DOFs associated with the closure of the specfied mesh region
        and function space (through `dofmap`)
    """
    mesh = residual.mesh()
    cell_func = residual.mesh_function('cell')
    cell_label_to_id = residual.mesh_function_label_to_value('cell')
    return process_meshlabel_to_dofs(mesh, cell_func, cell_label_to_id, dofmap)
