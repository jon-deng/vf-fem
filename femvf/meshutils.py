"""
Functionality for dealing with meshes
"""

from typing import Any, Callable
from numpy.typing import NDArray

from os import path
import warnings

import meshio as mio
import numpy as np
import dolfin as dfn

# if typing
# from femvf.residuals.base import FenicsResidual

## Functions for loading `dfn.Mesh` objects from other mesh formats

MeshFunctions = list[dfn.MeshFunction]
MeshSubdomainData = dict[str, int]
MeshSubdomainsData = list[MeshSubdomainData]

def mesh_element_type_dim(element_type: str | int) -> int:
    """
    Return the dimension of an element type

    This can be used to index into a list of mesh functions or mesh subdomains

    Parameters
    ----------
    element_type: str | int
        The element type

        If a string, it can be one of vertex, edge, facet, cell
        If an integer, it is interpreted as the dimension itself

    Returns
    -------
    int
        The dimension of the mesh element
    """
    ELEMENT_TYPE_TO_IDX = {
        'vertex': 0, 'edge': 1, 'facet': -2, 'cell': -1
    }
    if isinstance(element_type, str):
        if element_type in ELEMENT_TYPE_TO_IDX:
            idx = ELEMENT_TYPE_TO_IDX[element_type]
        else:
            raise ValueError(
                "`mesh_element_type` must be one of "
                f"{ELEMENT_TYPE_TO_IDX.keys()}`"
            )
    elif isinstance(element_type, int):
        idx = element_type
    else:
        raise TypeError(
            f"`mesh_element_type` must be `str` or `int`, not "
            f"`{type(element_type)}`"
        )
    return idx

def load_fenics_gmsh(
    mesh_path: str
) -> tuple[dfn.Mesh, MeshFunctions, MeshSubdomainsData]:
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
    mesh_funcs: MeshFunctions
        A `dfn.MeshFunction` for each type of mesh entity (vertex, line, triangle, tetrahedron)

        For example:
        - `mesh_funcs[0]` is a mesh function for vertices
        - `mesh_funcs[1]` is a mesh function for lines, etc.

        Each mesh function represents an integer for value for each mesh entity
    mesh_subdomains: MeshSubdomainsData
        A dictionary of tagged mesh values for each type of mesh entity (vertex, line, triangle, tetrahedron)

        For example:
        - `mesh_subdomains[0]` is a dictionary of tagged vertex values
        - `mesh_subdomains[1]` is a dictionary of tagged line values, etc.

        If `mesh_subdomains[0] == {'A': 3, 'B': 5, 'C': 10}` then:
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
    submesh_cell_types = ('vertex', 'line', 'triangle', 'tetra')[:max_dim+1]
    submesh_dims = (0, 1, 2, 3)[:max_dim+1]
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
    mesh_funcs = [dfn.MeshFunction('size_t', dfn_mesh, vc) for vc in vcs]

    # Load mappings of 'field data' These associate labels to mesh function values
    mesh_subdomains = [
        {
            key: value
            for key, (value, entity_dim) in mio_mesh.field_data.items()
            if entity_dim == dim
        }
        for dim in range(max_dim + 1)
    ]

    return dfn_mesh, tuple(mesh_funcs), tuple(mesh_subdomains)


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


## Filter mesh entities

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

def filter_mesh_entities_by_subdomain(
    mesh_entities: list[dfn.MeshEntity],
    mesh_function: dfn.MeshFunction,
    filtering_mesh_values: set[int]
):
    """
    Return a subset of mesh entities incident to given subdomains

    Parameters
    ----------
    mesh_entities: list[dfn.MeshEntity]
        An iterable of mesh entities to filter
    subdomains: list[str]
        A list of subdomain names
    mesh_function: dfn.MeshFunction
        A mesh function specifying the subdomains

        The mesh function/subdomain dimension does not have to match the
        mesh entity dimension.
    mesh_subdomain_data: dict[str, int]
        A mapping from subdomain names to mesh function values
    """
    def is_incident(
        mesh_entity: dfn.MeshEntity,
        mesh_function: dfn.MeshFunction,
        mesh_values: set[int]
    ):
        """
        Return whether `mesh_entity` is incident to the subdomain
        """
        # Determine if `mesh_entity` is incident to the given subdomains
        subdomain_dim = mesh_function.dim()
        incident_entities = [ent for ent in dfn.entities(mesh_entity, subdomain_dim)]
        incident_entity_values = [mesh_function[ent] for ent in incident_entities]
        return any(
            value in mesh_values for value in incident_entity_values
        )

    return filter_mesh_entities(
        mesh_entities,
        lambda entity: is_incident(entity, mesh_function, filtering_mesh_values)
    )

def filter_mesh_entities_by_plane(
    mesh_entities: list[dfn.MeshEntity],
    origin: NDArray[np.float64]=np.zeros(3),
    normal: NDArray[np.float64]=np.array([0, 0, 1])
):
    """
    Return a subset of mesh entities with midpoints on a plane

    Parameters
    ----------
    mesh_entities: list[dfn.MeshEntity]
        An iterable of mesh entities to filter
    origin: NDArray[np.float64]
        The plane origin
    normal: NDArray[np.float64]
        The plane normal
    """
    def on_plane(mesh_entity):
        midpoint = mesh_entity.midpoint().array()
        normal_distance = np.dot(midpoint-origin, normal)
        return np.isclose(normal_distance, 0)

    return filter_mesh_entities(mesh_entities, on_plane)

def filter_mesh_entities(
    mesh_entities: list[dfn.MeshEntity],
    filter: Callable[[dfn.MeshEntity], bool]
):
    """
    Return a subset of mesh entities incident satisfying a condition

    Parameters
    ----------
    mesh_entities: list[dfn.MeshEntity]
        An iterable of mesh entities to filter
    filter: Callable[[dfn.MeshEntity], bool]
        A function which determines if the mesh entity is included

        If filter returns `True`, then the mesh entity is included.
    """
    return [
        mesh_entity for mesh_entity in mesh_entities if filter(mesh_entity)
    ]


## Functions for sorting medial surface coordinates in a stream-wise manner
# This is needed for getting 1D fluid model coordinates

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
    label_to_meshfunc: dict[str, int],
    dofmap: dfn.DofMap,
) -> dict[str, np.ndarray]:
    """
    Return a mapping from mesh region labels to associated DOFs

    Parameters
    ----------
    mesh: dfn.Mesh
    mesh_func: dfn.MeshFunction
        The mesh function use to specify the mesh region
    label_to_meshfunc: dict[str, int]
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
    residual: "FenicsResidual", dofmap: dfn.DofMap
) -> dict[str, np.ndarray]:
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
    cell_label_to_id = residual.mesh_subdomain('cell')
    return process_meshlabel_to_dofs(mesh, cell_func, cell_label_to_id, dofmap)
