"""
Writes out vertex values from a statefile to xdmf
"""

from typing import Union, Tuple, Optional

import os
from os import path

from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree

import h5py
import numpy as np
import dolfin as dfn

# from .. import statefile as sf

AxisSize = int
Shape = Tuple[AxisSize, ...]

AxisIndex = Union[int, slice, type(Ellipsis)]
AxisIndices = Tuple[AxisIndex, ...]

def xdmf_shape(shape: Shape) -> str:
    """
    Return a shape tuple as an XDMF string
    """
    return r' '.join(str(dim) for dim in shape)

class XDMFArrayIndex:
    """
    Return XDMF slice strings from an array

    Parameters
    ----------
    shape: Shape
        The shape of the array
    """

    def __init__(self, shape: Shape):
        self._shape = shape

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self.shape)

    @staticmethod
    def expand_axis_indices(axis_indices: AxisIndices, ndim: int):
        """
        Expand any missing axis indices in an index tuple
        """
        if not isinstance(axis_indices, tuple):
            axis_indices = (axis_indices,)

        assert axis_indices.count(Ellipsis) < 2

        if Ellipsis in axis_indices:
            # This is the number of missing, explicit, axis indices
            ndim_expand = ndim - len(axis_indices) + 1
            # If an ellipsis exists, then add missing axis indices at the
            # ellipsis
            ii_split = axis_indices.index(Ellipsis)
        else:
            # This is the number of missing, explicit, axis indices
            ndim_expand = ndim - len(axis_indices)
            # If no ellipsis exists, then add missing axis indices starting at 0
            ii_split = 0

        # Here add `[:]` slices to all missing axis indices
        expanded_axis_indices = (
            axis_indices[:ii_split]
            + ndim_expand*(slice(None),)
            + axis_indices[ii_split+1:]
        )
        return expanded_axis_indices

    @staticmethod
    def get_start(axis_index: AxisIndex, axis_size: int):
        """
        Return the start of an axis index
        """
        if isinstance(axis_index, slice):
            if axis_index.start is None:
                start = 0
            else:
                start = axis_index.start
        elif isinstance(axis_index, int):
            start = axis_index
        elif axis_index is Ellipsis:
            raise TypeError("Invalid `Ellipsis` axis index")
        return start

    @staticmethod
    def get_stop(axis_index: AxisIndex, axis_size: int):
        """
        Return the stop of an axis index
        """
        if isinstance(axis_index, slice):
            if axis_index.stop is None:
                stop = axis_size
            else:
                stop = axis_index.stop
        elif isinstance(axis_index, int):
            stop = axis_index + 1
        elif axis_index is Ellipsis:
            raise TypeError("Invalid `Ellipsis` axis index")
        return stop

    @staticmethod
    def get_step(axis_index: AxisIndex, axis_size: int):
        """
        Return the step of an axis index
        """
        if isinstance(axis_index, slice):
            if axis_index.step is None:
                step = 1
            else:
                step = axis_index.step
        elif isinstance(axis_index, int):
            step = 1
        elif axis_index is Ellipsis:
            raise TypeError("Invalid `Ellipsis` axis index")
        return step

    def __getitem__(self, axis_indices: AxisIndices):
        """
        Return the XDMF array slice string representation of `index`
        """
        axis_indices = self.expand_axis_indices(axis_indices, self.ndim)

        starts = [
            str(self.get_start(axis_index, axis_size))
            for axis_index, axis_size in zip(axis_indices, self.shape)
        ]
        stops = [
            str(self.get_stop(axis_index, axis_size))
            for axis_index, axis_size in zip(axis_indices, self.shape)
        ]
        steps = [
            str(self.get_step(axis_index, axis_size))
            for axis_index, axis_size in zip(axis_indices, self.shape)
        ]
        col_widths = [
            max(len(start), len(stop), len(step))
            for start, stop, step in zip(starts, stops, steps)
        ]

        row = ' '.join([f'{{:>{width}s}}' for width in col_widths])
        return (
            row.format(*starts) + '\n'
            + row.format(*steps) + '\n'
            + row.format(*stops)
        )


def export_vertex_values(model, state_file, export_path, post_file=None):
    """
    Exports vertex values from a state file to another h5 file
    """
    solid = model.solid
    if os.path.isfile(export_path):
        os.remove(export_path)

    ## Input data
    with h5py.File(export_path, mode='w') as fo:

        ## Write the mesh and timing info out
        fo.create_dataset(
            'mesh/solid/coordinates',
            data=state_file.file['mesh/solid/coordinates']
        )
        fo.create_dataset(
            'mesh/solid/connectivity',
            data=state_file.file['mesh/solid/connectivity']
        )

        fo.create_dataset('time', data=state_file.file['time'])

        solid = model.solid
        fspace_dg0 = dfn.FunctionSpace(solid.residual.mesh(), 'DG', 0)
        fspace_cg1_scalar = solid.residual.form['coeff.fsi.p1'].function_space()
        fspace_cg1_vector = solid.residual.form['coeff.state.u1'].function_space()
        vert_to_sdof = dfn.vertex_to_dof_map(fspace_cg1_scalar)
        vert_to_vdof = dfn.vertex_to_dof_map(fspace_cg1_vector)

        ## Make empty functions to store vector values
        scalar_func = dfn.Function(fspace_cg1_scalar)
        vector_func = dfn.Function(fspace_cg1_vector)

        ## Prepare constant variables describing the shape
        N_TIME = state_file.size
        N_VERT = solid.residual.mesh().num_vertices()
        VECTOR_VALUE_SHAPE = tuple(vector_func.value_shape())
        SCALAR_VALUE_SHAPE = tuple(scalar_func.value_shape())

        ## Initialize solid/fluid state variables
        vector_labels = ['state/u', 'state/v', 'state/a']
        for label in vector_labels:
            fo.create_dataset(
                label, shape=(N_TIME, N_VERT, *VECTOR_VALUE_SHAPE),
                dtype=np.float64
            )

        scalar_labels = ['p']
        for label in scalar_labels:
            fo.create_dataset(
                label, shape=(N_TIME, N_VERT, *SCALAR_VALUE_SHAPE),
                dtype=np.float64
            )

        ## Write solid/fluid state variables in vertex order
        for ii in range(N_TIME):
            state = state_file.get_state(ii)
            model.set_fin_state(state)
            model.set_ini_state(state)

            u, v, a = state['u'], state['v'], state['a']
            for label, vector in zip(vector_labels, [u, v, a]):
                vector_func.vector()[:] = vector
                fo[label][ii, ...] = vector_func.vector()[vert_to_vdof].reshape(-1, *VECTOR_VALUE_SHAPE)

            p = model.solid.control['p']
            for label, scalar in zip(scalar_labels, [p]):
                scalar_func.vector()[:] = scalar
                fo[label][ii, ...] = scalar_func.vector()[vert_to_sdof].reshape((-1, *SCALAR_VALUE_SHAPE))

        ## Write (q, p) vertex values (pressure only defined)

        ## Write post-processed scalars
        if post_file is not None:
            labels = ['field.tavg_strain_energy', 'field.tavg_viscous_rate', 'field.vswell']
            for label in labels:
                fo[label] = post_file[label][:]

def write_xdmf(model, h5file_path, xdmf_name=None):
    """
    Parameters
    ----------
    h5file_path : str
        path to a file with exported vertex values
    """

    root_dir = path.split(h5file_path)[0]
    h5file_name = path.split(h5file_path)[1]

    with h5py.File(h5file_path, mode='r') as f:

        N_TIME = f['state/u'].shape[0]
        N_VERT = f['mesh/solid/coordinates'].shape[0]
        N_CELL = f['mesh/solid/connectivity'].shape[0]

        # breakpoint()

        root = Element('Xdmf')
        root.set('version', '2.0')

        domain = SubElement(root, 'Domain')

        # Grid for static data
        grid = SubElement(domain, 'Grid', {'GridType': 'Uniform'})

        # Handle options for 2D/3D meshes
        mesh = model.solid.residual.mesh()
        mesh_dim = mesh.topology().dim()

        add_xdmf_grid_topology(grid, f, h5file_path, mesh_dim)
        add_xdmf_grid_geometry(grid, f, h5file_path, mesh_dim)

        for label in ['state/u']:
            add_xdmf_array(
                grid, label, h5file_path, f[label], (slice(0, 1), ...),
                value_type='vector', value_center='node'
            )

        # scalar_labels = [
        #     'field.tavg_viscous_rate', 'field.tavg_strain_energy', 'field.vswell'
        # ]

        # for label in scalar_labels:
        #     add_xdmf_array(
        #         grid, label, h5file_name, f[label], (slice(None),),
        #         value_type='scalar', value_center='cell'
        #     )

        ## Temporal data
        temporal_grid = SubElement(
            domain, 'Grid', {
                'GridType': 'Collection',
                'CollectionType': 'Temporal'
            }
        )
        for ii in range(N_TIME):
            ## Make the grid (they always reference the same h5 dataset)
            grid = SubElement(temporal_grid, 'Grid', {'GridType': 'Uniform'})

            time = SubElement(
                grid, 'Time', {
                    'TimeType': 'Single',
                    'Value': f"{f['time'][ii]}"
                }
            )

            ## Set the mesh topology

            # Handle options for 2D/3D meshes
            mesh = model.solid.residual.mesh()
            if mesh.topology().dim() == 3:
                topology_type = 'Tetrahedron'
                geometry_type = 'XYZ'
            else:
                topology_type = 'Triangle'
                geometry_type = 'XY'

            add_xdmf_grid_topology(grid, f, h5file_path, mesh_dim)
            add_xdmf_grid_geometry(grid, f, h5file_path, mesh_dim)

            ## Write u, v, a data to xdmf
            solid_labels = ['state/u', 'state/v', 'state/a']
            # solid_labels = []
            for label in solid_labels:

                ## This assumes the data is the raw fenics data

                ## This assumes data is in vertex order

                comp = add_xdmf_array(
                    grid, label, h5file_name, f[label], (slice(ii, ii+1), ...),
                    value_type='vector', value_center='node'
                )

            # Write q, p data to xdmf
            scalar_labels = ['p']
            for label in scalar_labels:
                add_xdmf_array(
                    grid, label, h5file_path, f[label], (slice(ii, ii+1), ...),
                    value_type='scalar', value_center='node'
                )

    ## Write the XDMF file
    lxml_root = etree.fromstring(ElementTree.tostring(root))
    etree.indent(lxml_root, space="    ")
    pretty_xml = etree.tostring(lxml_root, pretty_print=True)

    if xdmf_name is None:
        xdmf_name = f'{path.splitext(h5file_name)[0]}.xdmf'

    with open(path.join(root_dir, xdmf_name), 'wb') as fxml:
        fxml.write(pretty_xml)

def add_xdmf_grid_topology(grid: Element, f, h5file_name: str, mesh_dim=2):

    if mesh_dim == 3:
        topology_type = 'Tetrahedron'
    else:
        topology_type = 'Triangle'

    N_CELL = f['mesh/solid/connectivity'].shape[0]

    topo = SubElement(
        grid, 'Topology', {
            'TopologyType': topology_type,
            'NumberOfElements': f'{N_CELL}'
        }
    )

    conn = SubElement(
        topo, 'DataItem', {
            'Name': 'MeshConnectivity',
            'ItemType': 'Uniform',
            'NumberType': 'Int',
            'Format': 'HDF',
            'Dimensions': xdmf_shape(f['mesh/solid/connectivity'].shape)
        }
    )
    conn.text = f'{h5file_name}:/mesh/solid/connectivity'

def add_xdmf_grid_geometry(grid: Element, f, h5file_name: str, mesh_dim=2):
    if mesh_dim == 3:
        geometry_type = 'XYZ'
    else:
        geometry_type = 'XY'

    geom = SubElement(grid, 'Geometry', {'GeometryType': geometry_type})

    coords = SubElement(
        geom, 'DataItem', {
            'Name': 'MeshCoordinates',
            'ItemType': 'Uniform',
            'NumberType': 'Float',
            'Precision': '8',
            'Format': 'HDF',
            'Dimensions': xdmf_shape(f['mesh/solid/coordinates'].shape)
        }
    )
    coords.text = f'{h5file_name}:/mesh/solid/coordinates'

def add_xdmf_array(
        grid: Element,
        label: str,
        dataset_fpath: str,
        dataset: h5py.Dataset,
        axis_indices: Optional[AxisIndices]=None,
        value_type='Vector',
        value_center='Node'
    ):
    comp = SubElement(
        grid, 'Attribute', {
            'Name': label,
            'AttributeType': value_type,
            'Center': value_center
        }
    )

    shape = dataset.shape

    data_subset = SubElement(
        comp, 'DataItem', {
            'ItemType': 'HyperSlab',
            'NumberType': 'Float',
            'Precision': '8',
            'Format': 'HDF',
            'Dimensions': xdmf_shape(dataset[axis_indices].shape)
        }
    )
    slice_sel = SubElement(
        data_subset, 'DataItem', {
            'Dimensions': f'3 {len(shape):d}',
            'Format': 'XML'
        }
    )
    xdmf_array = XDMFArrayIndex(shape)
    slice_sel.text = xdmf_array[axis_indices]

    slice_data = SubElement(
        data_subset, 'DataItem', {
            'Dimensions': xdmf_shape(shape),
            'Format': 'HDF'
        }
    )

    slice_data.text = f'{dataset_fpath}:{dataset.name}'
    return comp

def add_xdmf_finite_element_function(
        grid, label,
        family='CG', degree=1, cell='triangle'
    ):
    comp = SubElement(
        grid, 'Attribute', {
            'Name': label,
            'AttributeType': 'Vector',
            'Center': 'Other',
            'ItemType': 'FiniteElementFunction',
            'ElementFamily': family,
            'ElementDegree': degree,
            'ElementCell': cell
        }
    )

    dofmap = SubElement(
        comp, 'DataItem', {
            'Name': 'dofmap',
            'ItemType': 'Uniform',
            'NumberType': 'Int',
            'Format': 'HDF',
            'Dimensions': format_shape_tuple(f['dofmap/CG1'].shape)
        }
    )
    dofmap.text = f'{h5file_name}:dofmap/CG1'

    data_subset = SubElement(
        comp, 'DataItem', {
            'ItemType': 'HyperSlab',
            'NumberType': 'Float',
            'Precision': '8',
            'Format': 'HDF',
            'Dimensions': format_shape_tuple(f[label][ii:ii+1, ...].shape)
        }
    )

    slice_sel = SubElement(
        data_subset, 'DataItem', {
            'Dimensions': '3 2',
            'Format': 'XML'
        }
    )
    slice_sel.text = (
        f"{ii} 0\n"
        "1 1\n"
        f"{ii+1} {f[label].shape[-1]}"
    )

    slice_data = SubElement(
        data_subset, 'DataItem', {
            'Dimensions': format_shape_tuple(f[label].shape),
            'Format': 'HDF'
        }
    )
    slice_data.text = f'{h5file_name}:{label}'
