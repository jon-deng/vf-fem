"""
Utilities for creating XDMF files
"""

from typing import Union, Tuple, Optional, List, Callable

from os import path

from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree

import h5py
import numpy as np
import dolfin as dfn

from femvf.models.transient.base import BaseTransientModel

# from .. import statefile as sf

Model = BaseTransientModel

AxisSize = int
Shape = Tuple[AxisSize, ...]

AxisIndex = Union[int, slice, type(Ellipsis)]
AxisIndices = Tuple[AxisIndex, ...]

# This is a tuple consisting of:
# an `h5py.Dataset` object containing the data
# a string (eg. 'vector', 'scalar') indicating whether the data is vector/scalar
# a string (eg. 'node', 'center') indicating where data is located
XDMFValueType = str
XDMFValueCenter = str
DatasetDescription = Tuple[h5py.Dataset, XDMFValueType, XDMFValueCenter]

class XDMFArray:
    """
    Represent an array as defined in the XDMF format

    Parameters
    ----------
    shape: Shape
        The shape of the array
    """

    def __init__(self, shape: Shape):
        self._shape = shape

    @property
    def shape(self) -> Shape:
        return self._shape

    @property
    def xdmf_shape(self) -> str:
        return r' '.join(str(dim) for dim in self.shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @staticmethod
    def expand_axis_indices(axis_indices: AxisIndices, ndim: int):
        """
        Expand any missing axis indices in an index tuple
        """
        assert axis_indices.count(Ellipsis) < 2

        # Here, we cut out a chunk `axis_indices[split_start:split_stop]`
        # and insert default 'slice(None)' slices to fill any missing axis
        # indices
        if Ellipsis in axis_indices:
            # This is the number of missing, explicit, axis indices
            ndim_expand = ndim - len(axis_indices) + 1
            # If an ellipsis exists, then add missing axis indices at the
            # ellipsis
            split_start = axis_indices.index(Ellipsis)
            split_stop = split_start+1
        else:
            # This is the number of missing, explicit, axis indices
            ndim_expand = ndim - len(axis_indices)
            # If no ellipsis exists, then add missing axis indices to the end
            split_start = len(axis_indices)
            split_stop = len(axis_indices)

        # Here add `[:]` slices to all missing axis indices
        expanded_axis_indices = (
            axis_indices[:split_start]
            + ndim_expand*(slice(None),)
            + axis_indices[split_stop:]
        )

        assert len(expanded_axis_indices) == ndim
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

    def to_hyperslab(self, axis_indices: AxisIndices):
        axis_indices = self.expand_axis_indices(axis_indices, self.ndim)

        starts = tuple(
            self.get_start(axis_index, axis_size)
            for axis_index, axis_size in zip(axis_indices, self.shape)
        )
        steps = tuple(
            self.get_step(axis_index, axis_size)
            for axis_index, axis_size in zip(axis_indices, self.shape)
        )
        stops = tuple(
            self.get_stop(axis_index, axis_size)
            for axis_index, axis_size in zip(axis_indices, self.shape)
        )
        counts = tuple(
            (stop-start)//step
            for start, stop, step in zip(starts, stops, steps)
        )
        return starts, steps, counts

    def to_xdmf_hyperslab_str(self, axis_indices: AxisIndices) -> str:
        """
        Return the XDMF array slice string representation of `index`
        """
        starts, steps, counts = self.to_hyperslab(axis_indices)
        starts = [str(start) for start in starts]
        steps = [str(step) for step in steps]
        counts = [str(count) for count in counts]
        col_widths = [
            max(len(start), len(step), len(stop))
            for start, step, stop in zip(starts, steps, counts)
        ]

        row = ' '.join([f'{{:>{width}s}}' for width in col_widths])
        return (
            row.format(*starts) + '\n'
            + row.format(*steps) + '\n'
            + row.format(*counts)
        )

Format = Union[None, dfn.FunctionSpace]
def export_mesh_values(
        datasets: List[Union[h5py.Dataset, h5py.Group]],
        formats: List[Format],
        output_group: h5py.Group,
        output_names: Optional[List[str]]=None
    ):
    """
    Export finite element and other data to mesh based data

    Parameters
    ----------
    datasets: List[Union[h5py.Dataset, h5py.Group]]
        A list of datasets
    formats: List[Format]
        A list of dataset formats

        If the dataset format is `None`, the dataset is assumed to represent
        raw array data. If the dataset format is `dfn.FunctionSpace`, the
        dataset is assumed to represent finite-element data. In this case,
        array values are extracted so that data is centred on mesh elements,
        which can be plotted by Paraview.
    output_group: h5py.Group
        The group to export data to

    Returns
    -------
    output_group: h5py.Group
        The group to export data to
    """
    if output_names is None:
        output_names = [dataset.name for dataset in datasets]

    for dataset_or_group, format, output_name in zip(datasets, formats, output_names):
        if isinstance(dataset_or_group, h5py.Dataset):
            dataset = dataset_or_group
            format_dataset = make_format_dataset(format)
            export_dataset(
                dataset, output_group,
                output_dataset_name=output_name, format_dataset=format_dataset
            )
        elif isinstance(dataset_or_group, h5py.Group):
            input_group = dataset_or_group
            export_group(
                input_group, output_group.require_group(output_name)
            )
        else:
            raise TypeError()

    return output_group

FormatDataset = Callable[[h5py.Dataset], np.ndarray]
def make_format_dataset(
        data_format: Union[dfn.FunctionSpace, None]
    ) -> FormatDataset:
    if isinstance(data_format, dfn.FunctionSpace):
        function_space: dfn.FunctionSpace = data_format
        dofmap = function_space.dofmap()
        mesh = function_space.mesh()

        # Plot CG spaces on mesh vertices and DG space in the mesh interior
        is_cg_space = (
            function_space.ufl_element().family()
            in ('Lagrange', 'CG')
        )
        is_dg_space = (
            function_space.ufl_element().family()
            in ('Discontinuous Lagrange', 'DG')
        )
        if is_cg_space:
            mesh_ent_dofs = np.arange(mesh.num_vertices())
            ent_dim = 0
        elif is_dg_space:
            mesh_ent_dofs = np.arange(mesh.num_cells())
            ent_dim = function_space.ufl_element().cell().topological_dimension()
            ent_dim = mesh.topology().dim()
        else:
            raise ValueError()
        mesh_to_dof = np.array(
            dofmap.entity_dofs(mesh, ent_dim, mesh_ent_dofs)
        )

        # This determines whether the function space is vector/scalar and
        # how many components
        value_dim = max(function_space.num_sub_spaces(), 1)
        def format_dataset(dataset: h5py.Dataset):
            array = dataset[()][..., mesh_to_dof]
            new_shape = (
                array.shape[:-1] + (array.shape[-1]//value_dim,) + (value_dim,)
            )
            array = np.reshape(array, new_shape)
            return array
    else:
        def format_dataset(dataset: h5py.Dataset):
            return dataset[()]
    return format_dataset

def export_dataset(
        input_dataset: h5py.Dataset,
        output_group: h5py.Group, output_dataset_name=None,
        format_dataset=None
    ):
    if output_dataset_name is None:
        output_dataset_name = input_dataset.name
    if format_dataset is None:
        format_dataset = lambda x: x

    dataset = output_group.create_dataset(
        output_dataset_name, data=format_dataset(input_dataset)
    )
    return dataset

def export_group(
        input_group: h5py.Group,
        output_group: h5py.Group,
        idx=None
    ):

    for key, dataset in input_group.items():
        if isinstance(dataset, h5py.Dataset):
            export_dataset(
                dataset, output_group, output_dataset_name=key,
                format_dataset=idx
            )
    return output_group


def write_xdmf(
        mesh_group: h5py.Group,
        static_dataset_descrs: List[DatasetDescription]=None,
        static_dataset_idxs: List[AxisIndices]=None,
        time_dataset: h5py.Dataset=None,
        temporal_dataset_descrs: List[DatasetDescription]=None,
        temporal_dataset_idxs: List[AxisIndices]=None,
        xdmf_fpath: Optional[str]=None
    ) -> str:
    """
    Create an XDMF file describing datasets

    Parameters
    ----------
    mesh_group: h5py.Group
        A group containing mesh information for the datasets
    static_dataset_descrs: List[DatasetDescription]
        A list of static datasets and info on how they are placed on the mesh

        Each element of the list should be a tuple with the format
        `(dataset, XDMFValueType, XDMFValueCenter)` ,
        where:
            - `dataset` is the `h5py.Dataset` containing the array values
            - `XDMFValueType` is a string indicating the value dimension
            ('scalar', 'vector', ...)
            - `XDFMValueCenter` is a string indicating where values are located
            on the mesh ('node', 'center', ...)
    static_dataset_idxs: List[AxisIndices]
        Indices into the datasets
    time_dataset: h5py.Dataset
        A dataset containing simulation times
    temporal_dataset_descrs: List[DatasetDescription]
        A list of temporal datasets and info on how they are placed on the mesh
    temporal_dataset_idxs: List[AxisIndices]
        Indices into the datasets
    xdmf_fpath: Optional[str]
        The path to the XDMF file

    Returns
    -------
    xdmf_fpath: str
        The path of the XDMF file written
    """
    # Set default empty data sets
    if static_dataset_descrs is None:
        static_dataset_descrs = []
    if static_dataset_idxs is None:
        static_dataset_idxs = []
    if temporal_dataset_descrs is None:
        temporal_dataset_descrs = []
    if temporal_dataset_idxs is None:
        temporal_dataset_idxs = []

    root = Element('Xdmf')
    root.set('version', '2.0')

    domain = SubElement(root, 'Domain')

    if xdmf_fpath is None:
        xdmf_basename = path.splitext(path.basename(mesh_group.file.filename))[0]
        xdmf_fpath = f'{xdmf_basename}.xdmf'
    xdmf_dir, xdmf_basename = path.split(xdmf_fpath)

    ## Add info for a static grid
    grid = add_xdmf_uniform_grid(
        domain, 'Static',
        mesh_group,
        static_dataset_descrs, static_dataset_idxs,
        xdmf_dir=xdmf_dir
    )

    ## Add info for a time-varying Grid
    if time_dataset is not None:
        n_time = time_dataset.size
        temporal_grid = SubElement(
            domain, 'Grid', {
                'GridType': 'Collection',
                'CollectionType': 'Temporal',
                'Name': 'Temporal'
            }
        )
        for ii in range(n_time):
            # Temporal dataset indices are assumed to apply to the non-time
            # axes and the time axis is assumed to be the first one
            _temporal_dataset_idxs = [
                (ii,)+idx for idx in temporal_dataset_idxs
            ]
            grid = add_xdmf_uniform_grid(
                temporal_grid, f'Time{ii}',
                mesh_group,
                temporal_dataset_descrs, _temporal_dataset_idxs,
                time=time_dataset[ii], xdmf_dir=xdmf_dir
            )

    ## Write the XDMF file
    lxml_root = etree.fromstring(ElementTree.tostring(root))
    etree.indent(lxml_root, space="    ")
    pretty_xml = etree.tostring(lxml_root, pretty_print=True)


    with open(xdmf_fpath, 'wb') as fxml:
        fxml.write(pretty_xml)

    return xdmf_fpath

def add_xdmf_uniform_grid(
        parent: Element,
        grid_name: str,
        mesh_group: h5py.Group,
        dataset_descrs: List[DatasetDescription],
        dataset_idxs: List[AxisIndices],
        time: float=None,
        xdmf_dir: str='.'
    ):
    grid = SubElement(
        parent, 'Grid', {
            'GridType': 'Uniform',
            'Name': grid_name
        }
    )

    if time is not None:
        time = SubElement(
            grid, 'Time', {
                'TimeType': 'Single',
                'Value': f"{time}"
            }
        )

    # Write mesh info to grid
    mesh_dim = mesh_group['dim'][()]
    add_xdmf_grid_topology(
        grid, mesh_group['connectivity'], mesh_dim, xdmf_dir=xdmf_dir
    )
    add_xdmf_grid_geometry(
        grid, mesh_group['coordinates'], mesh_dim, xdmf_dir=xdmf_dir
    )

    # Write arrays to grid
    for (dataset, value_type, value_center), idx in zip(
            dataset_descrs, dataset_idxs
        ):
        add_xdmf_grid_array(
            grid, dataset.name, dataset, idx,
            value_type=value_type, value_center=value_center,
            xdmf_dir=xdmf_dir
        )

    return grid

def add_xdmf_grid_topology(
        grid: Element, dataset: h5py.Dataset, mesh_dim=2, xdmf_dir='.'
    ):

    if mesh_dim == 3:
        topology_type = 'Tetrahedron'
    else:
        topology_type = 'Triangle'

    N_CELL = dataset.shape[0]

    topo = SubElement(
        grid, 'Topology', {
            'TopologyType': topology_type,
            'NumberOfElements': f'{N_CELL}'
        }
    )

    xdmf_array = XDMFArray(dataset.shape)
    conn = SubElement(
        topo, 'DataItem', {
            'Name': 'MeshConnectivity',
            'ItemType': 'Uniform',
            'NumberType': 'Int',
            'Format': 'HDF',
            'Dimensions': xdmf_array.xdmf_shape
        }
    )
    conn.text = (
        f'{path.relpath(dataset.file.filename, start=xdmf_dir)}'
        f':{dataset.name}'
    )

def add_xdmf_grid_geometry(
        grid: Element, dataset: h5py.Dataset, mesh_dim=2, xdmf_dir='.'
    ):
    if mesh_dim == 3:
        geometry_type = 'XYZ'
    else:
        geometry_type = 'XY'

    geom = SubElement(grid, 'Geometry', {'GeometryType': geometry_type})

    xdmf_array = XDMFArray(dataset.shape)
    coords = SubElement(
        geom, 'DataItem', {
            'Name': 'MeshCoordinates',
            'ItemType': 'Uniform',
            'NumberType': 'Float',
            'Precision': '8',
            'Format': 'HDF',
            'Dimensions': xdmf_array.xdmf_shape
        }
    )
    coords.text = (
        f'{path.relpath(dataset.file.filename, start=xdmf_dir)}'
        f':{dataset.name}'
    )

def add_xdmf_grid_array(
        grid: Element,
        label: str,
        dataset: h5py.Dataset,
        axis_indices: Optional[AxisIndices]=None,
        value_type='Vector',
        value_center='Node',
        xdmf_dir='.'
    ):
    comp = SubElement(
        grid, 'Attribute', {
            'Name': label,
            'AttributeType': value_type,
            'Center': value_center
        }
    )

    shape = dataset.shape

    xdmf_array = XDMFArray(dataset[axis_indices].shape)
    data_subset = SubElement(
        comp, 'DataItem', {
            'ItemType': 'HyperSlab',
            'NumberType': 'Float',
            'Precision': '8',
            'Format': 'HDF',
            'Dimensions': xdmf_array.xdmf_shape
        }
    )
    slice_sel = SubElement(
        data_subset, 'DataItem', {
            'Dimensions': f'3 {len(shape):d}',
            'Format': 'XML'
        }
    )
    xdmf_array = XDMFArray(shape)
    slice_sel.text = xdmf_array.to_xdmf_hyperslab_str(axis_indices)

    slice_data = SubElement(
        data_subset, 'DataItem', {
            'Dimensions': xdmf_array.xdmf_shape,
            'Format': 'HDF'
        }
    )

    slice_data.text = (
        f'{path.relpath(dataset.file.filename, start=xdmf_dir)}'
        f':{dataset.name}'
    )
    return comp

def add_xdmf_grid_finite_element_function(
        grid: Element,
        label: str,
        dataset: h5py.Dataset,
        dataset_dofmap: h5py.Dataset,
        axis_indices: Optional[AxisIndices]=None,
        elem_family='CG', elem_degree=1, elem_cell='triangle',
        elem_value_type='vector',
        xdmf_dir='.'
    ):
    comp = SubElement(
        grid, 'Attribute', {
            'Name': label,
            'AttributeType': elem_value_type,
            'Center': 'Other',
            'ItemType': 'FiniteElementFunction',
            'ElementFamily': elem_family,
            'ElementDegree': elem_degree,
            'ElementCell': elem_cell
        }
    )

    xdmf_array = XDMFArray(dataset_dofmap.shape)
    dofmap = SubElement(
        comp, 'DataItem', {
            'Name': 'dofmap',
            'ItemType': 'Uniform',
            'NumberType': 'Int',
            'Format': 'HDF',
            'Dimensions': xdmf_array.xdmf_shape
        }
    )
    dofmap.text = f'{dataset_dofmap.file.filename}:{dataset_dofmap.name}'

    xdmf_array = XDMFArray(dataset[axis_indices].shape)
    data_subset = SubElement(
        comp, 'DataItem', {
            'ItemType': 'HyperSlab',
            'NumberType': 'Float',
            'Precision': '8',
            'Format': 'HDF',
            'Dimensions': xdmf_array.xdmf_shape
        }
    )

    shape = dataset.shape
    slice_sel = SubElement(
        data_subset, 'DataItem', {
            'Dimensions': f'3 {len(shape)}',
            'Format': 'XML'
        }
    )
    xdmf_array = XDMFArray(shape)
    slice_sel.text = xdmf_array.to_xdmf_hyperslab_str(axis_indices)

    slice_data = SubElement(
        data_subset, 'DataItem', {
            'Dimensions': xdmf_array.xdmf_shape,
            'Format': 'HDF'
        }
    )
    slice_data.text = (
        f'{path.relpath(dataset.file.filename, start=xdmf_dir)}'
        f':{dataset.name}'
    )
