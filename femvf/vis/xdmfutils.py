"""
Writes out vertex values from a statefile to xdmf
"""
import sys
import os
from os import path

from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree

import h5py
import numpy as np
import dolfin as dfn
# import xml

sys.path.append('../')
from femvf import statefile as sf

def export_vertex_values(model, statefile_path, export_path):
    """
    Exports vertex values from a state file to another h5 file
    """
    solid = model.solid
    if os.path.isfile(export_path):
        os.remove(export_path)

    with sf.StateFile(model, statefile_path, mode='r') as fi:
        with h5py.File(export_path, mode='w') as fo:

            ## Write the mesh and timing info out
            fo.create_dataset('mesh/solid/coordinates', data=fi.root_group['mesh/solid/coordinates'])
            fo.create_dataset('mesh/solid/connectivity', data=fi.root_group['mesh/solid/connectivity'])

            fo.create_dataset('time', data=fi.root_group['time'])

            vert_to_sdof = solid.vert_to_sdof
            vert_to_vdof = solid.vert_to_vdof

            ## Make empty functions to store vector values
            scalar_func = dfn.Function(solid.scalar_fspace)
            vector_func = dfn.Function(solid.vector_fspace)

            ## Prepare constant variables describing the shape
            N_TIME = fi.size
            N_VERT = solid.mesh.num_vertices()

            ## Write (u, v, a) vertex values
            VALUE_SHAPE = tuple(vector_func.value_shape())

            for label in ['u', 'v', 'a']:
                fo.create_dataset(label, shape=(N_TIME, N_VERT, *VALUE_SHAPE), dtype=np.float64)

            for ii in range(N_TIME):
                u, v, a = fi.get_state(ii)
                for label, vector in zip(['u', 'v', 'a'], [u, v, a]):
                    vector_func.vector()[:] = vector
                    fo[label][ii, ...] = vector_func.vector()[vert_to_vdof].reshape(-1, *VALUE_SHAPE)

            ## Write (q, p) vertex values (pressure only defined)
            VALUE_SHAPE = tuple(scalar_func.value_shape())

            for label in ['pressure']:
                fo.create_dataset(label, shape=(N_TIME, N_VERT, *VALUE_SHAPE), dtype=np.float64)

            for ii in range(N_TIME):
                _, p = fi.get_fluid_state(ii)

                for label, vector in zip(['pressure'], [p]):
                    scalar_func.vector()[:] = model.pressure_fluidord_to_solidord(p)
                    fo[label][ii, ...] = scalar_func.vector()[vert_to_sdof].reshape((-1, *VALUE_SHAPE))

def write_xdmf(model, h5file_path, xdmf_path=None):
    """
    Parameters
    ----------
    h5file_path : str
        path to a file with exported vertex values
    """

    with h5py.File(h5file_path, mode='r') as f:

        N_TIME = f['u'].shape[0]
        N_VERT = f['mesh/solid/coordinates'].shape[0]
        N_CELL = f['mesh/solid/connectivity'].shape[0]

        root = Element('Xdmf')
        root.set('version', '2.0')

        domain = SubElement(root, 'Domain')

        temporal_grid = SubElement(domain, 'Grid', {'GridType': 'Collection',
                                                    'CollectionType': 'Temporal'})

        for ii in range(N_TIME):
            ## Make the grid (they always reference the same h5 dataset)
            grid = SubElement(temporal_grid, 'Grid', {'GridType': 'Uniform'})

            time = SubElement(grid, 'Time', {'TimeType': 'Single', 'Value': f"{f['time'][ii]}"})

            ## Set the mesh topology
            topo = SubElement(grid, 'Topology', {'TopologyType': 'Triangle',
                                                'NumberOfElements': f'{N_CELL}'})

            conn = SubElement(topo, 'DataItem', {
                'Name': 'Connectivity',
                'ItemType': 'Uniform',
                'NumberType': 'Int',
                'Format': 'HDF',
                'Dimensions': tuple_shape_to_xdmf(f['mesh/solid/connectivity'].shape)})
            conn.text = f'{h5file_path}:/mesh/solid/connectivity'

            geom = SubElement(grid, 'Geometry', {'GeometryType': 'XY'})

            coords = SubElement(geom, 'DataItem', {
                'Name': 'Coordinates',
                'ItemType': 'Uniform',
                'NumberType': 'Float',
                'Precision': '8',
                'Format': 'HDF',
                'Dimensions': tuple_shape_to_xdmf(f['mesh/solid/coordinates'].shape)
            })
            coords.text = f'{h5file_path}:/mesh/solid/coordinates'

            ## Write u, v, a data to xdmf
            for label in ['u', 'v', 'a']:
                comp = SubElement(grid, 'Attribute', {
                    'Name': label,
                    'AttributeType': 'Vector',
                    'Center': 'Node'})

                slice = SubElement(comp, 'DataItem', {
                    'ItemType': 'HyperSlab',
                    'NumberType': 'Float',
                    'Precision': '8',
                    'Format': 'HDF',
                    'Dimensions': tuple_shape_to_xdmf(f[label][ii:ii+1, ...].shape)})

                slice_sel = SubElement(slice, 'DataItem', {
                    'Dimensions': '3 3',
                    'Format': 'XML'})
                slice_sel.text = f"{ii} 0 0\n 1 1 1\n 1 {tuple_shape_to_xdmf(f[label].shape[-2:])}"

                slice_data = SubElement(slice, 'DataItem', {
                    'Dimensions': tuple_shape_to_xdmf(f[label].shape),
                    'Format': 'HDF'})
                slice_data.text = f'{h5file_path}:{label}'

            # Write q, p data to xdmf
            for label in ['pressure']:
                comp = SubElement(grid, 'Attribute', {
                    'Name': label,
                    'AttributeType': 'Scalar',
                    'Center': 'Node'})

                slice = SubElement(comp, 'DataItem', {
                    'ItemType': 'HyperSlab',
                    'NumberType': 'Float',
                    'Precision': '8',
                    'Format': 'HDF',
                    'Dimensions': tuple_shape_to_xdmf(f[label][ii:ii+1, ...].shape)})

                slice_sel = SubElement(slice, 'DataItem', {
                    'Dimensions': '3 2',
                    'Format': 'XML'})
                slice_sel.text = f"{ii} 0\n1 1\n1 {tuple_shape_to_xdmf(f[label].shape[-1:])}"

                slice_data = SubElement(slice, 'DataItem', {
                    'Dimensions': tuple_shape_to_xdmf(f[label].shape),
                    'Format': 'HDF'})
                slice_data.text = f'{h5file_path}:{label}'

    ## Write the Xdmf file
    lxml_root = etree.fromstring(ElementTree.tostring(root))
    etree.indent(lxml_root, space="    ")
    pretty_xml = etree.tostring(lxml_root, pretty_print=True)

    if xdmf_path is None:
        xdmf_path = f'{path.splitext(h5file_path)[0]}.xdmf'

    with open(xdmf_path, 'wb') as fxml:
        fxml.write(pretty_xml)

# def export_vertex_values_from_opt(model, p, opt_path, export_path):
#     r"""
#     Every optimization file is organized as

#     /iter0/p : The parameterization vector. All values under `iter0` are calculated with p
#     /iter0/obj : The value of the objective function
#     /iter0/grad_obj : The gradient of the objective function
#     /iter0/constraints : The vector of constraints
#     /iter0/jac_constraints : The jacobian of constraints
#     """

#     with h5py.File(opt_path, mode='r') as fi:
#         with h5py.File(export_path, mode='w') as fo:
#             n = 10
#             for ii in range(n):
#                 ## Write the parameterization
#                 p.vector = fi[f'iter{n}/p'][:]
#                 uva, solid_props, fluid_props, timing_props = p.convert()

#                 ## Write datasets for the parameterization
#                 for key in p:
#                     if p.TYPE[key][0] == 'field':

# # def dataset_to_xdmf(path, shape, ):
# #     """
# #     """


# def xdmf_h5_slice(group, slice):
#     """
#     Parameters
#     ----------
#     f : h5py.File
#         h5py File object
#     name : str
#         Name of the data set
#     slice :
#         A slice object to extract from the data set
#     """
#     data_shape = group[ii:ii+1, ...].shape
#     data_item = Element('DataItem', {
#                     'ItemType': 'HyperSlab',
#                     'NumberType': 'Float',
#                     'Precision': '8',
#                     'Format': 'HDF',
#                     'Dimensions': tuple_shape_to_xdmf()})

#     slice_sel = SubElement(slice, 'DataItem', {
#         'Dimensions': '3 3',
#         'Format': 'XML'})
#     slice_sel.text = f"{ii} 0 0\n 1 1 1\n 1 {tuple_shape_to_xdmf(group.shape[-2:])}"

#     slice_data = SubElement(slice, 'DataItem', {
#         'Dimensions': tuple_shape_to_xdmf(f[label].shape),
#         'Format': 'HDF'})
#     slice_data.text = f'{h5file_path}:{label}'

# def tuple_shape_to_xdmf(shape):
#             return r' '.join(str(dim) for dim in shape)