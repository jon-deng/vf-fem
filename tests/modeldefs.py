"""
This module contains code that loads some "standard" models for testing
"""

import os
from time import perf_counter

import unittest

import numpy as np
import matplotlib.pyplot as plt
import dolfin as dfn

from femvf import linalg
from femvf.models import (
    Rayleigh, KelvinVoigt, Bernoulli, WRAnalog)
from femvf.load import load_fsi_model, load_fsai_model
from femvf.constants import PASCAL_TO_CGS

def load_fsi_rayleigh_model(coupling='explicit'):
    ## Set the mesh to be used and initialize the forward model
    mesh_dir = '../meshes'
    mesh_base_filename = 'geometry2'
    mesh_base_filename = 'M5-3layers-refined'

    mesh_path = os.path.join(mesh_dir, mesh_base_filename + '.xml')
    model = load_fsi_model(mesh_path, None, SolidType=Rayleigh, FluidType=Bernoulli, coupling=coupling)

    ## Set the fluid/solid parameters
    emod = 2.5e3 * PASCAL_TO_CGS

    k_coll = 1e11
    y_gap = 0.02
    y_coll_offset = 0.01
    zeta_amin, zeta_sep, zeta_ainv = 1/3000, 1/50, 0.002

    fluid_props = model.fluid.get_properties_vec()
    fluid_props['y_midline'][()] = np.max(model.solid.mesh.coordinates()[..., 1]) + y_gap
    fluid_props['zeta_amin'][()] = zeta_amin
    fluid_props['zeta_sep'][()] = zeta_sep
    fluid_props['zeta_ainv'][()] = zeta_ainv

    solid_props = model.solid.get_properties_vec()
    solid_props['emod'][:] = emod
    solid_props['rayleigh_m'][()] = 0.0
    solid_props['rayleigh_k'][()] = 3e-4
    solid_props['kcontact'][()] = k_coll
    solid_props['ycontact'][()] = fluid_props['y_midline'][()] - y_coll_offset

    return model, linalg.concatenate(solid_props, fluid_props)

def load_fsi_kelvinvoigt_model(coupling='explicit'):
    ## Set the mesh to be used and initialize the forward model
    mesh_dir = '../meshes'
    mesh_base_filename = 'geometry2'
    mesh_base_filename = 'M5-3layers-medial-surface-refinement'

    mesh_path = os.path.join(mesh_dir, mesh_base_filename + '.xml')
    model = load_fsi_model(mesh_path, None, SolidType=KelvinVoigt, FluidType=Bernoulli, coupling=coupling)

    ## Set the fluid/solid parameters
    emod = 6e3 * PASCAL_TO_CGS

    k_coll = 1e13
    y_gap = 0.02
    y_gap = 0.1
    y_coll_offset = 0.01

    y_gap = 0.01
    y_coll_offset = 0.0025
    zeta_amin, zeta_sep, zeta_ainv = 1/3000, 1/50, 0.002

    fluid_props = model.fluid.get_properties_vec()
    fluid_props['y_midline'][()] = np.max(model.solid.mesh.coordinates()[..., 1]) + y_gap
    fluid_props['zeta_amin'][()] = zeta_amin
    fluid_props['zeta_sep'][()] = zeta_sep
    fluid_props['zeta_ainv'][()] = zeta_ainv
    fluid_props['ygap_lb'][()] = y_coll_offset
    # fluid_props['ygap_lb'][()] = -10000

    solid_props = model.solid.get_properties_vec()
    solid_props['emod'][:] = emod
    solid_props['eta'][()] = 3.0
    solid_props['kcontact'][()] = k_coll
    # solid_props['ycontact'][()] = fluid_props['y_midline'][()] - y_coll_offset
    solid_props['ycontact'][()] = fluid_props['y_midline'] - y_coll_offset

    return model, linalg.concatenate(solid_props, fluid_props)

def load_fsai_rayleigh_model(coupling='explicit'):
    mesh_dir = '../meshes'
    mesh_base_filename = 'geometry2'
    mesh_base_filename = 'M5-3layers-refined'
    mesh_path = os.path.join(mesh_dir, mesh_base_filename + '.xml')

    ## Configure the model and its parameters
    acoustic = WRAnalog(44)
    model = load_fsai_model(mesh_path, None, acoustic, SolidType=Rayleigh, FluidType=Bernoulli,
                            coupling='explicit')

    # Set the properties
    y_gap = 0.01
    zeta_amin, zeta_sep, zeta_ainv = 1/3000, 1/50, 0.002

    fl_props = model.fluid.get_properties_vec(set_default=True)
    fl_props['y_midline'][()] = np.max(model.solid.mesh.coordinates()[..., 1]) + y_gap
    fl_props['zeta_amin'][()] = zeta_amin
    fl_props['zeta_sep'][()] = zeta_sep
    fl_props['zeta_ainv'][()] = zeta_ainv

    sl_props = model.solid.get_properties_vec(set_default=True)
    xy = model.solid.scalar_fspace.tabulate_dof_coordinates()
    x = xy[:, 0]
    y = xy[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    sl_props['emod'][:] = 1/2*5.0e3*PASCAL_TO_CGS*((x-x_min)/(x_max-x_min) + (y-y_min)/(y_max-y_min)) + 2.5e3*PASCAL_TO_CGS
    sl_props['rayleigh_m'][()] = 0
    sl_props['rayleigh_k'][()] = 4e-3
    sl_props['kcontact'][()] = 1e11
    sl_props['ycontact'][()] = fl_props['y_midline'] - y_gap*1/2

    ac_props = model.acoustic.get_properties_vec(set_default=True)
    ac_props['area'][:] = 4.0
    ac_props['length'][:] = 12.0
    ac_props['soundspeed'][:] = 340*100

    props = linalg.concatenate(sl_props, fl_props, ac_props)
    
    return model, props

