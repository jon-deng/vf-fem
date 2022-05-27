"""
Tests that the solid model forms run
"""

import os
import dolfin as dfn
from pprint import pprint

from femvf.meshutils import load_fenics_xmlmesh
from femvf.models import solid as smd
from femvf.models import solidforms as fsmd

dfn.set_log_level(30)

mesh_dir = '../meshes'
mesh_name = 'M5-3layers'
mesh_path = os.path.join(mesh_dir, mesh_name + '.xml')

mesh, (vertex_func, facet_func, cell_func), (vertex_labels, facet_label_to_id, cell_label_to_id) = \
    load_fenics_xmlmesh(mesh_path)

def test_base_form_definitions():
    try:
        fsmd.base_form_definitions(
            mesh, facet_func, facet_label_to_id, cell_func, cell_label_to_id,
            ['pressure'], ['fixed'])
    except:
        raise

def test_hopf_form_definitions():
    try:
        forms = \
            fsmd.add_surface_pressure_form(
            fsmd.add_inertial_form(
            fsmd.add_kv_viscous_form(
            fsmd.add_isotropic_elastic_form(
            fsmd.base_form_definitions(
            mesh, facet_func, facet_label_to_id, cell_func, cell_label_to_id,
            ['pressure'], ['fixed'])))))
        fsmd.gen_hopf_forms(forms)
        pprint(forms)
    except:
        raise

if __name__ == '__main__':
    test_hopf_form_definitions()

