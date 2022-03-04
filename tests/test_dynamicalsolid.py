"""
Test that the Hopf solid models will run
"""

import os
import dolfin as dfn
from pprint import pprint

from femvf.meshutils import load_fenics_xmlmesh
# from femvf.models import solid as smd
from femvf.dynamicalmodels import solid as dynsol

dfn.set_log_level(30)

mesh_dir = '../meshes'
mesh_name = 'M5-3layers'
mesh_path = os.path.join(mesh_dir, mesh_name + '.xml')

mesh, facet_func, cell_func, (vertex_labels, facet_label_to_id, cell_label_to_id) = \
    load_fenics_xmlmesh(mesh_path)

def test_dynamical_kelvin_voigt():
    # breakpoint()
    fsi_facet_labels = ['pressure']
    fixed_facet_labels = ['fixed']
    model = dynsol.KelvinVoigt(
        mesh, facet_func, facet_label_to_id, cell_func, cell_label_to_id, 
        fsi_facet_labels, fixed_facet_labels, residual_form_name='f1uva')

    model.assem_res()
    model.assem_dres_dstate()
    model.assem_dres_dcontrol()

if __name__ == '__main__':
    print(test_dynamical_kelvin_voigt())