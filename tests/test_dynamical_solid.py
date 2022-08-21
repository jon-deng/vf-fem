"""
Test that the Hopf solid models will run
"""

import os
import dolfin as dfn

from femvf.meshutils import load_fenics_xml
from femvf.models.dynamical import solid as dynsol

dfn.set_log_level(30)

def test_dynamical_kelvin_voigt(mesh_path):
    """
    Test that the Kelvin Voigt model can be loaded
    """
    mesh, (vertex_func, facet_func, cell_func), (vertex_labels, facet_labels, cell_labels) = \
        load_fenics_xml(mesh_path)
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
    mesh_dir = '../meshes'
    mesh_name = 'M5-3layers'
    mesh_path = os.path.join(mesh_dir, mesh_name + '.xml')
    test_dynamical_kelvin_voigt(mesh_path)
