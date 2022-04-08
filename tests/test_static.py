"""
Tests that solving for static configurations works
"""

import os
import numpy as np

from femvf.models import solid as smd, fluid as fmd #, acoustic as amd
from femvf.load import load_transient_fsi_model
from femvf.meshutils import process_meshlabel_to_dofs
from femvf import static

### Specify model
mesh_dir = '../meshes'
mesh_name = 'M5-3layers'
mesh_path = os.path.join(mesh_dir, mesh_name + '.xml')
model = load_transient_fsi_model(
    mesh_path, None, SolidType=smd.KelvinVoigt, FluidType=fmd.Bernoulli, coupling='explicit')

### Specify control
control = model.get_control_vec()
control['psub'][:] = 2000.0 * 10
control['psup'][:] = 0.0 * 10

### Specify properties
props = model.get_properties_vec()

mesh = model.solid.forms['mesh.mesh']
cell_func = model.solid.forms['mesh.cell_function']
func_space = model.solid.forms['fspace.scalar']
cell_label_to_id = model.solid.forms['mesh.cell_label_to_id']
region_to_dofs = process_meshlabel_to_dofs(mesh, cell_func, func_space, cell_label_to_id)
dofs_cover = region_to_dofs['cover']
dofs_body = region_to_dofs['body']

# Set the layer moduli
ECOV = 5e3*10
EBODY = 15e3*10
props['emod'][dofs_cover] = ECOV
props['emod'][dofs_body] = EBODY
props['rho'][:] = 1.0
props['nu'][:] = 0.45

# contact and midline properties
y_max = np.max(model.solid.mesh.coordinates()[..., 1])
y_gap = 0.01
y_gap = 0.5
y_contact_offset = 1/10*y_gap

props['y_midline'][:] = y_max + y_gap
props['ycontact'][:] = y_max + y_gap - y_contact_offset
props['kcontact'][:] = 1e16

# separation point smoothing properties
ZETA = 1e-4
R_SEP = 1.0
props['r_sep'][:] = R_SEP
props['ygap_lb'][:] = y_contact_offset
props['zeta_lb'][:] = 1e-6
props['zeta_amin'][:] = ZETA
props['zeta_sep'][:] = ZETA
props['zeta_ainv'][:] = ZETA

### Set the control and properties for the model
model.set_control(control)
model.set_properties(props)

def test_static_configuration_coupled_newton():
    x_n, info = static.static_configuration_coupled_newton(model)
    print(x_n.norm())
    print(info)


def test_static_configuration_coupled_picard():
    x_n, info = static.static_configuration_coupled_picard(model)
    print(x_n.norm())
    print(info)

if __name__ == '__main__':
    test_static_configuration_coupled_picard()
    # test_static_configuration_coupled_newton()
