"""
Tests functionality of `femvf.static`

Functions in `femvf.static` solve statics problems instead of the the transient
problems in `femvf.forward`.
"""

import os
import numpy as np
import dolfin as dfn

from femvf.models.transient import (
    solid as tsmd,
    fluid as tfmd,
    # acoustic as amd
)
from femvf.load import load_transient_fsi_model
from femvf.meshutils import process_meshlabel_to_dofs
from femvf import static

### Specify model
mesh_dir = '../meshes'
mesh_name = 'M5-3layers'
mesh_path = os.path.join(mesh_dir, mesh_name + '.xml')
model = load_transient_fsi_model(
    mesh_path, None, SolidType=tsmd.KelvinVoigt, FluidType=tfmd.BernoulliAreaRatioSep, coupling='explicit')

### Specify control
control = model.control.copy()
control['psub'][:] = 2000.0 * 10
control['psup'][:] = 0.0 * 10

### Specify properties
prop = model.prop.copy()

mesh = model.solid.forms['mesh.mesh']
cell_func = model.solid.forms['mesh.cell_function']
func_space = model.solid.forms['fspace.scalar']
cell_label_to_id = model.solid.forms['mesh.cell_label_to_id']
region_to_dofs = process_meshlabel_to_dofs(mesh, cell_func, cell_label_to_id, func_space.dofmap())
dofs_cover = region_to_dofs['cover']
dofs_body = region_to_dofs['body']

# Set the layer moduli
ECOV = 5e3*10
EBODY = 15e3*10
prop['emod'] = ECOV
prop['emod'][dofs_cover] = ECOV
prop['emod'][dofs_body] = EBODY
prop['rho'][:] = 1.0
prop['nu'][:] = 0.45

# contact and midline properties
y_max = np.max(model.solid.mesh.coordinates()[..., 1])
y_gap = 0.01
y_gap = 0.5
y_contact_offset = 1/10*y_gap

prop['ymid'][:] = y_max + y_gap
prop['ycontact'][:] = y_max + y_gap - y_contact_offset
prop['kcontact'][:] = 1e16

# separation point smoothing properties
ZETA = 1e-4
R_SEP = 1.0
prop['r_sep'][:] = R_SEP
prop['area_lb'][:] = 2*y_contact_offset
# prop['zeta_lb'][:] = 1e-6
# prop['zeta_min'][:] = ZETA
# prop['zeta_sep'][:] = ZETA
# prop['zeta_inv'][:] = ZETA

### Set the control and properties for the model
model.set_control(control)
model.set_prop(prop)
breakpoint()

def test_static_solid_configuration():
    """
    Test `static_solid_configuration`
    """
    solid = model.solid
    solid.dt = 100.0
    _p = np.zeros(solid.control['p'].size)
    _p[model.fsimap.dofs_solid[:10]] = 500*10
    dfn.as_backend_type(solid.control['p'])[:] = _p

    x_n, info = static.static_solid_configuration(solid, solid.control, solid.prop)
    print('\n', x_n.norm())
    print(info)

def test_static_configuration_coupled_newton():
    """
    Test `static_coupled_configuration_newton`
    """
    x_n, info = static.static_coupled_configuration_newton(model, control, prop)
    print(x_n.norm())
    print(info)

def test_static_configuration_coupled_picard():
    """
    Test `static_coupled_configuration_picard`
    """
    x_n, info = static.static_coupled_configuration_picard(model, control, prop)
    print(x_n.norm())
    print(info)

if __name__ == '__main__':
    test_static_solid_configuration()
    test_static_configuration_coupled_picard()
    # test_static_configuration_coupled_newton()
