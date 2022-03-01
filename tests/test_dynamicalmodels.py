"""
This modules contains tests for dynamical models 
"""

from os import path
# from functools import partial
import warnings
import numpy as np
import dolfin as dfn

from blocklinalg import linalg as bla
from blocklinalg import genericops as gops
# from femvf import dynamicalmodels as dynmod
from femvf.dynamicalmodels import solid as slmodel, fluid as flmodel
from femvf import load

# warnings.filterwarnings('error', 'RuntimeWarning')
np.seterr(invalid='raise')

mesh_name = 'M5-3layers'
mesh_path = path.join('../meshes', mesh_name+'.xml')

solid_mesh = mesh_path
fluid_mesh = None
SolidType = slmodel.KelvinVoigt
FluidType = flmodel.Bernoulli1DDynamicalSystem
model_coupled = load.load_dynamical_fsi_model(
    solid_mesh, fluid_mesh, SolidType, FluidType, 
    fsi_facet_labels=('pressure',), fixed_facet_labels=('fixed',))

model_solid = model_coupled.solid
model_fluid = model_coupled.fluid

props = model_coupled.properties.copy()
props['psub'] = 800*10
props['psup'] = 0.0
props['ycontact'] = np.max(model_solid.XREF.vector()[1::2]) + 1 # 1 cm above the maximum y extent
model_coupled.ymid = props['ycontact'][0]
model_coupled.set_properties(props)

model = model_solid
# model = model_fluid
# model = model_coupled

def gen_res(x, set_x, assem_resx):
    set_x(x)
    return assem_resx()

def gen_jac(x, set_x, assem_jacx):
    set_x(x)
    return assem_jacx()

def test_assem_dres_dstate():
    res = lambda state: gen_res(state, model.set_state, model.assem_res)
    jac = lambda state: gen_jac(state, model.set_state, model.assem_dres_dstate)

    x0 = model.state.copy()
    dx = x0.copy()
    if 'u' in dx and 'v' in dx:
        dxu = model_solid.state['u'].copy()
        dxu[:] = 1e-3
        dxu[:] = 1e-3*np.arange(dxu[:].size)
        model.forms['bc.dirichlet'].apply(dxu)
        gops.set_vec(dx['u'], dxu)

        dxv = model_solid.state['v'].copy()
        dxv[:] = 0.0
        model.forms['bc.dirichlet'].apply(dxv)
        gops.set_vec(dx['v'], dxv)
    if 'q' in dx:
        dx.set(1e-3)
    x1 = x0 + dx
    model.set_state(x1)
    breakpoint()

    dres_exact = res(x1) - res(x0)
    # breakpoint()
    A = jac(x0)
    dres_linear = bla.mult_mat_vec(jac(x0), dx)
    print(dres_linear.norm(), dres_exact.norm())

def test_assem_dres_dstatet():
    res = lambda state: gen_res(state, model.set_statet, model.assem_res)
    jac = lambda state: gen_jac(state, model.set_statet, model.assem_dres_dstatet)

    x0 = model.statet.copy()
    dx = x0.copy()
    if 'u' in dx:
        gops.set_vec(dx['u'][:], 1e-3)
        gops.set_vec(dx['v'][:], 0)
    elif 'q' in dx:
        dx.set(1e-3)
    x1 = x0 + dx

    dres_exact = res(x1) - res(x0)
    dres_linear = bla.mult_mat_vec(jac(x0), dx)
    # breakpoint()
    print(dres_linear.norm(), dres_exact.norm())

def test_assem_dres_dicontrol():
    res = lambda state: gen_res(state, model.set_icontrol, model.assem_res)
    jac = lambda state: gen_jac(state, model.set_icontrol, model.assem_dres_dicontrol)

    x0 = model.icontrol.copy()
    dx = x0.copy()
    dx.set(1e-4)
    x1 = x0 + dx

    dres_exact = res(x1) - res(x0)
    dres_linear = bla.mult_mat_vec(jac(x0), dx)
    print(dres_linear.norm(), dres_exact.norm())

def test_assem_dres_dprops():
    res = lambda state: gen_res(state, model.set_properties, model.assem_res)
    jac = lambda state: gen_jac(state, model.set_properties, model.assem_dres_dprops)

    x0 = model.properties.copy()
    dx = x0.copy()
    dx.set(1e-4)
    x1 = x0 + dx

    dres_exact = res(x1) - res(x0)
    dres_linear = bla.mult_mat_vec(jac(x0), dx)
    print(dres_exact)
    print(dres_linear.norm(), dres_exact.norm())

if __name__ == '__main__':
    test_assem_dres_dstate()
    test_assem_dres_dstatet()
    # test_assem_dres_dicontrol()
    # print("yoyo whatup")