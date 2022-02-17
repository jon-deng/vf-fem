"""
This modules contains tests for dynamical models 
"""

from os import path
from functools import partial

from blocklinalg import linalg as bla
from femvf import dynamicalmodels as dynmod
from femvf.dynamicalmodels import solid as slmodel, fluid as flmodel
from femvf import load


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

model = model_solid

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
    dx.set(1e-3)
    dx['u'][:] = 1e-3
    dx['v'][:] = 0.0
    x1 = x0 + dx

    dres_exact = res(x1) - res(x0)
    dres_linear = bla.mult_mat_vec(jac(x0), dx)
    print(dres_linear.norm(), dres_exact.norm())

def test_assem_dres_dicontrol():
    res = lambda state: gen_res(state, model.set_icontrol, model.assem_res)
    jac = lambda state: gen_jac(state, model.set_icontrol, model.assem_dres_dicontrol)

    x0 = model.icontrol.copy()
    dx = x0.copy()
    dx.set(1e-3)
    x1 = x0 + dx

    dres_exact = res(x1) - res(x0)
    dres_linear = bla.mult_mat_vec(jac(x0), dx)
    print(dres_linear.norm(), dres_exact.norm())

def test_assem_dres_dstatet():
    res = lambda state: gen_res(state, model.set_statet, model.assem_res)
    jac = lambda state: gen_jac(state, model.set_statet, model.assem_dres_dstatet)

    x0 = model.statet.copy()
    dx = x0.copy()
    dx.set(1e-3)
    dx['u'][:] = 1e-3
    dx['v'][:] = 0
    x1 = x0 + dx

    dres_exact = res(x1) - res(x0)
    dres_linear = bla.mult_mat_vec(jac(x0), dx)
    # breakpoint()
    print(dres_linear.norm(), dres_exact.norm())

def test_assem_dres_dprops():
    res = lambda state: gen_res(state, model.set_properties, model.assem_res)
    jac = lambda state: gen_jac(state, model.set_properties, model.assem_dres_dprops)

    state_0 = model.properties.copy()
    dstate = state_0.copy()
    dstate.set(1e-3)
    state_1 = state_0 + dstate

    dres_exact = res(state_1) - res(state_0)
    dres_linear = bla.mult_mat_vec(jac(state_0), dstate)
    print(dres_exact)
    print(dres_linear.norm(), dres_exact.norm())

if __name__ == '__main__':
    test_assem_dres_dstate()
    test_assem_dres_dstatet()
    test_assem_dres_dicontrol()
    # print("yoyo whatup")