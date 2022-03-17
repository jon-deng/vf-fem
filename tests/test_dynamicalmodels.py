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

### Configuration ###
## Loading the model to test
mesh_name = 'M5-3layers'
mesh_path = path.join('../meshes', mesh_name+'.xml')

solid_mesh = mesh_path
fluid_mesh = None
# Base residual
SolidType = slmodel.KelvinVoigt
FluidType = flmodel.Bernoulli1DDynamicalSystem
# Linearized residual
SolidType = slmodel.LinearStateKelvinVoigt
FluidType = flmodel.LinearStateBernoulli1DDynamicalSystem
# Linearized residual
SolidType = slmodel.LinearStatetKelvinVoigt
FluidType = flmodel.LinearStatetBernoulli1DDynamicalSystem
# Linearized residual
# SolidType = slmodel.LinearControlKelvinVoigt
# FluidType = flmodel.LinearControlBernoulli1DDynamicalSystem
model_coupled = load.load_dynamical_fsi_model(
    solid_mesh, fluid_mesh, SolidType, FluidType, 
    fsi_facet_labels=('pressure',), fixed_facet_labels=('fixed',))

model_solid = model_coupled.solid
model_fluid = model_coupled.fluid
model = model_solid
# model = model_fluid
# model = model_coupled

## Set the model properties/parameters
props = model_coupled.properties.copy()
props['emod'].array[:] = 1.0
props['psub'][:] = 800*10
props['psup'][:] = 0.0
props['ycontact'][:] = np.max(model_solid.XREF.vector()[1::2]) + 1 # 1 cm above the maximum y extent
model_coupled.ymid = props['ycontact'][0]
model_coupled.set_properties(props)

## Set the linearization point and linearization directions to test along
# state
state0 = model.state.copy()
dstate = state0.copy()
if 'u' in dstate and 'v' in dstate:
    dxu = model_solid.state['u'].copy()
    dxu[:] = 1e-3*np.arange(dxu[:].size)
    # dxu[:] = 0
    model_solid.forms['bc.dirichlet'].apply(dxu)
    gops.set_vec(dstate['u'], dxu)

    dxv = model_solid.state['v'].copy()
    dxv[:] = 0.0
    model_solid.forms['bc.dirichlet'].apply(dxv)
    gops.set_vec(dstate['v'], dxv)
if 'q' in dstate:
    gops.set_vec(dstate['q'], 1e-3)
    # gops.set_vec(dstate['q'], 0.0)
if 'p' in dstate:
    gops.set_vec(dstate['p'], 1e-3)
    # gops.set_vec(dstate['p'], 0.0)

statet0 = state0.copy()
dstatet = dstate.copy()

if hasattr(model, 'control'):
    control0 = model.control.copy()
    dcontrol = control0.copy()
    if 'psub' in control0:
        control0['psub'][:] = 500
    if 'psup' in control0:
        control0['psup'][:] = 0
    dcontrol.set(1e-5)

props0 = model.properties.copy()
dprops = props0.copy()
dprops.set(1e-4)

def gen_res(x, set_x, assem_resx):
    set_x(x)
    return assem_resx()

def gen_jac(x, set_x, assem_jacx):
    set_x(x)
    return assem_jacx()

def test_assem_dres_dstate():
    res = lambda state: gen_res(state, model.set_state, model.assem_res)
    jac = lambda state: gen_jac(state, model.set_state, model.assem_dres_dstate)

    x0 = state0
    dx = 1e-5*dstate
    x1 = x0 + dx
    # model.set_state(x1)

    dres_exact = res(x1) - res(x0)
    dres_linear = bla.mult_mat_vec(jac(x0), dx)
    print(dres_linear.norm(), dres_exact.norm())

def test_assem_dres_dstatet():
    res = lambda state: gen_res(state, model.set_statet, model.assem_res)
    jac = lambda state: gen_jac(state, model.set_statet, model.assem_dres_dstatet)

    x0 = statet0
    dx = 1e-5*dstatet
    x1 = x0 + dx

    dres_exact = res(x1) - res(x0)
    dres_linear = bla.mult_mat_vec(jac(x0), dx)
    print(dres_linear.norm(), dres_exact.norm())

def test_assem_dres_dcontrol():
    # model_fluid.control['psub'][:] = 1
    # model_fluid.control['psup'][:] = 0
    res = lambda state: gen_res(state, model.set_control, model.assem_res)
    jac = lambda state: gen_jac(state, model.set_control, model.assem_dres_dcontrol)

    x0 = control0
    dx = dcontrol
    x1 = x0 + dx

    dres_exact = res(x1) - res(x0)
    dres_linear = bla.mult_mat_vec(jac(x0), dx)
    # breakpoint()
    print(dres_linear.norm(), dres_exact.norm())

def test_assem_dres_dprops():
    res = lambda state: gen_res(state, model.set_properties, model.assem_res)
    jac = lambda state: gen_jac(state, model.set_properties, model.assem_dres_dprops)

    x0 = props0
    dx = dprops
    x1 = x0 + dx

    dres_exact = res(x1) - res(x0)
    dres_linear = bla.mult_mat_vec(jac(x0), dx)
    print(dres_exact)
    print(dres_linear.norm(), dres_exact.norm())

if __name__ == '__main__':
    breakpoint()
    test_assem_dres_dstate()
    test_assem_dres_dstatet()
    if hasattr(model, 'control'):
        test_assem_dres_dcontrol()