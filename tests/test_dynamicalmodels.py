"""
This module contains tests for dynamical models 
"""

from os import path
# from functools import partial
import warnings
import numpy as np
import dolfin as dfn

from blocktensor import linalg as bla
from blocktensor import subops as gops
# from femvf import dynamicalmodels as dynmod
from femvf.dynamicalmodels import solid as slmodel, fluid as flmodel
from femvf import load

# warnings.filterwarnings('error', 'RuntimeWarning')
# np.seterr(invalid='raise')

### Configuration ###
## Loading the model to test
mesh_name = 'M5-3layers'
mesh_name = 'BC-dcov5.00e-02-cl1.00'
mesh_path = path.join('../meshes', mesh_name+'.xml')

solid_mesh = mesh_path
fluid_mesh = None

SolidType = slmodel.KelvinVoigt
FluidType = flmodel.Bernoulli1DDynamicalSystem
# SolidType = slmodel.LinearStateKelvinVoigt
# FluidType = flmodel.LinearStateBernoulli1DDynamicalSystem
# SolidType = slmodel.LinearStatetKelvinVoigt
# FluidType = flmodel.LinearStatetBernoulli1DDynamicalSystem
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
props['emod'].array[:] = 5e3*10
props['rho'].array[:] = 1.0

ymax = np.max(model_solid.XREF.vector()[1::2])
ygap = 0.01 # gap between VF and symmetry plane
ymid = ymax + ygap
ycontact = ymid - 0.1*ygap
props['ycontact'][:] = ycontact # 1 cm above the maximum y extent
model_coupled.ymid = ymid
model_coupled.set_properties(props)

props['zeta_sep'][0] = 1e-3
props['zeta_min'][0] = 1e-3
props['rho_air'][0] = 1.0e-3

## Set the linearization point and linearization directions to test along
state0 = model.state.copy()
dstate = state0.copy()
if 'u' in dstate and 'v' in dstate:
    dxu = model_solid.state['u'].copy()
    dxu[:] = 1e-3*np.arange(dxu[:].size)
    dxu[:] = 1e-8
    # dxu[:] = 0
    model_solid.forms['bc.dirichlet'].apply(dxu)
    gops.set_vec(dstate['u'], dxu)

    dxv = model_solid.state['v'].copy()
    dxv[:] = 1e-8
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
        control0['psub'][:] = 800*10
    if 'psup' in control0:
        control0['psup'][:] = 0
    dcontrol.set(1e0)
model.set_control(control0)

props0 = model.properties.copy()
dprops = props0.copy()
dprops.set(1e-4)

def gen_res(x, set_x, assem_resx):
    set_x(x)
    return assem_resx()

def gen_jac(x, set_x, assem_jacx):
    set_x(x)
    return assem_jacx()

def _test_taylor(x0, dx, res, jac):
    """
    Test that the Taylor convergence order is 2
    """
    res0 = res(x0)
    alphas = 2**np.arange(4)[::-1] # start with the largest step and move to original
    dres_exacts = [res(x0+alpha*dx)-res0 for alpha in alphas] 
    dres_linear = bla.mult_mat_vec(jac(x0), dx)
    errs = [
        (dres_exact-alpha*dres_linear).norm() 
        for dres_exact, alpha in zip(dres_exacts, alphas)]
    with np.errstate(invalid='ignore'):
        conv_rates = [
            np.log(err_0/err_1)/np.log(alpha_0/alpha_1)
            for err_0, err_1, alpha_0, alpha_1 
            in zip(errs[:-1], errs[1:], alphas[:-1], alphas[1:])]

    print("")
    print(f"||dres_linear||, ||dres_exact|| = {dres_linear.norm()}, {dres_exacts[-1].norm()}")
    print("Errors: ", errs)
    print("Convergence rates: ", conv_rates)

def test_assem_dres_dstate():
    res = lambda state: gen_res(state, model.set_state, model.assem_res)
    jac = lambda state: gen_jac(state, model.set_state, model.assem_dres_dstate)

    x0 = state0
    dx = dstate

    _test_taylor(x0, dx, res, jac)

def test_assem_dres_dstatet():
    res = lambda state: gen_res(state, model.set_statet, model.assem_res)
    jac = lambda state: gen_jac(state, model.set_statet, model.assem_dres_dstatet)

    x0 = statet0
    dx = 1e-5*dstatet

    _test_taylor(x0, dx, res, jac)

def test_assem_dres_dcontrol():
    # model_fluid.control['psub'][:] = 1
    # model_fluid.control['psup'][:] = 0
    res = lambda state: gen_res(state, model.set_control, model.assem_res)
    jac = lambda state: gen_jac(state, model.set_control, model.assem_dres_dcontrol)

    x0 = control0
    dx = dcontrol

    _test_taylor(x0, dx, res, jac)

def test_assem_dres_dprops():
    res = lambda state: gen_res(state, model.set_properties, model.assem_res)
    jac = lambda state: gen_jac(state, model.set_properties, model.assem_dres_dprops)

    x0 = props0
    dx = dprops

    _test_taylor(x0, dx, res, jac)

if __name__ == '__main__':
    # breakpoint()
    print("-- Testing dres/dstate --")
    test_assem_dres_dstate()

    print("\n-- Testing dres/dstatet --")
    test_assem_dres_dstatet()

    print("\n-- Testing dres/dcontrol --")
    test_assem_dres_dcontrol()
