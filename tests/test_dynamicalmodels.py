"""
This module contains tests for dynamical models
"""

from os import path
import numpy as np
import dolfin as dfn

from blocktensor import linalg as bla
from blocktensor import subops as gops
from femvf.dynamicalmodels import solid as slmodel, fluid as flmodel
from femvf import load

# warnings.filterwarnings('error', 'RuntimeWarning')
# np.seterr(invalid='raise')
def _set_dirichlet_bvec(dirichlet_bc, bvec):
    for label in ['u', 'v']:
        if label in bvec:
            dirichlet_bc.apply(dfn.PETScVector(bvec[label]))
    return bvec

def setup_models():
    """
    Setup the dynamical model objects
    """
    mesh_name = 'M5-3layers'
    mesh_name = 'BC-dcov5.00e-02-cl1.00'
    mesh_path = path.join('../meshes', mesh_name+'.xml')

    solid_mesh = mesh_path
    fluid_mesh = None

    SolidType = slmodel.KelvinVoigt
    FluidType = flmodel.Bernoulli1DDynamicalSystem
    model_coupled = load.load_dynamical_fsi_model(
        solid_mesh, fluid_mesh, SolidType, FluidType,
        fsi_facet_labels=('pressure',), fixed_facet_labels=('fixed',))

    SolidType = slmodel.LinearStateKelvinVoigt
    FluidType = flmodel.LinearStateBernoulli1DDynamicalSystem
    model_coupled_linear_state = load.load_dynamical_fsi_model(
        solid_mesh, fluid_mesh, SolidType, FluidType,
        fsi_facet_labels=('pressure',), fixed_facet_labels=('fixed',))

    SolidType = slmodel.LinearStatetKelvinVoigt
    FluidType = flmodel.LinearStatetBernoulli1DDynamicalSystem
    model_coupled_linear_statet = load.load_dynamical_fsi_model(
        solid_mesh, fluid_mesh, SolidType, FluidType,
        fsi_facet_labels=('pressure',), fixed_facet_labels=('fixed',))

    # model = model_coupled.fluid
    # model_linear_state = model_coupled_linear_state.fluid
    # model_linear_statet = model_coupled_linear_statet.fluid

    # model = model_coupled.solid
    # model_linear_state = model_coupled_linear_state.solid
    # model_linear_statet = model_coupled_linear_statet.solid

    model = model_coupled
    model_linear_state = model_coupled_linear_state
    model_linear_statet = model_coupled_linear_statet
    return model, model_linear_state, model_linear_statet

model, model_linear_state, model_linear_statet = setup_models()

def setup_parameter_base():
    ## Set model properties/control/linearization directions
    model_solid = model.solid

    # (linearization directions for linearized residuals)
    props0 = model.properties.copy()
    props0['emod'].array[:] = 5e3*10
    props0['rho'].array[:] = 1.0

    ymax = np.max(model_solid.XREF.vector()[1::2])
    ygap = 0.01 # gap between VF and symmetry plane
    ymid = ymax + ygap
    ycontact = ymid - 0.1*ygap
    props0['ycontact'][:] = ycontact

    for _model in [model, model_linear_state, model_linear_statet]:
        _model.ymid = ymid

    props0['zeta_sep'][0] = 1e-4
    props0['zeta_min'][0] = 1e-4
    props0['rho_air'][0] = 1.2e-3
    model.set_properties(props0)

    control0 = model.control.copy()
    control0.set(1.0)
    if 'psub' in control0:
        control0['psub'][:] = 800*10
    if 'psup' in control0:
        control0['psup'][:] = 0
    model.set_control(control0)

    del_state = model.state.copy()
    del_state.set(0.0)
    del_state['u'] = 1.0
    _set_dirichlet_bvec(model_solid.forms['bc.dirichlet'], del_state)
    model.set_dstate(del_state)

    del_statet = model.state.copy()
    del_statet.set(1.0e4)
    _set_dirichlet_bvec(model_solid.forms['bc.dirichlet'], del_statet)
    model.set_dstatet(del_statet)

    state0 = model.state.copy()

    statet0 = state0.copy()

    return state0, statet0, control0, props0, del_state, del_statet

def setup_parameter_perturbation():
    model_solid = model.solid

    dstate = model.state.copy()
    if 'u' in dstate and 'v' in dstate:
        dxu = model_solid.state['u'].copy()
        dxu[:] = 1e-3*np.arange(dxu[:].size)
        dxu[:] = 1e-8
        # dxu[:] = 0
        # model_solid.forms['bc.dirichlet'].apply(dxu)
        gops.set_vec(dstate['u'], dxu)

        dxv = model_solid.state['v'].copy()
        dxv[:] = 1e-8
        # model_solid.forms['bc.dirichlet'].apply(dxv)
        gops.set_vec(dstate['v'], dxv)
    if 'q' in dstate:
        gops.set_vec(dstate['q'], 1e-3)
        # gops.set_vec(dstate['q'], 0.0)
    if 'p' in dstate:
        gops.set_vec(dstate['p'], 1e-3)
        # gops.set_vec(dstate['p'], 0.0)
    _set_dirichlet_bvec(model_solid.forms['bc.dirichlet'], dstate)

    dstatet = dstate.copy()
    dstatet.set(1e-6)
    _set_dirichlet_bvec(model_solid.forms['bc.dirichlet'], dstatet)

    props0 = model.properties.copy()
    dprops = props0.copy()
    dprops.set(1e-4)

    dcontrol = model.control.copy()
    dcontrol.set(1e0)
    return dstate, dstatet, dcontrol, dprops

state0, statet0, control0, props0, del_state, del_statet = setup_parameter_base()
for _model in [model, model_linear_state, model_linear_statet]:
    _model.set_state(state0)
    _model.set_statet(statet0)
    _model.set_control(control0)
    _model.set_properties(props0)
    _model.set_dstate(del_state)
    _model.set_dstatet(del_statet)

dstate, dstatet, dcontrol, dprops  = setup_parameter_perturbation()

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
    alphas = 2**np.arange(4)[::-1] # start with the largest step and move to original
    res_ns = [res(x0+alpha*dx) for alpha in alphas]
    res_0 = res(x0)

    dres_exacts = [res_n-res_0 for res_n in res_ns]
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
    print("Errors: ", np.array(errs))
    print("Convergence rates: ", np.array(conv_rates))

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

def test_dres_dstate_vs_dres_state():
    """
    Test consistency between `model` and `model_linear_state`

    `model` represents a residual F(...)
    `model_linear_state` represents the linearized residual (dF/dstate * del_state)(...)
    This test checks that:
        dF/dstate(...) * del_state    (computed from `model`)
        is equal to
        (dF/dstate * del_state)(...)  (computed from `model_linear_state`)
    """
    # compute the linearized residual from `model`
    breakpoint()
    dres_dstate = gen_jac(state0, model.set_state, model.assem_dres_dstate)
    dres_state_a = bla.mult_mat_vec(dres_dstate, del_state)

    dres_state_b = gen_res(state0, model_linear_state.set_state, model_linear_state.assem_res)
    err = dres_state_a - dres_state_b

    for vec, name in zip([dres_state_a, dres_state_b, err], ["from model", "from linear_state_model", "error"]):
        print(f"\n{name}")
        for key, subvec in vec.items():
            print(key, subvec.norm())
    breakpoint()

if __name__ == '__main__':
    # breakpoint()
    # print("-- Test dres/dstate --")
    # test_assem_dres_dstate()

    # print("\n-- Test dres/dstatet --")
    # test_assem_dres_dstatet()

    # print("\n-- Test dres/dcontrol --")
    # test_assem_dres_dcontrol()

    print("\n-- Test dres/dstate vs dres_state --")
    test_dres_dstate_vs_dres_state()
