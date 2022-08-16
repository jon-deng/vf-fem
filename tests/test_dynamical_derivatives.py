"""
This module tests correctness of the dynamical model's derivatives

Correctness is tested by comparing finite difference derivatives against
implemented derivatives.
"""

from os import path
import numpy as np
import dolfin as dfn

from blockarray import linalg as bla, blockmat as bmat
from blockarray import subops as gops
from femvf.models.dynamical import solid as slmodel, fluid as flmodel
from femvf import load

# pylint: disable=redefined-outer-name

# warnings.filterwarnings('error', 'RuntimeWarning')
# np.seterr(invalid='raise')

def _set_dirichlet_bvec(dirichlet_bc, bvec):
    for label in ['u', 'v']:
        if label in bvec:
            dirichlet_bc.apply(dfn.PETScVector(bvec.sub[label]))
    return bvec

def setup_coupled_models():
    """
    Setup the dynamical model objects
    """
    mesh_name = 'M5-3layers'
    mesh_name = 'BC-dcov5.00e-02-cl1.00'
    mesh_path = path.join('../meshes', mesh_name+'.xml')

    solid_mesh = mesh_path
    fluid_mesh = None

    SolidType = slmodel.KelvinVoigt
    FluidType = flmodel.BernoulliSmoothMinSep
    model_coupled = load.load_dynamical_fsi_model(
        solid_mesh, fluid_mesh, SolidType, FluidType,
        fsi_facet_labels=('pressure',), fixed_facet_labels=('fixed',))

    SolidType = slmodel.LinearizedKelvinVoigt
    FluidType = flmodel.LinearizedBernoulliSmoothMinSep
    model_coupled_linear = load.load_dynamical_fsi_model(
        solid_mesh, fluid_mesh, SolidType, FluidType,
        fsi_facet_labels=('pressure',), fixed_facet_labels=('fixed',))

    return model_coupled, model_coupled_linear

def setup_coupled_parameter_base(model):
    """
    Return base parameters to compute derivatives at
    """
    ## Set model properties/control/linearization directions
    model_solid = model.solid

    # (linearization directions for linearized residuals)
    props0 = model.props.copy()
    props0['emod'] = 5e3*10
    props0['rho'] = 1.0

    ymax = np.max(model_solid.XREF[1::2])
    ygap = 0.01 # gap between VF and symmetry plane
    ymid = ymax + ygap
    ycontact = ymid - 0.1*ygap
    props0['ycontact'] = ycontact

    for _model in [model, model_linear]:
        _model.ymid = ymid

    props0['zeta_sep'] = 1e-4
    props0['zeta_min'] = 1e-4
    props0['rho_air'] = 1.2e-3
    model.set_props(props0)

    control0 = model.control.copy()
    control0[:] = 1.0
    if 'psub' in control0:
        control0['psub'] = 800*10
    if 'psup' in control0:
        control0['psup'] = 0
    model.set_control(control0)

    del_state = model.state.copy()
    del_state[:] = 0.0
    del_state['u'] = 1.0
    _set_dirichlet_bvec(model_solid.forms['bc.dirichlet'], del_state)
    model.set_dstate(del_state)

    del_statet = model.state.copy()
    del_statet[:] = 1.0e4
    _set_dirichlet_bvec(model_solid.forms['bc.dirichlet'], del_statet)
    model.set_dstatet(del_statet)

    state0 = model.state.copy()
    # Make the initial displacement a pure shear motion
    xref = model.solid.forms['coeff.ref.x'].vector().copy()
    xx = xref[:-1:2]
    yy = xref[1::2]
    state0['u'][:-1:2] = 0.01*(yy/yy.max())
    state0['u'][1::2] = 0.0 * yy
    model.set_state(state0)

    statet0 = state0.copy()
    model.set_statet(statet0)

    return state0, statet0, control0, props0, del_state, del_statet

def setup_coupled_parameter_perturbation(model):
    """
    Return parameters pertrubations to compute directional derivatives along
    """
    model_solid = model.solid

    dstate = model.state.copy()
    if 'u' in dstate and 'v' in dstate:
        dxu = model_solid.state['u'].copy()
        dxu[:] = 1e-3*np.arange(dxu[:].size)
        dxu[:] = 1e-8
        # dxu[:] = 0
        # model_solid.forms['bc.dirichlet'].apply(dxu)
        dstate['u'] = dxu

        dxv = model_solid.state['v'].copy()
        dxv[:] = 1e-8
        # model_solid.forms['bc.dirichlet'].apply(dxv)
        dstate['v'] = dxv
    if 'q' in dstate:
        dstate['q'] = 1e-3
    if 'p' in dstate:
        dstate['p'] = 1e-3
    _set_dirichlet_bvec(model_solid.forms['bc.dirichlet'], dstate)

    dstatet = dstate.copy()
    dstatet[:] = 1e-6
    _set_dirichlet_bvec(model_solid.forms['bc.dirichlet'], dstatet)

    props0 = model.props.copy()
    dprops = props0.copy()
    dprops[:] = 0
    dprops['emod'] = 1.0


    dcontrol = model.control.copy()
    dcontrol[:] = 1e0
    return dstate, dstatet, dcontrol, dprops


def _reset_parameter_base(model, state0, statet0, control0, props0, del_state, del_statet):
    model.set_state(state0)
    model.set_statet(statet0)
    model.set_control(control0)
    model.set_props(props0)
    model.set_dstate(del_state)
    model.set_dstatet(del_statet)

def _set_and_assemble(x, set_x, assem):
    set_x(x)
    return assem()

def _test_taylor(x0, dx, res, jac):
    """
    Test that the Taylor convergence order is 2
    """
    alphas = 2**np.arange(4)[::-1] # start with the largest step and move to original
    res_ns = [res(x0+float(alpha)*dx) for alpha in alphas]
    res_0 = res(x0)

    dres_exacts = [res_n-res_0 for res_n in res_ns]
    dres_linear = bla.mult_mat_vec(jac(x0), dx)

    errs = [
        (dres_exact-float(alpha)*dres_linear).norm()
        for dres_exact, alpha in zip(dres_exacts, alphas)
    ]
    magnitudes = [
        1/2*(dres_exact+float(alpha)*dres_linear).norm()
        for dres_exact, alpha in zip(dres_exacts, alphas)
    ]
    with np.errstate(invalid='ignore'):
        conv_rates = [
            np.log(err_0/err_1)/np.log(alpha_0/alpha_1)
            for err_0, err_1, alpha_0, alpha_1
            in zip(errs[:-1], errs[1:], alphas[:-1], alphas[1:])]
        rel_errs = np.array(errs)/np.array(magnitudes)*100

    print("")
    print(f"||dres_linear||, ||dres_exact|| = {dres_linear.norm()}, {dres_exacts[-1].norm()}")
    print("Relative errors: ", rel_errs)
    print("Convergence rates: ", np.array(conv_rates))

def test_assem_dres_dstate(model, x0, dx):
    """Test dF/dx is correct"""
    res = lambda state: _set_and_assemble(state, model.set_state, model.assem_res)
    jac = lambda state: _set_and_assemble(state, model.set_state, model.assem_dres_dstate)

    _test_taylor(x0, dx, res, jac)

def test_assem_dres_dstatet(model, x0, dx):
    """Test dF/dxt is correct"""
    res = lambda state: _set_and_assemble(state, model.set_statet, model.assem_res)
    jac = lambda state: _set_and_assemble(state, model.set_statet, model.assem_dres_dstatet)

    _test_taylor(x0, dx, res, jac)

def test_assem_dres_dcontrol(model, x0, dx):
    """Test dF/dg is correct"""
    # model_fluid.control['psub'][:] = 1
    # model_fluid.control['psup'][:] = 0
    res = lambda state: _set_and_assemble(state, model.set_control, model.assem_res)
    jac = lambda state: _set_and_assemble(state, model.set_control, model.assem_dres_dcontrol)

    _test_taylor(x0, dx, res, jac)

def test_assem_dres_dprops(model, x0, dx):
    """Test dF/dprops is correct"""
    res = lambda state: _set_and_assemble(state, model.set_props, model.assem_res)
    jac = lambda state: _set_and_assemble(state, model.set_props, model.assem_dres_dprops)

    _test_taylor(x0, dx, res, jac)

def test_dres_dstate_vs_dres_state(model, model_linear, x0, del_x):
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
    dres_dstate = _set_and_assemble(x0, model.set_state, model.assem_dres_dstate)
    dres_state_a = bla.mult_mat_vec(dres_dstate, del_x)

    model_linear.set_dstate(del_x)
    _zero_del_xt = model_linear.dstatet.copy()
    _zero_del_xt[:] = 0
    model_linear.set_dstatet(_zero_del_xt)

    dres_state_b = _set_and_assemble(x0, model_linear.set_state, model_linear.assem_res)
    err = dres_state_a - dres_state_b

    for vec, name in zip([dres_state_a, dres_state_b, err], ["from model", "from linear_state_model", "error"]):
        print(f"\n{name}")
        for key, subvec in vec.items():
            print(key, subvec.norm())
    breakpoint()

def test_dres_dstatet_vs_dres_statet(model, model_linear, x0, del_xt):
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
    dres_dstatet = _set_and_assemble(x0, model.set_state, model.assem_dres_dstatet)
    dres_statet_a = bla.mult_mat_vec(dres_dstatet, del_xt)

    model_linear.set_dstatet(del_xt)
    _zero_del_x = model_linear.dstate.copy()
    _zero_del_x[:] = 0
    model_linear.set_dstate(_zero_del_x)

    dres_statet_b = _set_and_assemble(x0, model_linear.set_state, model_linear.assem_res)
    err = dres_statet_a - dres_statet_b

    for vec, name in zip([dres_statet_a, dres_statet_b, err], ["from model", "from linear_state_model", "error"]):
        print(f"\n{name}")
        for key, subvec in vec.items():
            print(key, subvec.norm())
    breakpoint()

if __name__ == '__main__':
    model, model_linear = setup_coupled_models()
    for _model in [model, model_linear]:
        state0, statet0, control0, props0, del_state, del_statet = setup_coupled_parameter_base(_model)
        base_parameters = (state0, statet0, control0, props0, del_state, del_statet)
        _reset_parameter_base(_model, *base_parameters)

        dstate, dstatet, dcontrol, dprops = setup_coupled_parameter_perturbation(_model)

    # breakpoint()
    for model_name, _model in zip(["Residual", "Linearized residual"], [model, model_linear]):
        print(model_name)
        print("-- Test dres/dstate --")
        test_assem_dres_dstate(_model, state0, dstate)

        print("\n-- Test dres/dstatet --")
        test_assem_dres_dstatet(_model, statet0, dstatet)

        print("\n-- Test dres/dcontrol --")
        test_assem_dres_dcontrol(_model, control0, dcontrol)

        print("\n-- Test dres/dprops --")
        test_assem_dres_dprops(model, props0, dprops)

    print("\n-- Test dres/dstate vs dres_state --")
    test_dres_dstate_vs_dres_state(model, model_linear, state0, del_state)

    print("\n-- Test dres/dstatet vs dres_statet --")
    test_dres_dstatet_vs_dres_statet(model, model_linear, state0, del_statet)
