"""
Contains definitions of different solid model forms
"""

import operator
import warnings
from functools import reduce
from typing import Tuple, Mapping

import numpy as np
import dolfin as dfn
import ufl

from . import newmark

def _depack_property_ufl_coeff(form_property):
    """
    Return the 'ufl.Coefficient' component from a stored 'coeff.prop.' value

    This mainly handles the shape parameter which is stored as a tuple
    of a function and an associated `ufl.Coefficient`.
    """
    if isinstance(form_property, tuple):
        return form_property[1]
    else:
        return form_property


def form_lin_iso_cauchy_stress(strain, emod, nu):
    """
    Returns the Cauchy stress for a small-strain displacement field

    Parameters
    ----------
    u : dfn.TrialFunction, ufl.Argument
        Trial displacement field
    emod : dfn.Function, ufl.Coefficient
        Elastic modulus
    nu : float
        Poisson's ratio
    """
    lame_lambda = emod*nu/(1+nu)/(1-2*nu)
    lame_mu = emod/2/(1+nu)
    return 2*lame_mu*strain + lame_lambda*ufl.tr(strain)*ufl.Identity(strain.ufl_shape[0])

def form_inf_strain(u):
    """
    Returns the strain tensor

    Parameters
    ----------
    u : dfn.TrialFunction, ufl.Argument
        Trial displacement field
    """
    spp = 1/2 * (ufl.grad(u) + ufl.grad(u).T)
    if u.geometric_dimension() == 2:
        return ufl.as_tensor(
            [[spp[0, 0], spp[0, 1], 0],
            [spp[1, 0], spp[1, 1], 0],
            [        0,         0, 0]]
        )
    else:
        return spp

def form_lin_green_strain(u0, u):
    """
    Returns the linearized Green-Lagrange strain tensor

    Parameters
    ----------
    u0 : dfn.TrialFunction, ufl.Argument
        Displacement to linearize about
    u : dfn.TrialFunction, ufl.Argument
        Trial displacement field
    """
    spp = 1/2*(ufl.grad(u).T*ufl.grad(u0) + ufl.grad(u0).T*ufl.grad(u))
    if u0.geometric_dimension() == 2:
        return ufl.as_tensor(
            [[spp[0, 0], spp[0, 1], 0],
            [spp[1, 0], spp[1, 1], 0],
            [        0,         0, 0]]
        )
    else:
        return spp

def form_penalty_contact_pressure(xref, u, k, ycoll, n=dfn.Constant([0.0, 1.0])):
    """
    Return the contact pressure expression according to the penalty method

    Parameters
    ----------
    xref : dfn.Function
        Reference configuration coordinates
    u : dfn.Function
        Displacement
    k : float or dfn.Constant
        Penalty contact spring value
    d : float or dfn.Constant
        y location of the contact plane
    n : dfn.Constant
        Contact plane normal, facing away from the vocal folds
    """
    gap = ufl.dot(xref+u, n) - ycoll
    positive_gap = (gap + abs(gap)) / 2

    # Uncomment/comment the below lines to choose between exponential or quadratic penalty springs
    return -k*positive_gap**3

def form_pressure_as_reference_traction(p, u, n):
    """

    Parameters
    ----------
    p : Pressure load
    u : displacement
    n : facet outer normal
    """
    deformation_gradient = ufl.grad(u) + ufl.Identity(2)
    deformation_cofactor = ufl.det(deformation_gradient) * ufl.inv(deformation_gradient).T

    return -p*deformation_cofactor*n

def form_pullback_area_normal(u, n):
    """

    Parameters
    ----------
    p : Pressure load
    u : displacement
    n : facet outer normal
    """
    deformation_gradient = ufl.grad(u) + ufl.Identity(2)
    deformation_cofactor = ufl.det(deformation_gradient) * ufl.inv(deformation_gradient).T

    return deformation_cofactor*n

def form_positive_gap(gap):
    """
    Return the positive gap
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            category=RuntimeWarning,
            message='invalid value encountered in add'
        )
        positive_gap = (gap + abs(gap)) / 2
    positive_gap = np.where(
        gap == -np.inf,
        0.0,
        positive_gap
    )
    return positive_gap

def form_cubic_penalty_pressure(gap, kcoll):
    """
    Return the cubic penalty pressure
    """
    positive_gap = form_positive_gap(gap)
    return kcoll*positive_gap**3

def dform_cubic_penalty_pressure(gap, kcoll):
    """
    Return derivatives of the cubic penalty pressure
    """
    positive_gap = form_positive_gap(gap)
    dpositive_gap = np.sign(gap)
    return kcoll*3*positive_gap**2 * dpositive_gap, positive_gap**3


def form_quad_penalty_pressure(gap, kcoll):
    positive_gap = (gap + abs(gap)) / 2
    return kcoll*positive_gap**2

# These functions are mainly for generating forms that are needed for solving
# the transient problem with a time discretization
def gen_residual_bilinear_forms(forms):
    """
    Generates bilinear forms representing derivatives of the residual wrt state variables

    If the residual is F(u, v, a; parameters, ...), this function generates
    bilinear forms dF/du, dF/dv, etc...
    """
    # Derivatives of the displacement residual form wrt all state variables
    initial_state_names = [f'coeff.state.{y}' for y in ('u0', 'v0', 'a0')]
    final_state_names = [f'coeff.state.{y}' for y in ('u1', 'v1', 'a1')]
    manual_state_var_names = [name for name in forms.keys() if 'coeff.state.manual' in name]

    # This section is for derivatives of the time-discretized residual
    # F(u0, v0, a0, u1; parameters, ...)
    for full_var_name in (
        initial_state_names
        + ['coeff.state.u1']
        + manual_state_var_names
        + ['coeff.time.dt', 'coeff.fsi.p1']):
        f = forms['form.un.f1']
        x = forms[full_var_name]

        var_name = full_var_name.split(".")[-1]
        form_name = f'form.bi.df1_d{var_name}'
        forms[form_name] = dfn.derivative(f, x)
        forms[f'{form_name}_adj'] = dfn.adjoint(forms[form_name])

    # This section is for derivatives of the original not time-discretized residual
    # F(u1, v1, a1; parameters, ...)
    for full_var_name in (
        final_state_names
        + manual_state_var_names
        + ['coeff.fsi.p1']):
        f = forms['form.un.f1uva']
        x = forms[full_var_name]

        var_name = full_var_name.split(".")[-1]
        form_name = f'form.bi.df1uva_d{var_name}'
        forms[form_name] = dfn.derivative(f, x)
        forms[f'{form_name}_adj'] = dfn.adjoint(forms[form_name])

def gen_residual_bilinear_property_forms(forms):
    """
    Return a dictionary of forms of derivatives of f1 with respect to the various solid parameters
    """
    df1_dsolid = {}
    property_labels = [
        form_name.split('.')[-1] for form_name in forms.keys()
        if 'coeff.prop' in form_name
    ]
    for prop_name in property_labels:
        prop_coeff = _depack_property_ufl_coeff(forms[f'coeff.prop.{prop_name}'])
        try:
            df1_dsolid[prop_name] = dfn.adjoint(
                dfn.derivative(forms['form.un.f1'], prop_coeff)
            )
        except RuntimeError:
            df1_dsolid[prop_name] = None

    return df1_dsolid

# These functions are mainly for generating derived forms that are needed for solving
# the hopf bifurcation problem
def gen_hopf_forms(forms):
    gen_unary_linearized_forms(forms)

    unary_form_names = ['f1uva', 'df1uva', 'df1uva_u1', 'df1uva_v1', 'df1uva_a1', 'df1uva_p1']
    for unary_form_name in unary_form_names:
        gen_jac_state_forms(unary_form_name, forms)
    for unary_form_name in unary_form_names:
        gen_jac_property_forms(unary_form_name, forms)

def gen_jac_state_forms(unary_form_name, forms):
    """
    Return the derivatives of a unary form wrt all solid properties
    """
    state_labels = ['u1', 'v1', 'a1']
    for state_name in state_labels:
        df_dx = dfn.derivative(forms[f'form.un.{unary_form_name}'], forms[f'coeff.state.{state_name}'])
        forms[f'form.bi.d{unary_form_name}_d{state_name}'] = df_dx

    state_labels = ['p1']
    for state_name in state_labels:
        df_dx = dfn.derivative(forms[f'form.un.{unary_form_name}'], forms[f'coeff.fsi.{state_name}'])
        forms[f'form.bi.d{unary_form_name}_d{state_name}'] = df_dx

    return forms

def gen_jac_property_forms(unary_form_name, forms):
    """
    Return the derivatives of a unary form wrt all solid properties
    """
    property_labels = [
        form_name.split('.')[-1] for form_name in forms.keys()
        if 'coeff.prop' in form_name
    ]
    for prop_name in property_labels:
        prop_coeff = _depack_property_ufl_coeff(forms[f'coeff.prop.{prop_name}'])
        try:
            df_dprop = dfn.derivative(
                forms[f'form.un.{unary_form_name}'], prop_coeff
            )
        except RuntimeError:
            df_dprop = None

        forms[f'form.bi.d{unary_form_name}_d{prop_name}'] = df_dprop

    return forms

def gen_unary_linearized_forms(forms):
    """
    Generate linearized forms representing linearization of the residual wrt different states

    These forms are needed for solving the Hopf bifurcation problem/conditions
    """
    # Specify the linearization directions
    for var_name in ['u1', 'v1', 'a1']:
        forms[f'coeff.dstate.{var_name}'] = dfn.Function(forms[f'coeff.state.{var_name}'].function_space())
    for var_name in ['p1']:
        forms[f'coeff.dfsi.{var_name}'] = dfn.Function(forms[f'coeff.fsi.{var_name}'].function_space())

    # Compute the jacobian bilinear forms
    unary_form_name = 'f1uva'
    for var_name in ['u1', 'v1', 'a1']:
        forms[f'form.bi.d{unary_form_name}_d{var_name}'] = dfn.derivative(forms[f'form.un.{unary_form_name}'], forms[f'coeff.state.{var_name}'])
    for var_name in ['p1']:
        forms[f'form.bi.d{unary_form_name}_d{var_name}'] = dfn.derivative(forms[f'form.un.{unary_form_name}'], forms[f'coeff.fsi.{var_name}'])

    # Take the action of the jacobian linear forms along states to get linearized unary forms
    # dF/dx * delta x, dF/dp * delta p, ...
    for var_name in ['u1', 'v1', 'a1']:
        unary_form_name = f'df1uva_{var_name}'
        df_dx = forms[f'form.bi.df1uva_d{var_name}']
        # print(len(df_dx.arguments()))
        # print(len(forms[f'form.un.f1uva'].arguments()))
        forms[f'form.un.{unary_form_name}'] = dfn.action(df_dx, forms[f'coeff.dstate.{var_name}'])

    for var_name in ['p1']:
        unary_form_name = f'df1uva_{var_name}'
        df_dx = forms[f'form.bi.df1uva_d{var_name}']
        forms[f'form.un.{unary_form_name}'] = dfn.action(df_dx, forms[f'coeff.dfsi.{var_name}'])

    # Compute the total linearized residual
    forms[f'form.un.df1uva'] = reduce(
        operator.add,
        [forms[f'form.un.df1uva_{var_name}'] for var_name in ('u1', 'v1', 'a1', 'p1')]
        )

def add_newmark_time_disc_form(forms):
    u0 = forms['coeff.state.u0']
    v0 = forms['coeff.state.v0']
    a0 = forms['coeff.state.a0']
    u1 = forms['coeff.state.u1']
    v1 = forms['coeff.state.v1']
    a1 = forms['coeff.state.a1']

    dt = dfn.Function(forms['fspace.scalar'])
    gamma = dfn.Constant(1/2)
    beta = dfn.Constant(1/4)
    v1_nmk = newmark.newmark_v(u1, u0, v0, a0, dt, gamma, beta)
    a1_nmk = newmark.newmark_a(u1, u0, v0, a0, dt, gamma, beta)

    forms['form.un.f1'] = ufl.replace(forms['form.un.f1uva'], {v1: v1_nmk, a1: a1_nmk})
    forms['coeff.time.dt'] = dt
    forms['coeff.time.gamma'] = gamma
    forms['coeff.time.beta'] = beta
    return forms

## These are the core form definitions
def base_form_definitions(
        mesh: dfn.Mesh,
        mesh_funcs: Tuple[dfn.MeshFunction],
        mesh_entities_label_to_value: Tuple[Mapping[str, int]],
        fsi_facet_labels: Tuple[str],
        fixed_facet_labels: Tuple[str]
    ):
    # Measures
    vertex_func, facet_func, cell_func = mesh_funcs
    vertex_label_to_id, facet_label_to_id, cell_label_to_id = mesh_entities_label_to_value
    dx = dfn.Measure('dx', domain=mesh, subdomain_data=cell_func)
    ds = dfn.Measure('ds', domain=mesh, subdomain_data=facet_func)
    _traction_ds = [ds(int(facet_label_to_id[facet_label])) for facet_label in fsi_facet_labels]
    traction_ds = reduce(operator.add, _traction_ds)

    # Function space
    scalar_fspace = dfn.FunctionSpace(mesh, 'CG', 1)
    vector_fspace = dfn.VectorFunctionSpace(mesh, 'CG', 1)
    scalar_dg0_fspace = dfn.FunctionSpace(mesh, 'DG', 0)

    # Trial/test function
    vector_trial = dfn.TrialFunction(vector_fspace)
    vector_test = dfn.TestFunction(vector_fspace)
    scalar_trial = dfn.TrialFunction(scalar_fspace)
    scalar_test = dfn.TestFunction(scalar_fspace)
    strain_test = form_inf_strain(vector_test)

    # Dirichlet BCs
    bc_base = dfn.DirichletBC(vector_fspace, dfn.Constant([0.0, 0.0]),
                              facet_func, facet_label_to_id['fixed'])

    # Basic kinematics
    u0 = dfn.Function(vector_fspace)
    v0 = dfn.Function(vector_fspace)
    a0 = dfn.Function(vector_fspace)

    u1 = dfn.Function(vector_fspace)
    v1 = dfn.Function(vector_fspace)
    a1 = dfn.Function(vector_fspace)

    xref = dfn.Function(vector_fspace)
    xref.vector()[:] = scalar_fspace.tabulate_dof_coordinates().reshape(-1).copy()

    forms = {
        'measure.dx': dx,
        'measure.ds': ds,
        'measure.ds_traction': traction_ds,
        'bc.dirichlet': bc_base,

        'geom.facet_normal': dfn.FacetNormal(mesh),

        'fspace.vector': vector_fspace,
        'fspace.scalar': scalar_fspace,
        'fspace.scalar_dg0': scalar_dg0_fspace,

        'test.scalar': scalar_test,
        'test.vector': vector_test,
        'test.strain': strain_test,
        'trial.vector': vector_trial,
        'trial.scalar': scalar_trial,

        'coeff.state.u0': u0,
        'coeff.state.v0': v0,
        'coeff.state.a0': a0,
        'coeff.state.u1': u1,
        'coeff.state.v1': v1,
        'coeff.state.a1': a1,
        'coeff.ref.x': xref,

        'expr.kin.inf_strain': form_inf_strain(u1),
        'expr.kin.inf_strain_rate': form_inf_strain(v1),

        'form.un.f1uva': 0.0,

        'mesh.mesh': mesh,
        'mesh.vertex_label_to_id': vertex_label_to_id,
        'mesh.facet_label_to_id': facet_label_to_id,
        'mesh.cell_label_to_id': cell_label_to_id,
        'mesh.vertex_function': vertex_func,
        'mesh.facet_function': facet_func,
        'mesh.cell_function': cell_func,
        'mesh.fsi_facet_labels': fsi_facet_labels,
        'mesh.fixed_facet_labels': fixed_facet_labels}
    return forms

# Inertial effect forms
def add_inertial_form(forms):
    dx = forms['measure.dx']
    vector_test = forms['test.vector']

    a = forms['coeff.state.a1']
    rho = dfn.Function(forms['fspace.scalar_dg0'])
    inertial_body_force = rho*a

    forms['form.un.f1uva'] += ufl.inner(inertial_body_force, vector_test) * dx
    forms['coeff.prop.rho'] = rho
    forms['expr.force_inertial'] = inertial_body_force
    return forms

# Elastic effect forms
def add_isotropic_elastic_form(forms):
    dx = forms['measure.dx']
    vector_test = forms['test.vector']
    strain_test = form_inf_strain(vector_test)

    u = forms['coeff.state.u1']
    inf_strain = form_inf_strain(u)
    emod = dfn.Function(forms['fspace.scalar_dg0'])
    nu = dfn.Constant(0.45)
    stress_elastic = form_lin_iso_cauchy_stress(inf_strain, emod, nu)

    forms['form.un.f1uva'] += ufl.inner(stress_elastic, strain_test) * dx
    forms['coeff.prop.emod'] = emod
    forms['coeff.prop.nu'] = nu
    forms['expr.stress_elastic'] = stress_elastic

    lame_lambda = emod*nu/(1+nu)/(1-2*nu)
    stress_zz = lame_lambda*ufl.tr(inf_strain)
    forms['expr.stress_elastic_zz'] = stress_zz
    return forms

def add_isotropic_elastic_with_incomp_swelling_form(forms):
    dx = forms['measure.dx']
    strain_test = forms['test.strain']

    emod = dfn.Function(forms['fspace.scalar_dg0'])
    nu = 0.5
    u = forms['coeff.state.u1']
    inf_strain = form_inf_strain(u)
    v_swelling = dfn.Function(forms['fspace.scalar_dg0'])
    k_swelling = dfn.Constant(1.0)
    v_swelling.vector()[:] = 1.0
    lame_mu = emod/2/(1+nu)
    stress_elastic = 2*lame_mu*inf_strain + k_swelling*(ufl.tr(inf_strain)-(v_swelling-1.0))*ufl.Identity(inf_strain.ufl_shape[0])

    forms['form.un.f1uva'] += ufl.inner(stress_elastic, strain_test) * dx
    forms['coeff.prop.emod'] = emod
    forms['coeff.prop.v_swelling'] = v_swelling
    forms['coeff.prop.k_swelling'] = k_swelling
    forms['expr.stress_elastic'] = stress_elastic
    return forms

def add_isotropic_elastic_with_swelling_form(forms):
    dx = forms['measure.dx']
    strain_test = forms['test.strain']
    u = forms['coeff.state.u1']

    lin_green_strain_test = form_lin_green_strain(u, forms['test.vector'])

    inf_strain = form_inf_strain(u)
    emod = dfn.Function(forms['fspace.scalar_dg0'])
    nu = dfn.Constant(0.45)
    v_swelling = dfn.Function(forms['fspace.scalar_dg0'])
    v_swelling.vector()[:] = 1.0
    m_swelling = dfn.Function(forms['fspace.scalar_dg0'])
    m_swelling.vector()[:] = 0.0

    lame_lambda = emod*nu/(1+nu)/(1-2*nu)
    lame_mu = emod/2/(1+nu)
    stress_initial = -(lame_lambda+2/3*lame_mu)*(v_swelling-1)*ufl.Identity(inf_strain.ufl_shape[0])
    stress_elastic = (m_swelling*(v_swelling-1) + 1)*form_lin_iso_cauchy_stress(inf_strain, emod, nu)
    stress_total = stress_initial + stress_elastic

    forms['form.un.f1uva'] += (
        ufl.inner(stress_total, strain_test) * dx
        + ufl.inner(stress_initial, lin_green_strain_test) * dx
    )
    forms['coeff.prop.emod'] = emod
    forms['coeff.prop.nu'] = nu
    forms['coeff.prop.v_swelling'] = v_swelling
    forms['coeff.prop.m_swelling'] = m_swelling
    forms['expr.stress_elastic'] = stress_elastic

    # lame_lambda = emod*nu/(1+nu)/(1-2*nu)
    # lame_mu = emod/2/(1+nu)
    return forms

# Surface forcing forms
def add_surface_pressure_form(forms):
    ds = forms['measure.ds_traction']
    vector_test = forms['test.vector']
    u = forms['coeff.state.u1']
    facet_normal = forms['geom.facet_normal']

    p = dfn.Function(forms['fspace.scalar'])
    reference_traction = -p * form_pullback_area_normal(u, facet_normal)

    forms['form.un.f1uva'] -= ufl.inner(reference_traction, vector_test) * ds
    forms['coeff.fsi.p1'] = p

    forms['expr.fluid_traction'] = reference_traction
    return forms

def add_manual_contact_traction_form(forms):
    ds = forms['measure.ds_traction']
    vector_test = forms['test.vector']

    # the contact traction must be manually linked with displacements and penalty parameters!
    ycontact = dfn.Constant(np.inf)
    kcontact = dfn.Constant(1.0)
    ncontact = dfn.Constant([0.0, 1.0])
    tcontact = dfn.Function(forms['fspace.vector'])
    traction_contact = ufl.inner(tcontact, vector_test) * ds

    forms['form.un.f1uva'] -= traction_contact
    forms['coeff.state.manual.tcontact'] = tcontact
    forms['coeff.prop.ncontact'] = ncontact
    forms['coeff.prop.kcontact'] = kcontact
    forms['coeff.prop.ycontact'] = ycontact
    return forms

# Surface membrane forms
def add_isotropic_membrane(forms):
    # Define the 8th order projector to get the planar strain component
    ds_traction = forms['measure.ds_traction']
    _n = forms['geom.facet_normal']
    n = ufl.as_tensor([_n[0], _n[1], 0.0])
    nn = ufl.outer(n, n)
    ident = ufl.Identity(n.ufl_shape[0])
    project_pp = ufl.outer(ident-nn, ident-nn)

    i, j, k, l = ufl.indices(4)

    vector_test = forms['test.vector']
    strain_test = form_inf_strain(vector_test)
    strain_pp_test = ufl.as_tensor(project_pp[i, j, k, l] * strain_test[j, k], (i, l))

    emod = dfn.Function(forms['fspace.scalar_dg0'])
    th_membrane = dfn.Function(forms['fspace.scalar'])
    nu = dfn.Constant(0.45)
    mu = emod/2/(1+nu)
    lmbda = emod*nu/(1+nu)/(1-2*nu)
    inf_strain = forms['expr.kin.inf_strain']
    inf_strain_pp = ufl.as_tensor(project_pp[i, j, k, l] * inf_strain[j, k], (i, l))

    # account for ambiguous 0/0 when emod=0
    lmbda_pp = ufl.conditional(ufl.eq(emod, 0), 0, 2*mu*lmbda/(lmbda+2*mu))
    stress_pp = 2*mu*inf_strain_pp + lmbda_pp*ufl.tr(inf_strain_pp)*(ident-nn)

    res = ufl.inner(stress_pp, strain_pp_test) * th_membrane*ds_traction

    forms['form.un.f1uva'] += res
    forms['coeff.prop.emod_membrane'] = emod
    forms['coeff.prop.nu_membrane'] = nu
    forms['coeff.prop.th_membrane'] = th_membrane
    return forms

def add_incompressible_epithelium_membrane(forms):
    # Define the 8th order projector to get the planar strain component
    ds_traction = forms['measure.ds_traction']
    n = forms['geom.facet_normal']
    nn = ufl.outer(n, n)
    ident = ufl.Identity(n.geometric_dimension())
    project_pp = ufl.outer(ident-nn, ident-nn)

    i, j, k, l = ufl.indices(4)

    vector_test = forms['test.vector']
    strain_test = form_inf_strain(vector_test)
    strain_pp_test = ufl.as_tensor(project_pp[i, j, k, l] * strain_test[j, k], (i, l))

    emod_membrane = dfn.Function(forms['fspace.scalar_dg0'])
    th_membrane = dfn.Function(forms['fspace.scalar'])
    nu = 0.5
    lame_mu = emod_membrane/2/(1+nu)
    inf_strain = forms['expr.kin.inf_strain']
    inf_strain_pp = ufl.as_tensor(project_pp[i, j, k, l] * inf_strain[j, k], (i, l))

    stress_pp = 2*lame_mu*inf_strain_pp + 2*lame_mu*ufl.tr(inf_strain_pp)*(ident-nn)

    res = ufl.inner(stress_pp, strain_pp_test) * th_membrane*ds_traction

    forms['form.un.f1uva'] += res
    forms['coeff.prop.emod_membrane'] = emod_membrane
    forms['coeff.prop.th_membrane'] = th_membrane
    return forms

# Viscous effect forms
def add_rayleigh_viscous_form(forms):
    dx = forms['measure.dx']
    vector_test = forms['test.vector']
    strain_test = forms['test.strain']
    u = forms['coeff.state.u1']
    v = forms['coeff.state.v1']
    a = forms['coeff.state.a1']

    rayleigh_m = dfn.Constant(1.0)
    rayleigh_k = dfn.Constant(1.0)
    stress_visco = rayleigh_k*ufl.replace(forms['expr.stress_elastic'], {u: v})
    force_visco = rayleigh_m*ufl.replace(forms['expr.force_inertial'], {a: v})

    damping = (ufl.inner(force_visco, vector_test) + ufl.inner(stress_visco, strain_test))*dx

    forms['form.un.f1uva'] += damping
    forms['coeff.prop.rayleigh_m'] = rayleigh_m
    forms['coeff.prop.rayleigh_k'] = rayleigh_k
    return forms

def add_kv_viscous_form(forms):
    dx = forms['measure.dx']
    strain_test = forms['test.strain']
    v = forms['coeff.state.v1']

    eta = dfn.Function(forms['fspace.scalar_dg0'])
    inf_strain_rate = form_inf_strain(v)
    stress_visco = eta*inf_strain_rate

    forms['form.un.f1uva'] += ufl.inner(stress_visco, strain_test) * dx
    forms['coeff.prop.eta'] = eta
    forms['expr.kv_stress'] = stress_visco
    forms['expr.kv_strain_rate'] = inf_strain_rate
    return forms

def add_ap_force_form(forms):
    dx = forms['measure.dx']
    vector_test = forms['test.vector']

    u1, v1 = forms['coeff.state.u1'], forms['coeff.state.v1']
    kv_eta = forms['coeff.prop.eta']
    emod = forms['coeff.prop.emod']
    nu = forms['coeff.prop.nu']
    lame_mu = emod/2/(1+nu)
    u_ant = dfn.Function(forms['fspace.vector']) # zero values by default
    u_pos = dfn.Function(forms['fspace.vector'])
    length = dfn.Function(forms['fspace.scalar'])
    muscle_stress = dfn.Function(forms['fspace.scalar'])
    d2u_dz2 = (u_ant - 2*u1 + u_pos) / length**2
    d2v_dz2 = (u_ant - 2*v1 + u_pos) / length**2
    force_elast_ap = (lame_mu+muscle_stress)*d2u_dz2
    force_visco_ap = 0.5*kv_eta*d2v_dz2
    stiffness = ufl.inner(force_elast_ap, vector_test) * dx
    viscous = ufl.inner(force_visco_ap, vector_test) * dx

    forms['form.un.f1uva'] += -stiffness - viscous
    forms['coeff.prop.length'] = length
    forms['coeff.prop.muscle_stress'] = muscle_stress
    return forms

# Add shape effect forms
def add_shape_form(forms):
    """
    Adds a shape parameter
    """
    umesh = dfn.Function(forms['fspace.vector'])

    # NOTE: To find the sensitivity w.r.t shape, UFL actually uses the parameters
    # `ufl.SpatialCoordinate(mesh)`
    # This doesn't have an associated function/vector of values so both are included
    # here
    # The code has to manually account for 'coeff.prop' cases that have both a
    # function/vector and ufl coefficient instance
    forms['coeff.prop.umesh'] = (umesh, ufl.SpatialCoordinate(forms['mesh.mesh']))
    forms['mesh.REF_COORDINATES'] = forms['mesh.mesh'].coordinates().copy()
    return forms

## Form models
def Rayleigh(
    mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels):
    return \
        add_newmark_time_disc_form(
        add_manual_contact_traction_form(
        add_surface_pressure_form(
        add_rayleigh_viscous_form(
        add_inertial_form(
        add_isotropic_elastic_form(
        add_shape_form(
        base_form_definitions(
            mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels))))))))

def KelvinVoigt(
    mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels):
    return \
        add_newmark_time_disc_form(
        add_manual_contact_traction_form(
        add_surface_pressure_form(
        add_kv_viscous_form(
        add_inertial_form(
        add_isotropic_elastic_form(
        add_shape_form(
        base_form_definitions(
            mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels))))))))

def KelvinVoigtWEpithelium(
    mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels):
    return  \
        add_newmark_time_disc_form(
        add_isotropic_membrane(
        add_manual_contact_traction_form(
        add_surface_pressure_form(
        add_kv_viscous_form(
        add_inertial_form(
        add_isotropic_elastic_form(
        add_shape_form(
        base_form_definitions(
            mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels)))))))))

def IncompSwellingKelvinVoigt(
    mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels):
    return \
        add_newmark_time_disc_form(
        add_manual_contact_traction_form(
        add_surface_pressure_form(
        add_kv_viscous_form(
        add_inertial_form(
        add_isotropic_elastic_with_incomp_swelling_form(
        add_shape_form(
        base_form_definitions(
            mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels))))))))

def SwellingKelvinVoigt(
    mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels):
    return \
        add_newmark_time_disc_form(
        add_manual_contact_traction_form(
        add_surface_pressure_form(
        add_kv_viscous_form(
        add_inertial_form(
        add_isotropic_elastic_with_swelling_form(
        add_shape_form(
        base_form_definitions(
            mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels))))))))

def SwellingKelvinVoigtWEpithelium(
    mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels):
    return \
        add_newmark_time_disc_form(
        add_isotropic_membrane(
        add_manual_contact_traction_form(
        add_surface_pressure_form(
        add_kv_viscous_form(
        add_inertial_form(
        add_isotropic_elastic_with_swelling_form(
        add_shape_form(
        base_form_definitions(
            mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels)))))))))

def SwellingKelvinVoigtWEpitheliumNoShape(
    mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels):
    return \
        add_newmark_time_disc_form(
        add_isotropic_membrane(
        add_manual_contact_traction_form(
        add_surface_pressure_form(
        add_kv_viscous_form(
        add_inertial_form(
        add_isotropic_elastic_with_swelling_form(
        base_form_definitions(
            mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels))))))))

def Approximate3DKelvinVoigt(
    mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels):
    return \
        add_newmark_time_disc_form(
        add_manual_contact_traction_form(
        add_surface_pressure_form(
        add_ap_force_form(
        add_kv_viscous_form(
        add_inertial_form(
        add_isotropic_elastic_form(
        add_shape_form(
        base_form_definitions(
            mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels)))))))))
