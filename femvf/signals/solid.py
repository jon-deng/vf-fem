"""
This module contains definitions of functionals over the solid state.
"""

import numpy as np
# import matplotlib.pyplot as plt
# import scipy.signal as sig

import dolfin as dfn
import ufl

from blocktensor import linalg
# from .base import AbstractFunctional
from ..models.solidforms import form_inf_strain
from .decorator import transform_to_make_signals

def make_glottal_width_smooth(model):
    def glottal_width_smooth(model, state, control, props):
        XREF = model.solid.scalar_fspace.tabulate_dof_coordinates()

        xcur = XREF.reshape(-1) + state['u'][:]
        widths = props['y_midline'] - xcur[1::2]
        gw = np.min(widths)
        return gw
    return glottal_width_smooth
make_sig_glottal_width_smooth = transform_to_make_signals(make_glottal_width_smooth)

def make_glottal_width_sharp(model):
    def glottal_width_sharp(model, state, control, props):
        XREF = model.solid.scalar_fspace.tabulate_dof_coordinates()

        xcur = XREF.reshape(-1) + state['u'][:]
        widths = props['y_midline'] - xcur[1::2]
        gw = np.min(widths)
        return gw
    return glottal_width_sharp
make_sig_glottal_width_sharp = transform_to_make_signals(make_glottal_width_sharp)

def make_peak_contact_pressure(model):
    def peak_contact_pressure(model, state, control, props):
        fsidofs = model.solid.vert_to_sdof[model.fsi_verts]
        
        model.set_fin_state(state)
        pcontact = -1*model.solid.forms['coeff.state.manual.tcontact'].vector()[1::2]
        return pcontact[fsidofs].max()
    return peak_contact_pressure
make_sig_peak_contact_pressure = transform_to_make_signals(make_peak_contact_pressure)

def make_piecewise_elastic_stress(model):
    V = dfn.TensorFunctionSpace(model.solid.mesh, 'DG', 0)

    forms = model.solid.forms
    stress = forms['expr.stress_elastic'] if 'expr.stress_elastic' in forms else 0.0
    def piecewise_elastic_stress(state, control, props):
        elementwise_stress = dfn.project(stress, V, solver_type='lu')
        return elementwise_stress.vector()
    return piecewise_elastic_stress
make_sig_piecewise_elastic_stress = transform_to_make_signals(make_piecewise_elastic_stress)

def make_contact_statistics(model):
    # FSI_DOFS = model.solid.vert_to_sdof[model.fsi_verts]
    
    ds = model.solid.forms['measure.ds_traction']
    tcontact = model.solid.forms['coeff.state.manual.tcontact']
    pcontact = ufl.inner(tcontact, tcontact)**0.5 # should be square of contact pressure
    contact_indicator = ufl.conditional(ufl.operators.ne(pcontact, 0.0), 1.0, 0.0)

    contact_area_expr = contact_indicator*ds
    total_pcontact_expr = pcontact*ds
    
    def contact_statistics(state, control, props):
        """
        Returns (max p contact, avg p contact, total p contact, contact area)
        """
        model.set_fin_state(state)
        tcontact_vec = tcontact.vector()[:].reshape(-1, 2) # [FSI_DOFS]
        pcontact = np.linalg.norm(tcontact_vec, axis=-1)

        # idx_min = np.argmin(tcontact_mag)
        idx_max = np.argmax(pcontact)
        total_pcontact = dfn.assemble(total_pcontact_expr)
        area_contact = dfn.assemble(contact_area_expr)
        max_pcontact = pcontact[idx_max]
        avg_pcontact = 0.0 if area_contact == 0.0 else total_pcontact/area_contact
        return (max_pcontact, avg_pcontact, total_pcontact, area_contact)
    return contact_statistics
make_sig_contact_statistics = transform_to_make_signals(make_contact_statistics)

def make_viscoelastic_dissipation_rate(model, dmeas):
    kv_stress = model.solid.forms['expr.kv_stress']
    kv_strain_rate = model.solid.forms['expr.kv_strain_rate']
    total_dissipation_rate = ufl.inner(kv_stress, kv_strain_rate)*dmeas

    assem_scalar_form = make_scalar_form(model, total_dissipation_rate)
    def viscoelastic_dissipation_rate(state, control, props):
        return assem_scalar_form(state, control, props)

    return viscoelastic_dissipation_rate
make_sig_viscoelastic_dissipation_rate = transform_to_make_signals(make_viscoelastic_dissipation_rate)

def make_stress_invariant_statistics(model, fspace, dmeas):
    """
    Return min/max/avg of the 3 stress invariants and the von-mises stress
    """
    kv_stress = model.solid.forms['expr.kv_stress']
    el_stress = model.solid.forms['expr.stress_elastic']
    S = el_stress + kv_stress

    I1 = ufl.tr(S)
    I2 = 1/2*(ufl.tr(S)**2-ufl.tr(S*S))
    I3 = ufl.det(S)

    j2 = 1/3*I1**2-I2
    SVONMISES = (3*j2)**0.5

    expressions = (I1, I2, I3, SVONMISES)
    projectors = tuple([make_project(expr, fspace, dmeas) for expr in expressions])
    expression_totals = tuple([expr*dmeas for expr in expressions])
    meas_total = dfn.assemble(1*dmeas)

    def stress_invariant_statistics(state, control, props):
        model.set_fin_state(state)
        model.set_control(control)
        model.set_properties(props)

        stats = []
        # For each expression, compute the min/max/average
        for expr, project, expr_total in zip(expressions, projectors, expression_totals):
            expr_vec = project().vector()
            _min = np.min(expr_vec[:])
            _max = np.max(expr_vec[:])
            _avg = dfn.assemble(expr_total)/meas_total
            stats += [_min, _max, _avg]
        return np.array(stats)

    return stress_invariant_statistics
make_sig_stress_invariant_statistics = transform_to_make_signals(make_stress_invariant_statistics)

def make_scalar_form(model, form):
    """
    Return a function that computes a scalar form for different coefficients
    """
    def scalar_form(state, control, props):
        model.set_fin_state(state)
        model.set_control(control)
        return dfn.assemble(form)

    return scalar_form

def make_scalar_expr_statistics(model, expr, fspace, dmeas):
    """
    Return a function that computes statistics from a scalar expression
    """

def make_project(expr, fspace, dmeas):
    """
    Project an expression onto the function space w.r.t the measure

    Parameters
    ----------
    expr: ufl.Expression
        An ufl expression
    fspace: 
        The function space that `expr` will be projected on
    dmeas: ufl.Measure
        The measure that the projection will be applied over
    """
    trial = dfn.TrialFunction(fspace)
    test = dfn.TestFunction(fspace)
    A = dfn.assemble(trial*test*dmeas, keep_diagonal=True, tensor=dfn.PETScMatrix())
    A.ident_zeros()
    lhs_expr = expr*test*dmeas

    x = dfn.Function(fspace)

    def project():
        """
        Project an expression onto the function space over the supplied measure
        """
        b = dfn.assemble(lhs_expr, tensor=dfn.PETScVector())
        dfn.solve(A, x.vector(), b, 'lu')
        return x
    return project
