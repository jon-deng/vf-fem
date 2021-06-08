"""
This module contains definitions of functionals over the solid state.
"""

import numpy as np
# import matplotlib.pyplot as plt
# import scipy.signal as sig

import dolfin as dfn
import ufl

from .. import linalg
# from .base import AbstractFunctional
from ..models.solid import form_inf_strain, Solid

def glottal_width_smooth(model, state, control, props):
    XREF = model.solid.scalar_fspace.tabulate_dof_coordinates()

    xcur = XREF.reshape(-1) + state['u'][:]
    widths = props['y_midline'] - xcur[1::2]
    gw = np.min(widths)
    return gw

def glottal_width_sharp(model, state, control, props):
    XREF = model.solid.scalar_fspace.tabulate_dof_coordinates()

    xcur = XREF.reshape(-1) + state['u'][:]
    widths = props['y_midline'] - xcur[1::2]
    gw = np.min(widths)
    return gw

def peak_contact_pressure(model, state, control, props):
    fsidofs = model.solid.vert_to_sdof[model.fsi_verts]
    
    model.set_fin_state(state)
    pcontact = -1*model.solid.forms['coeff.state.manual.tcontact'].vector()[1::2]
    return pcontact[fsidofs].max()

def make_piecewise_elastic_stress(model):
    V = dfn.TensorFunctionSpace(model.solid.mesh, 'DG', 0)

    forms = model.solid.forms
    stress = forms['expr.stress_elastic'] if 'expr.stress_elastic' in forms else 0.0
    def piecewise_elastic_stress(state, control, props):
        elementwise_stress = dfn.project(stress, V, solver_type='lu')
        return elementwise_stress.vector()
    return piecewise_elastic_stress

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

def make_viscoelastic_dissipation_rate(model, dmeas):
    kv_stress = model.solid.forms['expr.kv_stress']
    kv_strain_rate = model.solid.forms['expr.kv_strain_rate']
    total_dissipation_rate = ufl.inner(kv_stress, kv_strain_rate)*dmeas

    assem_scalar_form = make_scalar_form(model, total_dissipation_rate)
    def viscoelastic_dissipation_rate(state, control, props):
        return assem_scalar_form(state, control, props)

    return viscoelastic_dissipation_rate


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
    """
    trial = dfn.Function(fspace)
    test = dfn.Function(fspace)
    A = dfn.assemble(trial*test, keep_diagonal=True, tensor=dfn.PETScMatrix())
    A.ident_zeros()

    x = dfn.Function(fspace)

    def project(expr_vec):
        """
        Project an expression onto the function space over the supplied measure
        """
        expr.vector()[:] = expr_vec
        b = dfn.assemble(expr*test*dmeas, tensor=dfn.PETScVector())

        dfn.solve(A, x, b, 'lu')
        return x.vector()
    return project
