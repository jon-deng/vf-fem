"""
This module contains definitions of functionals over the solid state.
"""

import numpy as np
# import matplotlib.pyplot as plt
# import scipy.signal as sig

import dolfin as dfn
import ufl

from .decorator import transform_to_make_signals, StateMeasure

class MinGlottalWidth(StateMeasure):

    def __init_measure_context__(self, *args, **kwargs):
        self.XREF = self.model.solid.scalar_fspace.tabulate_dof_coordinates()

    def __call__(self, state, control, props):
        xcur = self.XREF.reshape(-1) + state['u'][:]
        widths = 2*(props['ymid'] - xcur[1::2])
        gw = np.min(widths)
        return gw

class MaxContactPressure(StateMeasure):

    def __init_measure_context__(self, *args, **kwargs):
        pass

    def __call__(self, state, control, props):
        fsidofs = self.model.solid.vert_to_sdof[self.model.fsi_verts]

        self.model.set_fin_state(state)
        pcontact = -1*self.model.solid.forms['coeff.state.manual.tcontact'].vector()[1::2]
        return pcontact[fsidofs].max()

class ElasticStressField(StateMeasure):
    # TODO: Pretty sure this one doesn't work
    def __init_measure_context__(self, *args, **kwargs):
        self.V = dfn.TensorFunctionSpace(self.model.solid.mesh, 'DG', 0)

        forms = self.model.solid.forms
        self.stress = forms['expr.stress_elastic'] if 'expr.stress_elastic' in forms else 0.0

    def __call__(self, state, control, props):
        self.model.set_state(state)
        self.model.set_control(control)

        stress = dfn.project(self.stress, self.V, solver_type='lu')
        return stress.vector()

class ContactStatistics(StateMeasure):

    def __init_measure_context__(self, *args, **kwargs):
        ds = self.model.solid.forms['measure.ds_traction']
        tcontact = self.model.solid.forms['coeff.state.manual.tcontact']
        pcontact = ufl.inner(tcontact, tcontact)**0.5 # should be square of contact pressure
        contact_indicator = ufl.conditional(ufl.operators.ne(pcontact, 0.0), 1.0, 0.0)

        self.contact_area_expr = contact_indicator*ds
        self.total_pcontact_expr = pcontact*ds
        self.tcontact = tcontact

    def __call__(self, state, control, props):
        self.model.set_fin_state(state)

        tcontact_vec = self.tcontact.vector()[:].reshape(-1, 2) # [FSI_DOFS]
        pcontact = np.linalg.norm(tcontact_vec, axis=-1)

        # idx_min = np.argmin(tcontact_mag)
        idx_max = np.argmax(pcontact)
        total_pcontact = dfn.assemble(self.total_pcontact_expr)
        area_contact = dfn.assemble(self.contact_area_expr)
        max_pcontact = pcontact[idx_max]
        avg_pcontact = 0.0 if area_contact == 0.0 else total_pcontact/area_contact
        return (max_pcontact, avg_pcontact, total_pcontact, area_contact)

class ViscousDissipationRate(StateMeasure):

    def __init_measure_context__(self, *args, **kwargs):
        model = self.model
        dmeas = kwargs.get('measure', model.solid.forms['measure.dx'])

        kv_stress = model.solid.forms['expr.kv_stress']
        kv_strain_rate = model.solid.forms['expr.kv_strain_rate']
        total_dissipation_rate = ufl.inner(kv_stress, kv_strain_rate)*dmeas

        self.assem_scalar_form = make_scalar_form(model, total_dissipation_rate)

    def __call__(self, state, control, props):
        return self.assem_scalar_form(state, control, props)

class StressI1Field(StateMeasure):
    def __init_measure_context__(self, *args, **kwargs):
        model = self.model

        kv_stress = model.solid.forms['expr.kv_stress']
        el_stress = model.solid.forms['expr.stress_elastic']
        S = el_stress + kv_stress

        self.I1 = ufl.tr(S)
        I2 = 1/2*(ufl.tr(S)**2-ufl.tr(S*S))
        I3 = ufl.det(S)

        self.fspace = kwargs.get(
            'fspace',
            dfn.FunctionSpace(model.solid.forms['mesh.mesh'], 'DG', 0)
        )

    def __call__(self, state, control, props):
        model = self.model
        model.set_fin_state(state)
        model.set_control(control)
        model.set_props(props)

        return dfn.project(self.I1, self.fspace)

class StressI2Field(StateMeasure):
    def __init_measure_context__(self, *args, **kwargs):
        model = self.model

        kv_stress = model.solid.forms['expr.kv_stress']
        el_stress = model.solid.forms['expr.stress_elastic']
        S = el_stress + kv_stress

        I1 = ufl.tr(S)
        self.I2 = 1/2*(ufl.tr(S)**2-ufl.tr(S*S))
        I3 = ufl.det(S)

        self.fspace = kwargs.get(
            'fspace',
            dfn.FunctionSpace(model.solid.forms['mesh.mesh'], 'DG', 0)
        )

    def __call__(self, state, control, props):
        model = self.model
        model.set_fin_state(state)
        model.set_control(control)
        model.set_props(props)

        return dfn.project(self.I2, self.fspace)

class StressI3Field(StateMeasure):
    def __init_measure_context__(self, *args, **kwargs):
        model = self.model

        kv_stress = model.solid.forms['expr.kv_stress']
        el_stress = model.solid.forms['expr.stress_elastic']
        S = el_stress + kv_stress

        I1 = ufl.tr(S)
        I2 = 1/2*(ufl.tr(S)**2-ufl.tr(S*S))
        self.I3 = ufl.det(S)

        self.fspace = kwargs.get(
            'fspace',
            dfn.FunctionSpace(model.solid.forms['mesh.mesh'], 'DG', 0)
        )

    def __call__(self, state, control, props):
        model = self.model
        model.set_fin_state(state)
        model.set_control(control)
        model.set_props(props)

        return dfn.project(self.I3, self.fspace)

class StressHydrostaticField(StateMeasure):
    def __init_measure_context__(self, *args, **kwargs):
        model = self.model

        kv_stress = model.solid.forms['expr.kv_stress']
        el_stress = model.solid.forms['expr.stress_elastic']
        S = el_stress + kv_stress

        self.phydro = -1/3*ufl.tr(S)

        self.fspace = kwargs.get(
            'fspace',
            dfn.FunctionSpace(model.solid.forms['mesh.mesh'], 'DG', 0)
        )

    def __call__(self, state, control, props):
        model = self.model
        model.set_fin_state(state)
        model.set_control(control)
        model.set_props(props)

        return dfn.project(self.phydro, self.fspace).vector()

class StressVonMisesField(StateMeasure):
    def __init_measure_context__(self, *args, **kwargs):
        model = self.model

        kv_stress = model.solid.forms['expr.kv_stress']
        el_stress = model.solid.forms['expr.stress_elastic']
        S = el_stress + kv_stress

        S_dev = S - 1/3*ufl.tr(S)*ufl.Identity(3)
        j2 = 0.5*ufl.tr(S_dev*S_dev)
        self.stress_field_vm = (3*j2)**(1/2)

        self.fspace = kwargs.get(
            'fspace',
            dfn.FunctionSpace(model.solid.forms['mesh.mesh'], 'DG', 0)
        )

    def __call__(self, state, control, props):
        model = self.model
        model.set_fin_state(state)
        model.set_control(control)
        model.set_props(props)

        return dfn.project(self.stress_field_vm, self.fspace).vector()

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
