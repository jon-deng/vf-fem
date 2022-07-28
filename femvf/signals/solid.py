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

class VertexGlottalWidth(StateMeasure):
    def __init_measure_context__(self, vertex_name=None):
        # Get the DOF/vertex number corresponding to `vertex_name`
        if vertex_name is None:
            raise ValueError("`vertex_name` must be supplied")
        vertlabel_to_id = self.model.solid.forms['mesh.vertex_label_to_id']
        vert_mf = self.model.solid.forms['mesh.vertex_function']
        idx_vertex = vert_mf.where_equal(vertlabel_to_id[vertex_name])
        if len(idx_vertex) == 0:
            raise ValueError(f"No vertex named `{vertex_name}` found")
        elif len(idx_vertex) > 1:
            raise ValueError(f"Multiple vertices names `{vertex_name}` found")
        else:
            idx_vertex = idx_vertex[0]

        vert_to_vdof = dfn.vertex_to_dof_map(self.model.solid.forms['fspace.vector'])
        # Get the y-displacement DOF
        self.idx_dof = vert_to_vdof[2*idx_vertex]

        self.XREF = self.model.solid.scalar_fspace.tabulate_dof_coordinates()

    def __call__(self, state, control, props):
        xcur = self.XREF.reshape(-1) + state['u'][:]
        widths = 2*(props['ymid'] - xcur[1::2])
        return widths[self.idx_dof]

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
        dx = forms['measure.dx']
        self.stress = forms['expr.stress_elastic'] if 'expr.stress_elastic' in forms else 0.0
        self.project = make_project(self.stress, self.V, dx)

    def __call__(self, state, control, props):
        self.model.set_state(state)
        self.model.set_control(control)

        stress = self.project(dfn.Function(self.V).vector())
        return stress

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
        self.model.set_props(props)
        self.model.set_fin_state(state)
        self.model.set_control(control)
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
        dx = model.solid.forms['measure.dx']

        self.project = make_project(self.I1, self.fspace, dx)

    def __call__(self, state, control, props):
        model = self.model
        model.set_fin_state(state)
        model.set_control(control)
        model.set_props(props)

        return self.project(dfn.Function(self.fspace).vector())

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
        dx = model.solid.forms['measure.dx']

        self.project = make_project(self.I2, self.fspace, dx)

    def __call__(self, state, control, props):
        model = self.model
        model.set_fin_state(state)
        model.set_control(control)
        model.set_props(props)

        return self.project(dfn.Function(self.fspace).vector())

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
        dx = model.solid.forms['measure.dx']

        self.project = make_project(self.I3, self.fspace, dx)

    def __call__(self, state, control, props):
        model = self.model
        model.set_fin_state(state)
        model.set_control(control)
        model.set_props(props)

        return self.project(dfn.Function(self.fspace).vector())

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

        dx = model.solid.forms['measure.dx']
        self.project = make_project(self.phydro, self.fspace, dx)

    def __call__(self, state, control, props):
        model = self.model
        model.set_fin_state(state)
        model.set_control(control)
        model.set_props(props)

        return self.project(dfn.Function(self.fspace).vector())

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
        dx = model.solid.forms['measure.dx']
        self.project = make_project(self.stress_field_vm, self.fspace, dx)

    def __call__(self, state, control, props):
        model = self.model
        model.set_fin_state(state)
        model.set_control(control)
        model.set_props(props)

        return self.project(dfn.Function(self.fspace).vector())

class StressVonMisesAverage(StressVonMisesField):
    def __init_measure_context__(self, *args, **kwargs):
        super().__init_measure_context__(self, *args, **kwargs)

        dx = kwargs.get('dx', self.model.solid.forms['measure.dx'])
        self.f = self.stress_field_vm*dx
        self.total_dx = dfn.assemble(1*dx)

    def __call__(self, state, control, props):
        model = self.model
        model.set_fin_state(state)
        model.set_control(control)
        model.set_props(props)

        return dfn.assemble(self.f) / self.total_dx

class StressVonMisesSpatialStats(StressVonMisesAverage):
    def __init_measure_context__(self, *args, **kwargs):
        super().__init_measure_context__(self, *args, **kwargs)

        self.vm_field = dfn.Function(self.fspace).vector()

    def __call__(self, state, control, props):
        model = self.model
        model.set_fin_state(state)
        model.set_control(control)
        model.set_props(props)

        field = self.project(self.vm_field)
        avg = dfn.assemble(self.f) / self.total_dx

        return np.min(field), np.max(field), dfn.assemble(self.f) / self.total_dx

class ContactPressureField(StateMeasure):

    def __init_measure_context__(self, *args, **kwargs):
        tcontact = self.model.solid.forms['coeff.state.manual.tcontact']
        pcontact = ufl.inner(tcontact, tcontact)**0.5 # should be square of contact pressure
        self.pcontact = pcontact
        self.fspace = self.model.solid.forms['fspace.scalar']

        dx = self.model.solid.forms['measure.ds_traction']
        self.project = make_project(self.pcontact, self.fspace, dx)

    def __call__(self, state, control, props):
        self.model.set_props(props)
        self.model.set_control(control)
        self.model.set_fin_state(state)

        return self.project(dfn.Function(self.fspace).vector())

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

    bvec = dfn.PETScVector()
    def project(x):
        """
        Project an expression onto the function space over the supplied measure
        """
        b = dfn.assemble(lhs_expr, tensor=bvec)
        dfn.solve(A, x, b, 'lu')
        return x
    return project
