"""
This module contains definitions of functionals over the solid state.
"""

from typing import Optional

import numpy as np
# import matplotlib.pyplot as plt
# import scipy.signal as sig

import dolfin as dfn
import ufl

from .base import transform_to_make_signals, StateMeasure

class MinGlottalWidth(StateMeasure):

    def __init_measure_context__(self, *args, **kwargs):
        self.XREF = self.model.solid.forms['fspace.scalar'].tabulate_dof_coordinates()

    def __call__(self, state, control, props):
        xcur = self.XREF.reshape(-1) + state.sub['u'][:]
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
        self.idx_dof = vert_to_vdof[2*idx_vertex+1]

        self.XREF = self.model.solid.forms['fspace.scalar'].tabulate_dof_coordinates()

    def __call__(self, state, control, props):
        xcur = self.XREF.reshape(-1) + state['u'][:]
        return 2*(props['ymid'][0] - xcur[self.idx_dof])

class MaxContactPressure(StateMeasure):

    def __init_measure_context__(self, *args, **kwargs):
        self.coeff_tcontact = self.model.solid.forms['coeff.state.manual.tcontact']

    def __call__(self, state, control, props):
        self.model.set_fin_state(state)
        fsidofs = self.model.solid.vert_to_sdof[self.model.fsi_verts]

        tcontact = self.coeff_tcontact.vector()[:].reshape(-1, 2) # [FSI_DOFS]
        pcontact = np.linalg.norm(tcontact, axis=-1)
        return pcontact[fsidofs].max()

class ContactStatistics(StateMeasure):

    def __init_measure_context__(self, *args, **kwargs):
        self.dx = kwargs.get('measure', self.model.solid.forms['measure.ds_traction'])

        tcontact = self.model.solid.forms['coeff.state.manual.tcontact']
        pcontact = ufl.inner(tcontact, tcontact)**0.5 # should be square of contact pressure
        contact_indicator = ufl.conditional(ufl.operators.ne(pcontact, 0.0), 1.0, 0.0)

        self.expr_contact_area = contact_indicator*self.dx
        self.expr_total_p_contact = pcontact*self.dx
        self.coeff_tcontact = tcontact

    def __call__(self, state, control, props):
        self.model.set_fin_state(state)

        tcontact_vec = np.array(self.coeff_tcontact.vector()[:]).reshape(-1, 2) # [FSI_DOFS]
        pcontact = np.linalg.norm(tcontact_vec, axis=-1)

        idx_max = np.argmax(pcontact)
        total_pcontact = dfn.assemble(self.expr_total_p_contact)
        area_contact = dfn.assemble(self.expr_contact_area)
        max_pcontact = pcontact[idx_max]
        avg_pcontact = 0.0 if area_contact == 0.0 else total_pcontact/area_contact
        return (max_pcontact, avg_pcontact, total_pcontact, area_contact)

class ViscousDissipationRate(StateMeasure):

    def __init_measure_context__(self, *args, **kwargs):
        self.dx = kwargs.get('measure', self.model.solid.forms['measure.dx'])

        kv_stress = self.model.solid.forms['expr.kv_stress']
        kv_strain_rate = self.model.solid.forms['expr.kv_strain_rate']
        self.expr_total_dissipation_rate = ufl.inner(kv_stress, kv_strain_rate)*self.dx

    def __call__(self, state, control, props):
        self.model.set_props(props)
        self.model.set_fin_state(state)
        self.model.set_control(control)
        return dfn.assemble(self.expr_total_dissipation_rate)

### Field type post-processing functions

class Field(StateMeasure):
    def __init_measure_context__(
            self,
            dx: Optional[dfn.Measure]=None,
            fspace: Optional[dfn.FunctionSpace]=None
        ):
        """
        Parameters
        ----------
        dx : dfn.Measure
            A measure to project the field expression over
        fspace : dfn.FunctionSpace
            A function space to project the field expression onto
        """
        if fspace is None:
            self.fspace = dfn.FunctionSpace(self.model.solid.forms['mesh.mesh'], 'DG', 0)
        else:
            self.fspace = fspace

        if dx is None:
            self.dx = self.model.solid.forms['measure.dx']
        else:
            self.dx = dx

class StressI1Field(Field):
    def __init_measure_context__(self, dx=None, fspace=None):
        super().__init_measure_context__(dx, fspace)

        model = self.model
        kv_stress = model.solid.forms['expr.kv_stress']
        el_stress = model.solid.forms['expr.stress_elastic']
        S = el_stress + kv_stress

        # This is the first invariant (I1)
        self.expression = ufl.tr(S)
        self.project = make_project(self.expression, self.fspace, self.dx)

    def __call__(self, state, control, props):
        model = self.model
        model.set_fin_state(state)
        model.set_control(control)
        model.set_props(props)

        return np.array(self.project()[:])

class StressI2Field(Field):
    def __init_measure_context__(self, dx=None, fspace=None):
        super().__init_measure_context__(dx, fspace)

        model = self.model

        kv_stress = model.solid.forms['expr.kv_stress']
        el_stress = model.solid.forms['expr.stress_elastic']
        S = el_stress + kv_stress

        # This is the second invariant (I2)
        self.expression = 1/2*(ufl.tr(S)**2-ufl.tr(S*S))
        self.project = make_project(self.expression, self.fspace, self.dx)

    def __call__(self, state, control, props):
        model = self.model
        model.set_fin_state(state)
        model.set_control(control)
        model.set_props(props)

        return np.array(self.project()[:])

class StressI3Field(Field):
    def __init_measure_context__(self, dx=None, fspace=None):
        super().__init_measure_context__(dx, fspace)

        model = self.model

        kv_stress = model.solid.forms['expr.kv_stress']
        el_stress = model.solid.forms['expr.stress_elastic']
        S = el_stress + kv_stress

        # This is the third invariant (I3)
        self.expression = ufl.det(S)
        self.project = make_project(self.expression, self.fspace, self.dx)

    def __call__(self, state, control, props):
        model = self.model
        model.set_fin_state(state)
        model.set_control(control)
        model.set_props(props)

        return np.array(self.project()[:])

class StressHydrostaticField(Field):
    def __init_measure_context__(self, dx=None, fspace=None):
        super().__init_measure_context__(dx, fspace)

        kv_stress = self.model.solid.forms['expr.kv_stress']
        el_stress = self.model.solid.forms['expr.stress_elastic']
        S = el_stress + kv_stress

        self.expression = -1/3*ufl.tr(S)
        self.project = make_project(self.expression, self.fspace, self.dx)

    def __call__(self, state, control, props):
        self.model.set_props(props)
        self.model.set_control(control)
        self.model.set_fin_state(state)

        return np.array(self.project()[:])

class StressVonMisesField(Field):
    def __init_measure_context__(self, dx=None, fspace=None):
        super().__init_measure_context__(dx, fspace)

        kv_stress = self.model.solid.forms['expr.kv_stress']
        el_stress = self.model.solid.forms['expr.stress_elastic']
        S = el_stress + kv_stress

        S_dev = S - 1/3*ufl.tr(S)*ufl.Identity(3)
        j2 = 0.5*ufl.tr(S_dev*S_dev)
        self.expression = (3*j2)**(1/2)
        self.project = make_project(self.expression, self.fspace, self.dx)

    def __call__(self, state, control, props):
        model = self.model
        model.set_props(props)
        model.set_control(control)
        model.set_fin_state(state)

        return np.array(self.project()[:])

class ElasticStressField(Field):
    def __init_measure_context__(self, dx=None, fspace=None):
        super().__init_measure_context__(dx, fspace)

        forms = self.model.solid.forms
        self.expression = forms['expr.stress_elastic']
        self.project = make_project(self.expression, self.fspace, self.dx)

    def __call__(self, state, control, props):
        self.model.set_state(state)
        self.model.set_control(control)
        return np.array(self.project()[:])

class ContactPressureField(Field):

    def __init_measure_context__(self, dx=None, fspace=None):
        super().__init_measure_context__(dx, fspace)

        tcontact = self.model.solid.forms['coeff.state.manual.tcontact']
        # `tcontact*tcontact` should be the square of contact pressure
        self.expression = ufl.inner(tcontact, tcontact)**0.5
        self.project = make_project(self.expression, self.fspace, self.dx)

    def __call__(self, state, control, props):
        self.model.set_props(props)
        self.model.set_control(control)
        self.model.set_fin_state(state)

        return np.array(self.project()[:])

class ViscousDissipationField(Field):
    def __init_measure_context__(self, dx=None, fspace=None):
        super().__init_measure_context__(dx, fspace)

        kv_stress = self.model.solid.forms['expr.kv_stress']
        kv_strain_rate = self.model.solid.forms['expr.kv_strain_rate']
        self.expression = ufl.inner(kv_stress, kv_strain_rate)*self.dx
        self.project = make_project(self.expression, self.fspace, self.dx)

    def __call__(self, state, control, props):
        self.model.set_props(props)
        self.model.set_control(control)
        self.model.set_fin_state(state)

        return np.array(self.project()[:])


class StressVonMisesAverage(StressVonMisesField):
    def __init_measure_context__(self, *args, **kwargs):
        super().__init_measure_context__(self, *args, **kwargs)

        self.expr_avg = self.expression*self.dx
        self.total_dx = dfn.assemble(1*self.dx)

    def __call__(self, state, control, props):
        model = self.model
        model.set_fin_state(state)
        model.set_control(control)
        model.set_props(props)

        return dfn.assemble(self.expr_avg) / self.total_dx

class StressVonMisesSpatialStats(StressVonMisesAverage):
    def __init_measure_context__(self, *args, **kwargs):
        super().__init_measure_context__(self, *args, **kwargs)

    def __call__(self, state, control, props):
        model = self.model
        model.set_fin_state(state)
        model.set_control(control)
        model.set_props(props)

        field = self.project()
        avg = dfn.assemble(self.expr_avg) / self.total_dx

        return np.min(field[:]), np.max(field[:]), avg


def make_scalar_form(model, form):
    """
    Return a function that computes a scalar form for different coefficients
    """
    def scalar_form(state, control, props):
        model.set_fin_state(state)
        model.set_control(control)
        return dfn.assemble(form)

    return scalar_form

def make_project(expr, fspace, measure, vec=None):
    """
    Project an expression onto the function space w.r.t the measure

    Parameters
    ----------
    expr : ufl.Expression
        An ufl expression
    fspace :
        The function space that `expr` will be projected on
    measure : ufl.Measure
        The measure that the projection will be applied over
    vec : dfn.GenericVector
        A vector to project the function into
    """
    trial = dfn.TrialFunction(fspace)
    test = dfn.TestFunction(fspace)
    A = dfn.assemble(trial*test*measure, keep_diagonal=True, tensor=dfn.PETScMatrix())
    A.ident_zeros()
    lhs_expr = expr*test*measure

    bvec = dfn.PETScVector()
    if vec is None:
        x = dfn.Function(fspace).vector()
    else:
        x = vec
    def project():
        """
        Project an expression onto the function space over the supplied measure
        """
        b = dfn.assemble(lhs_expr, tensor=bvec)
        dfn.solve(A, x, b, 'lu')
        return x
    return project
