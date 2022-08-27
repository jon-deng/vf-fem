"""
This module contains definitions of functionals over the solid state.
"""

from typing import Optional

import numpy as np
# import matplotlib.pyplot as plt
# import scipy.signal as sig

import dolfin as dfn
import ufl

from femvf.models.transient.base import BaseTransientModel
from .base import BaseStateMeasure, BaseDerivedStateMeasure


### Field type post-processing functions

class BaseFieldMeasure(BaseStateMeasure):
    def __init__(
            self,
            model: BaseTransientModel,
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
        super().__init__(model)
        if fspace is None:
            self.fspace = dfn.FunctionSpace(self.model.solid.forms['mesh.mesh'], 'DG', 0)
        else:
            self.fspace = fspace

        if dx is None:
            self.dx = self.model.solid.forms['measure.dx']
        else:
            self.dx = dx

        self.expression = self._init_expression(self)
        self.project = make_project(self.expression, self.fspace, self.dx)

    def _init_expression(self):
        raise NotImplementedError("Method must be implemented by subclasses")

    def assem(self):
        raise NotImplementedError("Method must be implemented by subclasses")

class StressI1Field(BaseFieldMeasure):
    def _init_expression(self):
        model = self.model
        kv_stress = model.solid.forms['expr.kv_stress']
        el_stress = model.solid.forms['expr.stress_elastic']
        S = el_stress + kv_stress

        # This is the first invariant (I1)
        return ufl.tr(S)

    def assem(self):
        return np.array(self.project()[:])

class StressI2Field(BaseFieldMeasure):
    def _init_expression(self):
        model = self.model

        kv_stress = model.solid.forms['expr.kv_stress']
        el_stress = model.solid.forms['expr.stress_elastic']
        S = el_stress + kv_stress

        # This is the second invariant (I2)
        return 1/2*(ufl.tr(S)**2-ufl.tr(S*S))

    def assem(self):
        return np.array(self.project()[:])

class StressI3Field(BaseFieldMeasure):
    def _init_expression(self):
        model = self.model

        kv_stress = model.solid.forms['expr.kv_stress']
        el_stress = model.solid.forms['expr.stress_elastic']
        S = el_stress + kv_stress

        # This is the third invariant (I3)
        return ufl.det(S)

    def assem(self):
        return np.array(self.project()[:])

class StressHydrostaticField(BaseFieldMeasure):
    def _init_expression(self):

        kv_stress = self.model.solid.forms['expr.kv_stress']
        el_stress = self.model.solid.forms['expr.stress_elastic']
        S = el_stress + kv_stress

        return -1/3*ufl.tr(S)

    def assem(self):
        return np.array(self.project()[:])

class StressVonMisesField(BaseFieldMeasure):
    def _init_expression(self):
        kv_stress = self.model.solid.forms['expr.kv_stress']
        el_stress = self.model.solid.forms['expr.stress_elastic']
        S = el_stress + kv_stress

        S_dev = S - 1/3*ufl.tr(S)*ufl.Identity(3)
        j2 = 0.5*ufl.tr(S_dev*S_dev)
        return (3*j2)**(1/2)

    def assem(self):
        return np.array(self.project()[:])

class ElasticStressField(BaseFieldMeasure):
    def _init_expression(self):
        forms = self.model.solid.forms
        return forms['expr.stress_elastic']

    def assem(self):
        return np.array(self.project()[:])

class ContactPressureField(BaseFieldMeasure):

    def _init_expression(self):
        tcontact = self.model.solid.forms['coeff.state.manual.tcontact']
        # `tcontact*tcontact` should be the square of contact pressure
        return ufl.inner(tcontact, tcontact)**0.5

    def assem(self):
        return np.array(self.project()[:])

class ViscousDissipationField(BaseFieldMeasure):
    def _init_expression(self):
        kv_stress = self.model.solid.forms['expr.kv_stress']
        kv_strain_rate = self.model.solid.forms['expr.kv_strain_rate']
        return ufl.inner(kv_stress, kv_strain_rate)

    def assem(self):
        return np.array(self.project()[:])

class ContactAreaDensityField(BaseFieldMeasure):
    def _init_expression(self):
        tcontact = self.model.solid.forms['coeff.state.manual.tcontact']
        pcontact = ufl.inner(tcontact, tcontact)**0.5 # should be square of contact pressure
        contact_indicator = ufl.conditional(ufl.operators.ne(pcontact, 0.0), 1.0, 0.0)

        return contact_indicator

    def assem(self):
        return np.array(self.project()[:])

class FluidTractionPowerDensity(BaseFieldMeasure):
    """
    Power area density due to fluid tractions
    """
    def _init_expression(self):
        forms = self.model.solid.forms
        fluid_traction = forms['expr.fluid_traction']
        velocity = forms['coeff.state.v1']
        return fluid_traction * velocity

    def assem(self):
        return np.array(self.project()[:])


### Field integral type

class BaseFieldIntegralMeasure(BaseStateMeasure):
    def __init__(
            self,
            model: BaseTransientModel,
            dx: Optional[dfn.Measure]=None
        ):
        """
        Parameters
        ----------
        dx : dfn.Measure
            A measure to project the field expression over
        """
        super().__init__(model)

        if dx is None:
            self.dx = self.model.solid.forms['measure.dx']
        else:
            self.dx = dx

        self.expression = self._init_expression(self)

    def _init_expression(
            self,
            dx: Optional[dfn.Measure]=None
        ):
        """
        Parameters
        ----------
        dx : dfn.Measure
            A measure to project the field expression over
        fspace : dfn.FunctionSpace
            A function space to project the field expression onto
        """
        if dx is None:
            self.dx = self.model.solid.forms['measure.dx']
        else:
            self.dx = dx

class ViscousDissipationRate(BaseFieldIntegralMeasure):

    def _init_expression(self):
        kv_stress = self.model.solid.forms['expr.kv_stress']
        kv_strain_rate = self.model.solid.forms['expr.kv_strain_rate']
        return ufl.inner(kv_stress, kv_strain_rate)*self.dx

    def assem(self):
        return dfn.assemble(self.expression)


### Custom post-processing functions that don't fit nicely into any type

class MinGlottalWidth(BaseStateMeasure):

    def __init__(self, model):
        super().__init__(model)
        self.XREF = self.model.solid.forms['fspace.scalar'].tabulate_dof_coordinates()

    def assem(self):
        xcur = self.XREF.reshape(-1) + self.model.state1.sub['u'][:]
        widths = 2*(self.model.props['ymid'] - xcur[1::2])
        gw = np.min(widths)
        return gw

class VertexGlottalWidth(BaseStateMeasure):
    def __init__(self, model, vertex_name=None):
        super().__init__(model)

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

    def assem(self):
        xcur = self.XREF.reshape(-1) + self.model.state1['u'][:]
        return 2*(self.model.props['ymid'][0] - xcur[self.idx_dof])

### Field statistics post-processing functions

class FieldStats(BaseDerivedStateMeasure):
    def __init__(self, field: BaseFieldMeasure):
        super().__init__(field)

        self.expr_total, self.expr_vol, self.dtype = self._init_expression()

    def _init_expression(self):
        dx = self.func.dx
        expr = self.func.expression

        expr_total = expr*dx
        expr_vol = 1*dx
        dtype = np.dtype([
            ('max', float),
            ('min', float),
            ('avg', float),
            ('total', float)
        ])
        return expr_total, expr_vol, dtype

    def assem(self):
        field_vec = self.func.assem()
        total = dfn.assemble(self.expr_total)
        vol = dfn.assemble(self.expr_vol)
        return np.array(
            (field_vec.max(), field_vec.min(), total/vol, total),
            dtype=self.dtype
        )


def make_project(expr, fspace, dx, vec=None):
    """
    Project an expression onto the function space w.r.t the measure

    Parameters
    ----------
    expr : ufl.Expression
        An ufl expression
    fspace :
        The function space that `expr` will be projected on
    dx : ufl.Measure
        The measure that the projection will be applied over
    vec : dfn.GenericVector
        A vector to project the function into
    """
    trial = dfn.TrialFunction(fspace)
    test = dfn.TestFunction(fspace)
    A = dfn.assemble(trial*test*dx, keep_diagonal=True, tensor=dfn.PETScMatrix())
    A.ident_zeros()
    lhs_expr = expr*test*dx

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
