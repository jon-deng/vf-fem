"""
Post-processing functionality for primarily solid models
"""

from typing import Optional, Callable

import numpy as np
# import matplotlib.pyplot as plt
# import scipy.signal as sig

from petsc4py import PETSc
import dolfin as dfn
import ufl
from blockarray.subops import solve_petsc_lu

from femvf.models.transient.base import BaseTransientModel
from .base import BaseStateMeasure, BaseDerivedStateMeasure


### Field type post-processing functions
def doc_field_measure_params(G: 'BaseFieldMeasure'):
    """
    Add the parameters docstring to a `BaseFieldMeasure` subclass
    """

    param_doc = \
    """
    Parameters
    ----------
    dx : dfn.Measure
        A measure to project the field expression over
    fspace : dfn.FunctionSpace
        A function space to project the field expression onto
    """
    G.__doc__ = G.__doc__ + param_doc
    return G

@doc_field_measure_params
class BaseFieldMeasure(BaseStateMeasure):
    """
    Base class for post-processing field quantities from a state

    Note field quantities are variable over space.

    """
    def __init__(
            self,
            model: BaseTransientModel,
            dx: Optional[dfn.Measure]=None,
            fspace: Optional[dfn.FunctionSpace]=None,
            **kwargs
        ):
        super().__init__(model)
        if fspace is None:
            self.fspace = dfn.FunctionSpace(self.model.solid.forms['mesh.mesh'], 'DG', 0)
        else:
            self.fspace = fspace

        if dx is None:
            self.dx = self.model.solid.forms['measure.dx']
        else:
            self.dx = dx

        self.expression = self._init_expression()
        self.project = make_project(self.expression, self.fspace, self.dx)

    def _init_expression(self):
        raise NotImplementedError("Method must be implemented by subclasses")

    def assem(self, state, control, prop):
        raise NotImplementedError("Method must be implemented by subclasses")

@doc_field_measure_params
class StressI1Field(BaseFieldMeasure):
    """
    Return the first invariant of the stress field

    """
    def _init_expression(self):
        model = self.model
        kv_stress = model.solid.forms['expr.kv_stress']
        el_stress = model.solid.forms['expr.stress_elastic']
        S = el_stress + kv_stress

        # This is the first invariant (I1)
        return ufl.tr(S)

    def assem(self, state, control, prop):
        return np.array(self.project()[:])

@doc_field_measure_params
class StressI2Field(BaseFieldMeasure):
    """
    Return the second invariant of the stress field

    """

    def _init_expression(self):
        model = self.model

        kv_stress = model.solid.forms['expr.kv_stress']
        el_stress = model.solid.forms['expr.stress_elastic']
        S = el_stress + kv_stress

        # This is the second invariant (I2)
        return 1/2*(ufl.tr(S)**2-ufl.tr(S*S))

    def assem(self, state, control, prop):
        return np.array(self.project()[:])

@doc_field_measure_params
class StressI3Field(BaseFieldMeasure):
    """
    Return the third invariant of the stress field

    """

    def _init_expression(self):
        model = self.model

        kv_stress = model.solid.forms['expr.kv_stress']
        el_stress = model.solid.forms['expr.stress_elastic']
        S = el_stress + kv_stress

        # This is the third invariant (I3)
        return ufl.det(S)

    def assem(self, state, control, prop):
        return np.array(self.project()[:])

@doc_field_measure_params
class StressHydrostaticField(BaseFieldMeasure):
    """
    Return the hydrostatic component of the stress field

    """

    def _init_expression(self):

        kv_stress = self.model.solid.forms['expr.kv_stress']
        el_stress = self.model.solid.forms['expr.stress_elastic']
        S = el_stress + kv_stress

        return -1/3*ufl.tr(S)

    def assem(self, state, control, prop):
        return np.array(self.project()[:])

@doc_field_measure_params
class StressVonMisesField(BaseFieldMeasure):
    """
    Return the von Mises stress

    """

    def _init_expression(self):
        kv_stress = self.model.solid.forms['expr.kv_stress']
        el_stress = self.model.solid.forms['expr.stress_elastic']
        S = el_stress + kv_stress

        S_dev = S - 1/3*ufl.tr(S)*ufl.Identity(3)
        j2 = 0.5*ufl.tr(S_dev*S_dev)
        return (3*j2)**(1/2)

    def assem(self, state, control, prop):
        return np.array(self.project()[:])

@doc_field_measure_params
class ElasticStressField(BaseFieldMeasure):
    """
    Return the elastic stress

    """

    def _init_expression(self):
        forms = self.model.solid.forms
        return forms['expr.stress_elastic']

    def assem(self, state, control, prop):
        return np.array(self.project()[:])

@doc_field_measure_params
class ContactPressureField(BaseFieldMeasure):
    """
    Return the penalty contact pressure

    """

    def __init__(self, model, dx=None, fspace=None, **kwargs):
        # The default `dx` measure should be a surface measure rather than
        # volume measure as used in `BaseFieldMeasure`
        if dx is None:
            dx = model.forms['measure.ds_traction']

        super().__init__(model, dx, fspace, **kwargs)

    def _init_expression(self):
        tcontact = self.model.solid.forms['coeff.state.manual.tcontact']
        # `tcontact*tcontact` should be the square of contact pressure
        return ufl.inner(tcontact, tcontact)**0.5

    def assem(self, state, control, prop):
        return np.array(self.project()[:])

@doc_field_measure_params
class ViscousDissipationField(BaseFieldMeasure):
    """
    Return the viscous dissipation density

    """

    def _init_expression(self):
        kv_stress = self.model.solid.forms['expr.kv_stress']
        kv_strain_rate = self.model.solid.forms['expr.kv_strain_rate']
        return ufl.inner(kv_stress, kv_strain_rate)

    def assem(self, state, control, prop):
        return np.array(self.project()[:])

@doc_field_measure_params
class ContactAreaDensityField(BaseFieldMeasure):
    """
    Return the contact area density

    """

    def _init_expression(self):
        tcontact = self.model.solid.forms['coeff.state.manual.tcontact']
        pcontact = ufl.inner(tcontact, tcontact)**0.5 # should be square of contact pressure
        contact_indicator = ufl.conditional(ufl.operators.ne(pcontact, 0.0), 1.0, 0.0)

        return contact_indicator

    def assem(self, state, control, prop):
        return np.array(self.project()[:])

@doc_field_measure_params
class FluidTractionPowerDensity(BaseFieldMeasure):
    """
    Return the power density due to fluid traction

    """
    def __init__(self, model, dx=None, fspace=None, **kwargs):
        # The default `dx` measure should be a surface measure rather than
        # volume measure as used in `BaseFieldMeasure`
        if dx is None:
            dx = model.solid.forms['measure.ds_traction']

        super().__init__(model, dx, fspace, **kwargs)

    def _init_expression(self):
        forms = self.model.solid.forms
        fluid_traction = forms['expr.fluid_traction']
        velocity = forms['coeff.state.v1']
        return ufl.inner(fluid_traction, velocity)

    def assem(self, state, control, prop):
        return np.array(self.project()[:])

@doc_field_measure_params
class XMomentum(BaseFieldMeasure):
    """
    Return x-momentum
    """
    def __init__(self, model, dx=None, fspace=None, **kwargs):
        # The default `dx` measure should be a surface measure rather than
        # volume measure as used in `BaseFieldMeasure`
        if dx is None:
            dx = model.solid.forms['measure.ds_traction']

        super().__init__(model, dx, fspace, **kwargs)

    def _init_expression(self):
        forms = self.model.solid.forms
        rho = forms['coeff.prop.rho']
        velocity = forms['coeff.state.v1']
        return ufl.inner(rho, velocity[0])

    def assem(self, state, control, prop):
        return np.array(self.project()[:])

@doc_field_measure_params
class YMomentum(BaseFieldMeasure):
    """
    Return y-momentum
    """
    def __init__(self, model, dx=None, fspace=None, **kwargs):
        # The default `dx` measure should be a surface measure rather than
        # volume measure as used in `BaseFieldMeasure`
        if dx is None:
            dx = model.solid.forms['measure.dx']

        super().__init__(model, dx, fspace, **kwargs)

    def _init_expression(self):
        forms = self.model.solid.forms
        rho = forms['coeff.prop.rho']
        velocity = forms['coeff.state.v1']
        return ufl.inner(rho, velocity[1])

    def assem(self, state, control, prop):
        return np.array(self.project()[:])

### Field integral type
def doc_field_integral_measure_params(G: 'BaseFieldIntegralMeasure'):
    """
    Add the parameters docstring to a `BaseFieldIntegralMeasure` subclass
    """

    param_doc = \
    """
    Parameters
    ----------
    dx : dfn.Measure
        A measure to compute the field integral expression over
    """
    G.__doc__ = G.__doc__ + param_doc
    return G

@doc_field_integral_measure_params
class BaseFieldIntegralMeasure(BaseStateMeasure):
    """
    Base class for post-processing field integral quantities from a state

    Note field quantities are variable over space.

    """

    def __init__(
            self,
            model: BaseTransientModel,
            dx: Optional[dfn.Measure]=None
        ):
        super().__init__(model)

        if dx is None:
            self.dx = self.model.solid.forms['measure.dx']
        else:
            self.dx = dx

        self.expression = self._init_expression()

    def _init_expression(self):
        """
        Parameters
        ----------
        dx : dfn.Measure
            A measure to project the field expression over
        fspace : dfn.FunctionSpace
            A function space to project the field expression onto
        """
        raise NotImplementedError("Child classes must implement this method")

@doc_field_integral_measure_params
class ViscousDissipationRate(BaseFieldIntegralMeasure):
    """
    Return the viscous dissipation rate

    """

    def _init_expression(self):
        kv_stress = self.model.solid.forms['expr.kv_stress']
        kv_strain_rate = self.model.solid.forms['expr.kv_strain_rate']
        return ufl.inner(kv_stress, kv_strain_rate)*self.dx

    def assem(self, state, control, prop):
        return dfn.assemble(self.expression)

### Field statistics post-processing functions

class FieldStats(BaseDerivedStateMeasure):
    """
    Return spatial statistics from a field post-processing function

    Parameters
    ----------
    field: BaseFieldMeasure
        The field post-processing function to return statistics from
    """
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

    def assem(self, state, control, prop):
        field_vec = self.func(state, control, prop)
        total = dfn.assemble(self.expr_total)
        vol = dfn.assemble(self.expr_vol)
        return np.array(
            (field_vec.max(), field_vec.min(), total/vol, total),
            dtype=self.dtype
        )

### Custom post-processing functions that don't fit nicely into any type

class MinGlottalWidth(BaseStateMeasure):
    """
    Return the minimum glottal width

    Parameters
    ----------
    model :
        The model to post process
    """

    def __init__(self, model: BaseTransientModel):
        super().__init__(model)
        self.XREF = np.array(self.model.solid.XREF[:])

    def assem(self, state, control, prop):
        xcur = self.XREF.reshape(-1) + self.model.state1.sub['u'][:]
        widths = 2*(self.model.prop['ymid'] - xcur[1::2])
        gw = np.min(widths)
        return gw

class VertexGlottalWidth(BaseStateMeasure):
    """
    Return the glottal width at a specified vertex

    Parameters
    ----------
    model :
        The model to post process
    vertex_name : str
        The name of the vertex
    """

    def __init__(self, model: BaseTransientModel, vertex_name: Optional[str]=None):
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

    def assem(self, state, control, prop):
        xcur = self.XREF.reshape(-1) + self.model.state1['u'][:]
        return 2*(self.model.prop['ymid'][0] - xcur[self.idx_dof])

def make_project(
        expr: ufl.core.expr.Expr,
        fspace: dfn.FunctionSpace,
        dx: dfn.Measure,
        vec: Optional[dfn.PETScVector]=None
    ) -> Callable[[], dfn.PETScVector]:
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

    ksp = PETSc.KSP().create()
    ksp.setType(ksp.Type.PREONLY)
    ksp.setOperators(A.mat())
    ksp.setUp()

    pc = ksp.getPC()
    pc.setType(pc.Type.LU)

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

        x_petsc = x.vec()
        b_petsc = b.vec()
        solve_petsc_lu(A.mat(), b_petsc, x_petsc, ksp=ksp)

        # dfn.solve(A, x, b, 'lu')
        return x
    return project
