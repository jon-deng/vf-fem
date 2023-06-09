"""
Contains definitions of different solid model forms
"""

import operator
import warnings
from functools import reduce
from typing import Tuple, Mapping, Callable, Union, Any

import numpy as np
import dolfin as dfn
import ufl

from . import newmark, base
from .uflcontinuum import *

DfnFunction = Union[ufl.Constant, dfn.Function]
CoefficientMapping = Mapping[str, DfnFunction]
FunctionSpaceMapping = Mapping[str, dfn.FunctionSpace]

## Function space specification

class BaseFunctionSpaceSpec:

    def __init__(self, *spec, default_value=0):
        self._spec = spec
        self._default_value = default_value

    @property
    def spec(self) -> Tuple[Any, ...]:
        return self._spec

    @property
    def default_value(self) -> Any:
        return self._default_value

    def generate_function(self, mesh: dfn.Mesh) -> Union[dfn.Constant, dfn.Function]:
        raise NotImplementedError()

class FunctionSpaceSpec(BaseFunctionSpaceSpec):

    def __init__(self, elem_family, elem_degree, value_dim, default_value=0):
        assert value_dim in {'vector', 'scalar'}
        super().__init__(elem_family, elem_degree, value_dim, default_value=default_value)

    def generate_function(self, mesh) -> dfn.Function:
        elem_family, elem_degree, value_dim = self.spec
        if value_dim == 'vector':
            return dfn.Function(dfn.VectorFunctionSpace(mesh, elem_family, elem_degree))
        elif value_dim == 'scalar':
            return dfn.Function(dfn.FunctionSpace(mesh, elem_family, elem_degree))

class ConstantFunctionSpaceSpec(BaseFunctionSpaceSpec):

    def __init__(self, value_dim, default_value=0):
        super().__init__(value_dim)
        self._default_value = default_value

    def generate_function(self, mesh: dfn.Mesh) -> dfn.Constant:
        value_dim, = self.spec
        if isinstance(value_dim, str):
            if value_dim == 'vector':
                return dfn.Constant(
                    mesh.geometric_dimension()*[self.default_value],
                    mesh.ufl_cell()
                )
            elif value_dim == 'scalar':
                return dfn.Constant(
                    self.default_value, mesh.ufl_cell()
                )
            else:
                raise ValueError()
        elif isinstance(value_dim, int):
            const = dfn.Constant(
                value_dim*[0], mesh.ufl_cell()
            )
            const.values()[:] = self.default_value
            return const
        else:
            raise TypeError()

def func_spec(elem_family, elem_degree, value_dim, default_value=0):
    return FunctionSpaceSpec(elem_family, elem_degree, value_dim, default_value=default_value)

def const_spec(value_dim, default_value=0):
    return ConstantFunctionSpaceSpec(value_dim, default_value=default_value)

## Form class

def set_fenics_function(function: Union[dfn.Function, dfn.Constant], value):
    """
    Set a constant value for a general fenics function
    """
    if isinstance(function, dfn.Constant):
        function.values()[:] = value
    else:
        function.vector()[:] = value

class FenicsForm:
    """
    Representation of a `dfn.Form` instance with associated coefficients

    Parameters
    ----------
    functional: dfn.Form
    coefficients: Mapping[str, dfn.Function]
    """

    def __init__(self, form: dfn.Form, coefficients: CoefficientMapping):
        self._form = form
        self._coefficients = coefficients

    @property
    def form(self):
        return self._form

    @property
    def coefficients(self) -> CoefficientMapping:
        return self._coefficients

    def arguments(self) -> list[ufl.Argument]:
        return self.form.arguments()

    ## Dict interface
    def keys(self) -> list[str]:
        return self.coefficients.keys()

    def values(self) -> list[DfnFunction]:
        return self.coefficients.values()

    def items(self) -> list[Tuple[str, DfnFunction]]:
        return self.coefficients.items()

    def __getitem__(self, key: str) -> DfnFunction:
        return self.coefficients[key]

    def __contains__(self, key: str) -> bool:
        return key in self.coefficients

    ## Basic math
    def __add__(self, other: 'FenicsForm') -> 'FenicsForm':
        return add_form(self,  other)

    def __radd__(self, other: 'FenicsForm') -> 'FenicsForm':
        return add_form(self,  other)

    def __sub__(self, other: 'FenicsForm') -> 'FenicsForm':
        return add_form(self,  -1.0*other)

    def __rsub__(self, other: 'FenicsForm') -> 'FenicsForm':
        return add_form(other, -1.0*self)

    def __mul__(self, other: float) -> 'FenicsForm':
        return mul_form(self, other)

    def __rmul__(self, other: float) -> 'FenicsForm':
        return mul_form(self, other)

def add_form(form_a: FenicsForm, form_b: FenicsForm) -> FenicsForm:
    """
    Return a new `FenicsForm` from a sum of other forms
    """
    # Check that form arguments are consistent and replace duplicated
    # consistent arguments
    new_form_a = form_a.form
    args_a, args_b = form_a.form.arguments(), form_b.form.arguments()
    for arg_a, arg_b in zip(args_a, args_b):
        if compare_function_space(arg_a.function_space(), arg_b.function_space()):
            new_form_a = ufl.replace(new_form_a, {arg_a: arg_b})
        else:
            raise ValueError(
                "Summed forms contain duplicate arguments with different function spaces."
            )
    residual = new_form_a + form_b.form

    # Replace duplicate coefficient keys by the `form_b` coefficient
    # only if the duplicate coefficients from `form_b` and `form_a` have the
    # same function space
    duplicate_coeff_keys = set.intersection(set(form_a.keys()), set(form_b.keys()))
    for key in list(duplicate_coeff_keys):
        coeff_a, coeff_b = form_a[key], form_b[key]

        if isinstance(coeff_a, dfn.Function) and isinstance(coeff_b, dfn.Function):
            if compare_function_space(coeff_a.function_space(), coeff_b.function_space()):
                residual = ufl.replace(residual, {coeff_a: coeff_b})
            else:
                raise ValueError(
                    "Summed forms contain duplicate coefficients with different function spaces."
                )
        elif isinstance(coeff_a, dfn.Constant) and isinstance(coeff_b, dfn.Constant):
            if coeff_a.values().shape == coeff_b.values().shape:
                residual = ufl.replace(residual, {coeff_a: coeff_b})
            else:
                raise ValueError(
                    "Summed forms contain duplicate coefficients with different constant value shapes."
                )
        else:
            raise TypeError(
                "Summed forms have duplicate coefficients mixing `dfn.Function` and `dfn.Constant`."
            )
    return FenicsForm(residual, {**form_a.coefficients, **form_b.coefficients})

def mul_form(form: FenicsForm, scalar: float) -> FenicsForm:
    """
    Return a new `FenicsForm` from a sum of other forms
    """
    # Check that form arguments are consistent and replace duplicated
    # consistent arguments
    new_form = scalar*form.form

    return FenicsForm(new_form, form.coefficients)

def compare_function_space(space_a: dfn.FunctionSpace, space_b: dfn.FunctionSpace) -> bool:
    """
    Return whether two function spaces are equivalent
    """
    if (
            space_a.element().signature() == space_b.element().signature()
            and space_a.mesh() == space_b.mesh()
        ):
        return True
    else:
        return False

## Pre-defined linear functionals

class PredefinedForm(FenicsForm):
    """
    A pre-defined linear functional has a pre-defined residual in the class
    """

    COEFFICIENT_SPEC: Mapping[str, BaseFunctionSpaceSpec] = {}
    _make_residual: Callable[[CoefficientMapping, dfn.Measure, dfn.Mesh], dfn.Form]

    def __init__(self,
            coefficients: CoefficientMapping,
            measure: dfn.Measure,
            mesh: dfn.Mesh
        ):

        # If a coefficient key is not supplied, generate a default coefficient
        # from `COEFFICIENT_SPEC`
        for key, spec in self.COEFFICIENT_SPEC.items():
            if key not in coefficients:
                coefficients[key] = spec.generate_function(mesh)

        residual = self._make_residual(coefficients, measure, mesh)
        super().__init__(residual, coefficients)

class InertialForm(PredefinedForm):
    """
    Linear functional representing an inertial force
    """

    COEFFICIENT_SPEC = {
        'coeff.state.a1': func_spec('CG', 1, 'vector'),
        'coeff.prop.rho': func_spec('DG', 0, 'scalar')
    }

    def _make_residual(self, coefficients, measure, mesh):
        vector_test = dfn.TestFunction(coefficients['coeff.state.a1'].function_space())

        acc = coefficients['coeff.state.a1']
        density = coefficients['coeff.prop.rho']
        inertial_body_force = density*acc

        return ufl.inner(inertial_body_force, vector_test) * measure
        # forms['expr.force_inertial'] = inertial_body_force

# Elastic effect forms

class IsotropicElasticForm(PredefinedForm):
    """
    Linear functional representing an isotropic elastic stress
    """

    COEFFICIENT_SPEC = {
        'coeff.state.u1': func_spec('CG', 1, 'vector'),
        'coeff.prop.emod': func_spec('DG', 0, 'scalar'),
        'coeff.prop.nu': const_spec('scalar', default_value=0.45)
    }

    def _make_residual(self, coefficients, measure, mesh):

        vector_test = dfn.TestFunction(coefficients['coeff.state.u1'].function_space())
        strain_test = strain_inf(vector_test)

        u = coefficients['coeff.state.u1']
        inf_strain = strain_inf(u)
        emod = coefficients['coeff.prop.emod']
        nu = coefficients['coeff.prop.nu']
        set_fenics_function(nu, 0.45)
        stress_elastic = stress_isotropic(inf_strain, emod, nu)

        # coefficients['expr.stress_elastic'] = stress_elastic
        return ufl.inner(stress_elastic, strain_test) * measure

class IsotropicIncompressibleElasticSwellingForm(PredefinedForm):
    """
    Linear functional representing an incompressible, isotropic elastic stress with swelling
    """

    COEFFICIENT_SPEC = {
        'coeff.state.u1': func_spec('CG', 1, 'vector'),
        'coeff.prop.emod': func_spec('DG', 0, 'scalar'),
        'coeff.prop.v_swelling': func_spec('DG', 0, 'scalar'),
        'coeff.prop.k_swelling': func_spec('DG', 0, 'scalar')
    }

    def _make_residual(self, coefficients, measure, mesh):

        vector_test = dfn.TestFunction(coefficients['coeff.state.u1'].function_space())
        strain_test = strain_inf(vector_test)

        emod = coefficients['coeff.prop.emod']
        nu = 0.5
        dis = coefficients['coeff.state.u1']
        inf_strain = strain_inf(dis)
        v_swelling = coefficients['coeff.prop.v_swelling']
        set_fenics_function(v_swelling, 1.0)
        k_swelling = coefficients['coeff.prop.k_swelling']
        set_fenics_function(k_swelling, 1.0)
        lame_mu = emod/2/(1+nu)
        stress_elastic = 2*lame_mu*inf_strain + k_swelling*(ufl.tr(inf_strain)-(v_swelling-1.0))*ufl.Identity(inf_strain.ufl_shape[0])

        return ufl.inner(stress_elastic, strain_test) * measure
        # forms['expr.stress_elastic'] = stress_elastic
        # return forms

class IsotropicElasticSwellingForm(PredefinedForm):
    """
    Linear functional representing an isotropic elastic stress with swelling
    """

    COEFFICIENT_SPEC = {
        'coeff.state.u1': func_spec('CG', 1, 'vector'),
        'coeff.prop.emod': func_spec('DG', 0, 'scalar'),
        'coeff.prop.nu': const_spec('scalar', default_value=0.45),
        'coeff.prop.v_swelling': func_spec('DG', 0, 'scalar'),
        'coeff.prop.m_swelling': func_spec('DG', 0, 'scalar')
    }

    def _make_residual(self, coefficients, measure, mesh):
        """
        Add an effect for isotropic elasticity with a swelling field
        """
        dx = measure

        u = coefficients['coeff.state.u1']

        vector_test = dfn.TestFunction(coefficients['coeff.state.u1'].function_space())
        DE = strain_lin_green_lagrange(u, vector_test)
        E = strain_green_lagrange(u)

        emod = coefficients['coeff.prop.emod']
        nu = dfn.Constant(0.45)
        v = coefficients['coeff.prop.v_swelling']
        v.vector()[:] = 1.0
        m = coefficients['coeff.prop.m_swelling']
        m.vector()[:] = 0.0

        E_v = v**(-2/3)*E + 1/2*(v**(-2/3)-1)*ufl.Identity(3)
        # Here write the factor $m(v)*v^(-2/3)$ as $m(v)*v^(-1) * v^(1/3)$
        # Then approximate the function $\hat{m} = m(v)*v^(-1)$ with a linear
        # approximation with slope `m`
        mhat = (m*(v-1) + 1)
        S = mhat*v**(1/3)*stress_isotropic(E_v, emod, nu)

        return ufl.inner(S, DE) * dx
        # # Make this the cauchy stress
        # F = form_def_grad(u)
        # J = ufl.det(F)
        # forms['expr.stress_elastic'] = (1/J)*F*S*F.T

        # # lame_lambda = emod*nu/(1+nu)/(1-2*nu)
        # # lame_mu = emod/2/(1+nu)
        # return forms

# Surface forcing forms

class SurfacePressureForm(PredefinedForm):
    """
    Linear functional representing a pressure follower load
    """

    COEFFICIENT_SPEC = {
        'coeff.state.u1': func_spec('CG', 1, 'vector'),
        'coeff.fsi.p1': func_spec('CG', 1, 'scalar')
    }

    def _make_residual(self, coefficients, measure, mesh):

        ds = measure

        dis = coefficients['coeff.state.u1']
        vector_test = dfn.TestFunction(coefficients['coeff.state.u1'].function_space())
        facet_normal = ufl.FacetNormal(mesh)

        p = coefficients['coeff.fsi.p1']
        reference_traction = -p * pullback_area_normal(dis, facet_normal)

        # coefficients['expr.fluid_traction'] = reference_traction
        return ufl.inner(reference_traction, vector_test) * ds

class ManualSurfaceContactTractionForm(PredefinedForm):
    """
    Linear functional representing a surface contact traction
    """

    COEFFICIENT_SPEC = {
        'coeff.state.u1': func_spec('CG', 1, 'vector'),
        'coeff.state.manual.tcontact': func_spec('CG', 1, 'vector'),
        'coeff.prop.ycontact': const_spec('scalar', np.inf),
        'coeff.prop.ncontact': const_spec(2, default_value=(0, 1)),
        'coeff.prop.kcontact': const_spec('scalar', 1.0)
    }

    def _make_residual(self, coefficients, measure, mesh):

        # The contact traction must be manually linked with displacements and penalty parameters!
        # These parameters are:
        # `ycontact = coefficients['coeff.prop.ycontact']`
        # `ncontact = coefficients['coeff.prop.ncontact']`
        # `kcontact = coefficients['coeff.prop.kcontact']`

        vector_test = dfn.TestFunction(coefficients['coeff.state.manual.tcontact'].function_space())
        tcontact = coefficients['coeff.state.manual.tcontact']
        return ufl.inner(tcontact, vector_test) * measure

# Surface membrane forms

class IsotropicMembraneForm(PredefinedForm):
    """
    Linear functional representing an isotropic elastic membrane
    """

    COEFFICIENT_SPEC = {
        'coeff.state.u1': func_spec('CG', 1, 'vector'),
        'coeff.prop.emod_membrane': func_spec('DG', 0, 'scalar'),
        'coeff.prop.nu_membrane': func_spec('DG', 0, 'scalar'),
        'coeff.prop.th_membrane': func_spec('DG', 0, 'scalar')
    }

    def _make_residual(self, coefficients, measure, mesh, large_def=False):
        vector_test = dfn.TestFunction(coefficients['coeff.state.u1'].function_space())

        # Define the 8th order projector to get the planar strain component
        facet_normal = ufl.FacetNormal(mesh)
        n = ufl.as_tensor([facet_normal[0], facet_normal[1], 0.0])
        nn = ufl.outer(n, n)
        ident = ufl.Identity(n.ufl_shape[0])
        project_pp = ufl.outer(ident-nn, ident-nn)

        i, j, k, l = ufl.indices(4)

        dis = coefficients['coeff.state.u1']
        if large_def:
            strain = strain_green_lagrange(dis)
            strain_test = strain_lin_green_lagrange(dis, vector_test)
        else:
            strain = strain_inf(dis)
            strain_test = strain_inf(vector_test)
        strain_pp_test = ufl.as_tensor(project_pp[i, j, k, l] * strain_test[j, k], (i, l))

        emod = coefficients['coeff.prop.emod_membrane']
        th_membrane = coefficients['coeff.prop.th_membrane']
        nu = coefficients['coeff.prop.nu_membrane']
        set_fenics_function(nu, 0.45)
        mu = emod/2/(1+nu)
        lmbda = emod*nu/(1+nu)/(1-2*nu)

        strain_pp = ufl.as_tensor(project_pp[i, j, k, l] * strain[j, k], (i, l))

        # account for ambiguous 0/0 when emod=0
        lmbda_pp = ufl.conditional(ufl.eq(emod, 0), 0, 2*mu*lmbda/(lmbda+2*mu))
        stress_pp = 2*mu*strain_pp + lmbda_pp*ufl.tr(strain_pp)*(ident-nn)

        return ufl.inner(stress_pp, strain_pp_test) * th_membrane*measure

        # forms['form.un.f1uva'] += res
        # forms['coeff.prop.nu_membrane'] = nu
        # return forms

class IsotropicIncompressibleMembraneForm(PredefinedForm):
    """
    Linear functional representing an incompressible isotropic elastic membrane
    """

    COEFFICIENT_SPEC = {
        'coeff.state.u1': func_spec('CG', 1, 'vector'),
        'coeff.prop.emod_membrane': func_spec('DG', 0, 'scalar'),
        'coeff.prop.th_membrane': func_spec('DG', 0, 'scalar')
    }

    def _make_residual(self, coefficients, measure, mesh, large_def=False):
        vector_test = dfn.TestFunction(coefficients['coeff.state.u1'].function_space())

        # Define the 8th order projector to get the planar strain component
        mesh = coefficients['coeff.state.u1'].function_space().mesh()
        facet_normal = ufl.FacetNormal(mesh)
        n = ufl.as_tensor([facet_normal[0], facet_normal[1], 0.0])
        nn = ufl.outer(n, n)
        ident = ufl.Identity(n.ufl_shape[0])
        project_pp = ufl.outer(ident-nn, ident-nn)
        i, j, k, l = ufl.indices(4)

        strain_test = strain_inf(vector_test)
        strain_pp_test = ufl.as_tensor(project_pp[i, j, k, l] * strain_test[j, k], (i, l))

        dis = coefficients['coeff.state.u1']
        if large_def:
            strain = strain_green_lagrange(dis)
            strain_test = strain_lin_green_lagrange(dis, vector_test)
        else:
            strain = strain_inf(dis)
            strain_test = strain_inf(vector_test)
        strain_pp_test = ufl.as_tensor(project_pp[i, j, k, l] * strain_test[j, k], (i, l))

        emod_membrane = coefficients['coeff.prop.emod_membrane']
        th_membrane = coefficients['coeff.prop.th_membrane']
        nu = 0.5
        lame_mu = emod_membrane/2/(1+nu)
        strain_pp = ufl.as_tensor(project_pp[i, j, k, l] * strain[j, k], (i, l))

        stress_pp = 2*lame_mu*strain_pp + 2*lame_mu*ufl.tr(strain_pp)*(ident-nn)

        return ufl.inner(stress_pp, strain_pp_test) * th_membrane * measure

# Viscous effect forms

class RayleighDampingForm(PredefinedForm):
    """
    Linear functional representing a Rayleigh damping viscous stress
    """

    COEFFICIENT_SPEC = {
        'coeff.state.v1': func_spec('CG', 1, 'vector'),
        'coeff.prop.rho': func_spec('DG', 0, 'scalar'),
        'coeff.prop.emod': func_spec('DG', 0, 'scalar'),
        'coeff.prop.nu': const_spec('scalar', 0.45),
        'coeff.prop.rayleigh_m': const_spec('scalar', 1.0),
        'coeff.prop.rayleigh_k': const_spec('scalar', 1.0)
    }

    def _make_residual(self, coefficients, measure, mesh, large_def=False):

        vector_test = dfn.TestFunction(coefficients['coeff.state.v1'].function_space())

        dx = measure
        strain_test = strain_inf(vector_test)
        v = coefficients['coeff.state.v1']

        rayleigh_m = coefficients['coeff.prop.rayleigh_m']
        rayleigh_k = coefficients['coeff.prop.rayleigh_k']

        emod = coefficients['coeff.prop.emod']
        nu = coefficients['coeff.prop.nu']
        inf_strain = strain_inf(v)
        stress_elastic = stress_isotropic(inf_strain, emod, nu)
        stress_visco = rayleigh_k*stress_elastic

        rho = coefficients['coeff.prop.rho']
        force_visco = rayleigh_m*rho*v

        return (ufl.inner(force_visco, vector_test) + ufl.inner(stress_visco, strain_test))*dx

        # coefficients['form.un.f1uva'] += damping
        # # coefficients['coeff.prop.nu'] = nu
        # # coefficients['coeff.prop.rayleigh_m'] = rayleigh_m
        # # coefficients['coeff.prop.rayleigh_k'] = rayleigh_k
        # return coefficients

class KelvinVoigtForm(PredefinedForm):
    """
    Linear functional representing a Kelvin-Voigt viscous stress
    """

    COEFFICIENT_SPEC = {
        'coeff.state.v1': func_spec('CG', 1, 'vector'),
        'coeff.prop.eta': func_spec('DG', 0, 'scalar')
    }

    def _make_residual(self, coefficients, measure, mesh):

        vector_test = dfn.TestFunction(coefficients['coeff.state.v1'].function_space())

        strain_test = strain_inf(vector_test)
        v = coefficients['coeff.state.v1']

        eta = coefficients['coeff.prop.eta']
        inf_strain_rate = strain_inf(v)
        stress_visco = eta*inf_strain_rate

        return ufl.inner(stress_visco, strain_test) * measure
        # forms['expr.kv_stress'] = stress_visco
        # forms['expr.kv_strain_rate'] = inf_strain_rate

class APForceForm(PredefinedForm):
    """
    Linear functional representing a anterior-posterior (AP) force
    """

    COEFFICIENT_SPEC = {
        'coeff.state.u1': func_spec('CG', 1, 'vector'),
        'coeff.state.v1': func_spec('CG', 1, 'vector'),
        'coeff.prop.eta': func_spec('DG', 0, 'scalar'),
        'coeff.prop.emod': func_spec('DG', 0, 'scalar'),
        'coeff.prop.nu': const_spec('scalar', default_value=0.45),
        'coeff.prop.u_ant': func_spec('DG', 0, 'scalar'),
        'coeff.prop.u_pos': func_spec('DG', 0, 'scalar'),
        'coeff.prop.length': func_spec('DG', 0, 'scalar'),
        'coeff.prop.muscle_stress': func_spec('DG', 0, 'scalar'),
    }

    def _make_residual(self, coefficients, measure, mesh):
        vector_test = dfn.TestFunction(coefficients['coeff.state.v1'].function_space())

        u1, v1 = coefficients['coeff.state.u1'], coefficients['coeff.state.v1']
        kv_eta = coefficients['coeff.prop.eta']
        emod = coefficients['coeff.prop.emod']
        nu = coefficients['coeff.prop.nu']
        lame_mu = emod/2/(1+nu)

        u_ant = coefficients['coeff.prop.u_ant'] # zero values by default
        u_pos = coefficients['coeff.prop.u_pos']
        length = coefficients['coeff.prop.length']
        muscle_stress = coefficients['coeff.prop.muscle_stress']

        d2u_dz2 = (u_ant - 2*u1 + u_pos) / length**2
        d2v_dz2 = (u_ant - 2*v1 + u_pos) / length**2
        force_elast_ap = (lame_mu+muscle_stress)*d2u_dz2
        force_visco_ap = 0.5*kv_eta*d2v_dz2
        stiffness = ufl.inner(force_elast_ap, vector_test) * measure
        viscous = ufl.inner(force_visco_ap, vector_test) * measure

        return -stiffness - viscous

# Add shape effect forms
class ShapeForm(PredefinedForm):
    """
    Linear functional that just adds a shape parameter

    TODO: This doesn't really work anymore after I updated the form class
    """

    COEFFICIENT_SPEC = {
        'coeff.prop.umesh': func_spec('CG', 1, 'vector')
    }

    def _make_residual(self, coefficients, measure, mesh):
        vector_test = dfn.TestFunction(coefficients['coeff.state.v1'].function_space())
        umesh = coefficients['coeff.prop.umesh']
        umesh_ufl = ufl.SpatialCoordinate(mesh)

        # NOTE: To find the sensitivity w.r.t shape, UFL actually uses the parameters
        # `ufl.SpatialCoordinate(mesh)`
        # This doesn't have an associated function/vector of values so both are included
        # here
        # The code has to manually account for 'coeff.prop' cases that have both a
        # function/vector and ufl coefficient instance
        # forms['coeff.prop.umesh'] = (umesh, ufl.SpatialCoordinate(mesh))
        # forms['mesh.REF_COORDINATES'] = mesh.coordinates().copy()
        return 0*ufl.inner(umesh_ufl, vector_test)*measure

## Residual definitions

class FenicsResidual(base.BaseResidual):
    """
    Representation of a (non-linear) residual in `Fenics`

    This includes additional information defining the residual such as the mesh,
    mesh functions, Dirichlet boundary conditions, etc.
    """

    def __init__(
            self,
            linear_form: FenicsForm,
            mesh: dfn.Mesh,
            mesh_functions: list[dfn.MeshFunction],
            mesh_functions_label_to_value: list[Mapping[str, int]],
            fsi_facet_labels: list[str],
            fixed_facet_labels: list[str]
        ):

        self._mesh = mesh
        self._form = linear_form

        self._mesh_functions = mesh_functions
        self._mesh_functions_label_to_value = mesh_functions_label_to_value

        bc_base = dfn.DirichletBC(
            self.form['coeff.state.u1'].function_space(), dfn.Constant([0.0, 0.0]),
            self.mesh_function('facet'), self.mesh_function_label_to_value('facet')['fixed']
        )
        self._dirichlet_bcs = (bc_base,)

        self._fsi_facet_labels = fsi_facet_labels
        self._fixed_facet_labels = fixed_facet_labels

    @property
    def form(self) -> FenicsForm:
        return self._form

    def mesh(self) -> dfn.Mesh:
        return self._mesh

    @staticmethod
    def _mesh_element_type_to_idx(mesh_element_type: Union[str, int]) -> int:
        if isinstance(mesh_element_type, str):
            if mesh_element_type == 'vertex':
                return 0
            elif mesh_element_type == 'facet':
                return 1
            elif mesh_element_type == 'cell':
                return 2
        elif isinstance(mesh_element_type, int):
            return mesh_element_type
        else:
            raise TypeError(
                f"`mesh_element_type` must be `str` or `int`, not `{type(mesh_element_type)}`"
            )

    def mesh_function(self, mesh_element_type: Union[str, int]) -> dfn.MeshFunction:
        idx = self._mesh_element_type_to_idx(mesh_element_type)
        return self._mesh_functions[idx]

    def mesh_function_label_to_value(self, mesh_element_type: Union[str, int]) -> Mapping[str, int]:
        idx = self._mesh_element_type_to_idx(mesh_element_type)
        return self._mesh_functions_label_to_value[idx]

    def measure(self, integral_type: str):
        if integral_type == 'dx':
            mf = self.mesh_function('cell')
        elif integral_type == 'ds':
            mf = self.mesh_function('facet')
        else:
            raise ValueError("Unknown `integral_type` '{integral_type}'")
        return dfn.Measure(integral_type, self.mesh(), subdomain_data=mf)

    @property
    def dirichlet_bcs(self) -> list[dfn.DirichletBC]:
        return self._dirichlet_bcs

    @property
    def fsi_facet_labels(self):
        return self._fsi_facet_labels

    @property
    def fixed_facet_labels(self):
        return self._fixed_facet_labels

class PredefinedFenicsResidual(FenicsResidual):
    """
    Class representing a pre-defined residual
    """

    def __init__(
            self,
            mesh: dfn.Mesh,
            mesh_functions: list[dfn.MeshFunction],
            mesh_functions_label_to_value: list[Mapping[str, int]],
            fsi_facet_labels: list[str],
            fixed_facet_labels: list[str]
        ):

        functional = self._make_functional(
            mesh, mesh_functions, mesh_functions_label_to_value, fsi_facet_labels, fixed_facet_labels
        )
        super().__init__(
            functional,
            mesh, mesh_functions, mesh_functions_label_to_value,
            fsi_facet_labels, fixed_facet_labels
        )

    def _make_functional(
            self,
            mesh: dfn.Mesh,
            mesh_functions: list[dfn.MeshFunction],
            mesh_functions_label_to_value: list[Mapping[str, int]],
            fsi_facet_labels: list[str],
            fixed_facet_labels: list[str]
        ) -> dfn.Form:
        raise NotImplementedError()

class Rayleigh(PredefinedFenicsResidual):

    def _make_functional(
            self,
            mesh: dfn.Mesh,
            mesh_functions: list[dfn.MeshFunction],
            mesh_functions_label_to_value: list[Mapping[str, int]],
            fsi_facet_labels: list[str],
            fixed_facet_labels: list[str]
        ):
        vertex_func, facet_func, cell_func = mesh_functions
        vertex_label_to_id, facet_label_to_id, cell_label_to_id = mesh_functions_label_to_value
        dx = dfn.Measure('dx', domain=mesh, subdomain_data=cell_func)
        ds = dfn.Measure('ds', domain=mesh, subdomain_data=facet_func)
        _traction_ds = [ds(int(facet_label_to_id[facet_label])) for facet_label in fsi_facet_labels]
        traction_ds = reduce(operator.add, _traction_ds)

        form = (
            InertialForm({}, dx, mesh)
            + IsotropicElasticForm({}, dx, mesh)
            + RayleighDampingForm({}, dx, mesh)
            - SurfacePressureForm({}, traction_ds, mesh)
            - ManualSurfaceContactTractionForm({}, traction_ds, mesh)
        )
        return form

class KelvinVoigt(PredefinedFenicsResidual):

    def _make_functional(
            self,
            mesh: dfn.Mesh,
            mesh_functions: list[dfn.MeshFunction],
            mesh_functions_label_to_value: list[Mapping[str, int]],
            fsi_facet_labels: list[str],
            fixed_facet_labels: list[str]
        ):
        vertex_func, facet_func, cell_func = mesh_functions
        vertex_label_to_id, facet_label_to_id, cell_label_to_id = mesh_functions_label_to_value
        dx = dfn.Measure('dx', domain=mesh, subdomain_data=cell_func)
        ds = dfn.Measure('ds', domain=mesh, subdomain_data=facet_func)
        _traction_ds = [ds(int(facet_label_to_id[facet_label])) for facet_label in fsi_facet_labels]
        traction_ds = reduce(operator.add, _traction_ds)

        form = (
            InertialForm({}, dx, mesh)
            + IsotropicElasticForm({}, dx, mesh)
            + KelvinVoigtForm({}, dx, mesh)
            - SurfacePressureForm({}, traction_ds, mesh)
            - ManualSurfaceContactTractionForm({}, traction_ds, mesh)
        )
        return form

class KelvinVoigtWEpithelium(PredefinedFenicsResidual):

    def _make_functional(
            self,
            mesh: dfn.Mesh,
            mesh_functions: list[dfn.MeshFunction],
            mesh_functions_label_to_value: list[Mapping[str, int]],
            fsi_facet_labels: list[str],
            fixed_facet_labels: list[str]
        ):
        vertex_func, facet_func, cell_func = mesh_functions
        vertex_label_to_id, facet_label_to_id, cell_label_to_id = mesh_functions_label_to_value
        dx = dfn.Measure('dx', domain=mesh, subdomain_data=cell_func)
        ds = dfn.Measure('ds', domain=mesh, subdomain_data=facet_func)
        _traction_ds = [ds(int(facet_label_to_id[facet_label])) for facet_label in fsi_facet_labels]
        traction_ds = reduce(operator.add, _traction_ds)

        form = (
            InertialForm({}, dx, mesh)
            + IsotropicMembraneForm({}, traction_ds, mesh)
            + IsotropicElasticForm({}, dx, mesh)
            + KelvinVoigtForm({}, dx, mesh)
            - SurfacePressureForm({}, traction_ds, mesh)
            - ManualSurfaceContactTractionForm({}, traction_ds, mesh)
        )
        return form

class IncompSwellingKelvinVoigt(PredefinedFenicsResidual):

    def _make_functional(
            self,
            mesh: dfn.Mesh,
            mesh_functions: list[dfn.MeshFunction],
            mesh_functions_label_to_value: list[Mapping[str, int]],
            fsi_facet_labels: list[str],
            fixed_facet_labels: list[str]
        ):
        vertex_func, facet_func, cell_func = mesh_functions
        vertex_label_to_id, facet_label_to_id, cell_label_to_id = mesh_functions_label_to_value
        dx = dfn.Measure('dx', domain=mesh, subdomain_data=cell_func)
        ds = dfn.Measure('ds', domain=mesh, subdomain_data=facet_func)
        _traction_ds = [ds(int(facet_label_to_id[facet_label])) for facet_label in fsi_facet_labels]
        traction_ds = reduce(operator.add, _traction_ds)

        form = (
            InertialForm({}, dx, mesh)
            + IsotropicIncompressibleElasticSwellingForm({}, dx, mesh)
            + KelvinVoigtForm({}, dx, mesh)
            - SurfacePressureForm({}, traction_ds, mesh)
            - ManualSurfaceContactTractionForm({}, traction_ds, mesh)
        )
        return form

class SwellingKelvinVoigt(PredefinedFenicsResidual):

    def _make_functional(
            self,
            mesh: dfn.Mesh,
            mesh_functions: list[dfn.MeshFunction],
            mesh_functions_label_to_value: list[Mapping[str, int]],
            fsi_facet_labels: list[str],
            fixed_facet_labels: list[str]
        ):
        vertex_func, facet_func, cell_func = mesh_functions
        vertex_label_to_id, facet_label_to_id, cell_label_to_id = mesh_functions_label_to_value
        dx = dfn.Measure('dx', domain=mesh, subdomain_data=cell_func)
        ds = dfn.Measure('ds', domain=mesh, subdomain_data=facet_func)
        _traction_ds = [ds(int(facet_label_to_id[facet_label])) for facet_label in fsi_facet_labels]
        traction_ds = reduce(operator.add, _traction_ds)

        form = (
            InertialForm({}, dx, mesh)
            + IsotropicElasticSwellingForm({}, dx, mesh)
            + KelvinVoigtForm({}, dx, mesh)
            - SurfacePressureForm({}, traction_ds, mesh)
            - ManualSurfaceContactTractionForm({}, traction_ds, mesh)
        )
        return form

class SwellingKelvinVoigtWEpithelium(PredefinedFenicsResidual):

    def _make_functional(
            self,
            mesh: dfn.Mesh,
            mesh_functions: list[dfn.MeshFunction],
            mesh_functions_label_to_value: list[Mapping[str, int]],
            fsi_facet_labels: list[str],
            fixed_facet_labels: list[str]
        ):
        vertex_func, facet_func, cell_func = mesh_functions
        vertex_label_to_id, facet_label_to_id, cell_label_to_id = mesh_functions_label_to_value
        dx = dfn.Measure('dx', domain=mesh, subdomain_data=cell_func)
        ds = dfn.Measure('ds', domain=mesh, subdomain_data=facet_func)
        _traction_ds = [ds(int(facet_label_to_id[facet_label])) for facet_label in fsi_facet_labels]
        traction_ds = reduce(operator.add, _traction_ds)

        form = (
            InertialForm({}, dx, mesh)
            + IsotropicMembraneForm({}, traction_ds, mesh)
            + IsotropicElasticSwellingForm({}, dx, mesh)
            + KelvinVoigtForm({}, dx, mesh)
            - SurfacePressureForm({}, traction_ds, mesh)
            - ManualSurfaceContactTractionForm({}, traction_ds, mesh)
        )
        return form

class SwellingKelvinVoigtWEpitheliumNoShape(PredefinedFenicsResidual):

    def _make_functional(
            self,
            mesh: dfn.Mesh,
            mesh_functions: list[dfn.MeshFunction],
            mesh_functions_label_to_value: list[Mapping[str, int]],
            fsi_facet_labels: list[str],
            fixed_facet_labels: list[str]
        ):
        vertex_func, facet_func, cell_func = mesh_functions
        vertex_label_to_id, facet_label_to_id, cell_label_to_id = mesh_functions_label_to_value
        dx = dfn.Measure('dx', domain=mesh, subdomain_data=cell_func)
        ds = dfn.Measure('ds', domain=mesh, subdomain_data=facet_func)
        _traction_ds = [ds(int(facet_label_to_id[facet_label])) for facet_label in fsi_facet_labels]
        traction_ds = reduce(operator.add, _traction_ds)

        form = (
            InertialForm({}, dx, mesh)
            + IsotropicMembraneForm({}, traction_ds, mesh)
            + IsotropicElasticSwellingForm({}, dx, mesh)
            + KelvinVoigtForm({}, dx, mesh)
            - SurfacePressureForm({}, traction_ds, mesh)
            - ManualSurfaceContactTractionForm({}, traction_ds, mesh)
        )
        return form

class Approximate3DKelvinVoigt(PredefinedFenicsResidual):

    def _make_functional(
            self,
            mesh: dfn.Mesh,
            mesh_functions: list[dfn.MeshFunction],
            mesh_functions_label_to_value: list[Mapping[str, int]],
            fsi_facet_labels: list[str],
            fixed_facet_labels: list[str]
        ):
        vertex_func, facet_func, cell_func = mesh_functions
        vertex_label_to_id, facet_label_to_id, cell_label_to_id = mesh_functions_label_to_value
        dx = dfn.Measure('dx', domain=mesh, subdomain_data=cell_func)
        ds = dfn.Measure('ds', domain=mesh, subdomain_data=facet_func)
        _traction_ds = [ds(int(facet_label_to_id[facet_label])) for facet_label in fsi_facet_labels]
        traction_ds = reduce(operator.add, _traction_ds)

        form = (
            InertialForm({}, dx, mesh)
            + IsotropicMembraneForm({}, traction_ds, mesh)
            + IsotropicElasticForm({}, dx, mesh)
            - APForceForm({}, dx, mesh)
            + KelvinVoigtForm({}, dx, mesh)
            - SurfacePressureForm({}, traction_ds, mesh)
            - ManualSurfaceContactTractionForm({}, traction_ds, mesh)
        )
        return form

## Form modifiers

def modify_newmark_time_discretization(form: FenicsForm) -> FenicsForm:
    u1 = form['coeff.state.u1']
    v1 = form['coeff.state.v1']
    a1 = form['coeff.state.a1']

    u0 = dfn.Function(form['coeff.state.u1'].function_space())
    v0 = dfn.Function(form['coeff.state.v1'].function_space())
    a0 = dfn.Function(form['coeff.state.a1'].function_space())

    dt = dfn.Function(form['coeff.prop.rho'].function_space())
    gamma = dfn.Constant(1/2)
    beta = dfn.Constant(1/4)
    v1_nmk = newmark.newmark_v(u1, u0, v0, a0, dt, gamma, beta)
    a1_nmk = newmark.newmark_a(u1, u0, v0, a0, dt, gamma, beta)

    new_coefficients = {
        'coeff.state.u0': u0,
        'coeff.state.v0': v0,
        'coeff.state.a0': a0,
        'coeff.time.dt': dt,
        'coeff.time.gamma': gamma,
        'coeff.time.beta': beta
    }

    coefficients = {**form.coefficients, **new_coefficients}

    new_functional = ufl.replace(form.form, {v1: v1_nmk, a1: a1_nmk})

    return FenicsForm(new_functional, coefficients)

def modify_unary_linearized_forms(form: FenicsForm) -> Mapping[str, dfn.Form]:
    """
    Generate linearized forms representing linearization of the residual wrt different states

    These forms are needed for solving the Hopf bifurcation problem/conditions
    """
    new_coefficients = {}

    # Create coefficients for linearization directions
    for var_name in ['u1', 'v1', 'a1']:
        new_coefficients[f'coeff.dstate.{var_name}'] = dfn.Function(
            form[f'coeff.state.{var_name}'].function_space()
        )
    for var_name in ['p1']:
        new_coefficients[f'coeff.dfsi.{var_name}'] = dfn.Function(
            form[f'coeff.fsi.{var_name}'].function_space()
        )

    # Compute the jacobian bilinear forms
    # unary_form_name = 'f1uva'
    # for var_name in ['u1', 'v1', 'a1']:
    #     form[f'form.bi.d{unary_form_name}_d{var_name}'] = dfn.derivative(form.form, form[f'coeff.state.{var_name}'])
    # for var_name in ['p1']:
    #     form[f'form.bi.d{unary_form_name}_d{var_name}'] = dfn.derivative(form.form, form[f'coeff.fsi.{var_name}'])

    # Take the action of the jacobian linear forms along states to get a new linear
    # dF/dx * delta x, dF/dp * delta p, ...
    linearized_forms = []
    for var_name in ['u1', 'v1', 'a1']:
        # unary_form_name = f'df1uva_{var_name}'
        df_dx = dfn.derivative(form.form, form[f'coeff.state.{var_name}'])
        # print(len(df_dx.arguments()))
        # print(len(forms[f'form.un.f1uva'].arguments()))
        linearized_form = dfn.action(df_dx, new_coefficients[f'coeff.dstate.{var_name}'])
        linearized_forms.append(linearized_form)

    for var_name in ['p1']:
        # unary_form_name = f'df1uva_{var_name}'
        # df_dx = form[f'form.bi.df1uva_d{var_name}']
        df_dx = dfn.derivative(form.form, form[f'coeff.fsi.{var_name}'])
        linearized_form = dfn.action(df_dx, new_coefficients[f'coeff.dfsi.{var_name}'])
        linearized_forms.append(linearized_form)

    # Compute the total linearized residual
    new_form = reduce(operator.add, linearized_forms)

    return FenicsForm(new_form, {**form.coefficients, **new_coefficients})

## Common functions

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

def dis_contact_gap(gap):
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

def pressure_contact_cubic_penalty(gap, kcoll):
    """
    Return the cubic penalty pressure
    """
    positive_gap = dis_contact_gap(gap)
    return kcoll*positive_gap**3

def dform_cubic_penalty_pressure(gap, kcoll):
    """
    Return derivatives of the cubic penalty pressure
    """
    positive_gap = dis_contact_gap(gap)
    dpositive_gap = np.sign(gap)
    return kcoll*3*positive_gap**2 * dpositive_gap, positive_gap**3



## Generation of new forms
# These functions are mainly for generating forms that are needed for solving
# the transient problem with a time discretization
def gen_residual_bilinear_forms(form: FenicsForm) -> Mapping[str, dfn.Form]:
    """
    Generates bilinear forms representing derivatives of the residual wrt state variables

    If the residual is F(u, v, a; parameters, ...), this function generates
    bilinear forms dF/du, dF/dv, etc...
    """
    bi_forms = {}
    # Derivatives of the displacement residual form wrt all state variables
    initial_state_names = [f'coeff.state.{y}' for y in ('u0', 'v0', 'a0')]
    final_state_names = [f'coeff.state.{y}' for y in ('u1', 'v1', 'a1')]
    manual_state_var_names = [name for name in form.keys() if 'coeff.state.manual' in name]

    # This section is for derivatives of the time-discretized residual
    # F(u0, v0, a0, u1; parameters, ...)
    for full_var_name in (
        initial_state_names
        + ['coeff.state.u1']
        + manual_state_var_names
        + ['coeff.time.dt', 'coeff.fsi.p1']):
        f = form.form
        x = form[full_var_name]

        var_name = full_var_name.split(".")[-1]
        form_name = f'form.bi.df1_d{var_name}'
        bi_forms[form_name] = dfn.derivative(f, x)
        bi_forms[f'{form_name}_adj'] = dfn.adjoint(bi_forms[form_name])

    # This section is for derivatives of the original not time-discretized residual
    # F(u1, v1, a1; parameters, ...)
    for full_var_name in (
        final_state_names
        + manual_state_var_names
        + ['coeff.fsi.p1']):
        f = form.form
        x = form[full_var_name]

        var_name = full_var_name.split(".")[-1]
        form_name = f'form.bi.df1uva_d{var_name}'
        bi_forms[form_name] = dfn.derivative(f, x)
        try:
            # TODO: This can fail if the form is not sensitive to a coefficient so the derivative
            # is 0
            bi_forms[f'{form_name}_adj'] = dfn.adjoint(bi_forms[form_name])
        except:
            pass

    return bi_forms

def gen_residual_bilinear_property_forms(form: FenicsForm) -> Mapping[str, dfn.Form]:
    """
    Return a dictionary of forms of derivatives of f1 with respect to the various solid parameters
    """
    df1_dsolid = {}
    property_labels = [
        form_name.split('.')[-1] for form_name in form.keys()
        if 'coeff.prop' in form_name
    ]
    for prop_name in property_labels:
        prop_coeff = _depack_property_ufl_coeff(form[f'coeff.prop.{prop_name}'])
        try:
            df1_dsolid[prop_name] = dfn.adjoint(
                dfn.derivative(form['form.un.f1'], prop_coeff)
            )
        except RuntimeError:
            df1_dsolid[prop_name] = None

    return df1_dsolid

# These functions are mainly for generating derived forms that are needed for solving
# the hopf bifurcation problem
def gen_hopf_forms(form: FenicsForm) -> Mapping[str, dfn.Form]:
    forms = {}
    # forms.update(modify_unary_linearized_forms(form))

    # unary_form_names = ['f1uva', 'df1uva', 'df1uva_u1', 'df1uva_v1', 'df1uva_a1', 'df1uva_p1']
    # for unary_form_name in unary_form_names:
    #     forms.update(gen_jac_state_forms(unary_form_name, form))
    # for unary_form_name in unary_form_names:
    #     forms.update(gen_jac_property_forms(unary_form_name, form))

    forms.update(gen_jac_state_forms(form.form))

    return forms

def gen_jac_state_forms(form: FenicsForm) -> Mapping[str, dfn.Form]:
    """
    Return the derivatives of a unary form wrt all states
    """
    forms = {}
    state_labels = ['u1', 'v1', 'a1']
    for state_name in state_labels:
        df_dx = dfn.derivative(form.form, form[f'coeff.state.{state_name}'])
        forms[f'form.bi.dres_d{state_name}'] = df_dx

    state_labels = ['p1']
    for state_name in state_labels:
        df_dx = dfn.derivative(form.form, form[f'coeff.fsi.{state_name}'])
        forms[f'form.bi.dres_d{state_name}'] = df_dx

    return forms

def gen_jac_property_forms(form: FenicsForm) -> Mapping[str, dfn.Form]:
    """
    Return the derivatives of a unary form wrt all solid properties
    """
    forms = {}
    property_labels = [
        form_name.split('.')[-1] for form_name in form.keys()
        if 'coeff.prop' in form_name
    ]
    for prop_name in property_labels:
        prop_coeff = form[f'coeff.prop.{prop_name}']
        try:
            df_dprop = dfn.derivative(form.form, prop_coeff)
        except RuntimeError:
            df_dprop = None

        forms[f'form.bi.dres_d{prop_name}'] = df_dprop

    return forms


