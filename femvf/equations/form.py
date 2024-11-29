"""
Define the `FenicsForm`, common forms and associated utilities

References
----------
[Gou2016] Gou, K., Pence, T.J. Hyperelastic modeling of swelling in fibrous soft tissue with application to tracheal angioedema. J. Math. Biol. 72, 499-526 (2016). https://doi.org/10.1007/s00285-015-0893-0
"""

from typing import Callable, Union, Any, Optional
import ufl.coefficient
from ufl.core.expr import Expr

import operator
import warnings
from functools import reduce

import numpy as np
import dolfin as dfn
import ufl

from . import newmark
from .uflcontinuum import *

DfnFunction = Union[ufl.Constant, dfn.Function]
FunctionLike = Union[ufl.Argument, dfn.Function, dfn.Constant]
FunctionSpace = Union[ufl.FunctionSpace, dfn.FunctionSpace]

## Utilities for handling Fenics functions

# These are used to treat `ufl.Constant` and `dfn.Function` uniformly


def set_fenics_function(function: DfnFunction, value) -> dfn.Function:
    """
    Set a value for a `dfn.Function` or `dfn.Constant` instance

    This is needed because, although both classes represent functions,
    they have different methods access their underlying coefficient vectors.
    """
    if isinstance(function, dfn.Constant):
        function.values()[:] = value
    elif isinstance(function, dfn.Function):
        function.vector()[:] = value
    else:
        raise TypeError(f"Unknown type {type(function)}")

    return function


def function_space(function: FunctionLike) -> dfn.FunctionSpace:
    """
    Return the function space of a given function
    """
    if isinstance(function, dfn.Function):
        space = function.function_space()
    elif isinstance(function, dfn.Constant):
        space = function.ufl_function_space()
    elif isinstance(function, ufl.Argument):
        space = function.function_space()
    else:
        raise TypeError(f"Unknown type {type(function)}")

    return space


# These are used to handle duplicate function spaces/functions


def compare_function_space(space_a: FunctionSpace, space_b: FunctionSpace) -> bool:

    both_are_ufl_function_space = isinstance(space_a, ufl.FunctionSpace) and isinstance(
        space_b, ufl.FunctionSpace
    )
    both_are_dfn_function_space = isinstance(space_a, dfn.FunctionSpace) and isinstance(
        space_b, dfn.FunctionSpace
    )

    if both_are_ufl_function_space:
        return compare_ufl_function_space(space_a, space_b)
    if both_are_dfn_function_space:
        return compare_dfn_function_space(space_a, space_b)
    else:
        assert False
        return False


def compare_dfn_function_space(
    space_a: dfn.FunctionSpace, space_b: dfn.FunctionSpace
) -> bool:
    """
    Return if two function spaces are equivalent
    """
    if (
        space_a.element().signature() == space_b.element().signature()
        and space_a.mesh() == space_b.mesh()
    ):
        return True
    else:
        return False


def compare_ufl_function_space(
    space_a: ufl.FunctionSpace, space_b: ufl.FunctionSpace
) -> bool:
    """
    Return if two function spaces are equivalent
    """
    if (
        space_a.ufl_element() == space_b.ufl_element()
        and space_a.ufl_domains() == space_b.ufl_domains()
    ):
        return True
    else:
        return False


def get_shared_function(
    function_a: FunctionLike, function_b: FunctionLike
) -> FunctionLike:
    """
    Return a shared function space for two `fenics` objects
    """
    if type(function_a) != type(function_b):
        raise TypeError("Functions must have the same type")

    if compare_function_space(function_space(function_a), function_space(function_b)):
        # TODO: Create a new function space for the shared function?
        shared_function = function_a
        return shared_function
    else:
        raise ValueError("Functions have different function spaces.")


## Function space specification


class BaseFunctionSpaceSpec:
    """
    Represents a `fenics` function space

    Parameters
    ----------
    spec:
        A tuple specifying the function space
    default_value: int
        The default value for the function
    """

    generate_function: Callable[[dfn.Mesh], Union[dfn.Constant, dfn.Function]]

    def __init__(self, *spec, default_value: int = 0):
        self._spec = spec
        self._default_value = default_value

    @property
    def spec(self) -> tuple[Any, ...]:
        return self._spec

    @property
    def default_value(self) -> Any:
        return self._default_value

    # def generate_function(self, mesh: dfn.Mesh) -> Union[dfn.Constant, dfn.Function]:
    #     raise NotImplementedError()


class FunctionSpaceSpec(BaseFunctionSpaceSpec):
    """
    Represents a `dolfin` function space

    Parameters
    ----------
    elem_family:
        The 'family' of the function space (see `dfn.cpp.function.FunctionSpace`)
    elem_degree:
        The 'degree' of the function space (see `dfn.cpp.function.FunctionSpace`)
    value_dim:
        The dimension of function value (see `dfn.cpp.function.FunctionSpace`)
    default_value: int
        The default value for the function
    """

    def __init__(
        self,
        elem_family: str,
        elem_degree: int,
        value_dim: Union[tuple[int, ...], str],
        default_value: int = 0,
    ):
        assert value_dim in {'vector', 'scalar'}
        super().__init__(
            elem_family, elem_degree, value_dim, default_value=default_value
        )

    def generate_function(self, mesh: dfn.Mesh) -> dfn.Function:
        elem_family, elem_degree, value_dim = self.spec
        # TODO: Add handling for case where `value_dim` being a tuple?
        if value_dim == 'vector':
            return dfn.Function(dfn.VectorFunctionSpace(mesh, elem_family, elem_degree))
        elif value_dim == 'scalar':
            return dfn.Function(dfn.FunctionSpace(mesh, elem_family, elem_degree))
        else:
            raise ValueError(f"Unknown `value_dim`, {value_dim}")


class ConstantFunctionSpaceSpec(BaseFunctionSpaceSpec):
    """
    Represents a `dolfin.Constant`

    Parameters
    ----------
    value_dim:
        The dimension of function value (see `dfn.cpp.function.FunctionSpace`)
    default_value: int
        The default value for the function
    """

    def __init__(self, value_dim: Union[tuple[int, ...], str], default_value: int = 0):
        super().__init__(value_dim, default_value=default_value)

    def generate_function(self, mesh: dfn.Mesh) -> dfn.Constant:
        (value_dim,) = self.spec
        if isinstance(value_dim, str):
            if value_dim == 'vector':
                return dfn.Constant(
                    mesh.geometric_dimension() * [self.default_value], mesh.ufl_cell()
                )
            elif value_dim == 'scalar':
                return dfn.Constant(self.default_value, mesh.ufl_cell())
            else:
                raise ValueError()
        elif isinstance(value_dim, tuple):
            const = dfn.Constant(value_dim, mesh.ufl_cell())
            const.values()[:] = self.default_value
            return const
        else:
            raise TypeError(f"Unknown `value_dim` type, {type(value_dim)}")


def func_spec(elem_family, elem_degree, value_dim, default_value=0):
    """
    Return a `FunctionSpaceSpec`
    """
    return FunctionSpaceSpec(
        elem_family, elem_degree, value_dim, default_value=default_value
    )


def const_spec(value_dim, default_value=0):
    """
    Return a `ConstantFunctionSpaceSpec`
    """
    return ConstantFunctionSpaceSpec(value_dim, default_value=default_value)


## Form class

CoefficientMapping = dict[str, DfnFunction]

# TODO: A `Form` should represent a ufl form expression *without* any associated
# mesh information (just element definitions for coefficients, etc.)
# You could then associate mesh information with a `Form` to get a numerically
# integrable object.
# Currently the mesh information is redunant because I'm using dolfin forms which
# contain mesh information. Future work could change this.
class Form:
    """
    Representation of a `ufl.Form` instance with associated coefficients

    Parameters
    ----------
    ufl_forms: dict[str, dfn.Form]
        The 'dfn.Form' instance
    coefficients: CoefficientMapping
        A mapping from string labels to `dfn.Coefficient` instances used in
        `ufl_forms`

    Attributes
    ----------
    ufl_forms: dict[str, dfn.Form]
        The 'dfn.Form' instance
    coefficients: CoefficientMapping
        A mapping from string labels to `dfn.Coefficient` instances used in
        `ufl_forms`
    expressions: CoefficientMapping
        A mapping from string labels to `dfn.Expression` instances, using the
        coefficients in `coefficients`
    """

    def __init__(
        self,
        ufl_forms: dict[str, dfn.Form],
        coefficients: CoefficientMapping,
        expressions: Optional[CoefficientMapping] = None,
    ):

        self._ufl_forms = ufl_forms
        self._coefficients = coefficients

        if expressions is None:
            expressions = {}
        self._expressions = expressions

    @property
    def ufl_forms(self) -> dict[str, dfn.Form]:
        return self._ufl_forms

    @property
    def coefficients(self) -> CoefficientMapping:
        return self._coefficients

    @property
    def expressions(self) -> CoefficientMapping:
        return self._expressions

    def arguments(self) -> list[ufl.Argument]:
        return {key: form.arguments() for key, form in self.ufl_forms.items()}

    ## Dict interface
    def __iter__(self):
        return self.coefficients.__iter__()

    def keys(self) -> list[str]:
        return self.coefficients.keys()

    def values(self) -> list[DfnFunction]:
        return self.coefficients.values()

    def items(self) -> list[tuple[str, DfnFunction]]:
        return self.coefficients.items()

    def __getitem__(self, key: str) -> DfnFunction:
        return self.coefficients[key]

    def __contains__(self, key: str) -> bool:
        return key in self.coefficients

    ## Basic math
    def __add__(self, other: 'Form') -> 'Form':
        return add_form(self, other)

    def __radd__(self, other: 'Form') -> 'Form':
        return add_form(self, other)

    def __sub__(self, other: 'Form') -> 'Form':
        return add_form(self, -1.0 * other)

    def __rsub__(self, other: 'Form') -> 'Form':
        return add_form(other, -1.0 * self)

    def __mul__(self, other: float) -> 'Form':
        return mul_form(self, other)

    def __rmul__(self, other: float) -> 'Form':
        return mul_form(self, other)


def add_form(form_a: Form, form_b: Form) -> Form:
    """
    Return a new `FenicsForm` from a sum of other forms

    This function:
        sums the two `ufl.Form` instances
        combines any coefficients with the same name
        adds any expressions with the same name together
    """
    # Ensure that the two forms have arguments with consistent function spaces.
    # Oftentimes, the function spaces from the arguments will be the same but
    # correspond to difference `FunctionSpace` instances; this code replaces
    # these with a single argument
    # NOTE: The form arguments are the test functions that forms are integrated
    # against for linear forms/functionals

    def add_ufl_form(a: dfn.Form, b: dfn.Form):
        # new_form_a = form_a
        # new_form_b = form_b
        args_a, args_b = a.arguments(), b.arguments()
        for arg_a, arg_b in zip(args_a, args_b):
            arg_shared = get_shared_function(arg_a, arg_b)
            a = ufl.replace(a, {arg_a: arg_shared})
            b = ufl.replace(b, {arg_b: arg_shared})
        return a + b

    # TODO: Handle case where form_a and form_b have form keys have non-shared keys
    assert form_a.ufl_forms.keys() == form_b.ufl_forms.keys()
    keys = form_a.ufl_forms.keys()
    new_forms = {
        key: add_ufl_form(form_a.ufl_forms[key], form_b.ufl_forms[key])
        for key in keys
    }

    # Sum any expressions with the same key
    new_expressions = {**form_a.expressions, **form_b.expressions}
    duplicate_expr_keys = set.intersection(
        set(form_a.expressions.keys()), set(form_b.expressions.keys())
    )
    duplicate_exprs = {
        key: (form_a.expressions[key], form_b.expressions[key])
        for key in list(duplicate_expr_keys)
    }
    for key, (expr_a, expr_b) in duplicate_exprs.items():
        new_expressions[key] = expr_a + expr_b

    # Link coefficients with the same key in forms and expressions to a single
    # shared `dfn.Function`
    def get_shared_coefficients(duplicate_coeffs):
        shared_coefficients = {
            coeff_key: get_shared_function(coeff_a, coeff_b)
            for coeff_key, (coeff_a, coeff_b) in duplicate_coeffs.items()
        }
        return shared_coefficients

    def update_coefficients(form_or_expr, duplicate_coeffs, shared_coefficients):
        new_form_or_expr = form_or_expr
        for coeff_key in duplicate_coeffs.keys():
            coeff_a, coeff_b = duplicate_coeffs[coeff_key]
            coeff_shared = shared_coefficients[coeff_key]
            coeff_shared = get_shared_function(coeff_a, coeff_b)
            new_form_or_expr = ufl.replace(
                new_form_or_expr, {coeff_a: coeff_shared, coeff_b: coeff_shared}
            )

        return new_form_or_expr

    new_coefficients = {**form_a.coefficients, **form_b.coefficients}
    duplicate_coeff_keys = set.intersection(set(form_a.keys()), set(form_b.keys()))
    duplicate_coeffs = {
        key: (form_a[key], form_b[key]) for key in list(duplicate_coeff_keys)
    }
    shared_coeffs = get_shared_coefficients(duplicate_coeffs)
    new_coefficients.update(shared_coeffs)
    new_forms = {
        key: update_coefficients(form, duplicate_coeffs, shared_coeffs)
        for key, form in new_forms.items()
    }

    new_expressions = {
        key: update_coefficients(expr, duplicate_coeffs, shared_coeffs)
        for key, expr in new_expressions.items()
    }

    return Form(new_forms, new_coefficients, new_expressions)


def mul_form(form: Form, scalar: float) -> Form:
    """
    Return a new `FenicsForm` from a sum of other forms
    """
    # Check that form arguments are consistent and replace duplicated
    # consistent arguments
    new_forms = {key: scalar * ufl_form for key, ufl_form in form.ufl_forms.items()}

    return Form(new_forms, form.coefficients, form.expressions)


## Pre-defined linear functionals

# TODO: Make predefined forms explicitly only depend on UFL forms? i.e. not require a mesh
class PredefinedForm(Form):
    """
    Represents a predefined `dfn.Form`

    The predefined form is defined through two class attributes (see below)
    which specify the coefficients in the form and return the form itself.

    Class Attributes
    ----------------
    COEFFICIENT_SPEC: dict[str, BaseFunctionSpaceSpec]
        A mapping defining all coefficients that are needed to create the form
    INIT_FORM: Callable[
            [CoefficientMapping, dfn.Measure, dfn.Mesh],
            tuple[dfn.Form, CoefficientMapping]
        ]
        A function that returns a `dfn.Form` instance using the coefficients
        given in `COEFFICIENT_SPEC` and additional mesh information

        Note that this function could use `dfn.Coefficient` instances that
        are not defined in `COEFFICIENT_SPEC` as well, but you will be unable to
        access these coefficients and modify their values using after the
        form has been created.

    Parameters
    ----------
    coefficients: CoefficientMapping
        A mapping from string labels to `dfn.Coefficient` instances to be used
        in the `dfn.Form` instance. These coefficient will be used when the
        `dfn.Form` instance is created.
    measure: dfn.Measure
        The measure to use in the form
    mesh: dfn.Mesh
        The measure to use in the form
    """

    COEFFICIENT_SPEC: dict[str, BaseFunctionSpaceSpec] = {}
    def init_form(
        self,
        coefficients: CoefficientMapping,
        measure: dfn.Measure,
        mesh: dfn.Mesh
    ) -> tuple[dict[str, dfn.Form], dict[str, Expr]]:
        raise NotImplementedError()

    def __init__(
        self, coefficients: CoefficientMapping, measure: dfn.Measure, mesh: dfn.Mesh
    ):
        # If a coefficient key is not supplied, generate a default coefficient
        # from `COEFFICIENT_SPEC`
        for key, spec in self.COEFFICIENT_SPEC.items():
            if key not in coefficients:
                coefficients[key] = spec.generate_function(mesh)

        form, expressions = self.init_form(coefficients, measure, mesh)
        super().__init__(form, coefficients, expressions)


class InertialForm(PredefinedForm):
    """
    Linear functional representing an inertial force
    """

    COEFFICIENT_SPEC = {
        'state/a1': func_spec('CG', 1, 'vector'),
        'prop/rho': func_spec('DG', 0, 'scalar'),
    }

    def init_form(self, coefficients, measure, mesh):
        vector_test = dfn.TestFunction(coefficients['state/a1'].function_space())

        acc = coefficients['state/a1']
        density = coefficients['prop/rho']
        inertial_body_force = density * acc

        return {'u': ufl.inner(inertial_body_force, vector_test) * measure}, {}
        # forms['expr.force_inertial'] = inertial_body_force


# Elastic effect forms


class IsotropicElasticForm(PredefinedForm):
    """
    Linear functional representing an isotropic elastic stress
    """

    COEFFICIENT_SPEC = {
        'state/u1': func_spec('CG', 1, 'vector'),
        'state/v1': func_spec('CG', 1, 'vector'),
        'prop/emod': func_spec('DG', 0, 'scalar'),
        'prop/nu': const_spec('scalar', default_value=0.45),
    }

    def init_form(self, coefficients, measure, mesh):

        vector_test = dfn.TestFunction(coefficients['state/u1'].function_space())
        strain_test = strain_inf(vector_test)

        u = coefficients['state/u1']
        inf_strain = strain_inf(u)
        emod = coefficients['prop/emod']
        nu = coefficients['prop/nu']
        set_fenics_function(nu, 0.45)
        stress_elastic = stress_isotropic(inf_strain, emod, nu)

        inf_strain_rate = strain_inf(coefficients['state/v1'])
        stress_elastic_rate = stress_isotropic(inf_strain_rate, emod, nu)

        expressions = {
            'expr.stress_elastic': stress_elastic,
            'expr.strain_energy': ufl.inner(stress_elastic, inf_strain),
            'expr.strain_energy_rate': 2*ufl.inner(stress_elastic_rate, inf_strain_rate),
        }
        return {'u': ufl.inner(stress_elastic, strain_test) * measure}, expressions


class IsotropicIncompressibleElasticSwellingForm(PredefinedForm):
    """
    Linear functional representing an incompressible, isotropic elastic stress with swelling
    """

    COEFFICIENT_SPEC = {
        'state/u1': func_spec('CG', 1, 'vector'),
        'prop/emod': func_spec('DG', 0, 'scalar'),
        'prop/v_swelling': func_spec('DG', 0, 'scalar'),
        'prop/k_swelling': func_spec('DG', 0, 'scalar'),
    }

    def init_form(self, coefficients, measure, mesh):

        vector_test = dfn.TestFunction(coefficients['state/u1'].function_space())
        strain_test = strain_inf(vector_test)

        emod = coefficients['prop/emod']
        nu = 0.5
        dis = coefficients['state/u1']
        inf_strain = strain_inf(dis)
        v_swelling = coefficients['prop/v_swelling']
        set_fenics_function(v_swelling, 1.0)
        k_swelling = coefficients['prop/k_swelling']
        set_fenics_function(k_swelling, 1.0)
        lame_mu = emod / 2 / (1 + nu)
        stress_elastic = 2 * lame_mu * inf_strain + k_swelling * (
            ufl.tr(inf_strain) - (v_swelling - 1.0)
        ) * ufl.Identity(inf_strain.ufl_shape[0])

        expressions = {
            'expr.stress_elastic': stress_elastic,
            'expr.strain_energy': ufl.inner(stress_elastic, inf_strain),
        }
        return {'u': ufl.inner(stress_elastic, strain_test) * measure}, expressions
        # return forms


class IsotropicElasticSwellingForm(PredefinedForm):
    """
    Linear functional representing an isotropic elastic stress with swelling
    """

    COEFFICIENT_SPEC = {
        'state/u1': func_spec('CG', 1, 'vector'),
        'prop/emod': func_spec('DG', 0, 'scalar'),
        'prop/nu': const_spec('scalar', default_value=0.45),
        'prop/v_swelling': func_spec('DG', 0, 'scalar'),
        'prop/m_swelling': func_spec('DG', 0, 'scalar'),
    }

    def init_form(self, coefficients, measure, mesh):
        """
        Add an effect for isotropic elasticity with a swelling field
        """
        dx = measure

        u = coefficients['state/u1']

        vector_test = dfn.TestFunction(coefficients['state/u1'].function_space())
        DE = strain_lin_green_lagrange(u, vector_test)
        E = strain_green_lagrange(u)

        emod = coefficients['prop/emod']
        nu = dfn.Constant(0.45)
        v = coefficients['prop/v_swelling']
        v.vector()[:] = 1.0
        m = coefficients['prop/m_swelling']
        m.vector()[:] = 0.0

        E_v = v ** (-2 / 3) * E + 1 / 2 * (v ** (-2 / 3) - 1) * ufl.Identity(3)
        # Here write the factor $m(v)*v^(-2/3)$ as $m(v)*v^(-1) * v^(1/3)$
        # Then approximate the function $\hat{m} = m(v)*v^(-1)$ with a linear
        # approximation with slope `m`
        mhat = m * (v - 1) + 1
        S = mhat * v ** (1 / 3) * stress_isotropic(E_v, emod, nu)

        F = def_grad(u)
        J = ufl.det(F)
        expressions = {
            # NOTE: The terms around `S` convert PK2 to Cauchy stress
            'expr.stress_elastic': (1 / J) * F * S * F.T,
            # NOTE: This should be true because this is a linear hyperelastic
            # material
            'expr.strain_energy': ufl.inner(S, E),
            'expr.stress_elastic_PK2': S,
            'expr.strain_green': E,
        }

        return {'u': ufl.inner(S, DE) * dx}, expressions


class IsotropicElasticSwellingPowerLawForm(PredefinedForm):
    """
    Linear functional representing an isotropic elastic stress with swelling

    The strain energy function is modified by a power law
    """

    COEFFICIENT_SPEC = {
        'state/u1': func_spec('CG', 1, 'vector'),
        'state/v1': func_spec('CG', 1, 'vector'),
        'prop/emod': func_spec('DG', 0, 'scalar'),
        'prop/nu': const_spec('scalar', default_value=0.45),
        'prop/v_swelling': func_spec('DG', 0, 'scalar'),
        'prop/m_swelling': func_spec('DG', 0, 'scalar'),
    }

    def init_form(self, coefficients, measure, mesh):
        """
        Add an effect for isotropic elasticity with a swelling field
        """
        dx = measure

        u = coefficients['state/u1']
        v = coefficients['state/v1']

        vector_test = dfn.TestFunction(coefficients['state/u1'].function_space())
        DE = strain_lin_green_lagrange(u, vector_test)
        E = strain_green_lagrange(u)
        E_rate = strain_green_lagrange(v)

        emod = coefficients['prop/emod']
        nu = dfn.Constant(0.45)
        v = coefficients['prop/v_swelling']
        v.vector()[:] = 1.0
        m = coefficients['prop/m_swelling']
        m.vector()[:] = 0.0

        E_v = v ** (-2 / 3) * E + 1 / 2 * (v ** (-2 / 3) - 1) * ufl.Identity(3)
        E_v_rate = v ** (-2 / 3) * E_rate + 1 / 2 * (v ** (-2 / 3) - 1) * ufl.Identity(3)
        # Here `mbar_v` corresponds to the scaling function 'm(v)/v' [Gou2016]
        # I used 'm(v)/v' (instead of 'm(v)') so that the coefficient
        # `'prop/m_swelling'` will correspond to the linear stiffness
        # change used for `IsotropicElasticSwellingForm` at no swelling
        mbar_v = v**m
        S = mbar_v * v ** (1 / 3) * stress_isotropic(E_v, emod, nu)
        S_rate = mbar_v * v ** (1 / 3) * stress_isotropic(E_v_rate, emod, nu)

        F = def_grad(u)
        J = ufl.det(F)
        expressions = {
            # NOTE: The terms around `S` convert PK2 to Cauchy stress
            'expr.stress_elastic': (1 / J) * F * S * F.T,
            # NOTE: This should be true because this is a linear hyperelastic
            # material
            'expr.strain_energy': ufl.inner(S, E),
            'expr.strain_energy_rate': ufl.inner(S, E_rate) + ufl.inner(S_rate, E),
            'expr.stress_elastic_PK2': S,
            'expr.strain_green': E,
        }

        return {'u': ufl.inner(S, DE) * dx}, expressions


# Surface forcing forms


class SurfacePressureForm(PredefinedForm):
    """
    Linear functional representing a pressure follower load
    """

    COEFFICIENT_SPEC = {
        'state/u1': func_spec('CG', 1, 'vector'),
        'coeff.fsi.p1': func_spec('CG', 1, 'scalar'),
    }

    def init_form(self, coefficients, measure, mesh):

        ds = measure

        dis = coefficients['state/u1']
        vector_test = dfn.TestFunction(coefficients['state/u1'].function_space())
        facet_normal = ufl.FacetNormal(mesh)

        p = coefficients['coeff.fsi.p1']
        reference_traction = -p * pullback_area_normal(dis, facet_normal)

        expressions = {}
        expressions['expr.fluid_traction'] = reference_traction
        return {'u': ufl.inner(reference_traction, vector_test) * ds}, expressions


class ManualSurfaceContactTractionForm(PredefinedForm):
    """
    Linear functional representing a surface contact traction
    """

    COEFFICIENT_SPEC = {
        'state/u1': func_spec('CG', 1, 'vector'),
        'control/tcontact': func_spec('CG', 1, 'vector'),
        'prop/ycontact': const_spec('scalar', np.inf),
        'prop/ncontact': const_spec('vector'),
        'prop/kcontact': const_spec('scalar', 1.0),
    }

    def init_form(self, coefficients, measure, mesh):

        # NOTE: The contact traction must be manually linked with displacements
        # and penalty parameters!
        # This manual linking is done through the class
        # `femvf.models.solid.NodalContactSolid`
        # The relevant penalty parameters are:
        # `ycontact = coefficients['prop/ycontact']`
        # `ncontact = coefficients['prop/ncontact']`
        # `kcontact = coefficients['prop/kcontact']`

        vector_test = dfn.TestFunction(
            coefficients['control/tcontact'].function_space()
        )
        tcontact = coefficients['control/tcontact']

        # Set a default y-dir contact surface direction
        ncontact = coefficients['prop/ncontact'].values()
        ncontact[1] = 1.0
        coefficients['prop/ncontact'].assign(dfn.Constant(ncontact))

        expressions = {}
        return {'u': ufl.inner(tcontact, vector_test) * measure}, expressions


# Surface membrane forms


class IsotropicMembraneForm(PredefinedForm):
    """
    Linear functional representing an isotropic elastic membrane
    """

    COEFFICIENT_SPEC = {
        'state/u1': func_spec('CG', 1, 'vector'),
        'prop/emod_membrane': func_spec('DG', 0, 'scalar'),
        'prop/nu_membrane': func_spec('DG', 0, 'scalar'),
        'prop/th_membrane': func_spec('DG', 0, 'scalar'),
    }

    def init_form(self, coefficients, measure, mesh, large_def=False):
        vector_test = dfn.TestFunction(coefficients['state/u1'].function_space())

        # Define the 8th order projector to get the planar strain component
        facet_normal = ufl.FacetNormal(mesh)
        if mesh.topology().dim() == 2:
            n = ufl.as_tensor([facet_normal[0], facet_normal[1], 0.0])
        else:
            n = facet_normal
        nn = ufl.outer(n, n)
        ident = ufl.Identity(n.ufl_shape[0])
        project_pp = ufl.outer(ident - nn, ident - nn)

        i, j, k, l = ufl.indices(4)

        dis = coefficients['state/u1']
        if large_def:
            strain = strain_green_lagrange(dis)
            strain_test = strain_lin_green_lagrange(dis, vector_test)
        else:
            strain = strain_inf(dis)
            strain_test = strain_inf(vector_test)
        strain_pp_test = ufl.as_tensor(
            project_pp[i, j, k, l] * strain_test[j, k], (i, l)
        )

        emod = coefficients['prop/emod_membrane']
        th_membrane = coefficients['prop/th_membrane']
        nu = coefficients['prop/nu_membrane']
        set_fenics_function(nu, 0.45)
        mu = emod / 2 / (1 + nu)
        lmbda = emod * nu / (1 + nu) / (1 - 2 * nu)

        strain_pp = ufl.as_tensor(project_pp[i, j, k, l] * strain[j, k], (i, l))

        # account for ambiguous 0/0 when emod=0
        lmbda_pp = ufl.conditional(
            ufl.eq(emod, 0), 0, 2 * mu * lmbda / (lmbda + 2 * mu)
        )
        stress_pp = 2 * mu * strain_pp + lmbda_pp * ufl.tr(strain_pp) * (ident - nn)

        expressions = {}

        return {'u': ufl.inner(stress_pp, strain_pp_test) * th_membrane * measure}, expressions

        # forms['form.un.f1uva'] += res
        # forms['prop/nu_membrane'] = nu
        # return forms


class IsotropicIncompressibleMembraneForm(PredefinedForm):
    """
    Linear functional representing an incompressible isotropic elastic membrane
    """

    COEFFICIENT_SPEC = {
        'state/u1': func_spec('CG', 1, 'vector'),
        'prop/emod_membrane': func_spec('DG', 0, 'scalar'),
        'prop/th_membrane': func_spec('DG', 0, 'scalar'),
    }

    def init_form(self, coefficients, measure, mesh, large_def=False):
        vector_test = dfn.TestFunction(coefficients['state/u1'].function_space())

        # Define the 8th order projector to get the planar strain component
        mesh = coefficients['state/u1'].function_space().mesh()
        facet_normal = ufl.FacetNormal(mesh)
        n = ufl.as_tensor([facet_normal[0], facet_normal[1], 0.0])
        nn = ufl.outer(n, n)
        ident = ufl.Identity(n.ufl_shape[0])
        project_pp = ufl.outer(ident - nn, ident - nn)
        i, j, k, l = ufl.indices(4)

        strain_test = strain_inf(vector_test)
        strain_pp_test = ufl.as_tensor(
            project_pp[i, j, k, l] * strain_test[j, k], (i, l)
        )

        dis = coefficients['state/u1']
        if large_def:
            strain = strain_green_lagrange(dis)
            strain_test = strain_lin_green_lagrange(dis, vector_test)
        else:
            strain = strain_inf(dis)
            strain_test = strain_inf(vector_test)
        strain_pp_test = ufl.as_tensor(
            project_pp[i, j, k, l] * strain_test[j, k], (i, l)
        )

        emod_membrane = coefficients['prop/emod_membrane']
        th_membrane = coefficients['prop/th_membrane']
        nu = 0.5
        lame_mu = emod_membrane / 2 / (1 + nu)
        strain_pp = ufl.as_tensor(project_pp[i, j, k, l] * strain[j, k], (i, l))

        stress_pp = 2 * lame_mu * strain_pp + 2 * lame_mu * ufl.tr(strain_pp) * (
            ident - nn
        )

        expressions = {}
        return {'u': ufl.inner(stress_pp, strain_pp_test) * th_membrane * measure}, expressions


# Viscous effect forms


class RayleighDampingForm(PredefinedForm):
    """
    Linear functional representing a Rayleigh damping viscous stress
    """

    COEFFICIENT_SPEC = {
        'state/v1': func_spec('CG', 1, 'vector'),
        'prop/rho': func_spec('DG', 0, 'scalar'),
        'prop/emod': func_spec('DG', 0, 'scalar'),
        'prop/nu': const_spec('scalar', 0.45),
        'prop/rayleigh_m': const_spec('scalar', 1.0),
        'prop/rayleigh_k': const_spec('scalar', 1.0),
    }

    def init_form(self, coefficients, measure, mesh, large_def=False):

        vector_test = dfn.TestFunction(coefficients['state/v1'].function_space())

        dx = measure
        strain_test = strain_inf(vector_test)
        v = coefficients['state/v1']

        rayleigh_m = coefficients['prop/rayleigh_m']
        rayleigh_k = coefficients['prop/rayleigh_k']

        emod = coefficients['prop/emod']
        nu = coefficients['prop/nu']
        inf_strain = strain_inf(v)
        stress_elastic = stress_isotropic(inf_strain, emod, nu)
        stress_visco = rayleigh_k * stress_elastic

        rho = coefficients['prop/rho']
        force_visco = rayleigh_m * rho * v

        expressions = {}
        form = (
            ufl.inner(force_visco, vector_test) + ufl.inner(stress_visco, strain_test)
        ) * dx
        return {'u': form}, expressions

        # coefficients['form.un.f1uva'] += damping
        # # coefficients['prop/nu'] = nu
        # # coefficients['prop/rayleigh_m'] = rayleigh_m
        # # coefficients['prop/rayleigh_k'] = rayleigh_k
        # return coefficients


class KelvinVoigtForm(PredefinedForm):
    """
    Linear functional representing a Kelvin-Voigt viscous stress
    """

    COEFFICIENT_SPEC = {
        'state/v1': func_spec('CG', 1, 'vector'),
        'prop/eta': func_spec('DG', 0, 'scalar'),
    }

    def init_form(self, coefficients, measure, mesh):

        vector_test = dfn.TestFunction(coefficients['state/v1'].function_space())

        strain_test = strain_inf(vector_test)
        v = coefficients['state/v1']

        eta = coefficients['prop/eta']
        inf_strain_rate = strain_inf(v)
        stress_visco = eta * inf_strain_rate

        expressions = {}
        expressions['expr.kv_stress'] = stress_visco
        expressions['expr.kv_strain_rate'] = inf_strain_rate

        return {'u': ufl.inner(stress_visco, strain_test) * measure}, expressions


class APForceForm(PredefinedForm):
    """
    Linear functional representing a anterior-posterior (AP) force
    """

    COEFFICIENT_SPEC = {
        'state/u1': func_spec('CG', 1, 'vector'),
        'state/v1': func_spec('CG', 1, 'vector'),
        'prop/eta': func_spec('DG', 0, 'scalar'),
        'prop/emod': func_spec('DG', 0, 'scalar'),
        'prop/nu': const_spec('scalar', default_value=0.45),
        'prop/u_ant': func_spec('DG', 0, 'scalar'),
        'prop/u_pos': func_spec('DG', 0, 'scalar'),
        'prop/length': func_spec('DG', 0, 'scalar'),
        'prop/muscle_stress': func_spec('DG', 0, 'scalar'),
    }

    def init_form(self, coefficients, measure, mesh):
        vector_test = dfn.TestFunction(coefficients['state/v1'].function_space())

        u1, v1 = coefficients['state/u1'], coefficients['state/v1']
        kv_eta = coefficients['prop/eta']
        emod = coefficients['prop/emod']
        nu = coefficients['prop/nu']
        lame_mu = emod / 2 / (1 + nu)

        u_ant = coefficients['prop/u_ant']  # zero values by default
        u_pos = coefficients['prop/u_pos']
        length = coefficients['prop/length']
        muscle_stress = coefficients['prop/muscle_stress']

        d2u_dz2 = (u_ant - 2 * u1 + u_pos) / length**2
        d2v_dz2 = (u_ant - 2 * v1 + u_pos) / length**2
        force_elast_ap = (lame_mu + muscle_stress) * d2u_dz2
        force_visco_ap = 0.5 * kv_eta * d2v_dz2
        stiffness = ufl.inner(force_elast_ap, vector_test) * measure
        viscous = ufl.inner(force_visco_ap, vector_test) * measure

        expressions = {}

        return {'u': -stiffness - viscous}, expressions


# Add shape effect forms
class ShapeForm(PredefinedForm):
    """
    Linear functional that just adds a shape parameter
    """

    COEFFICIENT_SPEC = {'prop/umesh': func_spec('CG', 1, 'vector')}

    def init_form(self, coefficients, measure, mesh):
        vector_test = dfn.TestFunction(
            coefficients['prop/umesh'].function_space()
        )
        umesh = coefficients['prop/umesh']
        umesh_ufl = ufl.SpatialCoordinate(mesh)

        # NOTE:
        # To find sensitivity w.r.t shape, UFL uses the object
        # `ufl.SpatialCoordinate(mesh)` rather than a `Function` instance.
        # This doesn't have an associated vector of values so you have to
        # store the ufl object and the coefficient vector separately.
        # Code has to manually account for this additional property,
        # for example, when taking shape derivatives
        coefficients['prop/umesh_ufl'] = umesh_ufl

        expressions = {}

        return {'u': 0 * ufl.inner(umesh_ufl, vector_test) * measure}, expressions


## Form modifiers

def modify_newmark_time_discretization(form: Form) -> Form:
    u1 = form['state/u1']
    v1 = form['state/v1']
    a1 = form['state/a1']

    u0 = dfn.Function(form['state/u1'].function_space())
    v0 = dfn.Function(form['state/v1'].function_space())
    a0 = dfn.Function(form['state/a1'].function_space())

    dt = dfn.Function(form['prop/rho'].function_space())
    gamma = dfn.Constant(1 / 2)
    beta = dfn.Constant(1 / 4)
    v1_nmk = newmark.newmark_v(u1, u0, v0, a0, dt, gamma, beta)
    a1_nmk = newmark.newmark_a(u1, u0, v0, a0, dt, gamma, beta)

    new_coefficients = {
        'state/u0': u0,
        'state/v0': v0,
        'state/a0': a0,
        'coeff.time.dt': dt,
        'coeff.time.gamma': gamma,
        'coeff.time.beta': beta,
    }

    coefficients = {**form.coefficients, **new_coefficients}

    ufl_forms = {
        key: ufl.replace(ufl_form, {v1: v1_nmk, a1: a1_nmk})
        for key, ufl_form in form.ufl_forms.items()
    }

    return Form(ufl_forms, coefficients, form.expressions)


def modify_unary_linearized_forms(form: Form) -> dict[str, dfn.Form]:
    """
    Generate linearized forms representing linearization of the residual wrt different states

    These forms are needed for solving the Hopf bifurcation problem/conditions
    """
    new_coefficients = {}

    # Create coefficients for linearization directions
    for var_name in ['u1', 'v1', 'a1']:
        new_coefficients[f'coeff.dstate.{var_name}'] = dfn.Function(
            form[f'state/{var_name}'].function_space()
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
    new_ufl_forms = {}
    for form_key, ufl_form in form.ufl_forms.items():
        linearized_forms = []
        for var_name in ['u1', 'v1', 'a1']:
            # unary_form_name = f'df1uva_{var_name}'
            df_dx = dfn.derivative(ufl_form, form[f'state/{var_name}'])
            # print(len(df_dx.arguments()))
            # print(len(forms[f'form.un.f1uva'].arguments()))
            linearized_form = dfn.action(
                df_dx, new_coefficients[f'coeff.dstate.{var_name}']
            )
            linearized_forms.append(linearized_form)

        for var_name in ['p1']:
            # unary_form_name = f'df1uva_{var_name}'
            # df_dx = form[f'form.bi.df1uva_d{var_name}']
            df_dx = dfn.derivative(ufl_form, form[f'coeff.fsi.{var_name}'])
            linearized_form = dfn.action(df_dx, new_coefficients[f'coeff.dfsi.{var_name}'])
            linearized_forms.append(linearized_form)

        # Compute the total linearized residual
        new_ufl_forms[form_key] = reduce(operator.add, linearized_forms)

    return Form(
        new_ufl_forms, {**form.coefficients, **new_coefficients}, form.expressions
    )


## Common functions

def dis_contact_gap(gap):
    """
    Return the positive gap
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            category=RuntimeWarning,
            message='invalid value encountered in add',
        )
        positive_gap = (gap + abs(gap)) / 2
    positive_gap = np.where(gap == -np.inf, 0.0, positive_gap)
    return positive_gap


def pressure_contact_cubic_penalty(gap, kcoll):
    """
    Return the cubic penalty pressure
    """
    positive_gap = dis_contact_gap(gap)
    return kcoll * positive_gap**3


def dform_cubic_penalty_pressure(gap, kcoll):
    """
    Return derivatives of the cubic penalty pressure
    """
    positive_gap = dis_contact_gap(gap)
    dpositive_gap = np.sign(gap)
    return kcoll * 3 * positive_gap**2 * dpositive_gap, positive_gap**3
