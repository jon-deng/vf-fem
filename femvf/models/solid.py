"""
Module for definitions of weak forms.

Units are in CGS

TODO: The form definitions have a lot of repeated code. Many of the form operations are copy-pasted.
You should think about what forms should be custom made for different solid governing equations
and what types of forms are always generated the same way, and refactor accordingly.
"""
import numpy as np
import dolfin as dfn
import ufl

from ..solverconst import DEFAULT_NEWTON_SOLVER_PRM
from ..parameters.properties import property_vecs
from ..constants import PASCAL_TO_CGS, SI_DENSITY_TO_CGS

from . import base
from . import newmark
from ..linalg import BlockVec, general_vec_set


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

    return 2*lame_mu*strain + lame_lambda*ufl.tr(strain)*ufl.Identity(strain.geometric_dimension())

def form_inf_strain(u):
    """
    Returns the strain tensor for a displacement field.

    Parameters
    ----------
    u : dfn.TrialFunction, ufl.Argument
        Trial displacement field
    """
    return 1/2 * (ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

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

def form_cubic_penalty_pressure(gap, kcoll):
    positive_gap = (gap + abs(gap)) / 2
    return kcoll*positive_gap**3

def dform_cubic_penalty_pressure(gap, kcoll):
    positive_gap = (gap + abs(gap)) / 2
    dpositive_gap = np.sign(gap)
    return kcoll*3*positive_gap**2 * dpositive_gap, positive_gap**3


def form_quad_penalty_pressure(gap, kcoll):
    positive_gap = (gap + abs(gap)) / 2
    return kcoll*positive_gap**2


def gen_residual_bilinear_forms(forms):
    """
    Add bilinear forms to a dictionary defining the residual and state variables
    """
    # Derivatives of the displacement residual form wrt all state variables
    manual_state_var_names = [name for name in forms.keys() if 'coeff.state.manual' in name]

    for full_var_name in (
        [f'coeff.state.{y}' for y in ['u0', 'v0', 'a0', 'u1']] + 
        manual_state_var_names + ['coeff.time.dt', 'coeff.fsi.p1']):
        f = forms['form.un.f1']
        x = forms[full_var_name]

        var_name = full_var_name.split(".")[-1]
        form_name = f'form.bi.df1_d{var_name}'
        forms[form_name] = dfn.derivative(f, x)
        forms[f'{form_name}_adj'] = dfn.adjoint(forms[form_name])

    # Derivatives of the u/v/a residual form wrt variables of interest
    for full_var_name in (
        [f'coeff.state.{y}' for y in ['u1', 'v1', 'a1']] + 
        manual_state_var_names + ['coeff.fsi.p1']):
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
    property_labels = [form_name.split('.')[-1] for form_name in forms.keys() 
                       if 'coeff.prop' in form_name]
    for prop_name in property_labels:
        try:
            df1_dsolid[prop_name] = dfn.adjoint(
                dfn.derivative(forms['form.un.f1'], forms[f'coeff.prop.{prop_name}']))
        except RuntimeError:
            df1_dsolid[prop_name] = None

    return df1_dsolid


def base_form_definitions(mesh, facet_func, facet_label_to_id, cell_func, cell_label_to_id,
                          fsi_facet_labels, fixed_facet_labels):
    # Measures
    dx = dfn.Measure('dx', domain=mesh, subdomain_data=cell_func)
    ds = dfn.Measure('ds', domain=mesh, subdomain_data=facet_func)
    _traction_ds = [ds(facet_label_to_id[facet_label]) for facet_label in fsi_facet_labels]
    traction_ds = sum(_traction_ds[1:], _traction_ds[0])

    # Function space
    scalar_fspace = dfn.FunctionSpace(mesh, 'CG', 1)
    vector_fspace = dfn.VectorFunctionSpace(mesh, 'CG', 1)

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

    forms = {
        'measure.dx': dx,
        'measure.ds': ds,
        'measure.ds_traction': traction_ds,
        'bcs.base': bc_base,

        'geom.facet_normal': dfn.FacetNormal(mesh),

        'fspace.vector': vector_fspace,
        'fspace.scalar': scalar_fspace,

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

        'expr.kin.inf_strain': form_inf_strain(u1),
        'expr.kin.inf_strain_rate': form_inf_strain(v1),
        
        'form.un.f1uva': 0.0
        # Add kinematic expressions?
        # 'expr.kin.'
        }
    return forms

def add_inertial_form(forms):
    dx = forms['measure.dx']
    vector_test = forms['test.vector']

    a = forms['coeff.state.a1']
    rho = dfn.Function(forms['fspace.scalar'])
    inertial_body_force = rho*a

    forms['form.un.f1uva'] += ufl.inner(inertial_body_force, vector_test) * dx
    forms['coeff.prop.rho'] = rho
    forms['expr.force_inertial'] = inertial_body_force
    return forms

def add_isotropic_elastic_form(forms):
    dx = forms['measure.dx']
    vector_test = forms['test.vector']
    strain_test = form_inf_strain(vector_test)

    inf_strain = forms['expr.kin.inf_strain']
    emod = dfn.Function(forms['fspace.scalar'])
    nu = dfn.Function(forms['fspace.scalar'])
    stress_elastic = form_lin_iso_cauchy_stress(inf_strain, emod, nu)

    forms['form.un.f1uva'] += ufl.inner(stress_elastic, strain_test) * dx
    forms['coeff.prop.emod'] = emod
    forms['coeff.prop.nu'] = nu
    forms['expr.stress_elastic'] = stress_elastic
    return forms

def add_surface_pressure_form(forms):
    ds = forms['measure.ds_traction']
    vector_test = forms['test.vector']
    u = forms['coeff.state.u1']
    facet_normal = forms['geom.facet_normal']

    p = dfn.Function(forms['fspace.scalar'])
    reference_traction = -p * form_pullback_area_normal(u, facet_normal)

    forms['form.un.f1uva'] -= ufl.inner(reference_traction, vector_test) * ds
    forms['coeff.fsi.p1'] = p
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


class Solid(base.Model):
    """
    Class representing the discretized governing equations of a solid
    """
    # Subclasses have to set these values
    PROPERTY_DEFAULTS = None

    def __init__(self, mesh, facet_func, facet_label_to_id, cell_func, cell_label_to_id, 
                 fsi_facet_labels, fixed_facet_labels):
        assert isinstance(fsi_facet_labels, (list, tuple))
        assert isinstance(fixed_facet_labels, (list, tuple))

        self._forms = self.form_definitions(mesh, facet_func, facet_label_to_id,
                                            cell_func, cell_label_to_id, fsi_facet_labels, fixed_facet_labels)
        gen_residual_bilinear_forms(self._forms)

        ## Store mesh related quantities
        self.mesh = mesh
        self.facet_func = facet_func
        self.cell_func = cell_func
        self.facet_label_to_id = facet_label_to_id
        self.cell_label_to_id = cell_label_to_id

        self.fsi_facet_labels = fsi_facet_labels
        self.fixed_facet_labels = fixed_facet_labels

        ## Store some key quantites related to the forms
        self.vector_fspace = self.forms['fspace.vector']
        self.scalar_fspace = self.forms['fspace.scalar']

        self.scalar_trial = self.forms['trial.scalar']
        self.vector_trial = self.forms['trial.vector']
        self.scalar_test = self.forms['test.scalar']
        self.vector_test = self.forms['test.vector']

        self.u0 = self.forms['coeff.state.u0']
        self.v0 = self.forms['coeff.state.v0']
        self.a0 = self.forms['coeff.state.a0']
        self.u1 = self.forms['coeff.state.u1']
        self.v1 = self.forms['coeff.state.v1']
        self.a1 = self.forms['coeff.state.a1']

        self.dt_form = self.forms['coeff.time.dt']

        self.f1 = self.forms['form.un.f1']
        self.df1_du1 = self.forms['form.bi.df1_du1']
        self.df1_dsolid = gen_residual_bilinear_property_forms(self.forms)

        ## Measures and boundary conditions
        self.dx = self.forms['measure.dx']
        self.ds = self.forms['measure.ds']
        self.bc_base = self.forms['bcs.base']

        ## Index mappings
        self.vert_to_vdof = dfn.vertex_to_dof_map(self.forms['fspace.vector'])
        self.vert_to_sdof = dfn.vertex_to_dof_map(self.forms['fspace.scalar'])
        self.vdof_to_vert = dfn.dof_to_vertex_map(self.forms['fspace.vector'])
        self.sdof_to_vert = dfn.dof_to_vertex_map(self.forms['fspace.scalar'])

        ## Define the state/controls/properties
        self.state0 = BlockVec((self.u0.vector(), self.v0.vector(), self.a0.vector()), ('u', 'v', 'a'))
        self.state1 = BlockVec((self.u1.vector(), self.v1.vector(), self.a1.vector()), ('u', 'v', 'a'))
        self.control = BlockVec((self.forms['coeff.fsi.p1'].vector(),), ('p',))
        self.properties = self.get_properties_vec(set_default=True)
        self.set_properties(self.properties)

        # TODO: You should move this to the solid since it's not the responsibility of this class to do solid specific stuff
        self.cached_form_assemblers = {
            'bilin.df1_du1_adj': CachedBiFormAssembler(self.forms['form.bi.df1_du1_adj']),
            'bilin.df1_du0_adj': CachedBiFormAssembler(self.forms['form.bi.df1_du0_adj']),
            'bilin.df1_dv0_adj': CachedBiFormAssembler(self.forms['form.bi.df1_dv0_adj']),
            'bilin.df1_da0_adj': CachedBiFormAssembler(self.forms['form.bi.df1_da0_adj'])
        }

    @property
    def solid(self):
        return self

    @property
    def forms(self):
        return self._forms

    # The dt property makes the time step form behave like a scalar
    @property
    def dt(self):
        return self.dt_form.vector()[0]

    @dt.setter
    def dt(self, value):
        self.dt_form.vector()[:] = value

    @property
    def XREF(self):
        xref = dfn.Function(self.vector_fspace)
        xref.vector()[:] = self.scalar_fspace.tabulate_dof_coordinates().reshape(-1).copy()
        return xref

    @staticmethod
    def form_definitions(mesh, facet_func, facet_label_to_id, cell_func, cell_label_to_id, 
                         fsi_facet_labels, fixed_facet_labels):
        """
        Return a dictionary of ufl forms representing the solid in Fenics.

        You have to implement this along with a description of the properties to make a subclass of
        the `Solid`.
        """
        raise NotImplementedError("Subclasses must implement this function")

    ## Functions for getting empty parameter vectors
    def get_state_vec(self):
        u = dfn.Function(self.vector_fspace).vector()
        v = dfn.Function(self.vector_fspace).vector()
        a = dfn.Function(self.vector_fspace).vector()
        return BlockVec((u, v, a), ('u', 'v', 'a')) 

    def get_control_vec(self):
        ret = self.control.copy()
        ret.set(0.0)
        return ret

    def get_properties_vec(self, set_default=True):
        labels = [form_name.split('.')[-1] for form_name in self.forms.keys() 
                  if 'coeff.prop' in form_name]
        vecs = []
        for label in labels:
            coefficient = self.forms['coeff.prop.'+label]

            vec = None
            # Generally the size of the vector comes directly from the property,
            # for examples, constants are size 1, scalar fields have size matching number of dofs
            # Time step is a special case since it is size 1 but is made to be a field as 
            # a workaround
            if isinstance(coefficient, dfn.function.constant.Constant) or label == 'dt':
                vec = np.ones(coefficient.values().size)
                vec[:] = coefficient.values()
            else:
                vec = coefficient.vector().copy()
            
            if set_default and label in self.PROPERTY_DEFAULTS:
                vec[:] = self.PROPERTY_DEFAULTS[label]

            vecs.append(vec)

        return BlockVec(vecs, labels)
    
    ## Parameter setting functions
    def set_ini_state(self, uva0):
        """
        Sets the initial state variables, (u, v, a)

        Parameters
        ----------
        u0, v0, a0 : array_like
        """
        self.forms['coeff.state.u0'].vector()[:] = uva0[0]
        self.forms['coeff.state.v0'].vector()[:] = uva0[1]
        self.forms['coeff.state.a0'].vector()[:] = uva0[2]

    def set_fin_state(self, uva1):
        """
        Sets the final state variables.

        Note that the solid forms are in displacement form so only the displacement is needed as an
        initial guess to solve the solid equations. The state `v1` and `a1` are specified explicitly
        by the Newmark relations once you solve for `u1`.

        Parameters
        ----------
        u1, v1, a1 : array_like
        """
        self.forms['coeff.state.u1'].vector()[:] = uva1[0]
        self.forms['coeff.state.v1'].vector()[:] = uva1[1]
        self.forms['coeff.state.a1'].vector()[:] = uva1[2]

    def set_control(self, p1):
        self.forms['coeff.fsi.p1'].vector()[:] = p1['p']

    def set_properties(self, props):
        """
        Sets the properties of the solid

        Properties are essentially all settable values that are not states. In UFL, all things that
        can have set values are `coefficients`, but this ditinguishes between state coefficients,
        and other property coefficients.

        Parameters
        ----------
        props : Property / dict-like
        """
        for key in props.keys:
            # TODO: Check types to make sure the input property is compatible with the solid type
            coefficient = self.forms['coeff.prop.'+key]

            # If the property is a field variable, values have to be assigned to every spot in
            # the vector
            if isinstance(coefficient, dfn.function.constant.Constant):
                coefficient.assign(dfn.Constant(np.squeeze(props[key])))
            else:
                coefficient.vector()[:] = props[key]

    ## Residual and sensitivity functions
    def assem(self, form_name):
        """
        Assembles the form given by label `form_name`
        """
        form_key = f'form.bi.{form_name}'

        if form_key in self.forms:
            return dfn.assemble(self.forms[form_key])
        else:
            raise ValueError(f"`{form_name}` is not a valid form label")

    def res(self):
        dt = self.dt
        u1, v1, a1 = self.u1.vector(), self.v1.vector(), self.a1.vector()
        u0, v0, a0 = self.u0.vector(), self.v0.vector(), self.a0.vector()

        res = self.get_state_vec()
        res['u'] = dfn.assemble(self.forms['form.un.f1'])
        self.bc_base.apply(res['u'])
        res['v'] = v1 - newmark.newmark_v(u1, u0, v0, a0, dt)
        res['a'] = a1 - newmark.newmark_a(u1, u0, v0, a0, dt)
        return res

    def solve_state1(self, state1, newton_solver_prm=None):
        if newton_solver_prm is None:
            newton_solver_prm = DEFAULT_NEWTON_SOLVER_PRM

        def linearized_subproblem(state):
            """
            Return a solver and residual corresponding to the linearized subproblem

            Returns
            -------
            assem_res : callable() -> type(state)
            solver : callable(type(state)) -> type(state)
            """
            self.set_fin_state(state)
            assem_res = self.res
            solve = self.solve_dres_dstate1
            return assem_res, solve

        state_n, solve_info = newton_solve(state1, linearized_subproblem, newton_solver_prm)
        return state_n, solve_info

    def solve_dres_dstate1(self, b):
        dt = self.dt
        dfu2_du2 = dfn.assemble(self.forms['form.bi.df1_du1'])
        dfv2_du2 = 0 - newmark.newmark_v_du1(dt)
        dfa2_du2 = 0 - newmark.newmark_a_du1(dt)

        # Solve A x = b
        bu, bv, ba = b.vecs
        x = self.get_state_vec()

        self.bc_base.apply(dfu2_du2)
        dfn.solve(dfu2_du2, x['u'], bu, 'petsc')

        x['v'][:] = bv - dfv2_du2*x['u']
        x['a'][:] = ba - dfa2_du2*x['u']
        
        return x
        
    def solve_dres_dstate1_adj(self, x):
        # Form key matrices
        dfu2_du2 = self.cached_form_assemblers['bilin.df1_du1_adj'].assemble()
        dfv2_du2 = 0 - newmark.newmark_v_du1(self.dt)
        dfa2_du2 = 0 - newmark.newmark_a_du1(self.dt)

        # Solve b^T A = x^T
        xu, xv, xa = x.vecs
        b = self.get_state_vec()
        b['a'][:] = xa
        b['v'][:] = xv

        rhs_u = xu - (dfv2_du2*b['v'] + dfa2_du2*b['a'])

        self.bc_base.apply(dfu2_du2, rhs_u)
        dfn.solve(dfu2_du2, b['u'], rhs_u, 'petsc')
        return b

    def apply_dres_dstate0(self, x):
        dt = self.dt

        dfu2_du1 = dfn.assemble(self.forms['form.bi.df1_du0'], tensor=dfn.PETScMatrix())
        dfu2_dv1 = dfn.assemble(self.forms['form.bi.df1_dv0'], tensor=dfn.PETScMatrix())
        dfu2_da1 = dfn.assemble(self.forms['form.bi.df1_da0'], tensor=dfn.PETScMatrix())
        for mat in (dfu2_du1, dfu2_dv1, dfu2_da1):
            self.bc_base.apply(mat)

        dfv2_du1 = 0 - newmark.newmark_v_du0(dt)
        dfv2_dv1 = 0 - newmark.newmark_v_dv0(dt)
        dfv2_da1 = 0 - newmark.newmark_v_da0(dt)

        dfa2_du1 = 0 - newmark.newmark_a_du0(dt)
        dfa2_dv1 = 0 - newmark.newmark_a_dv0(dt)
        dfa2_da1 = 0 - newmark.newmark_a_da0(dt)

        ## Do the matrix vector multiplication that gets the RHS for the adjoint equations
        # Allocate a vector the for fluid side mat-vec multiplication
        b = self.get_state_vec()
        b['u'][:] = (dfu2_du1*x['u'] + dfu2_dv1*x['v'] + dfu2_da1*x['a'])
        b['v'][:] = (dfv2_du1*x['u'] + dfv2_dv1*x['v'] + dfv2_da1*x['a'])
        b['a'][:] = (dfa2_du1*x['u'] + dfa2_dv1*x['v'] + dfa2_da1*x['a'])
        return b
    
    def apply_dres_dstate0_adj(self, b):
        dt = self.dt

        dfu2_du1 = self.cached_form_assemblers['bilin.df1_du0_adj'].assemble()
        dfu2_dv1 = self.cached_form_assemblers['bilin.df1_dv0_adj'].assemble()
        dfu2_da1 = self.cached_form_assemblers['bilin.df1_da0_adj'].assemble()

        dfv2_du1 = 0 - newmark.newmark_v_du0(dt)
        dfv2_dv1 = 0 - newmark.newmark_v_dv0(dt)
        dfv2_da1 = 0 - newmark.newmark_v_da0(dt)

        dfa2_du1 = 0 - newmark.newmark_a_du0(dt)
        dfa2_dv1 = 0 - newmark.newmark_a_dv0(dt)
        dfa2_da1 = 0 - newmark.newmark_a_da0(dt)

        ## Do the matrix vector multiplication that gets the RHS for the adjoint equations
        # Allocate a vector the for fluid side mat-vec multiplication
        x = b.copy()
        x['u'][:] = (dfu2_du1*b['u'] + dfv2_du1*b['v'] + dfa2_du1*b['a'])
        x['v'][:] = (dfu2_dv1*b['u'] + dfv2_dv1*b['v'] + dfa2_dv1*b['a'])
        x['a'][:] = (dfu2_da1*b['u'] + dfv2_da1*b['v'] + dfa2_da1*b['a'])
        return x

    def apply_dres_dcontrol(self, x):
        raise NotImplementedError

    def apply_dres_dcontrol_adj(self, x):
        raise NotImplementedError
    
    def apply_dres_dp(self, x):
        raise NotImplementedError

    def apply_dres_dp_adj(self, x):
        b = self.get_properties_vec(set_default=False)
        for prop_name, vec in zip(b.keys, b.vecs):
            # assert self.df1_dsolid[key] is not None
            df1_dprop = None
            if self.df1_dsolid[prop_name] is None:
                df1_dprop = 0.0
            else:
                df1_dprop = dfn.assemble(self.df1_dsolid[prop_name])
            val = df1_dprop*x['u']

            # Note this is a workaround because some properties are scalar values but stored as 
            # vectors in order to take their derivatives. This is the case for time step, `dt`
            if vec.size == 1:
                val = val.sum()
                
            vec[:] = val
        return b

    def apply_dres_ddt(self, x):
        dfu_ddt = dfn.assemble(self.forms['form.bi.df1_ddt'], tensor=dfn.PETScMatrix())
        dfv_ddt = 0 - newmark.newmark_v_dt(self.state1[0], *self.state0.vecs, self.dt)
        dfa_ddt = 0 - newmark.newmark_a_dt(self.state1[0], *self.state0.vecs, self.dt)

        ddt = x
        ddt_vec = dfn.PETScVector(dfu_ddt.mat().getVecRight())
        ddt_vec[:] = ddt
        self.bc_base.zero(dfu_ddt)

        dres = self.get_state_vec()
        dres['u'][:] = dfu_ddt*ddt_vec 
        dres['v'][:] = dfv_ddt*ddt
        dres['a'][:] = dfa_ddt*ddt
        return dres

    def apply_dres_ddt_adj(self, b):
        # Note that dfu_ddt is a matrix because fenics doesn't allow taking derivative w.r.t a scalar
        # As a result, the time step is defined for each vertex. This is why 'ddt' is computed weirdly
        # below
        dfu_ddt = dfn.assemble(self.forms['form.bi.df1_ddt_adj'], tensor=dfn.PETScMatrix())
        dfv_ddt = 0 - newmark.newmark_v_dt(self.state1[0], *self.state0.vecs, self.dt)
        dfa_ddt = 0 - newmark.newmark_a_dt(self.state1[0], *self.state0.vecs, self.dt)

        bu, bv, ba = b
        ddt = (dfu_ddt*bu).sum() + dfv_ddt.inner(bv) + dfa_ddt.inner(ba)
        return ddt


class NodalContactSolid(Solid):
    """
    This class modifies the default behaviour of the solid to implement contact pressures
    interpolated with the displacement function space. This involves manual modification of the 
    matrices generated by Fenics.
    """
    def set_fin_state(self, state):
        # This sets the 'standard' state variables u/v/a
        super().set_fin_state(state)

        # This updates nodal values of an additional contact pressure
        XREF = self.XREF.vector()
        kcontact = self.forms['coeff.prop.kcontact'].values()[0]
        u1 = self.forms['coeff.state.u1'].vector()
        tcon = self.forms['coeff.state.manual.tcontact'].vector()

        ncontact = self.forms['coeff.prop.ncontact'].values()
        ncontact = np.array([0, 1])
        gap = np.dot((XREF+u1)[:].reshape(-1, 2), ncontact) - self.forms['coeff.prop.ycontact'].values()[0]
        tcon[:] = (-form_cubic_penalty_pressure(gap, kcontact)[:, None]*ncontact).reshape(-1).copy()

    def _assem_dres_du(self):
        ## dres_du has two components: one due to the standard u/v/a variables  
        ## and an additional effect due to contact pressure
        dfu2_du2_nocontact = dfn.assemble(self.forms['form.bi.df1_du1'])
        
        # Compute things needed to find sensitivities of contact pressure
        dfu2_dtcontact = dfn.as_backend_type(dfn.assemble(self.forms['form.bi.df1_dtcontact']))
        XREF = self.XREF.vector()
        kcontact = self.forms['coeff.prop.kcontact'].values()[0]
        u1 = self.forms['coeff.state.u1'].vector()

        ncontact = self.forms['coeff.prop.ncontact'].values()
        ncontact = np.array([0, 1])
        gap = np.dot((XREF+u1)[:].reshape(-1, 2), ncontact) - self.forms['coeff.prop.ycontact'].values()[0]
        dgap_du = ncontact

        # FIXME: This code below only works if n is aligned with the x/y axes.
        # for a general collision plane normal, the operation 'df_dtc*dtc_du' will 
        # have to be represented by a block diagonal dtc_du (need to loop in python to do this). It 
        # reduces to a diagonal if n is aligned with a coordinate axis.
        dtcontact_du2 = dfn.Function(self.vector_fspace).vector()
        dpcontact_dgap, _ = dform_cubic_penalty_pressure(gap, kcontact)
        dtcontact_du2[:] = np.array((-dpcontact_dgap[:, None]*dgap_du).reshape(-1))

        dfu2_dtcontact.mat().diagonalScale(None, dtcontact_du2.vec())
        dfu2_du2_contact = dfu2_dtcontact
        dfu2_du2 = dfu2_du2_nocontact + dfu2_du2_contact
        return dfu2_du2

    def _assem_dres_du_adj(self):
        ## dres_du has two components: one due to the standard u/v/a variables  
        ## and an additional effect due to contact pressure
        dfu2_du2_nocontact = dfn.assemble(self.forms['form.bi.df1_du1_adj'])
        
        # Compute things needed to find sensitivities of contact pressure
        dfu2_dtcontact = dfn.as_backend_type(dfn.assemble(self.forms['form.bi.df1_dtcontact_adj']))
        XREF = self.XREF.vector()
        kcontact = self.forms['coeff.prop.kcontact'].values()[0]
        u1 = self.forms['coeff.state.u1'].vector()

        ncontact = self.forms['coeff.prop.ncontact'].values()
        ncontact = np.array([0, 1])
        gap = np.dot((XREF+u1)[:].reshape(-1, 2), ncontact) - self.forms['coeff.prop.ycontact'].values()[0]
        dgap_du = ncontact

        # FIXME: This code below only works if n is aligned with the x/y axes.
        # for a general collision plane normal, the operation 'df_dtc*dtc_du' will 
        # have to be represented by a block diagonal dtc_du (need to loop). It reduces to a diagonal 
        # if n is aligned with a coordinate axis.
        dtcontact_du2 = dfn.Function(self.vector_fspace).vector()
        dpcontact_dgap, _ = dform_cubic_penalty_pressure(gap, kcontact)
        dtcontact_du2[:] = np.array((-dpcontact_dgap[:, None]*dgap_du).reshape(-1))

        dfu2_dtcontact.mat().diagonalScale(dtcontact_du2.vec(), None)
        dfu2_du2_contact = dfu2_dtcontact
        dfu2_du2 = dfu2_du2_nocontact + dfu2_du2_contact
        return dfu2_du2

    # TODO: refactor this copy-paste
    def _assem_dresuva_du(self):
        ## dres_du has two components: one due to the standard u/v/a variables  
        ## and an additional effect due to contact pressure
        dfu2_du2_nocontact = dfn.assemble(self.forms['form.bi.df1uva_du1'])
        
        # Compute things needed to find sensitivities of contact pressure
        dfu2_dtcontact = dfn.as_backend_type(dfn.assemble(self.forms['form.bi.df1uva_dtcontact']))
        XREF = self.XREF.vector()
        kcontact = self.forms['coeff.prop.kcontact'].values()[0]
        u1 = self.forms['coeff.state.u1'].vector()

        ncontact = self.forms['coeff.prop.ncontact'].values()
        ncontact = np.array([0, 1])
        gap = np.dot((XREF+u1)[:].reshape(-1, 2), ncontact) - self.forms['coeff.prop.ycontact'].values()[0]
        dgap_du = ncontact

        # FIXME: This code below only works if n is aligned with the x/y axes.
        # for a general collision plane normal, the operation 'df_dtc*dtc_du' will 
        # have to be represented by a block diagonal dtc_du (need to loop in python to do this). It 
        # reduces to a diagonal if n is aligned with a coordinate axis.
        dtcontact_du2 = dfn.Function(self.vector_fspace).vector()
        dpcontact_dgap, _ = dform_cubic_penalty_pressure(gap, kcontact)
        dtcontact_du2[:] = np.array((-dpcontact_dgap[:, None]*dgap_du).reshape(-1))

        dfu2_dtcontact.mat().diagonalScale(None, dtcontact_du2.vec())
        dfu2_du2_contact = dfu2_dtcontact
        dfu2_du2 = dfu2_du2_nocontact + dfu2_du2_contact
        return dfu2_du2


    def solve_dres_dstate1(self, b):
        dt = self.dt
        # Contact will change this matrix!
        dfu2_du2 = self._assem_dres_du()

        dfv2_du2 = 0 - newmark.newmark_v_du1(dt)
        dfa2_du2 = 0 - newmark.newmark_a_du1(dt)

        # Solve A x = b
        bu, bv, ba = b.vecs
        x = self.get_state_vec()

        self.bc_base.apply(dfu2_du2)
        dfn.solve(dfu2_du2, x['u'], bu, 'petsc')

        x['v'][:] = bv - dfv2_du2*x['u']
        x['a'][:] = ba - dfa2_du2*x['u']
        
        return x
        
    def solve_dres_dstate1_adj(self, x):
        # Form key matrices
        dfu2_du2 = self.cached_form_assemblers['bilin.df1_du1_adj'].assemble()
        dfv2_du2 = 0 - newmark.newmark_v_du1(self.dt)
        dfa2_du2 = 0 - newmark.newmark_a_du1(self.dt)

        # Solve b^T A = x^T
        xu, xv, xa = x.vecs
        b = self.get_state_vec()
        b['a'][:] = xa
        b['v'][:] = xv

        rhs_u = xu - (dfv2_du2*b['v'] + dfa2_du2*b['a'])

        self.bc_base.apply(dfu2_du2, rhs_u)
        dfn.solve(dfu2_du2, b['u'], rhs_u, 'petsc')
        return b


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

class Rayleigh(NodalContactSolid):
    """
    Represents the governing equations of Rayleigh damped solid
    """
    PROPERTY_DEFAULTS = {
        'emod': 10e3 * PASCAL_TO_CGS,
        'nu': 0.49,
        'rho': 1000 * SI_DENSITY_TO_CGS,
        'rayleigh_m': 10,
        'rayleigh_k': 1e-3,
        'ycontact': 0.61-0.001,
        'kcontact': 1e11}

    @staticmethod
    def form_definitions(mesh, facet_func, facet_label_to_id, cell_func, cell_label_to_id,
                          fsi_facet_labels, fixed_facet_labels):
        return \
            add_newmark_time_disc_form(
            add_manual_contact_traction_form(
            add_surface_pressure_form(
            add_rayleigh_viscous_form(
            add_inertial_form(
            add_isotropic_elastic_form(
            base_form_definitions(
                mesh, facet_func, facet_label_to_id, cell_func, cell_label_to_id,
                fsi_facet_labels, fixed_facet_labels)))))))


def add_kv_viscous_form(forms):
    dx = forms['measure.dx']
    strain_test = forms['test.strain']
    v = forms['coeff.state.v1']

    eta = dfn.Function(forms['fspace.scalar'])
    inf_strain_rate = form_inf_strain(v)
    stress_visco = eta*inf_strain_rate

    forms['form.un.f1uva'] += ufl.inner(stress_visco, strain_test) * dx
    forms['coeff.prop.eta'] = eta
    return forms

class KelvinVoigt(NodalContactSolid):
    """
    Represents the governing equations of a Kelvin-Voigt damped solid
    """
    PROPERTY_DEFAULTS = {
        'emod': 10e3 * PASCAL_TO_CGS,
        'nu': 0.49,
        'rho': 1000 * SI_DENSITY_TO_CGS,
        'eta': 3.0,
        'ycontact': 0.61-0.001,
        'kcontact': 1e11}

    @staticmethod
    def form_definitions(mesh, facet_func, facet_label_to_id, cell_func, cell_label_to_id,
                          fsi_facet_labels, fixed_facet_labels):
        return \
            add_newmark_time_disc_form(
            add_manual_contact_traction_form(
            add_surface_pressure_form(
            add_kv_viscous_form(
            add_inertial_form(
            add_isotropic_elastic_form(
            base_form_definitions(
                mesh, facet_func, facet_label_to_id, cell_func, cell_label_to_id,
                fsi_facet_labels, fixed_facet_labels)))))))


def add_incompressible_isotropic_elastic_form(forms):
    dx = forms['measure.dx']
    strain_test = forms['test.strain']

    emod = dfn.Function(forms['fspace.scalar'])
    nu = 0.5
    u = forms['coeff.state.u1']
    inf_strain = form_inf_strain(u)
    v_swelling = dfn.Function(forms['fspace.scalar'])
    k_swelling = dfn.Constant(1.0)
    v_swelling.vector()[:] = 1.0
    lame_mu = emod/2/(1+nu)
    stress_elastic = 2*lame_mu*inf_strain + k_swelling*(ufl.tr(inf_strain)-(v_swelling-1.0))*ufl.Identity(u.geometric_dimension())

    forms['form.un.f1uva'] += ufl.inner(stress_elastic, strain_test) * dx
    forms['coeff.prop.emod'] = emod
    forms['coeff.prop.v_swelling'] = v_swelling
    forms['coeff.prop.k_swelling'] = k_swelling
    forms['expr.stress_elastic'] = stress_elastic
    return forms

class IncompSwellingKelvinVoigt(NodalContactSolid):
    """
    Kelvin Voigt model allowing for a swelling field
    """
    PROPERTY_DEFAULTS = {
        'emod': 10e3 * PASCAL_TO_CGS,
        'v_swelling': 1.0,
        'k_swelling': 1000.0 * 10e3 * PASCAL_TO_CGS,
        'rho': 1000 * SI_DENSITY_TO_CGS,
        'eta': 3.0,
        'ycontact': 0.61-0.001,
        'kcontact': 1e11}

    @staticmethod
    def form_definitions(mesh, facet_func, facet_label_to_id, cell_func, cell_label_to_id,
                          fsi_facet_labels, fixed_facet_labels):
        return \
            add_newmark_time_disc_form(
            add_manual_contact_traction_form(
            add_surface_pressure_form(
            add_kv_viscous_form(
            add_inertial_form(
            add_incompressible_isotropic_elastic_form(
            base_form_definitions(
                mesh, facet_func, facet_label_to_id, cell_func, cell_label_to_id,
                fsi_facet_labels, fixed_facet_labels)))))))


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

    forms['form.un.f1uva'] += stiffness + viscous
    forms['coeff.prop.length'] = length
    forms['coeff.prop.muscle_stress'] = muscle_stress
    return forms

class Approximate3DKelvinVoigt(NodalContactSolid):
    PROPERTY_DEFAULTS = {
        'emod': 10e3 * PASCAL_TO_CGS,
        'nu': 0.49,
        'rho': 1000 * SI_DENSITY_TO_CGS,
        'eta': 3.0,
        'ycontact': 0.61-0.001,
        'kcontact': 1e11}

    @staticmethod
    def form_definitions(mesh, facet_func, facet_label_to_id, cell_func, cell_label_to_id,
                         fsi_facet_labels, fixed_facet_labels):
        return \
            add_newmark_time_disc_form(
            add_manual_contact_traction_form(
            add_surface_pressure_form(
            add_ap_force_form(
            add_kv_viscous_form(
            add_inertial_form(
            add_isotropic_elastic_form(
            base_form_definitions(
                mesh, facet_func, facet_label_to_id, cell_func, cell_label_to_id,
                fsi_facet_labels, fixed_facet_labels))))))))


class CachedBiFormAssembler:
    """
    Assembles a bilinear form using a cached sparsity pattern

    Parameters
    ----------
    form : ufl.Form
    keep_diagonal : bool, optional
        Whether to preserve diagonals in the form
    """

    def __init__(self, form, keep_diagonal=True):
        self._form = form

        self._tensor = dfn.assemble(form, keep_diagonal=keep_diagonal, tensor=dfn.PETScMatrix())
        self._tensor.zero()

    @property
    def tensor(self):
        return self._tensor

    @property
    def form(self):
        return self._form

    def assemble(self):
        out = self.tensor.copy()
        return dfn.assemble(self.form, tensor=out)

def newton_solve(x0, linearized_subproblem, params=None):
    """
    Solve a non-linear problem with Newton-Raphson

    Parameters
    ----------
    x0 : A
        Initial guess
    linearized_subproblem : fn(A) -> (fn() -> A, fn(A) -> A)
        Callable returning a residual and linear solver about a state (x).
    params : dict
        Dictionary of parameters

    Returns
    -------
    xn
    """
    if params is None:
        params = DEFAULT_NEWTON_SOLVER_PRM
    n = 0
    state_n = x0
    assem_res_n, solve_n = linearized_subproblem(state_n)

    max_iter = params['maximum_iterations']
    
    abs_err_0 = 1.0
    abs_tol, rel_tol = params['absolute_tolerance'], params['relative_tolerance']
    while True:
        assem_res_n, solve_n = linearized_subproblem(state_n)
        res_n = assem_res_n()

        res_n['u'].norm('l2')
        res_n['v'].norm('l2')
        res_n['a'].norm('l2')
        abs_err = abs(res_n.norm())
        if n == 0:
            abs_err_0 = max(abs_err, 1.0)
        rel_err = abs_err/abs_err_0

        if abs_err <= abs_tol or rel_err <= rel_tol or n > max_iter:
            break
        else:
            dstate_n = solve_n(res_n)
            state_n = state_n - dstate_n
            n += 1
    return state_n, {'numiter': n, 'abserr': abs_err, 'relerr': rel_err}
