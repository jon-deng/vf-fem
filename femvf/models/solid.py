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
    # Derivatives of the displacement residual form wrt variables of interest
    for full_var_name in (
        [f'coeff.state.{y}' for y in ['u0', 'v0', 'a0', 'u1']] + 
        ['coeff.time.dt', 'coeff.fsi.p1']):
        f = forms['form.un.f1']
        x = forms[full_var_name]

        var_name = full_var_name.split(".")[-1]
        form_name = f'form.bi.df1_d{var_name}'
        forms[form_name] = dfn.derivative(f, x)
        forms[f'{form_name}_adj'] = dfn.adjoint(forms[form_name])

    # Derivatives of the u/v/a residual form wrt variables of interest
    for full_var_name in (
        [f'coeff.state.{y}' for y in ['u1', 'v1', 'a1']] + 
        ['coeff.fsi.p1']):
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
        self.set_properties(self.get_properties_vec(set_default=True))

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
        xref = dfn.Function(model.vector_fspace)
        xref.vector()[:] = model.vector_fspace.tabulate_dof_coordinates()
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
        return {}

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
                vec = np.ones(1)
            else:
                vec = coefficient.vector().copy()
            
            if set_default:
                vec[:] = self.PROPERTY_DEFAULTS.get(label, 0.0)

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
                coefficient.assign(props[key][()])
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

        n = 0
        state_n = state1
        assem_res_n, solve_n = linearized_subproblem(state_n)
    
        max_iter = newton_solver_prm['maximum_iterations']
        
        abs_err_0 = 1.0
        abs_tol, rel_tol = newton_solver_prm['absolute_tolerance'], newton_solver_prm['relative_tolerance']

        while True:
            assem_res_n, solve_n = linearized_subproblem(state_n)
            res_n = assem_res_n()

            abs_err = abs(res_n.norm())
            if n == 0:
                abs_err_0 = abs_err
            rel_err = abs_err/abs_err_0

            # breakpoint()
            if abs_err <= abs_tol or rel_err <= rel_tol or n > max_iter:
                break
            else:
                dstate_n = solve_n(res_n)
                state_n = state_n - dstate_n
                n += 1
                
        return state_n, {}

    def _solve_state1(self, state1, newton_solver_prm=None):
        if newton_solver_prm is None:
            newton_solver_prm = DEFAULT_NEWTON_SOLVER_PRM
        dt = self.dt

        uva1 = self.get_state_vec()
        self.set_fin_state(state1)
        dfn.solve(self.f1 == 0, self.u1, bcs=self.bc_base, J=self.df1_du1,
                  solver_parameters={"newton_solver": newton_solver_prm})

        uva1['u'][:] = self.u1.vector()
        uva1['v'][:] = newmark.newmark_v(uva1['u'], *self.state0.vecs, dt)
        uva1['a'][:] = newmark.newmark_a(uva1['u'], *self.state0.vecs, dt)
        return uva1, {}

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

class Rayleigh(Solid):
    """
    Represents the governing equations of Rayleigh damped solid
    """
    PROPERTY_DEFAULTS = {
        'emod': 10e3 * PASCAL_TO_CGS,
        'nu': 0.49,
        'rho': 1000 * SI_DENSITY_TO_CGS,
        'rayleigh_m': 10,
        'rayleigh_k': 1e-3,
        'y_collision': 0.61-0.001,
        'k_collision': 1e11}

    @staticmethod
    def form_definitions(mesh, facet_func, facet_label_to_id, cell_func, cell_label_to_id, 
                         fsi_facet_labels, fixed_facet_labels):
        dx = dfn.Measure('dx', domain=mesh, subdomain_data=cell_func)
        ds = dfn.Measure('ds', domain=mesh, subdomain_data=facet_func)

        scalar_fspace = dfn.FunctionSpace(mesh, 'CG', 1)
        vector_fspace = dfn.VectorFunctionSpace(mesh, 'CG', 1)

        vector_trial = dfn.TrialFunction(vector_fspace)
        vector_test = dfn.TestFunction(vector_fspace)
        strain_test = form_inf_strain(vector_test)

        scalar_trial = dfn.TrialFunction(scalar_fspace)
        scalar_test = dfn.TestFunction(scalar_fspace)

        vert_to_vdof = dfn.vertex_to_dof_map(vector_fspace)
        XREF = dfn.Function(vector_fspace)
        XREF.vector()[vert_to_vdof.reshape(-1)] = mesh.coordinates().reshape(-1)

        # Newmark update parameters
        gamma = dfn.Constant(1/2)
        beta = dfn.Constant(1/4)

        # Solid material properties
        y_collision = dfn.Constant(1.0)
        k_collision = dfn.Constant(1.0)
        rho = dfn.Constant(1.0)
        nu = dfn.Constant(1.0)
        rayleigh_m = dfn.Constant(1.0)
        rayleigh_k = dfn.Constant(1.0)
        emod = dfn.Function(scalar_fspace)

        emod.vector()[:] = 1.0

        # NOTE: Fenics doesn't support form derivatives w.r.t Constant. By making time step
        # vary in space, you can take the derivative. As a result you have to set the time step 
        # the same at every DOF
        dt = dfn.Function(scalar_fspace)
        dt.vector()[:] = 1e-4

        # Initial and final states
        # u: displacement, v: velocity, a: acceleration
        u0 = dfn.Function(vector_fspace)
        v0 = dfn.Function(vector_fspace)
        a0 = dfn.Function(vector_fspace)

        u1 = dfn.Function(vector_fspace)
        v1 = dfn.Function(vector_fspace)
        a1 = dfn.Function(vector_fspace)

        v1_nmk = newmark.newmark_v(u1, u0, v0, a0, dt, gamma, beta)
        a1_nmk = newmark.newmark_a(u1, u0, v0, a0, dt, gamma, beta)

        # Surface pressure
        pcontact = dfn.Function(scalar_fspace)
        p1 = dfn.Function(scalar_fspace)

        ## Symbolic calculations to get the variational form for a linear-elastic solid
        inf_strain = form_inf_strain(u1)
        force_inertial = rho*a1 
        stress_elastic = form_lin_iso_cauchy_stress(inf_strain, emod, nu)
        force_visco = rayleigh_m * ufl.replace(force_inertial, {a1: v1})
        stress_visco = rayleigh_k * ufl.replace(stress_elastic, {u1: v1})

        inertia = ufl.inner(force_inertial, vector_test) * dx
        stiffness = ufl.inner(stress_elastic, strain_test) * dx
        damping = (ufl.inner(force_visco, vector_test) + ufl.inner(stress_visco, strain_test))*dx

        # Compute the pressure loading Neumann boundary condition on the reference configuration
        # using Nanson's formula. This is because the 'total lagrangian' formulation is used.
        facet_normal = dfn.FacetNormal(mesh)
        traction_dss = [ds(facet_label_to_id[facet_label]) for facet_label in fsi_facet_labels]
        traction_ds = sum(traction_dss[1:], traction_dss[0])
        reference_traction = form_pressure_as_reference_traction(p1, u1, facet_normal)

        traction = ufl.inner(reference_traction, vector_test) * traction_ds

        # Use the penalty method to account for collision
        ncoll = dfn.Constant([0.0, 1.0])
        gap = ufl.dot(XREF+u1, ncoll) - y_collision
        contact_pressure = form_cubic_penalty_pressure(gap, k_collision)
        penalty = ufl.inner(-contact_pressure*ncoll, vector_test)*traction_ds

        f1_uva = inertia + stiffness + damping - traction - penalty

        f1 = ufl.replace(f1_uva, {v1: v1_nmk, a1:a1_nmk})

        ## Boundary conditions
        # Specify DirichletBC at the VF base
        bc_base = dfn.DirichletBC(vector_fspace, dfn.Constant([0.0, 0.0]),
                                  facet_func, facet_label_to_id['fixed'])

        forms = {
            'measure.dx': dx,
            'measure.ds': ds,
            'bcs.base': bc_base,

            'fspace.vector': vector_fspace,
            'fspace.scalar': scalar_fspace,

            'test.vector': vector_test,
            'test.scalar': scalar_test,
            'trial.vector': vector_trial,
            'trial.scalar': scalar_trial,

            'coeff.time.dt': dt,
            'coeff.time.gamma': gamma,
            'coeff.time.beta': beta,

            'coeff.state.u0': u0,
            'coeff.state.v0': v0,
            'coeff.state.a0': a0,
            'coeff.state.u1': u1,
            'coeff.state.v1': v1,
            'coeff.state.a1': a1,
            
            'coeff._state.pcontact': pcontact,
            'coeff.fsi.p1': p1,

            'coeff.prop.rho': rho,
            'coeff.prop.nu': nu,
            'coeff.prop.emod': emod,
            'coeff.prop.rayleigh_m': rayleigh_m,
            'coeff.prop.rayleigh_k': rayleigh_k,
            'coeff.prop.y_collision': y_collision,
            'coeff.prop.k_collision': k_collision,

            'expr.contact_pressure': contact_pressure,

            'form.un.f1uva': f1_uva,
            'form.un.f1': f1}

        return forms

class KelvinVoigt(Solid):
    """
    Represents the governing equations of a Kelvin-Voigt damped solid
    """
    PROPERTY_DEFAULTS = {
        'emod': 10e3 * PASCAL_TO_CGS,
        'nu': 0.49,
        'rho': 1000 * SI_DENSITY_TO_CGS,
        'eta': 3.0,
        'y_collision': 0.61-0.001,
        'k_collision': 1e11}

    @staticmethod
    def form_definitions(mesh, facet_func, facet_label_to_id, cell_func, cell_label_to_id,
                         fsi_facet_labels, fixed_facet_labels):
        dx = dfn.Measure('dx', domain=mesh, subdomain_data=cell_func)
        ds = dfn.Measure('ds', domain=mesh, subdomain_data=facet_func)

        scalar_fspace = dfn.FunctionSpace(mesh, 'CG', 1)
        vector_fspace = dfn.VectorFunctionSpace(mesh, 'CG', 1)

        vector_trial = dfn.TrialFunction(vector_fspace)
        vector_test = dfn.TestFunction(vector_fspace)

        scalar_trial = dfn.TrialFunction(scalar_fspace)
        scalar_test = dfn.TestFunction(scalar_fspace)
        strain_test = form_inf_strain(vector_test)

        vert_to_vdof = dfn.vertex_to_dof_map(vector_fspace)
        XREF = dfn.Function(vector_fspace)
        XREF.vector()[vert_to_vdof.reshape(-1)] = mesh.coordinates().reshape(-1)

        # Newmark update parameters
        gamma = dfn.Constant(1/2)
        beta = dfn.Constant(1/4)

        # Solid material properties
        y_collision = dfn.Constant(1.0)
        k_collision = dfn.Constant(1.0)
        rho = dfn.Constant(1.0)
        nu = dfn.Constant(1.0)
        emod = dfn.Function(scalar_fspace)
        kv_eta = dfn.Function(scalar_fspace)

        emod.vector()[:] = 1.0

        # Initial and final states
        dt = dfn.Function(scalar_fspace)
        dt.vector()[:] = 1e-4

        u0 = dfn.Function(vector_fspace)
        v0 = dfn.Function(vector_fspace)
        a0 = dfn.Function(vector_fspace)

        u1 = dfn.Function(vector_fspace)
        v1 = dfn.Function(vector_fspace)
        a1 = dfn.Function(vector_fspace)

        v1_nmk = newmark.newmark_v(u1, u0, v0, a0, dt, gamma, beta)
        a1_nmk = newmark.newmark_a(u1, u0, v0, a0, dt, gamma, beta)

        # Surface pressures
        pcontact = dfn.Function(scalar_fspace)
        p1 = dfn.Function(scalar_fspace)

        # Symbolic calculations to get the variational form for a linear-elastic solid
        inf_strain = form_inf_strain(u1)
        force_inertial = rho*a1 
        stress_elastic = form_lin_iso_cauchy_stress(inf_strain, emod, nu)
        stress_visco = kv_eta*form_inf_strain(v1)

        inertia = ufl.inner(force_inertial, vector_test) * dx
        stiffness = ufl.inner(stress_elastic, strain_test) * dx
        damping = ufl.inner(stress_visco, strain_test) * dx

        # Compute the pressure loading using Neumann boundary conditions on the reference configuration
        # using Nanson's formula. This is because the 'total lagrangian' formulation is used.
        facet_normal = dfn.FacetNormal(mesh)
        traction_dss = [ds(facet_label_to_id[facet_label]) for facet_label in fsi_facet_labels]
        traction_ds = sum(traction_dss[1:], traction_dss[0])
        reference_traction = form_pressure_as_reference_traction(p1, u1, facet_normal)

        traction = ufl.inner(reference_traction, vector_test) * traction_ds

        # Use the penalty method to account for collision
        ncoll = dfn.Constant([0.0, 1.0])
        gap = ufl.dot(XREF+u1, ncoll) - y_collision
        contact_pressure = form_cubic_penalty_pressure(gap, k_collision)
        penalty = ufl.inner(-contact_pressure*ncoll, vector_test) * traction_ds

        f1_uva = inertia + stiffness + damping - traction - penalty
        f1 = ufl.replace(f1_uva, {v1: v1_nmk, a1: a1_nmk})

        ## Boundary conditions
        # Specify DirichletBC at the VF base
        bc_base = dfn.DirichletBC(vector_fspace, dfn.Constant([0.0, 0.0]),
                                  facet_func, facet_label_to_id['fixed'])

        forms = {
            'measure.dx': dx,
            'measure.ds': ds,
            'bcs.base': bc_base,

            'fspace.vector': vector_fspace,
            'fspace.scalar': scalar_fspace,

            'test.vector': vector_test,
            'test.scalar': scalar_test,
            'trial.vector': vector_trial,
            'trial.scalar': scalar_trial,

            'coeff.time.dt': dt,
            'coeff.time.gamma': gamma,
            'coeff.time.beta': beta,

            'coeff.state.u0': u0,
            'coeff.state.v0': v0,
            'coeff.state.a0': a0,
            'coeff.state.u1': u1,
            'coeff.state.v1': v1,
            'coeff.state.a1': a1,

            'coeff._state.pcontact': pcontact,
            'coeff.fsi.p1': p1,

            'coeff.prop.rho': rho,
            'coeff.prop.eta': kv_eta,
            'coeff.prop.emod': emod,
            'coeff.prop.nu': nu,
            'coeff.prop.y_collision': y_collision,
            'coeff.prop.k_collision': k_collision,

            'expr.contact_pressure': contact_pressure,

            'form.un.f1uva': f1_uva,
            'form.un.f1': f1}
        return forms

class IncompSwellingKelvinVoigt(Solid):
    """
    Kelvin Voigt model allowing for a swelling field
    """
    PROPERTY_DEFAULTS = {
        'emod': 10e3 * PASCAL_TO_CGS,
        'v_swelling': 1.0,
        'k_swelling': 1000.0 * 10e3 * PASCAL_TO_CGS,
        'rho': 1000 * SI_DENSITY_TO_CGS,
        'eta': 3.0,
        'y_collision': 0.61-0.001,
        'k_collision': 1e11}

    @staticmethod
    def form_definitions(mesh, facet_func, facet_label_to_id, cell_func, cell_label_to_id,
                         fsi_facet_labels, fixed_facet_labels):
        dx = dfn.Measure('dx', domain=mesh, subdomain_data=cell_func)
        ds = dfn.Measure('ds', domain=mesh, subdomain_data=facet_func)

        scalar_fspace = dfn.FunctionSpace(mesh, 'CG', 1)
        vector_fspace = dfn.VectorFunctionSpace(mesh, 'CG', 1)

        vector_trial = dfn.TrialFunction(vector_fspace)
        vector_test = dfn.TestFunction(vector_fspace)

        scalar_trial = dfn.TrialFunction(scalar_fspace)
        scalar_test = dfn.TestFunction(scalar_fspace)
        strain_test = form_inf_strain(vector_test)

        vert_to_vdof = dfn.vertex_to_dof_map(vector_fspace)
        XREF = dfn.Function(vector_fspace)
        XREF.vector()[vert_to_vdof.reshape(-1)] = mesh.coordinates().reshape(-1)

        # Newmark update parameters
        gamma = dfn.Constant(1/2)
        beta = dfn.Constant(1/4)

        # Solid material properties
        v_swelling = dfn.Function(scalar_fspace)
        k_swelling = dfn.Constant(1.0)
        y_collision = dfn.Constant(1.0)
        k_collision = dfn.Constant(1.0)
        rho = dfn.Function(scalar_fspace)
        # nu = dfn.Constant(1.0)
        emod = dfn.Function(scalar_fspace)
        kv_eta = dfn.Function(scalar_fspace)

        rho.vector()[:] = 1.0
        v_swelling.vector()[:] = 1.0
        emod.vector()[:] = 1.0

        # Initial and final states
        dt = dfn.Function(scalar_fspace)
        dt.vector()[:] = 1e-4

        u0 = dfn.Function(vector_fspace)
        v0 = dfn.Function(vector_fspace)
        a0 = dfn.Function(vector_fspace)

        u1 = dfn.Function(vector_fspace)
        v1 = dfn.Function(vector_fspace)
        a1 = dfn.Function(vector_fspace)

        v1_nmk = newmark.newmark_v(u1, u0, v0, a0, dt, gamma, beta)
        a1_nmk = newmark.newmark_a(u1, u0, v0, a0, dt, gamma, beta)

        # Surface pressures
        p1 = dfn.Function(scalar_fspace)

        # Symbolic calculations to get the variational form for a linear-elastic solid
        inf_strain = form_inf_strain(u1)
        force_inertial = rho*a1 

        lame_mu = emod/2/(1+0.5) # This is a poisson's ratio of 0.5
        stress_elastic = 2*lame_mu*inf_strain + k_swelling*(ufl.tr(inf_strain)-(v_swelling-1.0))*ufl.Identity(u1.geometric_dimension())
        stress_visco = kv_eta*form_inf_strain(v1)

        inertia = ufl.inner(force_inertial, vector_test) * dx
        stiffness = ufl.inner(stress_elastic, strain_test) * dx
        damping = ufl.inner(stress_visco, strain_test) * dx

        # Compute the pressure loading using Neumann boundary conditions on the reference configuration
        # using Nanson's formula. This is because the 'total lagrangian' formulation is used.
        facet_normal = dfn.FacetNormal(mesh)
        traction_dss = [ds(facet_label_to_id[facet_label]) for facet_label in fsi_facet_labels]
        traction_ds = sum(traction_dss[1:], traction_dss[0])
        reference_traction = form_pressure_as_reference_traction(p1, u1, facet_normal)

        traction = ufl.inner(reference_traction, vector_test) * traction_ds

        # Use the penalty method to account for collision
        ncoll = dfn.Constant([0.0, 1.0])
        gap = ufl.dot(XREF+u1, ncoll) - y_collision
        contact_pressure = form_cubic_penalty_pressure(gap, k_collision)
        penalty = ufl.inner(-contact_pressure*ncoll, vector_test) * traction_ds

        f1_uva = inertia + stiffness + damping - traction - penalty
        f1 = ufl.replace(f1_uva, {v1: v1_nmk, a1: a1_nmk})

        ## Boundary conditions
        # Specify DirichletBC at the VF base
        bc_base = dfn.DirichletBC(vector_fspace, dfn.Constant([0.0, 0.0]),
                                  facet_func, facet_label_to_id['fixed'])

        forms = {
            'measure.dx': dx,
            'measure.ds': ds,
            'bcs.base': bc_base,

            'fspace.vector': vector_fspace,
            'fspace.scalar': scalar_fspace,

            'test.vector': vector_test,
            'test.scalar': scalar_test,
            'trial.vector': vector_trial,
            'trial.scalar': scalar_trial,

            'coeff.time.dt': dt,
            'coeff.time.gamma': gamma,
            'coeff.time.beta': beta,

            'coeff.state.u0': u0,
            'coeff.state.v0': v0,
            'coeff.state.a0': a0,
            'coeff.state.u1': u1,
            'coeff.state.v1': v1,
            'coeff.state.a1': a1,

            'coeff.fsi.p1': p1,

            'coeff.prop.rho': rho,
            'coeff.prop.v_swelling': v_swelling,
            'coeff.prop.k_swelling': k_swelling,
            'coeff.prop.eta': kv_eta,
            'coeff.prop.emod': emod,
            # 'coeff.prop.nu': nu,
            'coeff.prop.y_collision': y_collision,
            'coeff.prop.k_collision': k_collision,

            'expr.contact_pressure': contact_pressure,

            'form.un.f1_uva': f1_uva,
            'form.un.f1': f1}
        return forms

class Approximate3DKelvinVoigt(Solid):
    PROPERTY_DEFAULTS = {
        'emod': 10e3 * PASCAL_TO_CGS,
        'nu': 0.49,
        'rho': 1000 * SI_DENSITY_TO_CGS,
        'eta': 3.0,
        'y_collision': 0.61-0.001,
        'k_collision': 1e11}

    def set_fin_state(self, state):
        super().set_fin_state(state)

        # Update nodal values of contact pressures using the penalty method 
        XREF = self.XREF.vector()
        k_collision = self.forms['coeff.prop.k_collision'].values()[0]
        u1 = self.forms['coeff.state.u1'].vector()
        pcon = self.forms['coeff._state.pcon'].vector()

        ncoll = np.array([0.0, 1.0])
        gap = np.dot(np.array(XREF+u1).reshape(-1, 2), ncoll) - self.forms['coeff.prop.y_collision'].values()[0]
        # gap = (XREF+u1)[1::2] - self.forms['coeff.prop.y_collision'].values()[0]
        pcon[:] = form_cubic_penalty_pressure(gap, k_collision)

    # TODO These three!
    def _assem_dres_du(self):
        dfu2_du2_nocontact = dfn.assemble(self.forms['form.bi.df1_du1'])
        dfu2_dpcontact = dfn.as_backend_type(dfn.assemble(self.forms['form.bi.df1_dpcontact']))

        # Compute things needed to find sensitivities of contact pressure
        XREF = self.XREF.vector()
        k_collision = self.forms['coeff.prop.k_collision'].values()[0]
        u1 = self.forms['coeff.state.u1'].vector()

        ncoll = np.array([0.0, 1.0])
        gap = np.dot(np.array(XREF+u1).reshape(-1, 2), ncoll) - self.forms['coeff.prop.y_collision'].values()[0]
        dgap_du = ncoll[None, :]

        dpcontact_du2 = dfn.Function(self.vector_fspace).vector()
        dpcontact_dgap, _ = dform_cubic_penalty_pressure(gap, k_collision)
        dpcontact_du2[:] = np.array((dpcontact_dgap*dgap_du).reshape(-1))

        dfu2_du2_contact = dfn.PETScMatrix(
            dfu2_dpcontact.mat().diagonalScale(None, dpcontact_du2))
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

    @staticmethod
    def form_definitions(mesh, facet_func, facet_label_to_id, cell_func, cell_label_to_id,
                         fsi_facet_labels, fixed_facet_labels):
        dx = dfn.Measure('dx', domain=mesh, subdomain_data=cell_func)
        ds = dfn.Measure('ds', domain=mesh, subdomain_data=facet_func)

        scalar_fspace = dfn.FunctionSpace(mesh, 'CG', 1)
        vector_fspace = dfn.VectorFunctionSpace(mesh, 'CG', 1)

        vector_trial = dfn.TrialFunction(vector_fspace)
        vector_test = dfn.TestFunction(vector_fspace)

        scalar_trial = dfn.TrialFunction(scalar_fspace)
        scalar_test = dfn.TestFunction(scalar_fspace)
        strain_test = form_inf_strain(vector_test)

        vert_to_vdof = dfn.vertex_to_dof_map(vector_fspace)
        XREF = dfn.Function(vector_fspace)
        XREF.vector()[vert_to_vdof.reshape(-1)] = mesh.coordinates().reshape(-1)

        # Length parameter to approximate 3D effect
        length = dfn.Function(scalar_fspace)
        length.vector()[:] = 1.0
        muscle_stress = dfn.Function(scalar_fspace)

        # Newmark update parameters
        gamma = dfn.Constant(1/2)
        beta = dfn.Constant(1/4)

        # Solid material properties
        y_collision = dfn.Constant(1.0)
        k_collision = dfn.Constant(1.0)
        rho = dfn.Constant(1.0)
        nu = dfn.Constant(1.0)
        emod = dfn.Function(scalar_fspace)
        kv_eta = dfn.Function(scalar_fspace)

        emod.vector()[:] = 1.0

        # Initial and final states
        dt = dfn.Function(scalar_fspace)
        dt.vector()[:] = 1e-4

        u0 = dfn.Function(vector_fspace)
        v0 = dfn.Function(vector_fspace)
        a0 = dfn.Function(vector_fspace)

        u1 = dfn.Function(vector_fspace)
        v1 = dfn.Function(vector_fspace)
        a1 = dfn.Function(vector_fspace)

        v1_nmk = newmark.newmark_v(u1, u0, v0, a0, dt, gamma, beta)
        a1_nmk = newmark.newmark_a(u1, u0, v0, a0, dt, gamma, beta)

        # Surface pressures
        p1 = dfn.Function(scalar_fspace)
        pcoll = dfn.Function(scalar_fspace)

        # Symbolic calculations to get the variational form for a linear-elastic solid
        inf_strain = form_inf_strain(u1)
        force_inertial = rho*a1 
        stress_elastic = form_lin_iso_cauchy_stress(inf_strain, emod, nu)
        stress_visco = kv_eta*form_inf_strain(v1)

        # Approximate 3D type effects using out-of-plane body forces
        # this is a second order finite difference approximation for displacements
        lame_mu = emod/2/(1+nu)
        u_ant = dfn.Function(vector_fspace) # zero values by default
        u_pos = dfn.Function(vector_fspace)  
        d2u_dz2 = (u_ant - 2*u1 + u_pos) / length**2
        d2v_dz2 = (u_ant - 2*v1 + u_pos) / length**2
        force_elast_ap = (lame_mu+muscle_stress)*d2u_dz2
        force_visco_ap = 0.5*kv_eta*d2v_dz2

        inertia = ufl.inner(force_inertial, vector_test) * dx
        stiffness = (ufl.inner(stress_elastic, strain_test) + 
                     ufl.inner(force_elast_ap, vector_test)) * dx
        damping = (ufl.inner(stress_visco, strain_test) + 
                   ufl.inner(force_visco_ap, vector_test)) * dx
        
        # Compute the pressure loading using Neumann boundary conditions on the reference configuration
        # using Nanson's formula. This is because the 'total lagrangian' formulation is used.
        facet_normal = dfn.FacetNormal(mesh)
        traction_dss = [ds(facet_label_to_id[facet_label]) for facet_label in fsi_facet_labels]
        traction_ds = sum(traction_dss[1:], traction_dss[0])
        reference_traction = form_pressure_as_reference_traction(p1, u1, facet_normal)

        traction = ufl.inner(reference_traction, vector_test) * traction_ds

        f1_uva = inertia + stiffness + damping - traction - penalty
        f1 = ufl.replace(f1_uva, {v1: v1_nmk, a1: a1_nmk})

        ## Boundary conditions
        # Specify DirichletBC at the VF base
        bc_base = dfn.DirichletBC(vector_fspace, dfn.Constant([0.0, 0.0]),
                                  facet_func, facet_label_to_id['fixed'])

        forms = {
            'measure.dx': dx,
            'measure.ds': ds,
            'bcs.base': bc_base,

            'fspace.vector': vector_fspace,
            'fspace.scalar': scalar_fspace,

            'test.vector': vector_test,
            'test.scalar': scalar_test,
            'trial.vector': vector_trial,
            'trial.scalar': scalar_trial,

            'coeff.time.dt': dt,
            'coeff.time.gamma': gamma,
            'coeff.time.beta': beta,

            'coeff.state.u0': u0,
            'coeff.state.v0': v0,
            'coeff.state.a0': a0,
            'coeff.state.u1': u1,
            'coeff.state.v1': v1,
            'coeff.state.a1': a1,

            'coeff._state.pcontact': pcoll,
            'coeff.fsi.p1': p1,

            'coeff.prop.rho': rho,
            'coeff.prop.eta': kv_eta,
            'coeff.prop.emod': emod,
            'coeff.prop.nu': nu,
            'coeff.prop.y_collision': y_collision,
            'coeff.prop.k_collision': k_collision,
            'coeff.prop.length': length,
            'coeff.prop.muscle_stress': muscle_stress,

            'expr.contact_pressure': contact_pressure,
            'expr.stress_elastic': stress_elastic,
            'expr.stress_visco': stress_visco,

            'form.un.f1uva': f1_uva,
            'form.un.f1': f1}
        return forms


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

def newton_solve(x0, linearized_subproblem, params):
    pass
