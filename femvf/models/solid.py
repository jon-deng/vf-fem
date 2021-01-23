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


def cauchy_stress(u, emod, nu):
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

    return 2*lame_mu*strain(u) + lame_lambda*ufl.tr(strain(u))*ufl.Identity(u.geometric_dimension())

def strain(u):
    """
    Returns the strain tensor for a displacement field.

    Parameters
    ----------
    u : dfn.TrialFunction, ufl.Argument
        Trial displacement field
    """
    return 1/2 * (ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

def biform_k(trial, test, emod, nu):
    """
    Return stiffness bilinear form

    Integrates linear_elasticity(a) with the strain(b)
    """
    return ufl.inner(cauchy_stress(trial, emod, nu), strain(test))*ufl.dx

def biform_m(trial, test, rho):
    """
    Return the mass bilinear form

    Integrates a with b
    """
    return rho*ufl.dot(trial, test) * ufl.dx

def inertia_2form(trial, test, rho):
    return biform_m(trial, test, rho)

def stiffness_2form(trial, test, emod, nu):
    return biform_k(trial, test, emod, nu)

def traction_1form(trial, test, pressure, facet_normal, traction_ds):
    deformation_gradient = ufl.grad(trial) + ufl.Identity(2)
    deformation_cofactor = ufl.det(deformation_gradient) * ufl.inv(deformation_gradient).T

    fluid_force = -pressure*deformation_cofactor*facet_normal

    traction = ufl.dot(fluid_force, test)*traction_ds
    return traction

# TODO: Remove PROPERTY_TYPES/PROPERTY_DEFAULTS class variables; properties can be defined based on what is exported from the UFL form representation
# The way it is now, you have to write what the properties are twice, and manually check that the names agree

class Solid(base.Model):
    """
    Class representing the discretized governing equations of a solid
    """
    # Subclasses have to set these values
    PROPERTY_TYPES = None
    PROPERTY_DEFAULTS = None

    def __init__(self, mesh, facet_func, facet_labels, cell_func, cell_labels, 
                 fsi_facet_labels, fixed_facet_labels):
        assert isinstance(fsi_facet_labels, (list, tuple))
        assert isinstance(fixed_facet_labels, (list, tuple))

        self._forms = self.form_definitions(mesh, facet_func, facet_labels,
                                            cell_func, cell_labels, fsi_facet_labels, fixed_facet_labels)

        ## Store mesh related quantities
        self.mesh = mesh
        self.facet_func = facet_func
        self.cell_func = cell_func
        self.facet_labels = facet_labels
        self.cell_labels = cell_labels

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
        self.df1_dsolid = get_df1_dsolid_forms(self.f1, self.forms, self.PROPERTY_TYPES.keys(), self.scalar_trial)

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
        self.state1 = BlockVec((self.u0.vector(), self.v0.vector(), self.a0.vector()), ('u', 'v', 'a'))
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

    @staticmethod
    def form_definitions(mesh, facet_func, facet_labels, cell_func, cell_labels, 
                         fsi_facet_labels, fixed_facet_labels):
        """
        Return a dictionary of ufl forms representing the solid in Fenics.

        You have to implement this along with a description of the properties to make a subclass of
        the `Solid`.
        """
        return NotImplementedError("Subclasses must implement this function")

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
        labels = tuple(self.PROPERTY_TYPES.keys())
        vecs = []
        for label in labels:
            coefficient = self.forms['coeff.prop.'+label]

            vec = None
            if isinstance(coefficient, dfn.function.constant.Constant):
                vec = np.ones(1)
            else:
                vec = coefficient.vector().copy()
            
            if set_default:
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

        dfv2_du1 = 0 - newmark.newmark_v_du0(dt)
        dfv2_dv1 = 0 - newmark.newmark_v_dv0(dt)
        dfv2_da1 = 0 - newmark.newmark_v_da0(dt)

        dfa2_du1 = 0 - newmark.newmark_a_du0(dt)
        dfa2_dv1 = 0 - newmark.newmark_a_dv0(dt)
        dfa2_da1 = 0 - newmark.newmark_a_da0(dt)

        ## Do the matrix vector multiplication that gets the RHS for the adjoint equations
        # Allocate a vector the for fluid side mat-vec multiplication
        b = x.copy()
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
        # breakpoint()
        for key, vec in zip(b.keys, b.vecs):
            # assert self.df1_dsolid[key] is not None
            df1_dkey = dfn.assemble(self.df1_dsolid[key])
            val = df1_dkey*x['u']
            if vec.size == 1:
                val = val.sum()
                # Note this is a hack because some properties are scalar values but stored as vectors
                # throughout the domain (specifically, the time step)
            vec[:] = val
        return b

    def apply_dres_ddt(self, x):
        dfu_ddt = dfn.assemble(self.forms['form.bi.df1_dt'], tensor=dfn.PETScMatrix())
        dfv_ddt = 0 - newmark.newmark_v_dt(self.state1[0], *self.state0, self.dt)
        dfa_ddt = 0 - newmark.newmark_a_dt(self.state1[0], *self.state1, self.dt)

        dt = x
        dt_vec = dfn.PETScVector(dfu_ddt.mat().getVecRight())
        dt_vec[:] = dt

        dres = self.get_state_vec()
        dres['u'] = dfu_ddt*dt_vec + dfv_ddt*dt + dfa_ddt*dt
        return dres

    def apply_dres_ddt_adj(self, b):
        # Note that dfu_ddt is a matrix because fenics doesn't allow taking derivative w.r.t a scalar
        # As a result, the time step is defined for each vertex. This is why 'ddt' is computed weirdly
        # below
        dfu_ddt = dfn.assemble(self.forms['form.bi.df1_dt_adj'], tensor=dfn.PETScMatrix())
        dfv_ddt = 0 - newmark.newmark_v_dt(self.state1[0], *self.state0, self.dt)
        dfa_ddt = 0 - newmark.newmark_a_dt(self.state1[0], *self.state1, self.dt)

        bu, bv, ba = b
        ddt = (dfu_ddt*bu).sum() + dfv_ddt.inner(bv) + dfa_ddt.inner(ba)
        return ddt

class Rayleigh(Solid):
    """
    Represents the governing equations of Rayleigh damped solid
    """
    PROPERTY_TYPES = {
        'emod': ('field', ()),
        'nu': ('const', ()),
        'rho': ('const', ()),
        'rayleigh_m': ('const', ()),
        'rayleigh_k': ('const', ()),
        'y_collision': ('const', ()),
        'k_collision': ('const', ())}

    PROPERTY_DEFAULTS = {
        'emod': 10e3 * PASCAL_TO_CGS,
        'nu': 0.49,
        'rho': 1000 * SI_DENSITY_TO_CGS,
        'rayleigh_m': 10,
        'rayleigh_k': 1e-3,
        'y_collision': 0.61-0.001,
        'k_collision': 1e11}

    @staticmethod
    def form_definitions(mesh, facet_func, facet_labels, cell_func, cell_labels, 
                         fsi_facet_labels, fixed_facet_labels):
        dx = dfn.Measure('dx', domain=mesh, subdomain_data=cell_func)
        ds = dfn.Measure('ds', domain=mesh, subdomain_data=facet_func)

        scalar_fspace = dfn.FunctionSpace(mesh, 'CG', 1)
        vector_fspace = dfn.VectorFunctionSpace(mesh, 'CG', 1)

        vector_trial = dfn.TrialFunction(vector_fspace)
        vector_test = dfn.TestFunction(vector_fspace)

        scalar_trial = dfn.TrialFunction(scalar_fspace)
        scalar_test = dfn.TestFunction(scalar_fspace)

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

        # NOTE: Fenics doesn't support form derivatives w.r.t Constant. This is a hack making time step
        # vary in space so you can take the derivative. You *must* set the time step equal at every DOF
        # dt = dfn.Constant(1e-4)
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

        # Surface pressures
        p0 = dfn.Function(scalar_fspace)
        p1 = dfn.Function(scalar_fspace)

        # Symbolic calculations to get the variational form for a linear-elastic solid
        def damping_2form(trial, test):
            return rayleigh_m * biform_m(trial, test, rho) \
                   + rayleigh_k * biform_k(trial, test, emod, nu)

        inertia = inertia_2form(a1, vector_test, rho)

        stiffness = stiffness_2form(u1, vector_test, emod, nu)

        damping = damping_2form(v1, vector_test)

        # Compute the pressure loading Neumann boundary condition on the reference configuration
        # using Nanson's formula. This is because the 'total lagrangian' formulation is used.
        ds = dfn.Measure('ds', domain=mesh, subdomain_data=facet_func)

        
        facet_normal = dfn.FacetNormal(mesh)
        traction_ds = ds(facet_labels[fsi_facet_labels[0]])
        for fsi_edge in fsi_facet_labels[1:]:
            traction_ds += ds(facet_labels[fsi_edge])
        traction = traction_1form(u1, vector_test, p1, facet_normal, traction_ds)

        # Use the penalty method to account for collision
        def penalty_1form(u):
            collision_normal = dfn.Constant([0.0, 1.0])
            x_reference = dfn.Function(vector_fspace)

            vert_to_vdof = dfn.vertex_to_dof_map(vector_fspace)
            x_reference.vector()[vert_to_vdof.reshape(-1)] = mesh.coordinates().reshape(-1)

            gap = ufl.dot(x_reference+u, collision_normal) - y_collision
            positive_gap = (gap + abs(gap)) / 2

            # Uncomment/comment the below lines to choose between exponential or quadratic penalty springs
            penalty = ufl.dot(k_collision*positive_gap**2*-1*collision_normal, vector_test) \
                      * traction_ds

            return penalty

        penalty = penalty_1form(u1)

        f1_linear = inertia + stiffness + damping
        f1_nonlin = -traction - penalty
        f1_uva = f1_linear + f1_nonlin

        f1 = ufl.replace(f1_uva, {v1: v1_nmk, a1:a1_nmk})

        df1_du1 = ufl.derivative(f1, u1, vector_trial)

        df1_dp1 = ufl.derivative(f1, p1, scalar_trial)

        ## Boundary conditions
        # Specify DirichletBC at the VF base
        bc_base = dfn.DirichletBC(vector_fspace, dfn.Constant([0.0, 0.0]),
                                  facet_func, facet_labels['fixed'])

        ## Adjoint forms
        df1_du0_adj = dfn.adjoint(ufl.derivative(f1, u0, vector_trial))

        df1_dv0_adj = dfn.adjoint(ufl.derivative(f1, v0, vector_trial))

        df1_da0_adj = dfn.adjoint(ufl.derivative(f1, a0, vector_trial))

        df1_du1_adj = dfn.adjoint(df1_du1)

        df1_demod = ufl.derivative(f1, emod, scalar_trial)
        df1_dp1_adj = dfn.adjoint(df1_dp1)
        df1_dt = ufl.derivative(f1, dt, scalar_trial)

        # Also define an 'f0' form that solves for a0, given u0 and v0
        # This is needed to solve for the first 'a0' given u0, v0 initial states
        f0 = inertia_2form(a0, vector_test, rho) + damping_2form(v0, vector_test) \
            + stiffness_2form(u0, vector_test, emod, nu) \
            - traction_1form(u0, vector_test, p0, facet_normal, traction_ds) \
            - penalty_1form(u0)
        df0_du0 = ufl.derivative(f0, u0, vector_trial)
        df0_dv0 = ufl.derivative(f0, v0, vector_trial)
        df0_da0 = ufl.derivative(f0, a0, vector_trial)
        df0_dp0 = ufl.derivative(f0, p0, scalar_trial)

        df0_du0_adj = dfn.adjoint(df0_du0)
        df0_dv0_adj = dfn.adjoint(df0_dv0)
        df0_da0_adj = dfn.adjoint(df0_da0)
        df0_dp0_adj = dfn.adjoint(df0_dp0)

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

            'coeff.fsi.p0': p0,
            'coeff.fsi.p1': p1,

            'coeff.prop.rho': rho,
            'coeff.prop.nu': nu,
            'coeff.prop.emod': emod,
            'coeff.prop.rayleigh_m': rayleigh_m,
            'coeff.prop.rayleigh_k': rayleigh_k,
            'coeff.prop.y_collision': y_collision,
            'coeff.prop.k_collision': k_collision,

            'form.un.f1_uva': f1_uva,
            'form.un.f1': f1,
            'form.un.f0': f0,

            'form.bi.df1_du1': df1_du1,
            'form.bi.df1_du1_adj': df1_du1_adj,
            'form.bi.df1_du0': dfn.adjoint(df1_du0_adj),
            'form.bi.df1_du0_adj': df1_du0_adj,
            'form.bi.df1_dp1': df1_dp1,
            'form.bi.df1_dp1_adj': df1_dp1_adj,
            'form.bi.df1_dv0': dfn.adjoint(df1_dv0_adj),
            'form.bi.df1_dv0_adj': df1_dv0_adj,
            'form.bi.df1_da0': dfn.adjoint(df1_da0_adj),
            'form.bi.df1_da0_adj': df1_da0_adj,
            # 'form.bi.df1_demod': df1_demod,
            'form.bi.df1_dt': df1_dt,
            'form.bi.df1_dt_adj': dfn.adjoint(df1_dt),

            'form.bi.df0_du0_adj': df0_du0_adj,
            'form.bi.df0_dv0_adj': df0_dv0_adj,
            'form.bi.df0_da0_adj': df0_da0_adj,
            'form.bi.df0_dp0_adj': df0_dp0_adj,
            'form.bi.df0_du0': df0_du0,
            'form.bi.df0_dv0': df0_dv0,
            'form.bi.df0_da0': df0_da0,
            'form.bi.df0_dp0': df0_dp0}
        return forms

class KelvinVoigt(Solid):
    """
    Represents the governing equations of a Kelvin-Voigt damped solid
    """
    PROPERTY_TYPES = {
        'emod': ('field', ()),
        'nu': ('const', ()),
        'rho': ('const', ()),
        'eta': ('field', ()),
        'y_collision': ('const', ()),
        'k_collision': ('const', ())}

    PROPERTY_DEFAULTS = {
        'emod': 10e3 * PASCAL_TO_CGS,
        'nu': 0.49,
        'rho': 1000 * SI_DENSITY_TO_CGS,
        'eta': 3.0,
        'y_collision': 0.61-0.001,
        'k_collision': 1e11}

    @staticmethod
    def form_definitions(mesh, facet_func, facet_labels, cell_func, cell_labels,
                         fsi_facet_labels, fixed_facet_labels):
        dx = dfn.Measure('dx', domain=mesh, subdomain_data=cell_func)
        ds = dfn.Measure('ds', domain=mesh, subdomain_data=facet_func)

        scalar_fspace = dfn.FunctionSpace(mesh, 'CG', 1)
        vector_fspace = dfn.VectorFunctionSpace(mesh, 'CG', 1)

        vector_trial = dfn.TrialFunction(vector_fspace)
        vector_test = dfn.TestFunction(vector_fspace)

        scalar_trial = dfn.TrialFunction(scalar_fspace)
        scalar_test = dfn.TestFunction(scalar_fspace)

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
        p0 = dfn.Function(scalar_fspace)
        p1 = dfn.Function(scalar_fspace)

        # Symbolic calculations to get the variational form for a linear-elastic solid
        def damping_2form(trial, test):
            kv_damping = ufl.inner(kv_eta*strain(trial), strain(test)) * ufl.dx
            return kv_damping

        inertia = inertia_2form(a1, vector_test, rho)
        stiffness = stiffness_2form(u1, vector_test, emod, nu)
        kv_damping = damping_2form(v1, vector_test)

        # Compute the pressure loading using Neumann boundary conditions on the reference configuration
        # using Nanson's formula. This is because the 'total lagrangian' formulation is used.
        ds = dfn.Measure('ds', domain=mesh, subdomain_data=facet_func)

        facet_normal = dfn.FacetNormal(mesh)
        traction_ds = ds(facet_labels[fsi_facet_labels[0]])
        for fsi_edge in fsi_facet_labels[1:]:
            traction_ds += ds(facet_labels[fsi_edge])
        traction = traction_1form(u1, vector_test, p1, facet_normal, traction_ds)

        # Use the penalty method to account for collision
        def penalty_1form(u):
            collision_normal = dfn.Constant([0.0, 1.0])
            x_reference = dfn.Function(vector_fspace)

            vert_to_vdof = dfn.vertex_to_dof_map(vector_fspace)
            x_reference.vector()[vert_to_vdof.reshape(-1)] = mesh.coordinates().reshape(-1)

            gap = ufl.dot(x_reference+u, collision_normal) - y_collision
            positive_gap = (gap + abs(gap)) / 2

            # Uncomment/comment the below lines to choose between exponential or quadratic penalty springs
            penalty = ufl.dot(k_collision*positive_gap**3*-1*collision_normal, vector_test) \
                      * traction_ds
            return penalty

        # Uncomment/comment the below lines to choose between exponential or quadratic penalty springs
        penalty = penalty_1form(u1)

        f1_linear = inertia + stiffness + kv_damping
        f1_nonlin = -traction - penalty
        f1_uva = f1_linear + f1_nonlin

        f1 = ufl.replace(f1_uva, {v1: v1_nmk, a1: a1_nmk})

        df1_du1 = ufl.derivative(f1, u1, vector_trial)

        df1_dp1 = ufl.derivative(f1, p1, scalar_trial)

        ## Boundary conditions
        # Specify DirichletBC at the VF base
        bc_base = dfn.DirichletBC(vector_fspace, dfn.Constant([0.0, 0.0]),
                                  facet_func, facet_labels['fixed'])

        ## Adjoint forms
        df1_du0_adj = dfn.adjoint(ufl.derivative(f1, u0, vector_trial))

        df1_dv0_adj = dfn.adjoint(ufl.derivative(f1, v0, vector_trial))

        df1_da0_adj = dfn.adjoint(ufl.derivative(f1, a0, vector_trial))

        # df1_du1_adj_linear = dfn.adjoint(df1_du1_linear)
        # df1_du1_adj_nonlin = dfn.adjoint(df1_du1_nonlin)
        df1_du1_adj = dfn.adjoint(df1_du1)

        df1_demod = ufl.derivative(f1, emod, scalar_trial)
        df1_dp1_adj = dfn.adjoint(df1_dp1)
        df1_dt = ufl.derivative(f1, dt, scalar_trial)

        f0 = inertia_2form(a0, vector_test, rho) + damping_2form(v0, vector_test) \
             + stiffness_2form(u0, vector_test, emod, nu) \
             - traction_1form(u0, vector_test, p0, facet_normal, traction_ds) \
             - penalty_1form(u0)
        df0_du0 = ufl.derivative(f0, u0, vector_trial)
        df0_dv0 = ufl.derivative(f0, v0, vector_trial)
        df0_da0 = ufl.derivative(f0, a0, vector_trial)
        df0_dp0 = ufl.derivative(f0, p0, scalar_trial)

        df0_du0_adj = dfn.adjoint(df0_du0)
        df0_dv0_adj = dfn.adjoint(df0_dv0)
        df0_da0_adj = dfn.adjoint(df0_da0)
        df0_dp0_adj = dfn.adjoint(df0_dp0)

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

            'coeff.fsi.p0': p0,
            'coeff.fsi.p1': p1,

            'coeff.prop.rho': rho,
            'coeff.prop.eta': kv_eta,
            'coeff.prop.emod': emod,
            'coeff.prop.nu': nu,
            'coeff.prop.y_collision': y_collision,
            'coeff.prop.k_collision': k_collision,

            'form.un.f1_uva': f1_uva,
            'form.un.f1': f1,
            'form.un.f0': f0,

            'form.bi.df1_du1': df1_du1,
            'form.bi.df1_du1_adj': df1_du1_adj,
            'form.bi.df1_du0': dfn.adjoint(df1_du0_adj),
            'form.bi.df1_du0_adj': df1_du0_adj,
            'form.bi.df1_dp1': df1_dp1,
            'form.bi.df1_dp1_adj': df1_dp1_adj,
            'form.bi.df1_dv0': dfn.adjoint(df1_dv0_adj),
            'form.bi.df1_dv0_adj': df1_dv0_adj,
            'form.bi.df1_da0': dfn.adjoint(df1_da0_adj),
            'form.bi.df1_da0_adj': df1_da0_adj,
            # 'form.bi.df1_demod': df1_demod,
            'form.bi.df1_dt': df1_dt,
            'form.bi.df1_dt_adj': dfn.adjoint(df1_dt),

            'form.bi.df0_du0_adj': df0_du0_adj,
            'form.bi.df0_dv0_adj': df0_dv0_adj,
            'form.bi.df0_da0_adj': df0_da0_adj,
            'form.bi.df0_dp0_adj': df0_dp0_adj,
            'form.bi.df0_du0': df0_du0,
            'form.bi.df0_dv0': df0_dv0,
            'form.bi.df0_da0': df0_da0,
            'form.bi.df0_dp0': df0_dp0}
        return forms

class IncompSwellingKelvinVoigt(Solid):
    """
    Kelvin Voigt model allowing for a swelling field
    """
    PROPERTY_TYPES = {
        'emod': ('field', ()),
        'k_swelling': ('const', ()),
        'v_swelling' : ('field', ()),
        'rho': ('const', ()),
        'eta': ('field', ()),
        'y_collision': ('const', ()),
        'k_collision': ('const', ())}

    PROPERTY_DEFAULTS = {
        'emod': 10e3 * PASCAL_TO_CGS,
        'v_swelling': 1.0,
        'k_swelling': 1000.0 * 10e3 * PASCAL_TO_CGS,
        'rho': 1000 * SI_DENSITY_TO_CGS,
        'eta': 3.0,
        'y_collision': 0.61-0.001,
        'k_collision': 1e11}

    @staticmethod
    def form_definitions(mesh, facet_func, facet_labels, cell_func, cell_labels,
                         fsi_facet_labels, fixed_facet_labels):
        dx = dfn.Measure('dx', domain=mesh, subdomain_data=cell_func)
        ds = dfn.Measure('ds', domain=mesh, subdomain_data=facet_func)

        scalar_fspace = dfn.FunctionSpace(mesh, 'CG', 1)
        vector_fspace = dfn.VectorFunctionSpace(mesh, 'CG', 1)

        vector_trial = dfn.TrialFunction(vector_fspace)
        vector_test = dfn.TestFunction(vector_fspace)

        scalar_trial = dfn.TrialFunction(scalar_fspace)
        scalar_test = dfn.TestFunction(scalar_fspace)

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
        p0 = dfn.Function(scalar_fspace)
        p1 = dfn.Function(scalar_fspace)

        # Symbolic calculations to get the variational form for a linear-elastic solid
        def damping_2form(trial, test):
            kv_damping = ufl.inner(kv_eta*strain(trial), strain(test)) * ufl.dx
            return kv_damping

        inertia = inertia_2form(a1, vector_test, rho)

        def stiffness_2form_swelling(trial, test, emod, vswell, kswell):
            nu = 0.5
            # lame_lambda = emod*nu/(1+nu)/(1-2*nu)
            lame_mu = emod/2/(1+nu)

            # normally the small stress-stain relation, would be given by:
            # cauchy_stress = 2*lame_mu*strain(u1) + lame_lambda*ufl.tr(strain(u))*ufl.Identity(u.geometric_dimension())

            # for incomp. swelling approximate by
            cauchy_stress = 2*lame_mu*strain(trial) + kswell*(ufl.tr(strain(trial)) - (vswell-1.0))*ufl.Identity(test.geometric_dimension())
            return ufl.inner(cauchy_stress, strain(test))*ufl.dx
        
        stiffness = stiffness_2form_swelling(u1, vector_test, emod, v_swelling, k_swelling)
        kv_damping = damping_2form(v1, vector_test)

        # Compute the pressure loading using Neumann boundary conditions on the reference configuration
        # using Nanson's formula. This is because the 'total lagrangian' formulation is used.
        ds = dfn.Measure('ds', domain=mesh, subdomain_data=facet_func)

        facet_normal = dfn.FacetNormal(mesh)
        traction_ds = ds(facet_labels[fsi_facet_labels[0]])
        for fsi_edge in fsi_facet_labels[1:]:
            traction_ds += ds(facet_labels[fsi_edge])
        traction = traction_1form(u1, vector_test, p1, facet_normal, traction_ds)

        # Use the penalty method to account for collision
        def penalty_1form(u):
            collision_normal = dfn.Constant([0.0, 1.0])
            x_reference = dfn.Function(vector_fspace)

            vert_to_vdof = dfn.vertex_to_dof_map(vector_fspace)
            x_reference.vector()[vert_to_vdof.reshape(-1)] = mesh.coordinates().reshape(-1)

            gap = ufl.dot(x_reference+u, collision_normal) - y_collision
            positive_gap = (gap + abs(gap)) / 2

            # Uncomment/comment the below lines to choose between exponential or quadratic penalty springs
            penalty = ufl.dot(k_collision*positive_gap**3*-1*collision_normal, vector_test) \
                      * traction_ds
            return penalty

        # Uncomment/comment the below lines to choose between exponential or quadratic penalty springs
        penalty = penalty_1form(u1)

        f1_linear = inertia + stiffness + kv_damping
        f1_nonlin = -traction - penalty
        f1_uva = f1_linear + f1_nonlin

        f1 = ufl.replace(f1_uva, {v1: v1_nmk, a1: a1_nmk})

        df1_du1 = ufl.derivative(f1, u1, vector_trial)

        df1_dp1 = ufl.derivative(f1, p1, scalar_trial)

        ## Boundary conditions
        # Specify DirichletBC at the VF base
        bc_base = dfn.DirichletBC(vector_fspace, dfn.Constant([0.0, 0.0]),
                                  facet_func, facet_labels['fixed'])

        ## Adjoint forms
        df1_du0_adj = dfn.adjoint(ufl.derivative(f1, u0, vector_trial))

        df1_dv0_adj = dfn.adjoint(ufl.derivative(f1, v0, vector_trial))

        df1_da0_adj = dfn.adjoint(ufl.derivative(f1, a0, vector_trial))

        # df1_du1_adj_linear = dfn.adjoint(df1_du1_linear)
        # df1_du1_adj_nonlin = dfn.adjoint(df1_du1_nonlin)
        df1_du1_adj = dfn.adjoint(df1_du1)

        df1_demod = ufl.derivative(f1, emod, scalar_trial)
        df1_dp1_adj = dfn.adjoint(df1_dp1)
        df1_dt = ufl.derivative(f1, dt, scalar_trial)

        f0 = inertia_2form(a0, vector_test, rho) + damping_2form(v0, vector_test) \
             + stiffness_2form_swelling(u0, vector_test, emod, v_swelling, k_swelling) \
             - traction_1form(u0, vector_test, p0, facet_normal, traction_ds) \
             - penalty_1form(u0)
        df0_du0 = ufl.derivative(f0, u0, vector_trial)
        df0_dv0 = ufl.derivative(f0, v0, vector_trial)
        df0_da0 = ufl.derivative(f0, a0, vector_trial)
        df0_dp0 = ufl.derivative(f0, p0, scalar_trial)

        df0_du0_adj = dfn.adjoint(df0_du0)
        df0_dv0_adj = dfn.adjoint(df0_dv0)
        df0_da0_adj = dfn.adjoint(df0_da0)
        df0_dp0_adj = dfn.adjoint(df0_dp0)

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

            'coeff.fsi.p0': p0,
            'coeff.fsi.p1': p1,

            'coeff.prop.rho': rho,
            'coeff.prop.v_swelling': v_swelling,
            'coeff.prop.k_swelling': k_swelling,
            'coeff.prop.eta': kv_eta,
            'coeff.prop.emod': emod,
            # 'coeff.prop.nu': nu,
            'coeff.prop.y_collision': y_collision,
            'coeff.prop.k_collision': k_collision,

            'form.un.f1_uva': f1_uva,
            'form.un.f1': f1,
            'form.un.f0': f0,

            'form.bi.df1_du1': df1_du1,
            'form.bi.df1_dp1': df1_dp1,
            'form.bi.df1_du1_adj': df1_du1_adj,
            'form.bi.df1_du0_adj': df1_du0_adj,
            'form.bi.df1_dv0_adj': df1_dv0_adj,
            'form.bi.df1_da0_adj': df1_da0_adj,
            'form.bi.df1_dp1_adj': df1_dp1_adj,
            # 'form.bi.df1_demod': df1_demod,
            'form.bi.df1_dt_adj': dfn.adjoint(df1_dt),

            'form.bi.df0_du0_adj': df0_du0_adj,
            'form.bi.df0_dv0_adj': df0_dv0_adj,
            'form.bi.df0_da0_adj': df0_da0_adj,
            'form.bi.df0_dp0_adj': df0_dp0_adj,
            'form.bi.df0_du0': df0_du0,
            'form.bi.df0_dv0': df0_dv0,
            'form.bi.df0_da0': df0_da0,
            'form.bi.df0_dp0': df0_dp0}
        return forms

def get_df1_dsolid_forms(f1, forms, property_labels, scalar_trial):
    """
    Return a dictionary of forms of derivatives of f1 with respect to the various solid parameters
    """
    df1_dsolid = {}
    for key in property_labels:
        try:
            df1_dsolid[key] = dfn.adjoint(ufl.derivative(f1, forms[f'coeff.prop.{key}'], scalar_trial))
        except RuntimeError:
            df1_dsolid[key] = None

        if df1_dsolid[key] is not None:
            try:
                dfn.assemble(df1_dsolid[key])
            except RuntimeError:
                df1_dsolid[key] = None
    return df1_dsolid

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

        self._tensor = dfn.assemble(form, keep_diagonal=keep_diagonal)
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
