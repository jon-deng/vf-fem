"""
Module for definitions of weak forms.

Units are in CGS

TODO: The form definitions have a lot of repeated code. Many of the form operations are copy-pasted.
You should think about what forms should be custom made for different solid governing equations
and what types of forms are always generated the same way, and refactor accordingly.
"""



import dolfin as dfn
import ufl




from .parameters.properties import SolidProperties
from .constants import PASCAL_TO_CGS, SI_DENSITY_TO_CGS

from .newmark import *


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

    # ds(facet_label['pressure']) should be replaced with traction_ds
    traction = ufl.dot(fluid_force, test)*traction_ds
    return traction

class Solid:
    """
    Class representing the discretized governing equations of a solid
    """
    # Subclasses have to set these values
    PROPERTY_TYPES = None
    DEFAULTS = None

    def __init__(self, mesh, facet_function, facet_labels, cell_function, cell_labels):

        self._forms = self.form_definitions(mesh, facet_function, facet_labels,
                                            cell_function, cell_labels)

        ## Store mesh related quantities
        self.mesh = mesh
        self.facet_function = facet_function
        self.cell_function = cell_function
        self.facet_labels = facet_labels
        self.cell_labels = cell_labels

        ## Store some key quantites related to the forms
        self.vector_fspace = self.forms['fspace.vector']
        self.scalar_fspace = self.forms['fspace.scalar']

        self.scalar_trial = self.forms['trial.scalar']
        self.vector_trial = self.forms['trial.vector']
        self.scalar_test = self.forms['test.scalar']
        self.vector_test = self.forms['test.vector']

        ## Measures
        self.dx = self.forms['measure.dx']
        self.ds = self.forms['measure.ds']

        self.vert_to_vdof = dfn.vertex_to_dof_map(self.forms['fspace.vector'])
        self.vert_to_sdof = dfn.vertex_to_dof_map(self.forms['fspace.scalar'])
        self.vdof_to_vert = dfn.dof_to_vertex_map(self.forms['fspace.vector'])
        self.sdof_to_vert = dfn.dof_to_vertex_map(self.forms['fspace.scalar'])

        self.u0 = self.forms['coeff.state.u0']
        self.v0 = self.forms['coeff.state.v0']
        self.a0 = self.forms['coeff.state.a0']
        self.u1 = self.forms['coeff.state.u1']

        self.dt = self.forms['coeff.time.dt']

        self.f1 = self.forms['form.un.f1']
        self.df1_du1 = self.forms['form.bi.df1_du1']

        self.bc_base = self.forms['bcs.base']

        # Set property values to defaults
        self.set_properties(SolidProperties(self))

    @property
    def forms(self):
        return self._forms

    @staticmethod
    def form_definitions(mesh, facet_function, facet_labels, cell_function, cell_labels):
        """
        Return a dictionary of ufl forms representing the solid in Fenics.

        You have to implement this along with a description of the properties to make a subclass of
        the `Solid`.
        """
        return NotImplementedError("Subclasses must implement this function")

    def set_ini_state(self, u0, v0, a0):
        """
        Sets the initial state variables, (u, v, a)

        Parameters
        ----------
        u0, v0, a0 : array_like
        """
        self.forms['coeff.state.u0'].vector()[:] = u0
        self.forms['coeff.state.v0'].vector()[:] = v0
        self.forms['coeff.state.a0'].vector()[:] = a0

    def set_fin_state(self, u1, v1, a1):
        """
        Sets the final state variables.

        Note that the solid forms are in displacement form so only the displacement is needed as an
        initial guess to solve the solid equations. The state `v1` and `a1` are specified explicitly
        by the Newmark relations once you solve for `u1`.

        Parameters
        ----------
        u1, v1, a1 : array_like
        """
        self.forms['coeff.state.u1'].vector()[:] = u1
        self.forms['coeff.state.v1'].vector()[:] = v1
        self.forms['coeff.state.a1'].vector()[:] = a1

    def set_ini_surf_pressure(self, p0):
        self.forms['coeff.fsi.p0'].vector()[:] = p0

    def set_fin_surf_pressure(self, p1):
        self.forms['coeff.fsi.p1'].vector()[:] = p1

    def set_time_step(self, dt):
        """
        Sets the time step
        """
        # The coefficient for time is a vector because of a hack; it's needed to take derivatives
        # but you should only pass a float to ensure each node has the same time step
        self.forms['coeff.time.dt'].vector()[:] = dt

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
        for key in props:
            # TODO: Check types to make sure the input property is compatible with the solid type
            coefficient = self.forms['coeff.prop.'+key]

            # If the property is a field variable, values have to be assigned to every spot in
            # the vector
            if isinstance(coefficient, dfn.function.constant.Constant):
                coefficient.assign(props[key][()])
            else:
                coefficient.vector()[:] = props[key]

    def set_iter_params(self, uva1, uva0, p1, p0, dt, props):
        """
        Sets all coefficient values to solve the model

        Parameters
        ----------
        uva1, uva0 : tuple of array_like
        p1, p0 : array_like
        dt : float
        props : dict_like
        """
        self.set_ini_state(*uva0)
        self.set_fin_state(*uva1)

        self.set_ini_surf_pressure(p0)
        self.set_fin_surf_pressure(p1)

        self.set_properties(props)
        self.set_time_step(dt)

    def assem(self, form_name):
        """
        Assembles the form given by label `form_name`
        """
        form_key = f'form.bi.{form_name}'

        if form_key in self.forms:
            return dfn.assemble(self.forms[form_key])
        else:
            raise ValueError(f"`{form_name}` is not a valid form label")

    def get_properties(self):
        """
        Returns the current values of the properties

        Returns
        -------
        properties : Properties
        """
        properties = SolidProperties(self)

        for key in properties:
            # TODO: Check types to make sure the input property is compatible with the solid type
            coefficient = self.forms['coeff.prop.'+key]

            if properties[key].shape == ():
                if isinstance(coefficient, dfn.function.constant.Constant):
                    # If the coefficient is a constant, then the property must be a float
                    properties[key][()] = coefficient.values()[0]
                else:
                    # If a vector, it's possible it's the 'hacky' version for a time step, where the
                    # actual property is a float but the coefficient is assigned to be a vector
                    # (which is done so you can differentiate it)
                    assert coefficient.vector().max() == coefficient.vector().min()
                    properties[key][()] = coefficient.vector().max()
            else:
                coefficient.vector()[:] = properties[key]

        return properties

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
    def form_definitions(mesh, facet_function, facet_labels, cell_function, cell_labels):
        dx = dfn.Measure('dx', domain=mesh, subdomain_data=cell_function)
        ds = dfn.Measure('ds', domain=mesh, subdomain_data=facet_function)

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
        # vary in space so you can take derivative. You *must* set the time step equal at every DOF
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

        v1_nmk = newmark_v(u1, u0, v0, a0, dt, gamma, beta)
        a1_nmk = newmark_a(u1, u0, v0, a0, dt, gamma, beta)

        # Surface pressures
        p0 = dfn.Function(scalar_fspace)
        p1 = dfn.Function(scalar_fspace)

        # Symbolic calculations to get the variational form for a linear-elastic solid
        def damping_2form(trial, test):
            return rayleigh_m * biform_m(trial, test, rho) \
                   + rayleigh_k * biform_k(trial, test, emod, nu)

        inertia = inertia_2form(a1_nmk, vector_test, rho)

        stiffness = stiffness_2form(u1, vector_test, emod, nu)

        damping = damping_2form(v1_nmk, vector_test)

        # Compute the pressure loading Neumann boundary condition on the reference configuration
        # using Nanson's formula. This is because the 'total lagrangian' formulation is used.
        ds = dfn.Measure('ds', domain=mesh, subdomain_data=facet_function)

        facet_normal = dfn.FacetNormal(mesh)
        traction_ds = ds(facet_labels['pressure'])
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
                      * ds(facet_labels['pressure'])

            return penalty

        penalty = penalty_1form(u1)

        f1_linear = inertia + stiffness + damping
        f1_nonlin = -traction - penalty
        f1 = f1_linear + f1_nonlin

        df1_du1_linear = ufl.derivative(f1_linear, u1, vector_trial)
        df1_du1_nonlin = ufl.derivative(f1_nonlin, u1, vector_trial)
        df1_du1 = df1_du1_linear + df1_du1_nonlin

        ## Boundary conditions
        # Specify DirichletBC at the VF base
        bc_base = dfn.DirichletBC(vector_fspace, dfn.Constant([0.0, 0.0]),
                                  facet_function, facet_labels['fixed'])

        ## Adjoint forms
        df1_du0_adj_linear = dfn.adjoint(ufl.derivative(f1_linear, u0, vector_trial))
        # df1_du0_adj_nonlin = dfn.adjoint(ufl.derivative(f1_nonlin, u0, vector_trial))
        df1_du0_adj_nonlin = 0
        df1_du0_adj = df1_du0_adj_linear + df1_du0_adj_nonlin

        df1_dv0_adj_linear = dfn.adjoint(ufl.derivative(f1_linear, v0, vector_trial))
        df1_dv0_adj_nonlin = 0
        df1_dv0_adj = df1_dv0_adj_linear + df1_dv0_adj_nonlin

        df1_da0_adj_linear = dfn.adjoint(ufl.derivative(f1_linear, a0, vector_trial))
        df1_da0_adj_nonlin = 0
        df1_da0_adj = df1_da0_adj_linear + df1_da0_adj_nonlin

        df1_du1_adj_linear = dfn.adjoint(df1_du1_linear)
        df1_du1_adj_nonlin = dfn.adjoint(df1_du1_nonlin)
        df1_du1_adj = df1_du1_adj_linear + df1_du1_adj_nonlin

        df1_demod = ufl.derivative(f1, emod, scalar_trial)
        df1_dp1_adj = dfn.adjoint(ufl.derivative(f1, p1, scalar_trial))
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

            'form.un.f1': f1,
            'form.un.f0': f0,

            'form.bi.df1_du1': df1_du1,
            'form.bi.df1_du1_adj': df1_du1_adj,
            'form.bi.df1_du0_adj': df1_du0_adj,
            'form.bi.df1_dv0_adj': df1_dv0_adj,
            'form.bi.df1_da0_adj': df1_da0_adj,
            'form.bi.df1_dp1_adj': df1_dp1_adj,
            'form.bi.df1_demod': df1_demod,
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
    def form_definitions(mesh, facet_function, facet_labels, cell_function, cell_labels):
        dx = dfn.Measure('dx', domain=mesh, subdomain_data=cell_function)
        ds = dfn.Measure('ds', domain=mesh, subdomain_data=facet_function)

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

        v1_nmk = newmark_v(u1, u0, v0, a0, dt, gamma, beta)
        a1_nmk = newmark_a(u1, u0, v0, a0, dt, gamma, beta)

        # Surface pressures
        p0 = dfn.Function(scalar_fspace)
        p1 = dfn.Function(scalar_fspace)

        # Symbolic calculations to get the variational form for a linear-elastic solid
        def damping_2form(trial, test):
            kv_damping = ufl.inner(kv_eta*strain(trial), strain(test)) * ufl.dx
            return kv_damping

        inertia = inertia_2form(a1_nmk, vector_test, rho)
        stiffness = stiffness_2form(u1, vector_test, emod, nu)
        kv_damping = damping_2form(v1_nmk, vector_test)

        # Compute the pressure loading using Neumann boundary conditions on the reference configuration
        # using Nanson's formula. This is because the 'total lagrangian' formulation is used.
        ds = dfn.Measure('ds', domain=mesh, subdomain_data=facet_function)

        facet_normal = dfn.FacetNormal(mesh)
        traction_ds = ds(facet_labels['pressure'])
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
                    * ds(facet_labels['pressure'])
            return penalty

        # Uncomment/comment the below lines to choose between exponential or quadratic penalty springs
        penalty = penalty_1form(u1)

        f1_linear = inertia + stiffness + kv_damping
        f1_nonlin = -traction - penalty
        f1 = f1_linear + f1_nonlin

        df1_du1_linear = ufl.derivative(f1_linear, u1, vector_trial)
        df1_du1_nonlin = ufl.derivative(f1_nonlin, u1, vector_trial)
        df1_du1 = df1_du1_linear + df1_du1_nonlin

        ## Boundary conditions
        # Specify DirichletBC at the VF base
        bc_base = dfn.DirichletBC(vector_fspace, dfn.Constant([0.0, 0.0]),
                                  facet_function, facet_labels['fixed'])

        ## Adjoint forms
        df1_du0_adj_linear = dfn.adjoint(ufl.derivative(f1_linear, u0, vector_trial))
        df1_du0_adj_nonlin = 0
        df1_du0_adj = df1_du0_adj_linear + df1_du0_adj_nonlin

        df1_dv0_adj_linear = dfn.adjoint(ufl.derivative(f1_linear, v0, vector_trial))
        df1_dv0_adj_nonlin = 0
        df1_dv0_adj = df1_dv0_adj_linear + df1_dv0_adj_nonlin

        df1_da0_adj_linear = dfn.adjoint(ufl.derivative(f1_linear, a0, vector_trial))
        df1_da0_adj_nonlin = 0
        df1_da0_adj = df1_da0_adj_linear + df1_da0_adj_nonlin

        df1_du1_adj_linear = dfn.adjoint(df1_du1_linear)
        df1_du1_adj_nonlin = dfn.adjoint(df1_du1_nonlin)
        df1_du1_adj = df1_du1_adj_linear + df1_du1_adj_nonlin

        df1_demod = ufl.derivative(f1, emod, scalar_trial)
        df1_dp1_adj = dfn.adjoint(ufl.derivative(f1, p1, scalar_trial))
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

            'form.un.f1': f1,
            'form.un.f0': f0,
            'form.bi.df1_du1': df1_du1,
            'form.bi.df1_du1_adj': df1_du1_adj,
            'form.bi.df1_du0_adj': df1_du0_adj,
            'form.bi.df1_dv0_adj': df1_dv0_adj,
            'form.bi.df1_da0_adj': df1_da0_adj,
            'form.bi.df1_dp1_adj': df1_dp1_adj,
            'form.bi.df1_demod': df1_demod,
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
