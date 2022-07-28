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
import warnings as wrn
from typing import Tuple, Mapping

from femvf.solverconst import DEFAULT_NEWTON_SOLVER_PRM
from femvf.parameters.properties import property_vecs
from femvf.constants import PASCAL_TO_CGS, SI_DENSITY_TO_CGS

from blockarray.blockmat import BlockMatrix
from blockarray.blockvec import BlockVector
from blockarray.subops import diag_mat, zero_mat

from . import base
from ..equations.solid import newmark

from ..equations.solid import solidforms
from ..assemblyutils import CachedFormAssembler

def properties_bvec_from_forms(forms, defaults=None):
    defaults = {} if defaults is None else defaults
    labels = [form_name.split('.')[-1] for form_name in forms.keys()
        if 'coeff.prop' in form_name]
    vecs = []
    for label in labels:
        coefficient = forms['coeff.prop.'+label]

        # vec = None
        # Generally the size of the vector comes directly from the property,
        # for examples, constants are size 1, scalar fields have size matching number of dofs
        # Time step is a special case since it is size 1 but is made to be a field as
        # a workaround
        if isinstance(coefficient, dfn.function.constant.Constant) or label == 'dt':
            vec = np.ones(coefficient.values().size)
            vec[:] = coefficient.values()
        else:
            vec = coefficient.vector().copy()

        if label in defaults:
            vec[:] = defaults[label]

        vecs.append(vec)

    return BlockVector(vecs, labels=[labels])


class Solid(base.Model):
    """
    Class representing the discretized governing equations of a solid
    """
    # Subclasses have to set these values
    PROPERTY_DEFAULTS = None

    def __init__(
            self,
            mesh: dfn.Mesh,
            mesh_funcs: Tuple[dfn.MeshFunction],
            mesh_entities_label_to_value: Tuple[Mapping[str, int]],
            fsi_facet_labels: Tuple[str],
            fixed_facet_labels: Tuple[str]
        ):
        assert isinstance(fsi_facet_labels, (list, tuple))
        assert isinstance(fixed_facet_labels, (list, tuple))

        self._forms = self.form_definitions(
            mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels,fixed_facet_labels)
        solidforms.gen_residual_bilinear_forms(self._forms)

        ## Store mesh related quantities
        vertex_func, facet_func, cell_func = mesh_funcs
        vertex_label_to_id, facet_label_to_id, cell_label_to_id = mesh_entities_label_to_value

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
        self.df1_dsolid = solidforms.gen_residual_bilinear_property_forms(self.forms)
        self.df1_dsolid_assemblers = {
            key: CachedFormAssembler(self.df1_dsolid[key])
            for key in self.df1_dsolid
            if self.df1_dsolid[key] is not None
        }

        ## Measures and boundary conditions
        self.dx = self.forms['measure.dx']
        self.ds = self.forms['measure.ds']
        self.bc_base = self.forms['bc.dirichlet']

        ## Index mappings
        self.vert_to_vdof = dfn.vertex_to_dof_map(self.forms['fspace.vector'])
        self.vert_to_sdof = dfn.vertex_to_dof_map(self.forms['fspace.scalar'])
        self.vdof_to_vert = dfn.dof_to_vertex_map(self.forms['fspace.vector'])
        self.sdof_to_vert = dfn.dof_to_vertex_map(self.forms['fspace.scalar'])

        ## Define the state/controls/properties
        self.state0 = BlockVector((self.u0.vector(), self.v0.vector(), self.a0.vector()), labels=[('u', 'v', 'a')])
        self.state1 = BlockVector((self.u1.vector(), self.v1.vector(), self.a1.vector()), labels=[('u', 'v', 'a')])
        self.control = BlockVector((self.forms['coeff.fsi.p1'].vector(),), labels=[('p',)])
        self.props = self.get_properties_vec(set_default=True)
        self.set_props(self.props)

        self.cached_form_assemblers = {
            key: CachedFormAssembler(self.forms[key]) for key in self.forms
            if 'form.' in key
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
        xref = self.u1.vector().copy()
        xref[:] = self.scalar_fspace.tabulate_dof_coordinates().reshape(-1).copy()
        return xref

    @staticmethod
    def form_definitions(
            mesh: dfn.Mesh,
            mesh_funcs: Tuple[dfn.MeshFunction],
            mesh_entities_label_to_value: Tuple[Mapping[str, int]],
            fsi_facet_labels: Tuple[str],
            fixed_facet_labels: Tuple[str]
        ):
        """
        Return a dictionary of ufl forms representing the solid in Fenics.

        You have to implement this along with a description of the properties to make a subclass of
        the `Solid`.
        """
        raise NotImplementedError("Subclasses must implement this function")

    ## Functions for getting empty parameter vectors
    def get_state_vec(self):
        ret = self.state1.copy()
        ret.set(0.0)
        return ret

    def get_control_vec(self):
        ret = self.control.copy()
        ret.set(0.0)
        return ret

    def get_properties_vec(self, set_default=True):
        defaults = self.PROPERTY_DEFAULTS if set_default else None
        return properties_bvec_from_forms(self.forms, defaults)

    ## Parameter setting functions
    def set_ini_state(self, uva0):
        """
        Sets the initial state variables, (u, v, a)

        Parameters
        ----------
        u0, v0, a0 : array_like
        """
        self.state0[:] = uva0

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
        self.state1[:] = uva1

    def set_control(self, p1):
        self.control[:] = p1

    def set_props(self, props):
        """
        Sets the properties of the solid

        Properties are essentially all settable values that are not states. In UFL, all things that
        can have set values are `coefficients`, but this ditinguishes between state coefficients,
        and other property coefficients.

        Parameters
        ----------
        props : Property / dict-like
        """
        for key in props.labels[0]:
            # TODO: Check types to make sure the input property is compatible with the solid type
            coefficient = self.forms['coeff.prop.'+key]

            # If the property is a field variable, values have to be assigned to every spot in
            # the vector
            if isinstance(coefficient, dfn.function.constant.Constant):
                coefficient.assign(dfn.Constant(np.squeeze(props[key])))
            else:
                coefficient.vector()[:] = props[key]

    ## Residual and sensitivity functions
    def res(self):
        dt = self.dt
        u1, v1, a1 = self.u1.vector(), self.v1.vector(), self.a1.vector()
        u0, v0, a0 = self.u0.vector(), self.v0.vector(), self.a0.vector()

        res = self.get_state_vec()
        res['u'] = self.cached_form_assemblers['form.un.f1'].assemble()
        self.bc_base.apply(res['u'])
        res['v'] = v1 - newmark.newmark_v(u1, u0, v0, a0, dt)
        res['a'] = a1 - newmark.newmark_a(u1, u0, v0, a0, dt)
        return res

    def assem_dres_dstate1(self):
        assert len(self.state1.bshape) == 1
        N = self.state1.bshape[0][0]
        dfu_du = self.cached_form_assemblers['form.bi.df1_du1'].assemble()
        dfu_dv = zero_mat(N, N)
        dfu_da = zero_mat(N, N)

        dfv_du = diag_mat(N, 0 - newmark.newmark_v_du1(self.dt))
        dfv_dv = diag_mat(N, 1)
        dfv_da = zero_mat(N, N)

        dfa_du = diag_mat(N, 0 - newmark.newmark_a_du1(self.dt))
        dfa_dv = zero_mat(N, N)
        dfa_da = diag_mat(N, 1)

        return BlockMatrix(
            [dfu_du, dfu_dv, dfu_da,
                dfv_du, dfv_dv, dfv_da,
                dfa_du, dfa_dv, dfa_da],
            shape=(3, 3),
            labels=2*self.state1.labels
        )

    def assem_dres_dstate0(self):
        assert len(self.state1.bshape) == 1
        N = self.state1.bshape[0][0]

        dfu_du = self.cached_form_assemblers['form.bi.df1_du0'].assemble()
        dfu_dv = self.cached_form_assemblers['form.bi.df1_dv0'].assemble()
        dfu_da = self.cached_form_assemblers['form.bi.df1_da0'].assemble()
        for mat in (dfu_du, dfu_dv, dfu_da):
            self.bc_base.apply(mat)

        dfv_du = diag_mat(N, 0 - newmark.newmark_v_du0(self.dt))
        dfv_dv = diag_mat(N, 0 - newmark.newmark_v_dv0(self.dt))
        dfv_da = diag_mat(N, 0 - newmark.newmark_v_da0(self.dt))

        dfa_du = diag_mat(N, 0 - newmark.newmark_a_du0(self.dt))
        dfa_dv = diag_mat(N, 0 - newmark.newmark_a_dv0(self.dt))
        dfa_da = diag_mat(N, 0 - newmark.newmark_a_da0(self.dt))

        return BlockMatrix(
            [dfu_du, dfu_dv, dfu_da,
                dfv_du, dfv_dv, dfv_da,
                dfa_du, dfa_dv, dfa_da],
            shape=(3, 3),
            labels=2*self.state1.labels
        )

    def assem_dres_dcontrol(self):
        pass

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
        dfu2_du2 = self.cached_form_assemblers['form.bi.df1_du1'].assemble()
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

        dfu2_du1 = self.cached_form_assemblers['form.bi.df1_du0'].assemble()
        dfu2_dv1 = self.cached_form_assemblers['form.bi.df1_dv0'].assemble()
        dfu2_da1 = self.cached_form_assemblers['form.bi.df1_da0'].assemble()
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
        for prop_name, vec in b.items():
            # assert self.df1_dsolid[key] is not None
            if self.df1_dsolid[prop_name] is None:
                df1_dprop = 0.0
            else:
                df1_dprop = self.df1_dsolid_assemblers[prop_name].assemble()
            val = df1_dprop*x['u']

            # Note this is a workaround because some properties are scalar values but stored as
            # vectors in order to take their derivatives. This is the case for time step, `dt`
            if vec.size == 1:
                val = val.sum()

            vec[:] = val
        return b

    def apply_dres_ddt(self, x):
        dfu_ddt = self.cached_form_assemblers['form.bi.df1_ddt'].assemble()
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
        dfu_ddt = self.cached_form_assemblers['form.bi.df1_ddt_adj'].assemble()
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
    def contact_traction(self, u):
        # This computes the nodal values of the contact traction function
        XREF = self.XREF
        ycontact = self.forms['coeff.prop.ycontact'].values()[0]
        ncontact = self.forms['coeff.prop.ncontact'].values()
        kcontact = self.forms['coeff.prop.kcontact'].values()[0]

        gap = np.dot((XREF+u)[:].reshape(-1, 2), ncontact) - ycontact
        tcontact = (-solidforms.form_cubic_penalty_pressure(gap, kcontact)[:, None]*ncontact).reshape(-1).copy()
        return tcontact

    def set_fin_state(self, state):
        # This sets the 'standard' state variables u/v/a
        super().set_fin_state(state)

        self.forms['coeff.state.manual.tcontact'].vector()[:] = self.contact_traction(state['u'])

    def _assem_dres_du(self, adjoint=False):
        ## dres_du has two components: one due to the standard u/v/a variables
        ## and an additional effect due to contact pressure
        dfu2_du2_nocontact = None
        if adjoint:
            dfu2_du2_nocontact = self.cached_form_assemblers['form.bi.df1_du1_adj'].assemble()
        else:
            dfu2_du2_nocontact = self.cached_form_assemblers['form.bi.df1_du1'].assemble()

        dfu2_du2 = dfu2_du2_nocontact + self._assem_dres_du_contact(adjoint)
        return dfu2_du2

    def _assem_dres_du_contact(self, adjoint=False):
        # Compute things needed to find sensitivities of contact pressure
        dfu2_dtcontact = None
        if adjoint:
            dfu2_dtcontact = self.cached_form_assemblers['form.bi.df1uva_dtcontact_adj'].assemble()
        else:
            dfu2_dtcontact = self.cached_form_assemblers['form.bi.df1uva_dtcontact'].assemble()

        XREF = self.XREF
        kcontact = self.forms['coeff.prop.kcontact'].values()[0]
        ycontact = self.forms['coeff.prop.ycontact'].values()[0]
        u1 = self.forms['coeff.state.u1'].vector()
        ncontact = self.forms['coeff.prop.ncontact'].values()
        gap = np.dot((XREF+u1)[:].reshape(-1, 2), ncontact) - ycontact
        dgap_du = ncontact

        # FIXME: This code below only works if n is aligned with the x/y axes.
        # for a general collision plane normal, the operation 'df_dtc*dtc_du' will
        # have to be represented by a block diagonal dtc_du (need to loop in python to do this). It
        # reduces to a diagonal if n is aligned with a coordinate axis.
        dtcontact_du2 = self.u1.vector().copy()
        dpcontact_dgap, _ = solidforms.dform_cubic_penalty_pressure(gap, kcontact)
        dtcontact_du2[:] = np.array((-dpcontact_dgap[:, None]*dgap_du).reshape(-1))

        if adjoint:
            dfu2_dtcontact.mat().diagonalScale(dtcontact_du2.vec(), None)
        else:
            dfu2_dtcontact.mat().diagonalScale(None, dtcontact_du2.vec())
        dfu2_du2_contact = dfu2_dtcontact
        return dfu2_du2_contact

    # TODO: refactor this copy-paste
    def _assem_dresuva_du(self, adjoint=False):
        ## dres_du has two components: one due to the standard u/v/a variables
        ## and an additional effect due to contact pressure
        dfu2_du2_nocontact = None
        if adjoint:
            dfu2_du2_nocontact = self.cached_form_assemblers['form.bi.df1uva_du1_adj'].assemble()
        else:
            dfu2_du2_nocontact = self.cached_form_assemblers['form.bi.df1uva_du1'].assemble()

        dfu2_du2_contact = self._assem_dres_du_contact(adjoint)

        return dfu2_du2_nocontact + dfu2_du2_contact

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
        dfu2_du2 = self._assem_dres_du(adjoint=True)
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
    def form_definitions(mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels,fixed_facet_labels):
        return \
            solidforms.Rayleigh(
                mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels,fixed_facet_labels)


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
    def form_definitions(mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels,fixed_facet_labels):
        return solidforms.KelvinVoigt(
                mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels,fixed_facet_labels)


class KelvinVoigtWEpithelium(KelvinVoigt):
    @staticmethod
    def form_definitions(mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels,fixed_facet_labels):
        return  solidforms.KelvinVoigtWEpithelium(
            mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels,fixed_facet_labels)


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
    def form_definitions(mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels,fixed_facet_labels):
        return \
            solidforms.IncompSwellingKelvinVoigt(
                mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels,fixed_facet_labels)

class SwellingKelvinVoigt(NodalContactSolid):
    """
    Kelvin Voigt model allowing for a swelling field
    """
    PROPERTY_DEFAULTS = {
        'emod': 10e3 * PASCAL_TO_CGS,
        'v_swelling': 1.0,
        'rho': 1000 * SI_DENSITY_TO_CGS,
        'eta': 3.0,
        'ycontact': 0.61-0.001,
        'kcontact': 1e11}

    @staticmethod
    def form_definitions(mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels,fixed_facet_labels):
        return \
            solidforms.SwellingKelvinVoigt(
                mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels,fixed_facet_labels)

class SwellingKelvinVoigtWEpithelium(NodalContactSolid):
    @staticmethod
    def form_definitions(mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels,fixed_facet_labels):
        return \
            solidforms.SwellingKelvinVoigtWEpithelium(
                mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels,fixed_facet_labels)

class Approximate3DKelvinVoigt(NodalContactSolid):
    PROPERTY_DEFAULTS = {
        'emod': 10e3 * PASCAL_TO_CGS,
        'nu': 0.49,
        'rho': 1000 * SI_DENSITY_TO_CGS,
        'eta': 3.0,
        'ycontact': 0.61-0.001,
        'kcontact': 1e11}

    @staticmethod
    def form_definitions(mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels,fixed_facet_labels):
        return \
            solidforms.Approximate3DKelvinVoigt(
                mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels,fixed_facet_labels)


def newton_solve(x0, linearized_subproblem, params=None):
    """
    Solve a non-linear problem with Newton-Raphson

    Parameters
    ----------
    x0 : A
        Initial guess
    linearized_subproblem : fn(A) -> (fn() -> A, fn(A) -> A)
        Callable returning a residual and linear solver about a linearizing state.
    params : dict
        Dictionary of parameters for newton solver
        {'absolute_tolerance', 'relative_tolerance', 'maximum_iterations'}

    Returns
    -------
    xn
    """
    if params is None:
        params = DEFAULT_NEWTON_SOLVER_PRM

    abs_tol, rel_tol = params['absolute_tolerance'], params['relative_tolerance']
    max_iter = params['maximum_iterations']

    abs_errs, rel_errs = [], []
    n = 0
    state_n = x0
    while True:
        assem_res_n, solve_n = linearized_subproblem(state_n)
        res_n = assem_res_n()

        abs_err = abs(res_n.norm())
        abs_errs.append(abs_err)
        rel_err = 0.0 if abs_errs[0] == 0 else abs_err/abs_errs[0]
        rel_errs.append(rel_err)

        if abs_err <= abs_tol or rel_err <= rel_tol or n > max_iter:
            break
        else:
            dstate_n = solve_n(res_n)
            state_n = state_n - dstate_n
            n += 1

    if n > max_iter:
        wrn.warn("Newton solve failed to converge before maximum "
                 "iteration count reached.", UserWarning)
    return state_n, {'num_iter': n, 'abs_err': abs_errs[-1], 'rel_err': rel_errs[-1]}
