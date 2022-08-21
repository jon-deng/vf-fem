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
import functools
from typing import Tuple, Mapping

from femvf.solverconst import DEFAULT_NEWTON_SOLVER_PRM
from femvf.parameters.properties import property_vecs
from femvf.constants import PASCAL_TO_CGS, SI_DENSITY_TO_CGS

from blockarray.blockmat import BlockMatrix
from blockarray.blockvec import BlockVector
from blockarray.subops import diag_mat, zero_mat

from nonlineq import newton_solve

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

        # Generally the size of the vector comes directly from the property;
        # i.e. constants are size 1, scalar fields have size matching number of dofs, etc.
        # The time step `dt` is a special case since it is size 1 but is specificied as a field as
        # a workaround in order to take derivatives
        if isinstance(coefficient, dfn.function.constant.Constant) or label == 'dt':
            vec = np.ones(coefficient.values().size)
            vec[:] = coefficient.values()
        else:
            vec = coefficient.vector().copy()

        if label in defaults:
            vec[:] = defaults[label]

        vecs.append(vec)

    return BlockVector(vecs, labels=[labels])


class BaseTransientSolid(base.BaseTransientModel):
    """
    Class representing the discretized governing equations of a solid
    """
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

        self._dt_form = self.forms['coeff.time.dt']

        ## Define the state/controls/properties
        u0 = self.forms['coeff.state.u0']
        v0 = self.forms['coeff.state.v0']
        a0 = self.forms['coeff.state.a0']
        u1 = self.forms['coeff.state.u1']
        v1 = self.forms['coeff.state.v1']
        a1 = self.forms['coeff.state.a1']

        self.state0 = BlockVector((u0.vector(), v0.vector(), a0.vector()), labels=[('u', 'v', 'a')])
        self.state1 = BlockVector((u1.vector(), v1.vector(), a1.vector()), labels=[('u', 'v', 'a')])
        self.control = BlockVector((self.forms['coeff.fsi.p1'].vector(),), labels=[('p',)])
        self.props = properties_bvec_from_forms(self.forms)
        self.set_props(self.props)

        self.cached_form_assemblers = {
            key: CachedFormAssembler(self.forms[key]) for key in self.forms
            if 'form.' in key
        }

    @property
    def mesh(self):
        return self.forms['mesh.mesh']

    @property
    def XREF(self):
        xref = self.state0.sub[0].copy()
        xref[:] = self.forms['fspace.scalar'].tabulate_dof_coordinates().reshape(-1).copy()
        return xref

    @property
    def solid(self):
        return self

    @property
    def forms(self):
        return self._forms

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

    ## Parameter setting functions
    @property
    def dt(self):
        return self._dt_form.vector()[0]

    @dt.setter
    def dt(self, value):
        self._dt_form.vector()[:] = value

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
        for key, value in props.sub_items():
            # TODO: Check types to make sure the input property is compatible with the solid type
            coefficient = self.forms['coeff.prop.'+key]

            # If the property is a field variable, values have to be assigned to every spot in
            # the vector
            if isinstance(coefficient, dfn.function.constant.Constant):
                coefficient.assign(dfn.Constant(np.squeeze(value)))
            else:
                coefficient.vector()[:] = value

    ## Residual and sensitivity functions
    def assem_res(self):
        dt = self.dt
        u1, v1, a1 = self.state1.sub_blocks.flat

        res = self.state1.copy()
        values = [
            self.cached_form_assemblers['form.un.f1'].assemble(),
            v1 - newmark.newmark_v(u1, *self.state0.sub_blocks, dt),
            a1 - newmark.newmark_a(u1, *self.state0.sub_blocks, dt)
        ]
        res[:] = values
        self.forms['bc.dirichlet'].apply(res.sub['u'])
        return res

    @functools.cached_property
    def _const_assem_dres_dstate1(self):
        # These are constant matrices that only have to be computed once
        N = self.state1.bshape[0][0]
        # dfu_du = dfn.PETScMatrix(diag_mat(N, 1))
        dfu_du = None
        dfu_dv = dfn.PETScMatrix(zero_mat(N, N))
        dfu_da = dfn.PETScMatrix(zero_mat(N, N))

        dfv_du = dfn.PETScMatrix(diag_mat(N, 1))
        dfv_dv = dfn.PETScMatrix(diag_mat(N, 1))
        dfv_da = dfn.PETScMatrix(zero_mat(N, N))

        dfa_du = dfn.PETScMatrix(diag_mat(N, 1))
        dfa_dv = dfn.PETScMatrix(zero_mat(N, N))
        dfa_da = dfn.PETScMatrix(diag_mat(N, 1))
        return (
            dfu_du, dfu_dv, dfu_da,
            dfv_du, dfv_dv, dfv_da,
            dfa_du, dfa_dv, dfa_da
        )

    def assem_dres_dstate1(self):
        dfu_du = self.cached_form_assemblers['form.bi.df1_du1'].assemble()
        (_, dfu_dv, dfu_da,
        dfv_du, dfv_dv, dfv_da,
        dfa_du, dfa_dv, dfa_da) = self._const_assem_dres_dstate1
        dfv_du = -newmark.newmark_v_du1(self.dt) * dfv_du
        dfa_du = -newmark.newmark_a_du1(self.dt) * dfa_du

        submats = [
            dfu_du, dfu_dv, dfu_da,
            dfv_du, dfv_dv, dfv_da,
            dfa_du, dfa_dv, dfa_da
        ]
        return BlockMatrix(submats, shape=(3, 3), labels=2*self.state1.labels, check_bshape=False)

    def assem_dres_dstate0(self):
        assert len(self.state1.bshape) == 1
        N = self.state1.bshape[0][0]

        dfu_du = self.cached_form_assemblers['form.bi.df1_du0'].assemble()
        dfu_dv = self.cached_form_assemblers['form.bi.df1_dv0'].assemble()
        dfu_da = self.cached_form_assemblers['form.bi.df1_da0'].assemble()
        for mat in (dfu_du, dfu_dv, dfu_da):
            self.forms['bc.dirichlet'].apply(mat)

        dfv_du = dfn.PETScMatrix(diag_mat(N, 0 - newmark.newmark_v_du0(self.dt)))
        dfv_dv = dfn.PETScMatrix(diag_mat(N, 0 - newmark.newmark_v_dv0(self.dt)))
        dfv_da = dfn.PETScMatrix(diag_mat(N, 0 - newmark.newmark_v_da0(self.dt)))

        dfa_du = dfn.PETScMatrix(diag_mat(N, 0 - newmark.newmark_a_du0(self.dt)))
        dfa_dv = dfn.PETScMatrix(diag_mat(N, 0 - newmark.newmark_a_dv0(self.dt)))
        dfa_da = dfn.PETScMatrix(diag_mat(N, 0 - newmark.newmark_a_da0(self.dt)))

        submats = [
            dfu_du, dfu_dv, dfu_da,
            dfv_du, dfv_dv, dfv_da,
            dfa_du, dfa_dv, dfa_da
        ]
        return BlockMatrix(submats, shape=(3, 3), labels=2*self.state1.labels)

    def assem_dres_dcontrol(self):
        N = self.state1.bshape[0][0]
        M = self.control.bshape[0][0]

        # It should be hardcoded that the control is just the surface pressure
        assert self.control.shape[0] == 1
        dfu_dcontrol = self.cached_form_assemblers['form.bi.df1_dp1'].assemble()
        dfv_dcontrol = dfn.PETScMatrix(zero_mat(N, M))

        submats = [
            [dfu_dcontrol],
            [dfv_dcontrol]
        ]
        return BlockMatrix(submats, labels=self.state1.labels+self.control.labels)

    def assem_dres_dprops(self):
        raise NotImplementedError("Not implemented yet!")

    ## Solver functions
    def solve_state1(self, state1, options=None):
        if options is None:
            options = DEFAULT_NEWTON_SOLVER_PRM

        x = state1.copy()
        def linearized_subproblem(state):
            """
            Return a solver and residual corresponding to the linearized subproblem

            Returns
            -------
            assem_res : callable() -> type(state)
            solver : callable(type(state)) -> type(state)
            """
            self.set_fin_state(state)
            assem_res = self.assem_res

            def solve(res):
                dres_dstate1 = self.assem_dres_dstate1()
                return self.solve_dres_dstate1(dres_dstate1, x, res)
            return assem_res, solve

        state_n, solve_info = newton_solve(state1, linearized_subproblem, params=options)
        return state_n, solve_info

    def solve_dres_dstate1(self, dres_dstate1, x, b):
        """
        Solve the linearized residual problem

        This solution has a special format due to the Newmark time discretization.
        As a result, the below code only has to do one matrix solve (for the 'u'
        residual).
        """
        # dres_dstate1 = self.assem_dres_dstate1()

        dfu1_du1 = dres_dstate1.sub['u', 'u']
        dfv1_du1 = dres_dstate1.sub['v', 'u']
        dfa1_du1 = dres_dstate1.sub['a', 'u']

        bu, bv, ba = b.sub_blocks

        xu = x.sub['u']
        dfn.solve(dfu1_du1, xu, bu, 'petsc')
        x['v'][:] = bv - dfv1_du1*xu
        x['a'][:] = ba - dfa1_du1*xu

        return x

    def solve_dres_dstate1_adj(self, dres_dstate1_adj, x, b):
        """
        Solve the linearized adjoint residual problem

        This solution has a special format due to the Newmark time discretization.
        As a result, the below code only has to do one matrix solve (for the 'u'
        residual).
        """
        # Form key matrices
        dfu_du = dres_dstate1_adj.sub['u', 'u']
        dfv_du = dres_dstate1_adj.sub['v', 'u']
        dfa_du = dres_dstate1_adj.sub['a', 'u']

        # Solve A^T b = x
        bu, bv, ba = b.sub_blocks
        x.sub['a'][:] = ba
        x.sub['v'][:] = bv

        rhs_u = bu - (dfv_du*b['v'] + dfa_du*b['a'])
        dfn.solve(dfu_du, x['u'], rhs_u, 'petsc')
        return x


class NodalContactSolid(BaseTransientSolid):
    """
    This class modifies the default behaviour of the solid to implement contact pressures
    interpolated with the displacement function space. This involves manual modification of the
    matrices generated by FEniCS.
    """

    def set_fin_state(self, state):
        # This sets the 'standard' state variables u/v/a
        super().set_fin_state(state)

        self.forms['coeff.state.manual.tcontact'].vector()[:] = self._contact_traction(state.sub['u'])

    def assem_dres_dstate1(self):
        dres_dstate1 = super().assem_dres_dstate1()

        _ = dres_dstate1.sub['u', 'u']
        _ += self._assem_dresu_du_contact()
        return dres_dstate1

    def _contact_traction(self, u):
        # This computes the nodal values of the contact traction function
        XREF = self.XREF
        ycontact = self.forms['coeff.prop.ycontact'].values()[0]
        ncontact = self.forms['coeff.prop.ncontact'].values()
        kcontact = self.forms['coeff.prop.kcontact'].values()[0]

        gap = np.dot((XREF+u)[:].reshape(-1, 2), ncontact) - ycontact
        tcontact = (-solidforms.form_cubic_penalty_pressure(gap, kcontact)[:, None]*ncontact).reshape(-1).copy()
        return tcontact

    def _assem_dresu_du_contact(self, adjoint=False):
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
        dtcontact_du2 = self.forms['coeff.state.u1'].vector().copy()
        dpcontact_dgap, _ = solidforms.dform_cubic_penalty_pressure(gap, kcontact)
        dtcontact_du2[:] = np.array((-dpcontact_dgap[:, None]*dgap_du).reshape(-1))

        if adjoint:
            dfu2_dtcontact.mat().diagonalScale(dtcontact_du2.vec(), None)
        else:
            dfu2_dtcontact.mat().diagonalScale(None, dtcontact_du2.vec())
        dfu2_du2_contact = dfu2_dtcontact
        return dfu2_du2_contact

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
