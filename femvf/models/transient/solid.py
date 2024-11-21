"""
Module definining a 'Model' class to represent finite-element (FE) based, transient, model equations
"""

import numpy as np
import dolfin as dfn
import functools
from typing import Tuple, Mapping, Union

from femvf.solverconst import DEFAULT_NEWTON_SOLVER_PRM
from femvf.constants import PASCAL_TO_CGS, SI_DENSITY_TO_CGS

from blockarray.blockmat import BlockMatrix
from blockarray.blockvec import BlockVector
from blockarray.subops import diag_mat, zero_mat

from nonlineq import newton_solve

from . import base
from femvf.equations import newmark
from femvf.equations import form
from femvf.residuals import solid
from ..assemblyutils import CachedFormAssembler


def depack_form_coefficient_function(form_coefficient):
    """
    Return the coefficient `Function` instance from a `forms` dict value

    This function mainly handles tuple coefficents which occurs
    for the shape parameter
    """
    #
    if isinstance(form_coefficient, tuple):
        # Tuple coefficients consist of a `(function, ufl_object)` tuple
        coefficient, _ = form_coefficient
    else:
        coefficient = form_coefficient
    return coefficient


def properties_bvec_from_forms(forms, defaults=None):
    defaults = {} if defaults is None else defaults
    prop_labels = [
        form_name.split('.')[-1]
        for form_name in forms.keys()
        if 'coeff.prop' == form_name[: len('coeff.prop')]
    ]
    vecs = []
    for prop_label in prop_labels:
        coefficient = depack_form_coefficient_function(
            forms['coeff.prop.' + prop_label]
        )

        # Generally the size of the vector comes directly from the property;
        # i.e. constants are size 1, scalar fields have size matching number of dofs, etc.
        # The time step `dt` is a special case since it is size 1 but is specificied as a field as
        # a workaround in order to take derivatives
        if (
            isinstance(coefficient, dfn.function.constant.Constant)
            or prop_label == 'dt'
        ):
            vec = np.ones(coefficient.values().size)
            vec[:] = coefficient.values()
        else:
            vec = coefficient.vector().copy()

        if prop_label in defaults:
            vec[:] = defaults[prop_label]

        vecs.append(vec)

    return BlockVector(vecs, labels=[prop_labels])


class Model(base.BaseTransientModel):
    """
    Class representing the discretized governing equations of a solid
    """

    def __init__(self, residual: solid.FenicsResidual):

        self._residual = residual

        bilinear_forms = form.gen_residual_bilinear_forms(self.residual.form)

        ## Define the state/controls/properties
        u0 = self.residual.form['coeff.state.u0']
        v0 = self.residual.form['coeff.state.v0']
        a0 = self.residual.form['coeff.state.a0']
        u1 = self.residual.form['coeff.state.u1']
        v1 = self.residual.form['coeff.state.v1']
        a1 = self.residual.form['coeff.state.a1']

        self.state0 = BlockVector(
            (u0.vector(), v0.vector(), a0.vector()), labels=[('u', 'v', 'a')]
        )
        self.state1 = BlockVector(
            (u1.vector(), v1.vector(), a1.vector()), labels=[('u', 'v', 'a')]
        )
        self.control = BlockVector(
            (self.residual.form['coeff.fsi.p1'].vector(),), labels=[('p',)]
        )
        self.prop = properties_bvec_from_forms(self.residual.form)
        self.set_prop(self.prop)

        self.cached_form_assemblers = {
            key: CachedFormAssembler(biform, keep_diagonal=True)
            for key, biform in bilinear_forms.items()
            if 'form.' in key
        }
        self.cached_form_assemblers['form.un.f1'] = CachedFormAssembler(
            self.residual.form.form
        )

    @property
    def residual(self) -> solid.FenicsResidual:
        return self._residual

    @property
    def XREF(self) -> dfn.Function:
        xref = self.state0.sub[0].copy()
        function_space = self.residual.form['coeff.state.u1'].function_space()
        n_subspace = function_space.num_sub_spaces()

        xref[:] = (
            function_space.tabulate_dof_coordinates()[::n_subspace, :]
            .reshape(-1)
            .copy()
        )
        return xref

    @property
    def solid(self) -> 'Model':
        return self

    ## Parameter setting functions
    @property
    def dt(self):
        return self.residual.form['coeff.time.dt'].vector()[0]

    @dt.setter
    def dt(self, value):
        self.residual.form['coeff.time.dt'].vector()[:] = value

    def set_ini_state(self, state):
        """
        Sets the initial state variables, (u, v, a)

        Parameters
        ----------
        u0, v0, a0 : array_like
        """
        self.state0[:] = state

    def set_fin_state(self, state):
        """
        Sets the final state variables.

        Note that the solid forms are in displacement form so only the displacement is needed as an
        initial guess to solve the solid equations. The state `v1` and `a1` are specified explicitly
        by the Newmark relations once you solve for `u1`.

        Parameters
        ----------
        u1, v1, a1 : array_like
        """
        self.state1[:] = state

    def set_control(self, p1):
        self.control[:] = p1

    def set_prop(self, prop):
        """
        Sets the properties of the solid

        Properties are essentially all settable values that are not states. In UFL, all things that
        can have set values are `coefficients`, but this ditinguishes between state coefficients,
        and other property coefficients.

        Parameters
        ----------
        prop : Property / dict-like
        """
        for key, value in prop.sub_items():
            # TODO: Check types to make sure the input property is compatible with the solid type
            coefficient = self.residual.form['coeff.prop.' + key]

            # If the property is a field variable, values have to be assigned to every spot in
            # the vector
            if isinstance(coefficient, dfn.function.constant.Constant):
                coefficient.assign(dfn.Constant(np.squeeze(value)))
            else:
                coefficient.vector()[:] = value

        # If a shape parameter exists, it needs special handling to update the mesh coordinates
        if 'coeff.prop.umesh' in self.residual.form:
            u_mesh_coeff = self.residual.form['coeff.prop.umesh']

            mesh = self.residual.mesh()
            fspace = self.residual.form['coeff.state.u1'].function_space()
            ref_mesh_coord = self.residual.ref_mesh_coords
            VERT_TO_VDOF = dfn.vertex_to_dof_map(fspace)
            dmesh_coords = np.array(u_mesh_coeff.vector()[VERT_TO_VDOF]).reshape(
                ref_mesh_coord.shape
            )
            mesh_coord = ref_mesh_coord + dmesh_coords
            mesh.coordinates()[:] = mesh_coord

    ## Residual and sensitivity functions
    def assem_res(self):
        dt = self.dt
        u1, v1, a1 = self.state1.sub_blocks.flat

        res = self.state1.copy()
        values = [
            self.cached_form_assemblers['form.un.f1'].assemble(),
            v1 - newmark.newmark_v(u1, *self.state0.sub_blocks, dt),
            a1 - newmark.newmark_a(u1, *self.state0.sub_blocks, dt),
        ]
        res[:] = values
        for bc in self.residual.dirichlet_bcs['coeff.state.u1']:
            bc.apply(res.sub['u'])
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
        return (dfu_du, dfu_dv, dfu_da, dfv_du, dfv_dv, dfv_da, dfa_du, dfa_dv, dfa_da)

    def assem_dres_dstate1(self):
        # BUG: Applying BCs to a tensor (`dfn.PETScMatrix()`) then
        # trying to reassemble into that tensor seems to cause problems.
        # This is done with the `cached_form_assembler` since it caches the
        # tensor it applies on
        dfu_du = dfn.assemble(
            self.cached_form_assemblers['form.bi.df1_du1'].form,
            tensor=dfn.PETScMatrix(),
        )
        # dfu_du = self.cached_form_assemblers['form.bi.df1_du1'].assemble()
        for bc in self.residual.dirichlet_bcs['coeff.state.u1']:
            bc.apply(dfu_du)

        (_, dfu_dv, dfu_da, dfv_du, dfv_dv, dfv_da, dfa_du, dfa_dv, dfa_da) = (
            self._const_assem_dres_dstate1
        )
        dfv_du = -newmark.newmark_v_du1(self.dt) * dfv_du
        dfa_du = -newmark.newmark_a_du1(self.dt) * dfa_du

        submats = [
            dfu_du,
            dfu_dv,
            dfu_da,
            dfv_du,
            dfv_dv,
            dfv_da,
            dfa_du,
            dfa_dv,
            dfa_da,
        ]
        return BlockMatrix(
            submats, shape=(3, 3), labels=2 * self.state1.labels, check_bshape=False
        )

    def assem_dres_dstate0(self):
        assert len(self.state1.bshape) == 1
        N = self.state1.bshape[0][0]

        dfu_du = self.cached_form_assemblers['form.bi.df1_du0'].assemble()
        dfu_dv = self.cached_form_assemblers['form.bi.df1_dv0'].assemble()
        dfu_da = self.cached_form_assemblers['form.bi.df1_da0'].assemble()
        for mat in (dfu_du, dfu_dv, dfu_da):
            for bc in self.residual.dirichlet_bcs['coeff.state.u1']:
                bc.apply(mat)

        dfv_du = dfn.PETScMatrix(diag_mat(N, 0 - newmark.newmark_v_du0(self.dt)))
        dfv_dv = dfn.PETScMatrix(diag_mat(N, 0 - newmark.newmark_v_dv0(self.dt)))
        dfv_da = dfn.PETScMatrix(diag_mat(N, 0 - newmark.newmark_v_da0(self.dt)))

        dfa_du = dfn.PETScMatrix(diag_mat(N, 0 - newmark.newmark_a_du0(self.dt)))
        dfa_dv = dfn.PETScMatrix(diag_mat(N, 0 - newmark.newmark_a_dv0(self.dt)))
        dfa_da = dfn.PETScMatrix(diag_mat(N, 0 - newmark.newmark_a_da0(self.dt)))

        submats = [
            dfu_du,
            dfu_dv,
            dfu_da,
            dfv_du,
            dfv_dv,
            dfv_da,
            dfa_du,
            dfa_dv,
            dfa_da,
        ]
        return BlockMatrix(submats, shape=(3, 3), labels=2 * self.state1.labels)

    def assem_dres_dcontrol(self):
        N = self.state1.bshape[0][0]
        M = self.control.bshape[0][0]

        # It should be hardcoded that the control is just the surface pressure
        assert self.control.shape[0] == 1
        dfu_dcontrol = self.cached_form_assemblers['form.bi.df1_dp1'].assemble()
        dfv_dcontrol = dfn.PETScMatrix(zero_mat(N, M))

        submats = [[dfu_dcontrol], [dfv_dcontrol]]
        return BlockMatrix(submats, labels=self.state1.labels + self.control.labels)

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

        state_n, solve_info = newton_solve(
            state1, linearized_subproblem, params=options
        )
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
        x['v'][:] = bv - dfv1_du1 * xu
        x['a'][:] = ba - dfa1_du1 * xu

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

        rhs_u = bu - (dfv_du * b['v'] + dfa_du * b['a'])
        dfn.solve(dfu_du, x['u'], rhs_u, 'petsc')
        return x


class PredefinedModel(Model):
    def __init__(
        self,
        mesh: dfn.Mesh,
        mesh_functions: Tuple[dfn.MeshFunction],
        mesh_subdomains: Tuple[Mapping[str, int]],
        fsi_facet_labels: Tuple[str],
        fixed_facet_labels: Tuple[str],
    ):
        residual = self._make_residual(
            mesh,
            mesh_functions,
            mesh_subdomains,
            fsi_facet_labels,
            fixed_facet_labels,
        )

        new_form = form.modify_newmark_time_discretization(residual.form)
        new_residual = solid.FenicsResidual(
            new_form,
            mesh,
            mesh_functions,
            mesh_subdomains,
            fsi_facet_labels,
            fixed_facet_labels,
        )
        super().__init__(new_residual)

    def _make_residual(
        self,
        mesh: dfn.Mesh,
        mesh_functions: Tuple[dfn.MeshFunction],
        mesh_subdomains: Tuple[Mapping[str, int]],
        fsi_facet_labels: Tuple[str],
        fixed_facet_labels: Tuple[str],
    ) -> solid.FenicsResidual:
        raise NotImplementedError()


class NodalContactSolid(PredefinedModel):
    """
    This class modifies the default behaviour of the solid to implement contact
    pressures interpolated with the displacement function space. This involves
    manual modification of the matrices generated by FEniCS.
    """

    def _make_residual(
        self,
        mesh: dfn.Mesh,
        mesh_functions: Tuple[dfn.MeshFunction],
        mesh_subdomains: Tuple[Mapping[str, int]],
        fsi_facet_labels: Tuple[str],
        fixed_facet_labels: Tuple[str],
    ) -> solid.FenicsResidual:
        raise NotImplementedError()

    def set_fin_state(self, state):
        # This sets the 'standard' state variables u/v/a
        super().set_fin_state(state)

        self.residual.form['coeff.state.manual.tcontact'].vector()[:] = (
            self._contact_traction(state.sub['u'])
        )

    def assem_dres_dstate1(self):
        dres_dstate1 = super().assem_dres_dstate1()

        _ = dres_dstate1.sub['u', 'u']
        _ += self._assem_dresu_du_contact()
        return dres_dstate1

    def _contact_traction(self, u):
        # This computes the nodal values of the contact traction function
        XREF = self.XREF
        ycontact = self.residual.form['coeff.prop.ycontact'].values()[0]
        ncontact = self.residual.form['coeff.prop.ncontact'].values()
        kcontact = self.residual.form['coeff.prop.kcontact'].values()[0]

        ndim = self.residual.form['coeff.state.u0'].ufl_shape[0]
        gap = np.dot((XREF + u)[:].reshape(-1, ndim), ncontact) - ycontact
        tcontact = (
            (-form.pressure_contact_cubic_penalty(gap, kcontact)[:, None] * ncontact)
            .reshape(-1)
            .copy()
        )
        return tcontact

    def _assem_dresu_du_contact(self, adjoint=False):
        # Compute things needed to find sensitivities of contact pressure
        dfu2_dtcontact = None
        if adjoint:
            dfu2_dtcontact = self.cached_form_assemblers[
                'form.bi.df1uva_dtcontact_adj'
            ].assemble()
        else:
            dfu2_dtcontact = self.cached_form_assemblers[
                'form.bi.df1uva_dtcontact'
            ].assemble()

        XREF = self.XREF
        kcontact = self.residual.form['coeff.prop.kcontact'].values()[0]
        ycontact = self.residual.form['coeff.prop.ycontact'].values()[0]
        u1 = self.residual.form['coeff.state.u1'].vector()
        ncontact = self.residual.form['coeff.prop.ncontact'].values()
        gap = (
            np.dot((XREF + u1)[:].reshape(-1, ncontact.shape[-1]), ncontact) - ycontact
        )
        dgap_du = ncontact

        # FIXME: This code below only works if n is aligned with the x/y axes.
        # for a general collision plane normal, the operation 'df_dtc*dtc_du' will
        # have to be represented by a block diagonal dtc_du (need to loop in python to do this). It
        # reduces to a diagonal if n is aligned with a coordinate axis.
        dtcontact_du2 = self.residual.form['coeff.state.u1'].vector().copy()
        dpcontact_dgap, _ = form.dform_cubic_penalty_pressure(gap, kcontact)
        dtcontact_du2[:] = np.array((-dpcontact_dgap[:, None] * dgap_du).reshape(-1))

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

    def _make_residual(
        self,
        mesh: dfn.Mesh,
        mesh_functions: Tuple[dfn.MeshFunction],
        mesh_subdomains: Tuple[Mapping[str, int]],
        fsi_facet_labels: Tuple[str],
        fixed_facet_labels: Tuple[str],
    ) -> solid.FenicsResidual:
        return solid.Rayleigh(
            mesh,
            mesh_functions,
            mesh_subdomains,
            fsi_facet_labels,
            fixed_facet_labels,
        )


class KelvinVoigt(NodalContactSolid):
    """
    Represents the governing equations of a Kelvin-Voigt damped solid
    """

    def _make_residual(
        self,
        mesh: dfn.Mesh,
        mesh_functions: Tuple[dfn.MeshFunction],
        mesh_subdomains: Tuple[Mapping[str, int]],
        fsi_facet_labels: Tuple[str],
        fixed_facet_labels: Tuple[str],
    ) -> solid.FenicsResidual:
        return solid.KelvinVoigt(
            mesh,
            mesh_functions,
            mesh_subdomains,
            fsi_facet_labels,
            fixed_facet_labels,
        )


class KelvinVoigtWEpithelium(NodalContactSolid):

    def _make_residual(
        self,
        mesh: dfn.Mesh,
        mesh_functions: Tuple[dfn.MeshFunction],
        mesh_subdomains: Tuple[Mapping[str, int]],
        fsi_facet_labels: Tuple[str],
        fixed_facet_labels: Tuple[str],
    ) -> solid.FenicsResidual:
        return solid.KelvinVoigtWEpithelium(
            mesh,
            mesh_functions,
            mesh_subdomains,
            fsi_facet_labels,
            fixed_facet_labels,
        )


class KelvinVoigtWShape(NodalContactSolid):
    """
    Represents the governing equations of a Kelvin-Voigt damped solid
    """

    def _make_residual(
        self,
        mesh: dfn.Mesh,
        mesh_functions: Tuple[dfn.MeshFunction],
        mesh_subdomains: Tuple[Mapping[str, int]],
        fsi_facet_labels: Tuple[str],
        fixed_facet_labels: Tuple[str],
    ) -> solid.FenicsResidual:
        return solid.KelvinVoigtWShape(
            mesh,
            mesh_functions,
            mesh_subdomains,
            fsi_facet_labels,
            fixed_facet_labels,
        )


class IncompSwellingKelvinVoigt(NodalContactSolid):
    """
    Kelvin Voigt model allowing for a swelling field
    """

    def _make_residual(
        self,
        mesh: dfn.Mesh,
        mesh_functions: Tuple[dfn.MeshFunction],
        mesh_subdomains: Tuple[Mapping[str, int]],
        fsi_facet_labels: Tuple[str],
        fixed_facet_labels: Tuple[str],
    ) -> solid.FenicsResidual:
        return solid.IncompSwellingKelvinVoigt(
            mesh,
            mesh_functions,
            mesh_subdomains,
            fsi_facet_labels,
            fixed_facet_labels,
        )


class SwellingKelvinVoigt(NodalContactSolid):
    """
    Kelvin Voigt model allowing for a swelling field
    """

    def _make_residual(
        self,
        mesh: dfn.Mesh,
        mesh_functions: Tuple[dfn.MeshFunction],
        mesh_subdomains: Tuple[Mapping[str, int]],
        fsi_facet_labels: Tuple[str],
        fixed_facet_labels: Tuple[str],
    ) -> solid.FenicsResidual:
        return solid.SwellingKelvinVoigt(
            mesh,
            mesh_functions,
            mesh_subdomains,
            fsi_facet_labels,
            fixed_facet_labels,
        )


class SwellingKelvinVoigtWEpithelium(NodalContactSolid):

    def _make_residual(
        self,
        mesh: dfn.Mesh,
        mesh_functions: Tuple[dfn.MeshFunction],
        mesh_subdomains: Tuple[Mapping[str, int]],
        fsi_facet_labels: Tuple[str],
        fixed_facet_labels: Tuple[str],
    ) -> solid.FenicsResidual:
        return solid.SwellingKelvinVoigtWEpithelium(
            mesh,
            mesh_functions,
            mesh_subdomains,
            fsi_facet_labels,
            fixed_facet_labels,
        )


class SwellingKelvinVoigtWEpitheliumNoShape(NodalContactSolid):

    def _make_residual(
        self,
        mesh: dfn.Mesh,
        mesh_functions: Tuple[dfn.MeshFunction],
        mesh_subdomains: Tuple[Mapping[str, int]],
        fsi_facet_labels: Tuple[str],
        fixed_facet_labels: Tuple[str],
    ) -> solid.FenicsResidual:
        return solid.SwellingKelvinVoigtWEpitheliumNoShape(
            mesh,
            mesh_functions,
            mesh_subdomains,
            fsi_facet_labels,
            fixed_facet_labels,
        )


class SwellingPowerLawKelvinVoigtWEpitheliumNoShape(NodalContactSolid):

    def _make_residual(
        self,
        mesh: dfn.Mesh,
        mesh_functions: Tuple[dfn.MeshFunction],
        mesh_subdomains: Tuple[Mapping[str, int]],
        fsi_facet_labels: Tuple[str],
        fixed_facet_labels: Tuple[str],
    ) -> solid.FenicsResidual:
        return solid.SwellingPowerLawKelvinVoigtWEpitheliumNoShape(
            mesh,
            mesh_functions,
            mesh_subdomains,
            fsi_facet_labels,
            fixed_facet_labels,
        )


class Approximate3DKelvinVoigt(NodalContactSolid):

    def _make_residual(
        self,
        mesh: dfn.Mesh,
        mesh_functions: Tuple[dfn.MeshFunction],
        mesh_subdomains: Tuple[Mapping[str, int]],
        fsi_facet_labels: Tuple[str],
        fixed_facet_labels: Tuple[str],
    ) -> solid.FenicsResidual:
        return solid.Approximate3DKelvinVoigt(
            mesh,
            mesh_functions,
            mesh_subdomains,
            fsi_facet_labels,
            fixed_facet_labels,
        )
