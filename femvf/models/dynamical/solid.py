"""
Solid nonlinear dynamical system class definitions

The nonlinear dynamical systems here are defined in "FEniCS" and then augmented
a bit manually. The basic dynamical system is represented by a nonlinear
residual with the block form:
    - `F(x, xt, g) = [Fu(x, xt, g, p), Fv(x, xt)]`
    - `x = [u, v]`
    - `xt = [ut, vt]`
where `F` denotes the nonlinear residual and `x` and `xt` are the 'state' and
'state time derivative', respectively. The variables `u` and `v` stand for
'position' and 'velocity'. The final two parameters `g` and `p` denote some
arbitrary collections of control and model parameters, respectively.

The two blocks of `F` are defined by:
    - `Fu(x, xt, g)` is defined symbolically in "FEniCS" with with the `ufl`
    form language.
    - `Fv(x, xt, g)` is defined by `v-ut` through the derivative trick used to
    convert second order ODEs to first order.

The classes represent either the residual `F` or its linearization `F`, and each
class gives methods to evaluate the residual and its derivatives w.r.t the
parameters `x`, `xt`, `g`, `p`. This is done in the classes
`SolidDynamicalSystem` and `LinearizedSolidDynamicalSystem` below.
"""

from typing import Tuple, Mapping

import numpy as np
import dolfin as dfn

from blockarray import blockvec as bv, blockmat as bm, subops

from .base import BaseDynamicalModel, BaseLinearizedDynamicalModel
from ..transient.solid import properties_bvec_from_forms, depack_form_coefficient_function
from ..equations import solid
from ..assemblyutils import CachedFormAssembler

# pylint: disable=abstract-method

def cast_output_bmat_to_petsc(func):
    def wrapped_func(*args, **kwargs):
        mat = func(*args, **kwargs)
        return bm.convert_subtype_to_petsc(mat)
    return wrapped_func

def cast_output_bvec_to_petsc(func):
    def wrapped_func(*args, **kwargs):
        vec = func(*args, **kwargs)
        return bv.convert_subtype_to_petsc(vec)
    return wrapped_func

class DynamicalSolidModelInterface:

    def __init__(self, residual: solid.FenicsResidual):

        self._residual = residual

        # assert isinstance(fsi_facet_labels, (list, tuple))
        # assert isinstance(fixed_facet_labels, (list, tuple))
        # self.residual_form_name = residual_form_name
        # self._forms = self.form_definitions(mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels)
        # bilinear_forms = gen_residual_bilinear_forms(self.residual.form)
        hopf_forms = solid.gen_jac_state_forms(self.residual.form)
        prop_jac_forms = solid.gen_jac_property_forms(self.residual.form)
        forms = {**hopf_forms, **prop_jac_forms}

        self.u = self.residual.form['coeff.state.u1']
        self.v = self.residual.form['coeff.state.v1']
        self.state = bv.BlockVector((self.u.vector(), self.v.vector()), labels=[('u', 'v')])
        self.state = bv.convert_subtype_to_petsc(self.state)

        self.ut = dfn.Function(self.residual.form['coeff.state.u1'].function_space())
        self.vt = self.residual.form['coeff.state.a1']
        self.statet = bv.BlockVector((self.ut.vector(), self.vt.vector()), labels=[('u', 'v')])
        self.statet = bv.convert_subtype_to_petsc(self.statet)

        self.control = bv.BlockVector((self.residual.form['coeff.fsi.p1'].vector(),), labels=[('p',)])
        self.control = bv.convert_subtype_to_petsc(self.control)

        self.prop = properties_bvec_from_forms(self.residual.form)
        self.prop = bv.convert_subtype_to_petsc(self.prop)
        self.set_prop(self.prop)

        self.cached_form_assemblers = {
            key: CachedFormAssembler(form) for key, form in forms.items()
            if ('form.' in key and form is not None)
        }

        self.cached_form_assemblers['form.un.res'] = CachedFormAssembler(self.residual.form.form)

    @property
    def residual(self) -> solid.FenicsResidual:
        return self._residual

    def set_state(self, state):
        self.state[:] = state

    def set_statet(self, statet):
        self.statet[:] = statet

    def set_control(self, control):
        self.control[:] = control

    def set_prop(self, prop):
        for key in prop.labels[0]:
            # TODO: Check types to make sure the input property is compatible with the solid type
            coefficient = depack_form_coefficient_function(self.residual.form['coeff.prop.'+key])

            # If the property is a field variable, values have to be assigned to every spot in
            # the vector
            if isinstance(coefficient, dfn.function.constant.Constant):
                coefficient.assign(dfn.Constant(np.squeeze(prop[key])))
            else:
                coefficient.vector()[:] = prop[key]

        # If a shape parameter exists, it needs special handling to update the mesh coordinates
        if 'coeff.prop.umesh' in self.residual.form:
            u_mesh_coeff = self.residual.form['coeff.prop.umesh']

            mesh = self.residual.mesh()
            fspace = self.residual.form['coeff.state.u1'].function_space()
            ref_mesh_coord = self.residual.ref_mesh_coords
            VERT_TO_VDOF = dfn.vertex_to_dof_map(fspace)
            dmesh_coords = np.array(
                u_mesh_coeff.vector()[VERT_TO_VDOF]
            ).reshape(ref_mesh_coord.shape)
            mesh_coord = ref_mesh_coord + dmesh_coords
            mesh.coordinates()[:] = mesh_coord

    # Convenience methods
    @property
    def XREF(self) -> dfn.Function:
        xref = self.state.sub[0].copy()
        function_space = self.residual.form['coeff.state.u1'].function_space()
        n_subspace = function_space.num_sub_spaces()

        xref[:] = function_space.tabulate_dof_coordinates()[::n_subspace, :].reshape(-1).copy()
        return xref


class Model(DynamicalSolidModelInterface, BaseDynamicalModel):

    @cast_output_bvec_to_petsc
    def assem_res(self):
        resu = self.cached_form_assemblers['form.un.res'].assemble()
        resv = self.v.vector() - self.ut.vector()
        return bv.BlockVector([resu, resv], labels=[['u',  'v']])

    @cast_output_bmat_to_petsc
    def assem_dres_dstate(self):
        dresu_du = self.cached_form_assemblers['form.bi.dres_du1'].assemble()
        dresu_dv = self.cached_form_assemblers['form.bi.dres_dv1'].assemble()

        n = self.v.vector().size()
        dresv_du = dfn.PETScMatrix(subops.zero_mat(n, n))
        dresv_dv = dfn.PETScMatrix(subops.ident_mat(n))

        mats = [
            [dresu_du, dresu_dv],
            [dresv_du, dresv_dv]]
        return bm.BlockMatrix(mats, labels=(['u', 'v'], ['u', 'v']))

    @cast_output_bmat_to_petsc
    def assem_dres_dstatet(self):
        n = self.u.vector().size()
        dresu_dut = dfn.PETScMatrix(subops.diag_mat(n, diag=0))
        dresu_dvt = self.cached_form_assemblers['form.bi.dres_da1'].assemble()

        dresv_dut = dfn.PETScMatrix(-1*subops.ident_mat(n))
        dresv_dvt = dfn.PETScMatrix(subops.diag_mat(n, diag=0))

        mats = [
            [dresu_dut, dresu_dvt],
            [dresv_dut, dresv_dvt]]
        return bm.BlockMatrix(mats, labels=(['u', 'v'], ['u', 'v']))

    @cast_output_bmat_to_petsc
    def assem_dres_dcontrol(self):
        n = self.u.vector().size()
        dresu_dcontrol = self.cached_form_assemblers['form.bi.dres_dp1'].assemble()

        dresv_dcontrol = dfn.PETScMatrix(subops.zero_mat(self.state['v'].size, self.control['p'].size))

        mats = [
            [dresu_dcontrol],
            [dresv_dcontrol]]
        return bm.BlockMatrix(mats, labels=self.state.labels+self.control.labels)

    @cast_output_bmat_to_petsc
    def assem_dres_dprop(self):
        nu, nv = self.state['u'].size, self.state['v'].size
        mats = [
            [subops.zero_mat(nu, prop_subvec.size) for prop_subvec in self.prop],
            [subops.zero_mat(nv, prop_subvec.size) for prop_subvec in self.prop]]

        j_emod = self.prop.labels[0].index('emod')
        mats[0][j_emod] = self.cached_form_assemblers['form.bi.dres_demod'].assemble()

        if 'umesh' in self.prop:
            j_shape = self.prop.labels[0].index('umesh')
            mats[0][j_shape] = self.cached_form_assemblers['form.bi.dres_dumesh'].assemble()

        return bm.BlockMatrix(
            mats, labels=(self.state.labels[0], self.prop.labels[0]))

class LinearizedModel(DynamicalSolidModelInterface, BaseLinearizedDynamicalModel):

    def __init__(self, residual: solid.FenicsResidual):

        new_form = solid.modify_unary_linearized_forms(residual.form)
        new_residual = solid.FenicsResidual(
            new_form,
            residual.mesh(),
            residual._mesh_functions,
            residual._mesh_functions_label_to_value,
            residual.fsi_facet_labels,
            residual.fixed_facet_labels
        )
        super().__init__(new_residual)

        self.du = self.residual.form['coeff.dstate.u1']
        self.dv = self.residual.form['coeff.dstate.v1']
        self.dstate = bv.BlockVector((self.du.vector(), self.dv.vector()), labels=[('u', 'v')])
        self.dstate = bv.convert_subtype_to_petsc(self.dstate)

        self.dut = dfn.Function(self.residual.form['coeff.dstate.u1'].function_space())
        self.dvt = self.residual.form['coeff.dstate.a1']
        self.dstatet = bv.BlockVector((self.dut.vector(), self.dvt.vector()), labels=[('u', 'v')])
        self.dstatet = bv.convert_subtype_to_petsc(self.dstatet)

        # self.p = self.forms['coeff.dfsi.p1']
        self.dcontrol = bv.BlockVector((self.residual.form['coeff.dfsi.p1'].vector(),), labels=[('p',)])
        self.dcontrol = bv.convert_subtype_to_petsc(self.dcontrol)

    def set_dstate(self, dstate):
        self.dstate[:] = dstate

    def set_dstatet(self, dstatet):
        self.dstatet[:] = dstatet

    def set_dcontrol(self, dcontrol):
        self.dcontrol[:] = dcontrol

    @cast_output_bvec_to_petsc
    def assem_res(self):
        resu = self.cached_form_assemblers['form.un.res'].assemble()
        resv = self.dv.vector() - self.dut.vector()
        return bv.BlockVector([resu, resv], labels=[['u',  'v']])

    @cast_output_bmat_to_petsc
    def assem_dres_dstate(self):
        dresu_du = self.cached_form_assemblers['form.bi.dres_du1'].assemble()
        dresu_dv = self.cached_form_assemblers['form.bi.dres_dv1'].assemble()

        n = self.u.vector().size()
        dresv_du = dfn.PETScMatrix(subops.zero_mat(n, n))
        dresv_dv = dfn.PETScMatrix(subops.zero_mat(n, n))

        mats = [
            [dresu_du, dresu_dv],
            [dresv_du, dresv_dv]]
        return bm.BlockMatrix(mats, labels=(['u', 'v'], ['u', 'v']))

    @cast_output_bmat_to_petsc
    def assem_dres_dstatet(self):
        n = self.u.vector().size()
        dresu_dut = dfn.PETScMatrix(subops.zero_mat(n, n))
        dresu_dvt = self.cached_form_assemblers['form.bi.dres_da1'].assemble()

        dresv_dut = dfn.PETScMatrix(subops.zero_mat(n, n))
        dresv_dvt = dfn.PETScMatrix(subops.zero_mat(n, n))

        mats = [
            [dresu_dut, dresu_dvt],
            [dresv_dut, dresv_dvt]]
        return bm.BlockMatrix(mats, labels=(['u', 'v'], ['u', 'v']))

    @cast_output_bmat_to_petsc
    def assem_dres_dcontrol(self):
        n = self.u.vector().size()
        m = self.control['p'].size
        dresu_dg = self.cached_form_assemblers['form.bi.dres_dp1'].assemble()

        dresv_dg = dfn.PETScMatrix(subops.zero_mat(n, m))

        mats = [
            [dresu_dg],
            [dresv_dg]]
        return bm.BlockMatrix(mats, labels=(['u', 'v'], ['g']))

    @cast_output_bmat_to_petsc
    def assem_dres_dprop(self):
        nu, nv = self.state['u'].size, self.state['v'].size
        mats = [
            [subops.zero_mat(nu, subvec.size) for subvec in self.prop],
            [subops.zero_mat(nv, subvec.size) for subvec in self.prop]
        ]

        j_emod = self.prop.labels[0].index('emod')
        mats[0][j_emod] = self.cached_form_assemblers['form.bi.dres_demod'].assemble()

        if 'umesh' in self.prop:
            j_shape = self.prop.labels[0].index('umesh')
            mats[0][j_shape] = self.cached_form_assemblers['form.bi.dres_dumesh'].assemble()

        return bm.BlockMatrix(
            mats, labels=self.state.labels+self.prop.labels
        )


class PredefinedModel(Model):
    def __init__(
                self,
                mesh: dfn.Mesh,
                mesh_functions: Tuple[dfn.MeshFunction],
                mesh_functions_label_to_value: Tuple[Mapping[str, int]],
                fsi_facet_labels: Tuple[str],
                fixed_facet_labels: Tuple[str]
            ):
            residual = self._make_residual(
                mesh,
                mesh_functions,
                mesh_functions_label_to_value,
                fsi_facet_labels,
                fixed_facet_labels
            )
            super().__init__(residual)

class PredefinedLinearizedModel(LinearizedModel):

    def __init__(
            self,
            mesh: dfn.Mesh,
            mesh_functions: Tuple[dfn.MeshFunction],
            mesh_functions_label_to_value: Tuple[Mapping[str, int]],
            fsi_facet_labels: Tuple[str],
            fixed_facet_labels: Tuple[str]
        ):
        residual = self._make_residual(
                mesh,
                mesh_functions,
                mesh_functions_label_to_value,
                fsi_facet_labels,
                fixed_facet_labels
            )
        super().__init__(residual)


class KelvinVoigt(PredefinedModel):

    def _make_residual(self, mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels):
        return solid.KelvinVoigt(
            mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels
        )

class KelvinVoigtWShape(PredefinedModel):

    def _make_residual(self, mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels):
        return solid.KelvinVoigtWShape(
            mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels
        )

class LinearizedKelvinVoigt(PredefinedLinearizedModel):

    def _make_residual(self, mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels):
        return solid.KelvinVoigt(
            mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels
        )

class LinearizedKelvinVoigtWShape(PredefinedLinearizedModel):

    def _make_residual(self, mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels):
        return solid.KelvinVoigtWShape(
            mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels
        )

class KelvinVoigtWEpithelium(PredefinedModel):

    def _make_residual(self, mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels):
        return solid.KelvinVoigtWEpithelium(
            mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels
        )

class LinearizedKelvinVoigtWEpithelium(PredefinedLinearizedModel):

    def _make_residual(self, mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels):
        return solid.KelvinVoigtWEpithelium(
            mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels
        )

class SwellingKelvinVoigtWEpithelium(PredefinedModel):

    def _make_residual(self, mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels):
        return solid.SwellingKelvinVoigtWEpithelium(
            mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels
        )

class LinearizedSwellingKelvinVoigtWEpithelium(PredefinedLinearizedModel):

    def _make_residual(self, mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels):
        return solid.SwellingKelvinVoigtWEpithelium(
            mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels
        )

class SwellingKelvinVoigtWEpitheliumNoShape(PredefinedModel):

    def _make_residual(self, mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels):
        return solid.SwellingKelvinVoigtWEpitheliumNoShape(
            mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels
        )

class LinearizedSwellingKelvinVoigtWEpitheliumNoShape(PredefinedLinearizedModel):

    def _make_residual(self, mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels):
        return solid.modify_unary_linearized_forms(
            solid.SwellingKelvinVoigtWEpitheliumNoShape(
                mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels
            )
        )

class Rayleigh(PredefinedModel):

    def _make_residual(self, mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels):
        return solid.Rayleigh(
            mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels
        )

class LinearizedRayleigh(PredefinedLinearizedModel):

    def _make_residual(self, mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels):
        return solid.Rayleigh(
            mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels
        )
