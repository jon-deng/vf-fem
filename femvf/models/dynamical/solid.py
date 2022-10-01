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

from .base import BaseDynamicalModel
from ..transient.solid import properties_bvec_from_forms, depack_form_coefficient_function
from ..equations.solid import solidforms
from ..equations.solid.solidforms import gen_residual_bilinear_forms, gen_hopf_forms
from ..assemblyutils import CachedFormAssembler

# pylint: disable=abstract-method

class BaseDynamicalSolid(BaseDynamicalModel):
    PROPERTY_DEFAULTS = {}
    def __init__(
            self,
            mesh: dfn.Mesh,
            mesh_funcs: Tuple[dfn.MeshFunction],
            mesh_entities_label_to_value: Tuple[Mapping[str, int]],
            fsi_facet_labels: Tuple[str],
            fixed_facet_labels: Tuple[str],
            residual_form_name='f1uva'
        ):

        assert isinstance(fsi_facet_labels, (list, tuple))
        assert isinstance(fixed_facet_labels, (list, tuple))
        self.residual_form_name = residual_form_name
        self._forms = self.form_definitions(mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels)
        gen_residual_bilinear_forms(self._forms)
        gen_hopf_forms(self._forms)

        self.u = self.forms['coeff.state.u1']
        self.v = self.forms['coeff.state.v1']
        self.state = bv.BlockVector((self.u.vector(), self.v.vector()), labels=[('u', 'v')])
        self.state = bv.convert_subtype_to_petsc(self.state)

        self.ut = dfn.Function(self.forms['coeff.state.u1'].function_space())
        self.vt = self.forms['coeff.state.a1']
        self.statet = bv.BlockVector((self.ut.vector(), self.vt.vector()), labels=[('u', 'v')])
        self.statet = bv.convert_subtype_to_petsc(self.statet)

        self.control = bv.BlockVector((self.forms['coeff.fsi.p1'].vector(),), labels=[('p',)])
        self.control = bv.convert_subtype_to_petsc(self.control)

        self.du = self.forms['coeff.dstate.u1']
        self.dv = self.forms['coeff.dstate.v1']
        self.dstate = bv.BlockVector((self.du.vector(), self.dv.vector()), labels=[('u', 'v')])
        self.dstate = bv.convert_subtype_to_petsc(self.dstate)

        self.dut = dfn.Function(self.forms['coeff.dstate.u1'].function_space())
        self.dvt = self.forms['coeff.dstate.a1']
        self.dstatet = bv.BlockVector((self.dut.vector(), self.dvt.vector()), labels=[('u', 'v')])
        self.dstatet = bv.convert_subtype_to_petsc(self.dstatet)

        # self.p = self.forms['coeff.dfsi.p1']
        self.dcontrol = bv.BlockVector((self.forms['coeff.dfsi.p1'].vector(),), labels=[('p',)])
        self.dcontrol = bv.convert_subtype_to_petsc(self.dcontrol)

        self.props = self.get_properties_vec(set_default=True)
        self.props = bv.convert_subtype_to_petsc(self.props)
        self.set_props(self.props)

        self.cached_form_assemblers = {
            key: CachedFormAssembler(self.forms[key]) for key in self.forms
            if ('form.' in key and self.forms[key] is not None)
        }

    @property
    def forms(self):
        return self._forms

    def form_definitions(
            self,
            mesh: dfn.Mesh,
            mesh_funcs: Tuple[dfn.MeshFunction],
            mesh_entities_label_to_value: Tuple[Mapping[str, int]],
            fsi_facet_labels: Tuple[str],
            fixed_facet_labels: Tuple[str]
        ):
        raise NotImplementedError("Must be implemented by child class")

    def get_properties_vec(self, set_default=True):
        defaults = self.PROPERTY_DEFAULTS if set_default else None
        return properties_bvec_from_forms(self.forms, defaults)

    def set_state(self, state):
        self.state[:] = state

    def set_statet(self, statet):
        self.statet[:] = statet

    def set_control(self, control):
        self.control[:] = control

    def set_props(self, props):
        for key in props.labels[0]:
            # TODO: Check types to make sure the input property is compatible with the solid type
            coefficient = depack_form_coefficient_function(self.forms['coeff.prop.'+key])

            # If the property is a field variable, values have to be assigned to every spot in
            # the vector
            if isinstance(coefficient, dfn.function.constant.Constant):
                coefficient.assign(dfn.Constant(np.squeeze(props[key])))
            else:
                coefficient.vector()[:] = props[key]


    def set_dstate(self, dstate):
        self.dstate[:] = dstate

    def set_dstatet(self, dstatet):
        self.dstatet[:] = dstatet

    def set_dcontrol(self, dcontrol):
        self.dcontrol[:] = dcontrol

    # Convenience methods
    @property
    def XREF(self):
        xref = self.state.sub['u'].copy()
        xref[:] = self.forms['fspace.scalar'].tabulate_dof_coordinates().reshape(-1).copy()
        return xref


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

class SolidDynamicalSystem(BaseDynamicalSolid):
    """
    Represents a dynamical system residual

    This residual has the form
    F(x, xt; g, ...) = [Fu, Fv],
    where 'x=[u,v]', 'Fu' is given by Fenics and 'Fv=v-ut'
    """
    # def form_definitions(self, *args):
    #     return super().form_definitions(*args)

    @cast_output_bvec_to_petsc
    def assem_res(self):
        resu = self.cached_form_assemblers['form.un.f1uva'].assemble()
        resv = self.v.vector() - self.ut.vector()
        return bv.BlockVector([resu, resv], labels=[['u',  'v']])

    @cast_output_bmat_to_petsc
    def assem_dres_dstate(self):
        dresu_du = self.cached_form_assemblers['form.bi.df1uva_du1'].assemble()
        dresu_dv = self.cached_form_assemblers['form.bi.df1uva_dv1'].assemble()

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
        dresu_dvt = self.cached_form_assemblers['form.bi.df1uva_da1'].assemble()

        dresv_dut = dfn.PETScMatrix(-1*subops.ident_mat(n))
        dresv_dvt = dfn.PETScMatrix(subops.diag_mat(n, diag=0))

        mats = [
            [dresu_dut, dresu_dvt],
            [dresv_dut, dresv_dvt]]
        return bm.BlockMatrix(mats, labels=(['u', 'v'], ['u', 'v']))

    @cast_output_bmat_to_petsc
    def assem_dres_dcontrol(self):
        n = self.u.vector().size()
        dresu_dcontrol = self.cached_form_assemblers['form.bi.df1uva_dp1'].assemble()

        dresv_dcontrol = dfn.PETScMatrix(subops.zero_mat(self.state['v'].size, self.control['p'].size))

        mats = [
            [dresu_dcontrol],
            [dresv_dcontrol]]
        return bm.BlockMatrix(mats, labels=self.state.labels+self.control.labels)

    @cast_output_bmat_to_petsc
    def assem_dres_dprops(self):
        nu, nv = self.state['u'].size, self.state['v'].size
        mats = [
            [subops.zero_mat(nu, prop_subvec.size) for prop_subvec in self.props],
            [subops.zero_mat(nv, prop_subvec.size) for prop_subvec in self.props]]

        j_emod = self.props.labels[0].index('emod')
        mats[0][j_emod] = self.cached_form_assemblers['form.bi.df1uva_demod'].assemble()

        j_shape = self.props.labels[0].index('umesh')
        mats[0][j_shape] = self.cached_form_assemblers['form.bi.df1uva_dumesh'].assemble()

        return bm.BlockMatrix(
            mats, labels=(self.state.labels[0], self.props.labels[0]))

class LinearizedSolidDynamicalSystem(BaseDynamicalSolid):
    """
    Represents a linearized dynamical system residual

    This residual represents
    dF/dx(x, xt, g; ...) * (dx, dxt, dg)
    where 'x=[u, v]', 'F=[Fu, Fv]=[Fu, v-ut]'
    """

    @cast_output_bvec_to_petsc
    def assem_res(self):
        resu = (
            self.cached_form_assemblers['form.un.df1uva_u1'].assemble()
            + self.cached_form_assemblers['form.un.df1uva_v1'].assemble()
            + self.cached_form_assemblers['form.un.df1uva_p1'].assemble()
            + self.cached_form_assemblers['form.un.df1uva_a1'].assemble()
        )
        resv = self.dv.vector() - self.dut.vector()
        return bv.BlockVector([resu, resv], labels=[['u',  'v']])

    @cast_output_bmat_to_petsc
    def assem_dres_dstate(self):
        dresu_du = (
            self.cached_form_assemblers['form.bi.ddf1uva_u1_du1'].assemble()
            + self.cached_form_assemblers['form.bi.ddf1uva_v1_du1'].assemble()
            + self.cached_form_assemblers['form.bi.ddf1uva_p1_du1'].assemble()
            + self.cached_form_assemblers['form.bi.ddf1uva_a1_du1'].assemble()
        )
        dresu_dv = (
            self.cached_form_assemblers['form.bi.ddf1uva_u1_dv1'].assemble()
            + self.cached_form_assemblers['form.bi.ddf1uva_v1_dv1'].assemble()
            + self.cached_form_assemblers['form.bi.ddf1uva_p1_dv1'].assemble()
            + self.cached_form_assemblers['form.bi.ddf1uva_a1_dv1'].assemble()
        )

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
        dresu_dvt = (
            self.cached_form_assemblers['form.bi.ddf1uva_u1_da1'].assemble()
            + self.cached_form_assemblers['form.bi.ddf1uva_v1_da1'].assemble()
            + self.cached_form_assemblers['form.bi.ddf1uva_p1_da1'].assemble()
            + self.cached_form_assemblers['form.bi.ddf1uva_a1_da1'].assemble()
        )

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
        dresu_dg = (
            self.cached_form_assemblers['form.bi.ddf1uva_u1_dp1'].assemble()
            + self.cached_form_assemblers['form.bi.ddf1uva_v1_dp1'].assemble()
            + self.cached_form_assemblers['form.bi.ddf1uva_p1_dp1'].assemble()
            + self.cached_form_assemblers['form.bi.ddf1uva_a1_dp1'].assemble()
        )

        dresv_dg = dfn.PETScMatrix(subops.zero_mat(n, m))

        mats = [
            [dresu_dg],
            [dresv_dg]]
        return bm.BlockMatrix(mats, labels=(['u', 'v'], ['g']))

    @cast_output_bmat_to_petsc
    def assem_dres_dprops(self):
        nu, nv = self.state['u'].size, self.state['v'].size
        mats = [
            [subops.zero_mat(nu, subvec.size) for subvec in self.props],
            [subops.zero_mat(nv, subvec.size) for subvec in self.props]
        ]

        j_emod = self.props.labels[0].index('emod')
        mats[0][j_emod] = self.cached_form_assemblers['form.bi.ddf1uva_demod'].assemble()

        j_shape = self.props.labels[0].index('umesh')
        mats[0][j_shape] = self.cached_form_assemblers['form.bi.ddf1uva_dumesh'].assemble()

        return bm.BlockMatrix(
            mats, labels=self.state.labels+self.props.labels
        )


class KelvinVoigt(SolidDynamicalSystem):
    PROPERTY_DEFAULTS = {}
    @staticmethod
    def form_definitions(mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels):
        return solidforms.KelvinVoigt(
                mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels)

class LinearizedKelvinVoigt(LinearizedSolidDynamicalSystem):
    PROPERTY_DEFAULTS = {}
    @staticmethod
    def form_definitions(mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels):
        return solidforms.KelvinVoigt(
                mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels)

class KelvinVoigtWEpithelium(SolidDynamicalSystem):
    PROPERTY_DEFAULTS = {}
    @staticmethod
    def form_definitions(mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels):
        return solidforms.KelvinVoigtWEpithelium(
            mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels
        )

class LinearizedKelvinVoigtWEpithelium(LinearizedSolidDynamicalSystem):
    PROPERTY_DEFAULTS = {}
    @staticmethod
    def form_definitions(mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels):
        return solidforms.KelvinVoigtWEpithelium(
            mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels
        )

class SwellingKelvinVoigtWEpithelium(SolidDynamicalSystem):
    PROPERTY_DEFAULTS = {}
    @staticmethod
    def form_definitions(mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels):
        return solidforms.SwellingKelvinVoigtWEpithelium(
            mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels
        )

class SwellingLinearizedKelvinVoigtWEpithelium(LinearizedSolidDynamicalSystem):
    PROPERTY_DEFAULTS = {}
    @staticmethod
    def form_definitions(mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels):
        return solidforms.SwellingKelvinVoigtWEpithelium(
            mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels
        )

class Rayleigh(SolidDynamicalSystem):
    PROPERTY_DEFAULTS = {}
    @staticmethod
    def form_definitions(mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels):
        return solidforms.Rayleigh(
                mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels)

class LinearizedRayleigh(LinearizedSolidDynamicalSystem):
    PROPERTY_DEFAULTS = {}
    @staticmethod
    def form_definitions(mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels):
        return solidforms.Rayleigh(
                mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels)