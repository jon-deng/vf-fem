"""
Contains a solid nonlinear dynamic system class definition

The nonlinear dynamic systems here are defined in Fenics and augmented a bit manually. The basic
dynamical system residual has a block form
F(x, xt, g) = [Fu(x, xt, g), Fv(x, xt)]
x = [u, v]
xt = [ut, vt]
and where
Fu(x, xt, g) : defined in Fenics with UFL (derivatives via UFL forms)
Fv(x, xt, g) = v-ut
"""

from typing import Tuple, Mapping

import numpy as np
import dolfin as dfn

from blockarray import blockvec as bvec, blockmat as bmat, subops

from .base import DynamicalSystem
from ..transient.solid import properties_bvec_from_forms
from ..equations.solid import solidforms
from ..equations.solid.solidforms import gen_residual_bilinear_forms, gen_hopf_forms
from ..assemble import CachedFormAssembler

# pylint: disable=abstract-method

class BaseSolidDynamicalSystem(DynamicalSystem):
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
        self.state = bvec.BlockVector((self.u.vector(), self.v.vector()), labels=[('u', 'v')])

        self.ut = dfn.Function(self.forms['coeff.state.u1'].function_space())
        self.vt = self.forms['coeff.state.a1']
        self.statet = bvec.BlockVector((self.ut.vector(), self.vt.vector()), labels=[('u', 'v')])

        self.control = bvec.BlockVector((self.forms['coeff.fsi.p1'].vector(),), labels=[('p',)])

        self.du = self.forms['coeff.dstate.u1']
        self.dv = self.forms['coeff.dstate.v1']
        self.dstate = bvec.BlockVector((self.du.vector(), self.dv.vector()), labels=[('u', 'v')])

        self.dut = dfn.Function(self.forms['coeff.dstate.u1'].function_space())
        self.dvt = self.forms['coeff.dstate.a1']
        self.dstatet = bvec.BlockVector((self.dut.vector(), self.dvt.vector()), labels=[('u', 'v')])

        # self.p = self.forms['coeff.dfsi.p1']
        self.dcontrol = bvec.BlockVector((self.forms['coeff.dfsi.p1'].vector(),), labels=[('p',)])


        self.props = self.get_properties_vec(set_default=True)
        self.set_props(self.props)

        self.cached_form_assemblers = {
            key: CachedFormAssembler(self.forms[key]) for key in self.forms
            if 'form.' in key
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

    def set_props(self, props):
        for key in props.labels[0]:
            # TODO: Check types to make sure the input property is compatible with the solid type
            coefficient = self.forms['coeff.prop.'+key]

            # If the property is a field variable, values have to be assigned to every spot in
            # the vector
            if isinstance(coefficient, dfn.function.constant.Constant):
                coefficient.assign(dfn.Constant(np.squeeze(props[key])))
            else:
                coefficient.vector()[:] = props[key]

    # Convenience methods
    @property
    def XREF(self):
        xref = dfn.Function(self.forms['fspace.vector'])
        xref.vector()[:] = self.forms['fspace.scalar'].tabulate_dof_coordinates().reshape(-1).copy()
        return xref


class SolidDynamicalSystem(BaseSolidDynamicalSystem):
    """
    Represents a dynamical system residual

    This residual has the form
    F(x, xt; g, ...) = [Fu, Fv],
    where 'x=[u,v]', 'Fu' is given by Fenics and 'Fv=v-ut'
    """
    # def form_definitions(self, *args):
    #     return super().form_definitions(*args)

    def assem_res(self):
        resu = self.cached_form_assemblers['form.un.f1uva'].assemble()
        resv = self.v.vector() - self.ut.vector()
        return bvec.BlockVector([resu, resv], labels=[['u',  'v']])

    def assem_dres_dstate(self):
        dresu_du = self.cached_form_assemblers['form.bi.df1uva_du1'].assemble()
        dresu_dv = self.cached_form_assemblers['form.bi.df1uva_dv1'].assemble()

        n = self.v.vector().size()
        dresv_du = dfn.PETScMatrix(subops.zero_mat(n, n))
        dresv_dv = dfn.PETScMatrix(subops.ident_mat(n))

        mats = [
            [dresu_du, dresu_dv],
            [dresv_du, dresv_dv]]
        return bvec.BlockMatrix(mats, labels=(['u', 'v'], ['u', 'v']))

    def assem_dres_dstatet(self):
        n = self.u.vector().size()
        dresu_dut = dfn.PETScMatrix(subops.zero_mat(n, n))
        dresu_dvt = self.cached_form_assemblers['form.bi.df1uva_da1'].assemble()

        dresv_dut = dfn.PETScMatrix(-1*subops.ident_mat(n))
        dresv_dvt = dfn.PETScMatrix(subops.zero_mat(n, n))

        mats = [
            [dresu_dut, dresu_dvt],
            [dresv_dut, dresv_dvt]]
        return bvec.BlockMatrix(mats, labels=(['u', 'v'], ['u', 'v']))

    def assem_dres_dcontrol(self):
        n = self.u.vector().size()
        dresu_dcontrol = self.cached_form_assemblers['form.bi.df1uva_dp1'].assemble()

        dresv_dcontrol = dfn.PETScMatrix(subops.zero_mat(self.state['v'].size(), self.control['p'].size()))

        mats = [
            [dresu_dcontrol],
            [dresv_dcontrol]]
        return bvec.BlockMatrix(mats, labels=self.state.labels+self.control.labels)

    def assem_dres_dprops(self):
        nu, nv = self.state['u'].size(), self.state['v'].size()
        mats = [
            [subops.zero_mat(nu, subops.size(prop_subvec)) for prop_subvec in self.props],
            [subops.zero_mat(nv, subops.size(prop_subvec)) for prop_subvec in self.props]]

        j_emod = self.props.labels[0].index('emod')
        mats[0][j_emod] = self.cached_form_assemblers['form.bi.df1uva_demod'].assemble()

        return bmat.BlockMatrix(
            mats, labels=(self.state.labels[0], self.props.labels[0]))

class LinearizedSolidDynamicalSystem(BaseSolidDynamicalSystem):
    """
    Represents a linearized dynamical system residual

    This residual represents
    dF/dx(x, xt, g; ...) * (dx, dxt, dg)
    where 'x=[u, v]', 'F=[Fu, Fv]=[Fu, v-ut]'
    """

    def assem_res(self):
        resu = (
            self.cached_form_assemblers['form.un.df1uva_u1'].assemble()
            + self.cached_form_assemblers['form.un.df1uva_v1'].assemble()
            + self.cached_form_assemblers['form.un.df1uva_p1'].assemble()
            + self.cached_form_assemblers['form.un.df1uva_a1'].assemble()
        )
        resv = self.dv.vector() - self.dut.vector()
        return bvec.BlockVector([resu, resv], labels=[['u',  'v']])

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
        return bvec.BlockMatrix(mats, labels=(['u', 'v'], ['u', 'v']))

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
        return bvec.BlockMatrix(mats, labels=(['u', 'v'], ['u', 'v']))

    def assem_dres_dcontrol(self):
        n = self.u.vector().size()
        m = self.control['p'].size()
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
        return bvec.BlockMatrix(mats, labels=(['u', 'v'], ['g']))

    def assem_dres_dprops(self):
        nu, nv = self.state['u'].size(), self.state['v'].size()
        mats = [
            [subops.zero_mat(nu, subops.size(prop_subvec)) for prop_subvec in self.props],
            [subops.zero_mat(nv, subops.size(prop_subvec)) for prop_subvec in self.props]]

        j_emod = self.props.labels[0].index('emod')
        mats[0][j_emod] = self.cached_form_assemblers['form.bi.ddf1uva_demod'].assemble()

        return bmat.BlockMatrix(
            mats, labels=self.state.labels+self.props.labels)


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