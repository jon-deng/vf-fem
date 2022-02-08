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

import numpy as np
import dolfin as dfn

import blocklinalg.linalg as bla

from .base import DynamicalSystem
from ..models.solid import properties_bvec_from_forms
from ..models import solidforms
from ..models.solidforms import gen_residual_bilinear_forms, gen_hopf_forms

# pylint: disable=abstract-method

class BaseSolidDynamicalSystem(DynamicalSystem):

    def __init__(
        self, mesh, facet_func, facet_label_to_id, cell_func, cell_label_to_id, 
        fsi_facet_labels, fixed_facet_labels, residual_form_name='f1uva'):


        assert isinstance(fsi_facet_labels, (list, tuple))
        assert isinstance(fixed_facet_labels, (list, tuple))
        self.residual_form_name = residual_form_name
        self._forms = self.form_definitions(mesh, facet_func, facet_label_to_id,
                                            cell_func, cell_label_to_id, fsi_facet_labels, fixed_facet_labels)
        gen_residual_bilinear_forms(self._forms)
        gen_hopf_forms(self._forms)

        self.u = self.forms['coeff.state.u1']
        self.v = self.forms['coeff.state.v1']
        self.state = bla.BlockVec((self.u.vector(), self.v.vector()), ('u', 'v'))

        self.ut = dfn.Function(self.forms['coeff.state.u1'].function_space())
        self.vt = self.forms['coeff.state.a1']
        self.statet = bla.BlockVec((self.ut.vector(), self.vt.vector()), ('ut', 'vt'))

        self.icontrol = bla.BlockVec((self.forms['coeff.fsi.p1'].vector(),), ('p',))

        self.du = self.forms['coeff.dstate.u1']
        self.dv = self.forms['coeff.dstate.v1']
        self.dstate = bla.BlockVec((self.du.vector(), self.dv.vector()), ('u', 'v'))

        self.dut = dfn.Function(self.forms['coeff.dstate.u1'].function_space())
        self.dvt = self.forms['coeff.dstate.a1']
        self.dstatet = bla.BlockVec((self.dut.vector(), self.dvt.vector()), ('ut', 'vt'))

        self.dicontrol = bla.BlockVec((self.forms['coeff.dfsi.p1'].vector(),), ('p',))


        self.properties = self.get_properties_vec(set_default=True)
        self.set_properties(self.properties)

    @property
    def forms(self):
        return self._forms

    def form_definitions(
        self, mesh, facet_func, facet_label_to_id, cell_func, cell_label_to_id,
        fsi_facet_labels, fixed_facet_labels):
        raise NotImplementedError("Must be implemented by child class")

    def get_properties_vec(self, set_default=True):
        defaults = self.PROPERTY_DEFAULTS if set_default else None
        return properties_bvec_from_forms(self.forms, defaults)

    def set_properties(self, props):
        for key in props.keys:
            # TODO: Check types to make sure the input property is compatible with the solid type
            coefficient = self.forms['coeff.prop.'+key]

            # If the property is a field variable, values have to be assigned to every spot in
            # the vector
            if isinstance(coefficient, dfn.function.constant.Constant):
                coefficient.assign(dfn.Constant(np.squeeze(props[key])))
            else:
                coefficient.vector()[:] = props[key]


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
        resu = dfn.assemble(
            self.forms[f'form.un.f1uva'], 
            tensor=dfn.PETScVector())
        resv = self.v - self.ut
        return bla.BlockVec([resu, resv], ['u',  'v'])

    def assem_dres_dstate(self):
        dresu_du = dfn.assemble(
            self.forms[f'form.bi.df1uva_du1'], 
            tensor=dfn.PETScMatrix())
        dresu_dv = dfn.assemble(
            self.forms[f'form.bi.df1uva_dv1'], 
            tensor=dfn.PETScMatrix())

        n = self.v.vector().size()
        dresv_du = bla.zero_mat(n, n)
        dresv_dv = bla.ident_mat(n)

        mats = [
            [dresu_du, dresu_dv],
            [dresv_du, dresv_dv]]
        return bla.BlockMat(mats, row_keys=['u', 'v'], col_keys=['u', 'v'])

    def assem_dres_dstatet(self):
        n = self.u.size()
        dresu_dut = bla.zero_mat(n, n)
        dresu_dvt = dfn.assemble(
            self.forms[f'form.bi.df1uva_da1'], 
            tensor=dfn.PETScMatrix())

        dresv_dut = -1*bla.ident_mat(n)
        dresv_dvt = bla.zero_mat(n, n)

        mats = [
            [dresu_dut, dresu_dvt],
            [dresv_dut, dresv_dvt]]
        return bla.BlockMat(mats, row_keys=['u', 'v'], col_keys=['u', 'v'])

    def assem_dres_dicontrol(self):
        n = self.u.vector().size()
        dresu_dicontrol = dfn.assemble(
            self.forms[f'form.bi.df1uva_dp1'], 
            tensor=dfn.PETScMatrix())

        dresv_dicontrol = bla.zero_mat(n, n)

        mats = [
            [dresu_dicontrol],
            [dresv_dicontrol]]
        return bla.BlockMat(mats, row_keys=['u', 'v'], col_keys=['icontrol'])

    def assem_dres_dprops(self):
        pass

class LinearStateSolidDynamicalSystem(BaseSolidDynamicalSystem):
    """
    Represents a linearized dynamical system residual

    This residual represents
    dF/dx(x, xt; g, ...) * dx
    where 'x=[u, v]', 'F=[Fu, Fv]=[Fu, v-ut]'
    """
    def assem_res(self):
        resu = (
            dfn.assemble(
                self.forms[f'form.un.df1uva_u1'], 
                tensor=dfn.PETScVector())
            + dfn.assemble(
                self.forms[f'form.un.df1uva_v1'],
                tensor=dfn.PETScVector())
            )
        resv = self.dv.vector()
        return bla.BlockVec([resu, resv], ['u',  'v'])

    def assem_dres_dstate(self):
        dresu_du = (
            dfn.assemble(
                self.forms[f'form.un.ddf1uva_u1_du1'], 
                tensor=dfn.PETScMatrix())
            + dfn.assemble(
                self.forms[f'form.un.ddf1uva_v1_du1'],
                tensor=dfn.PETScMatrix())
            )
        dresu_dv = (
            dfn.assemble(
                self.forms[f'form.un.ddf1uva_u1_dv1'], 
                tensor=dfn.PETScMatrix())
            + dfn.assemble(
                self.forms[f'form.un.ddf1uva_v1_dv1'],
                tensor=dfn.PETScMatrix())
            )

        n = self.u.vector().size()
        dresv_du = bla.zero_mat(n, n)
        dresv_dv = bla.zero_mat(n, n)

        mats = [
            [dresu_du, dresu_dv],
            [dresv_du, dresv_dv]]
        return bla.BlockMat(mats, row_keys=['u', 'v'], col_keys=['u', 'v'])

    def assem_dres_dstatet(self):
        n = self.u.vector().size()
        dresu_dut = bla.zero_mat(n, n)
        dresu_dvt = (
            dfn.assemble(
                self.forms[f'form.un.ddf1uva_u1_da1'], 
                tensor=dfn.PETScMatrix())
            + dfn.assemble(
                self.forms[f'form.un.ddf1uva_v1_da1'],
                tensor=dfn.PETScMatrix())
            )

        dresv_dut = bla.zero_mat(n, n)
        dresv_dvt = bla.zero_mat(n, n)

        mats = [
            [dresu_dut, dresu_dvt],
            [dresv_dut, dresv_dvt]]
        return bla.BlockMat(mats, row_keys=['u', 'v'], col_keys=['u', 'v'])

    def assem_dres_dicontrol(self):
        n = self.u.vector().size()
        dresu_dg = (
            dfn.assemble(
                self.forms[f'form.un.ddf1uva_u1_dp1'], 
                tensor=dfn.PETScMatrix())
            + dfn.assemble(
                self.forms[f'form.un.ddf1uva_v1_dp1'],
                tensor=dfn.PETScMatrix())
            )

        dresv_dg = bla.zero_mat(n, n)

        mats = [
            [dresu_dg],
            [dresv_dg]]
        return bla.BlockMat(mats, row_keys=['u', 'v'], col_keys=['g'])

class LinearStatetSolidDynamicalSystem(BaseSolidDynamicalSystem):
    """
    Represents a linearized dynamical system residual

    This residual represents
    dF/dxt(x, xt; g, ...) * dxt
    where 'x=[u, v]', 'F=[Fu, Fv]'
    """
    def assem_res(self):
        resu = (
            dfn.assemble(
                self.forms[f'form.un.df1uva_a1'], 
                tensor=dfn.PETScVector())
            )
        resv = -self.dut.vector()
        return bla.BlockVec([resu, resv], ['u',  'v'])

    def assem_dres_dstate(self):
        dresu_du = (
            dfn.assemble(
                self.forms[f'form.un.ddf1uva_a1_du1'], 
                tensor=dfn.PETScMatrix()))
        dresu_dv = (
            dfn.assemble(
                self.forms[f'form.un.ddf1uva_a1_dv1'], 
                tensor=dfn.PETScMatrix()))

        n = self.u.vector().size()
        dresv_du = bla.zero_mat(n, n)
        dresv_dv = bla.zero_mat(n, n)

        mats = [
            [dresu_du, dresu_dv],
            [dresv_du, dresv_dv]]
        return bla.BlockMat(mats, row_keys=['u', 'v'], col_keys=['u', 'v'])

    def assem_dres_dstatet(self):
        n = self.u.vector().size()
        dresu_dut = bla.zero_mat(n, n)
        dresu_dvt = (
            dfn.assemble(
                self.forms[f'form.un.ddf1uva_a1_da1'], 
                tensor=dfn.PETScMatrix()))

        dresv_dut = bla.zero_mat(n, n)
        dresv_dvt = bla.zero_mat(n, n)

        mats = [
            [dresu_dut, dresu_dvt],
            [dresv_dut, dresv_dvt]]
        return bla.BlockMat(mats, row_keys=['u', 'v'], col_keys=['u', 'v'])

    def assem_dres_dicontrol(self):
        n = self.u.vector().size()
        dresu_dg = (
            dfn.assemble(
                self.forms[f'form.un.ddf1uva_a1_dp1'], 
                tensor=dfn.PETScMatrix()))

        dresv_dg = bla.zero_mat(n, n)

        mats = [
            [dresu_dg],
            [dresv_dg]]
        return bla.BlockMat(mats, row_keys=['u', 'v'], col_keys=['g'])

class LinearIcontrolSolidDynamicalSystem(BaseSolidDynamicalSystem):
    """
    Represents a linearized dynamical system residual

    This residual represents
    dF/dg(x, xt; g, ...) * dg,
    where 'x=[u, v]', 'F=[Fu, Fv]'
    """
    def assem_res(self):
        resu = dfn.assemble(
            self.forms[f'form.un.df1uva_p1'], 
            tensor=dfn.PETScVector())
        resv = -self.v.vector().copy()
        resv.zero()
        return bla.BlockVec([resu, resv], ['u',  'v'])

    def assem_dres_dstate(self):
        dresu_du = dfn.assemble(
            self.forms[f'form.un.ddf1uva_p1_du1'], 
            tensor=dfn.PETScMatrix())
        dresu_dv = dfn.assemble(
            self.forms[f'form.un.ddf1uva_p1_dv1'], 
            tensor=dfn.PETScMatrix())

        n = self.u.vector().size()
        dresv_du = bla.zero_mat(n, n)
        dresv_dv = bla.zero_mat(n, n)
        
        mats = [
            [dresu_du, dresu_dv],
            [dresv_du, dresv_dv]]
        return bla.BlockMat(mats, row_keys=['u', 'v'], col_keys=['u', 'v'])

    def assem_dres_dstatet(self):
        n = self.u.vector().size()
        dresu_dut = bla.zero_mat(n, n)
        dresu_dvt = dfn.assemble(
            self.forms[f'form.un.ddf1uva_p1_da1'], 
            tensor=dfn.PETScMatrix())

        dresv_dut = bla.zero_mat(n, n)
        dresv_dvt = bla.zero_mat(n, n)

        mats = [
            [dresu_dut, dresu_dvt],
            [dresv_dut, dresv_dvt]]
        return bla.BlockMat(mats, row_keys=['u', 'v'], col_keys=['u', 'v'])

    def assem_dres_dicontrol(self):
        n = self.u.vector().size()
        m = self.forms['coeff.fsi.p1'].vector().size()
        dresu_dg = dfn.assemble(
            self.forms[f'form.un.ddf1uva_p1_dp1'], 
            tensor=dfn.PETScMatrix())
        dresv_dg = bla.zero_mat(n, m)


class KelvinVoigt(SolidDynamicalSystem):
    PROPERTY_DEFAULTS = {}
    @staticmethod
    def form_definitions(mesh, facet_func, facet_label_to_id, cell_func, cell_label_to_id,
                         fsi_facet_labels, fixed_facet_labels):
        return solidforms.KelvinVoigt(
                mesh, facet_func, facet_label_to_id, cell_func, cell_label_to_id,
                fsi_facet_labels, fixed_facet_labels)
