"""
Contains a solid nonlinear dynamic system class definition
"""

import numpy as np
import dolfin as dfn

import blocklinalg.linalg as bla

from .base import DynamicalSystemResidual
from ..models.solid import properties_bvec_from_forms
from ..models import solidforms
from ..models.solidforms import gen_residual_bilinear_forms, gen_hopf_forms

class SolidDynamicalSystemResidual(DynamicalSystemResidual):

    def __init__(
        self, mesh, facet_func, facet_label_to_id, cell_func, cell_label_to_id, 
        fsi_facet_labels, fixed_facet_labels):

        assert isinstance(fsi_facet_labels, (list, tuple))
        assert isinstance(fixed_facet_labels, (list, tuple))

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
        self.properties = self.get_properties_vec(set_default=True)
        self.set_properties(self.properties)

    @property
    def forms(self):
        return self._forms

    def form_definitions(self, *args):
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

    def assem_res(self):
        res_u = dfn.assemble(self.forms['form.un.f1uva'], tensor=dfn.PETScVector())
        res_v = self.v - self.ut
        return bla.BlockVec([res_u, res_v], ['u',  'v'])

    def assem_dres_dstate(self):
        dres_u_du = dfn.assemble(self.forms['form.bi.df1uva_du1'], tensor=dfn.PETScMatrix())
        dres_u_dv = dfn.assemble(self.forms['form.bi.df1uva_dv1'], tensor=dfn.PETScMatrix())

        n = self.v.vector().size()
        dres_v_du = bla.zero_mat(n, n)
        dres_v_dv = bla.ident_mat(n)

        mats = [
            [dres_u_du, dres_u_dv],
            [dres_v_du, dres_v_dv]]
        return bla.BlockMat(mats, row_keys=['u', 'v'], col_keys=['u', 'v'])

    def assem_dres_dstatet(self):
        n = self.u.size()
        dres_u_dut = bla.zero_mat(n, n)
        dres_u_dvt = dfn.assemble(self.forms['form.bi.df1uva_da1'], tensor=dfn.PETScMatrix())

        dres_v_dut = -1*bla.ident_mat(n)
        dres_v_dvt = bla.zero_mat(n, n)

        mats = [
            [dres_u_du, dres_u_dv],
            [dres_v_du, dres_v_dv]]
        return bla.BlockMat(mats, row_keys=['u', 'v'], col_keys=['u', 'v'])

    def assem_dres_dprops(self):
        pass

class KelvinVoigt(SolidDynamicalSystemResidual):
    PROPERTY_DEFAULTS = {}
    @staticmethod
    def form_definitions(mesh, facet_func, facet_label_to_id, cell_func, cell_label_to_id,
                         fsi_facet_labels, fixed_facet_labels):
        return solidforms.KelvinVoigt(
                mesh, facet_func, facet_label_to_id, cell_func, cell_label_to_id,
                fsi_facet_labels, fixed_facet_labels)
