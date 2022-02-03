"""
Contains a solid nonlinear dynamic system class definition
"""

from .base import DynamicalSystemResidual

from ..models.solid import properties_bvec_from_forms

class SolidDynamicalSystemResidual(DynamicalSystemResidual):

    def __init__(
        self, mesh, facet_func, facet_label_to_id, cell_func, cell_label_to_id, 
        fsi_facet_labels, fixed_facet_labels):

        assert isinstance(fsi_facet_labels, (list, tuple))
        assert isinstance(fixed_facet_labels, (list, tuple))

        self._forms = self.form_definitions(mesh, facet_func, facet_label_to_id,
                                            cell_func, cell_label_to_id, fsi_facet_labels, fixed_facet_labels)
        gen_residual_bilinear_forms(self._forms)

        self.u = self.forms['coeff.state.u1']
        self.v = self.forms['coeff.state.v1']
        self.state = BlockVec((self.u.vector(), self.v.vector()), ('u', 'v'))

        self.ut = dfn.Function(self.forms['coeff.state.u1'].function_space())
        self.vt = self.forms['coeff.state.a1']
        self.statet = BlockVec((self.ut.vector(), self.vt.vector()), ('ut', 'vt'))

        self.icontrol = BlockVec((self.forms['coeff.fsi.p1'].vector(),), ('p',))
        self.properties = self.get_properties_vec(set_default=True)
        self.set_properties(self.properties)

    @property
    def forms(self):
        return self._forms

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