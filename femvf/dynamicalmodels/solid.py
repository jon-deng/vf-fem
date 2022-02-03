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

       
        self.state1 = BlockVec((self.u1.vector(), self.v1.vector(), self.a1.vector()), ('u', 'v', 'a'))
        self.icontrol1 = BlockVec((self.forms['coeff.fsi.p1'].vector(),), ('p',))
        self.properties = self.get_properties_vec(set_default=True)
        self.set_properties(self.properties)

    # def form_defin

    def get_properties_vec(self, set_default=True):
        defaults = self.PROPERTY_DEFAULTS if set_default else None
        return properties_bvec_from_forms(self.forms, defaults)