
import pytest

import numpy as np

from femvf.models.transient import solid, fluid
from femvf.load import load_transient_fsi_model
from femvf import meshutils
from femvf.parameters import parameterization

from blockarray import linalg

class TestParameterization:

    @pytest.fixture
    def setup_model(self):
        mesh_path = '../meshes/M5-3layers.xml'
        model = load_transient_fsi_model(
            mesh_path,
            None,
            SolidType=solid.KelvinVoigt,
            FluidType=fluid.Bernoulli
        )
        return model

    def test_layer_moduli(self, setup_model):
        model = setup_model
        cell_label_to_dofs = meshutils.process_celllabel_to_dofs_from_forms(
            model.solid.forms,
            model.solid.forms['coeff.prop.emod'].function_space()
        )

        layer_moduli = parameterization.LayerModuli(model, model.props)

        x = layer_moduli.x.copy()
        x['cover'][0] = 1.0
        x['body'][0] = 2.0
        x.print_summary()

        y = layer_moduli.apply(x)
        assert all(np.all(x[label] == y['emod'][cell_label_to_dofs[label]]) for label in x.labels[0])

    def test_identity(self, setup_model):
        model = setup_model
        identity = parameterization.Identity(model, model.props)

        x = identity.x.copy()
        x['emod'][:] = 1.0
        x['rho'][:] = 2.0

        y = identity.apply(x)
        assert all(np.all(x[label] == y[label]) for label in x.labels[0])

# if __name__ == '__main__':