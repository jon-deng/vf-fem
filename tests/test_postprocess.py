"""
Test `femvf.postprocess`
"""

import pytest

from femvf.load import load_transient_fsi_model
from femvf.postprocess import (solid, base)
from femvf.models.transient import (solid as tsmd, fluid as tfmd)

class TestStateMeasure:
    """
    Test `BaseStateMeasure` subclasses
    """

    @pytest.fixture
    def setup_model(self):
        """
        Return a model (subclass of `BaseTransientModel`)
        """
        model = load_transient_fsi_model(
            '../meshes/M5-3layers.msh', None,
            SolidType=tsmd.KelvinVoigtWEpithelium,
            FluidType=tfmd.BernoulliAreaRatioSep,
            fsi_facet_labels=['pressure'],
            fixed_facet_labels=['fixed']
        )
        return model

    _StateMeasures = [
        solid.StressI1Field,
        solid.StressI2Field,
        solid.StressI3Field,
        solid.StressHydrostaticField,
        solid.StressVonMisesField,
        solid.ElasticStressField,
        solid.ViscousDissipationField,
        solid.ContactAreaDensityField,
        solid.FluidTractionPowerDensity,
    ]
    @pytest.fixture(params=_StateMeasures)
    def setup_state_measure(self, setup_model, request):
        """
        Return a `BaseStateMeasure` subclass
        """
        model = setup_model
        StateMeasure = request.param
        return StateMeasure(model)

    @staticmethod
    def args_from_model(model):
        """
        Return a `(state, control, props)` tuple
        """
        state = model.state1.copy()

        control = model.control.copy()

        props = model.props.copy()

        return state, control, props

    def test_state_measure(self, setup_state_measure):
        """
        Test if the state measure processes without errors

        This doesn't check for correctness of the measure yet.
        """
        state_measure = setup_state_measure
        state, control, props = self.args_from_model(state_measure.model)

        # Run the state measure for a given `(state, control, props)` tuple
        # and check that no errors are raised
        state_measure(state, control, props)
        assert True
