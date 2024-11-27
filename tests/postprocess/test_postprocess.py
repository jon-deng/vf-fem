"""
Test `femvf.postprocess`
"""

import pytest

from femvf import statefile as sf
from femvf.load import load_transient_fsi_model
from femvf.postprocess import solid, base
from femvf.models import transient


@pytest.fixture
def setup_model():
    """
    Return a model (subclass of `BaseTransientModel`)
    """
    model = load_transient_fsi_model(
        '../meshes/M5-3layers.msh',
        None,
        SolidResidual=transient.KelvinVoigtWEpithelium,
        FluidResidual=transient.BernoulliAreaRatioSep,
        fsi_facet_labels=['pressure'],
        fixed_facet_labels=['fixed'],
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
def setup_state_measure(setup_model, request):
    """
    Return a `BaseStateMeasure` subclass
    """
    model = setup_model
    StateMeasure = request.param
    return StateMeasure(model)


def args_from_model(model):
    """
    Return a `(state, control, prop)` tuple
    """
    state = model.state1.copy()

    control = model.control.copy()

    prop = model.prop.copy()

    return state, control, prop


class TestStateMeasure:
    """
    Test `BaseStateMeasure` subclasses
    """

    def test_state_measure(self, setup_state_measure):
        """
        Test if the state measure processes without errors

        This doesn't check for correctness of the measure yet.
        """
        state_measure = setup_state_measure
        state, control, prop = args_from_model(state_measure.model)

        # Run the state measure for a given `(state, control, prop)` tuple
        # and check that no errors are raised
        state_measure(state, control, prop)
        assert True


class TestStateHistoryMeasure:
    """
    Test `BaseStateHistoryMeasure` subclasses
    """

    @pytest.fixture
    def setup_time_series_measure(self, setup_state_measure):
        """
        Return a `TimeSeries` measure instance
        """
        state_measure = setup_state_measure
        return base.TimeSeries(state_measure)

    def test_time_series_measure(self, setup_time_series_measure, tmp_path):
        """
        Test if `TimeSeries` measures run
        """
        fpath = tmp_path / "test.h5"
        time_series_measure = setup_time_series_measure
        model = time_series_measure.model
        with sf.StateFile(model, str(fpath), mode='a') as f:
            # Add simple state history to the `StateFile` instance
            state, control, prop = args_from_model(model)
            for n in range(10):
                f.append_state(state)
                f.append_control(control)
                f.append_prop(prop)

            # Test if the `TimeSeries` measure runs w/o error
            time_series_measure(f)
            assert True
