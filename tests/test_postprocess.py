"""
Test `femvf.postprocess`
"""

import pytest

from femvf.load import load_transient_fsi_model
from femvf.postprocess import (solid, base)

class TestStateMeasure:
    """
    Test `BaseStateMeasure` subclasses
    """

    @pytest.fixture
    def setup_model(self):
        """
        Return a model (subclass of `BaseTransientModel`)
        """

    @pytest.fixture
    def setup_args(self):
        """
        Return a `(state, control, props)` tuple
        """
