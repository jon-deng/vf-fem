"""
Contains base classes for different functionals
"""

from typing import Optional, Iterable

import numpy as np

from femvf import statefile as sf
from femvf.models.transient.base import BaseTransientModel

class BaseStateMeasure():
    """
    A post-processing function that returns an output from (state, control, props)

    Parameters
    ----------
    model : BaseTransientModel
        The transient model to post-process
    kwargs :
        Optional keyword arguments for controlling the post-processed measure
        calculation
    """
    def __init__(self, model: BaseTransientModel, **kwargs):
        self._model = model

    def __call__(self, state, control, props):
        model = self.model
        model.set_props(props)
        model.set_fin_state(state)
        model.set_control(control)
        return self.assem()

    @property
    def model(self):
        return self._model

    def assem(self):
        raise NotImplementedError("Method must be implemented by subclasses")


class BaseDerivedStateMeasure(BaseStateMeasure):
    """
    Returns measures derived from post-processed data at single instants

    Parameters
    ----------
    func : Callable with signature `func(state, control, props)`
    """

    def __init__(self, func: BaseStateMeasure):
        self._func = func
        super().__init__(func.model)

    @property
    def func(self):
        return self._func

    def __call__(self, state, control, props):
        raise NotImplementedError("This function must be implemented by child classes")

class Derived():
    """
    Returns measures derived from post-processed data at single instants

    Parameters
    ----------
    func : Callable with signature `func(state, control, props)`
    """

    def __init__(self, func):
        self._func = func

    @property
    def func(self):
        return self._func

class TimeSeries(Derived):
    """
    Returns time series data

    Parameters
    ----------
    func : Callable with signature `func(state, control, props)`
    """

    def __call__(self, f: sf.StateFile, ns: Optional[Iterable]=None):
        props = f.get_props()
        self.func.model.set_props(props)

        signals = [
            self.func(f.get_state(ii), f.get_control(ii), props)
            for ii in range(f.size)
        ]
        return np.array(signals)

class TimeSeriesStats(Derived):
    """
    Returns time series statistics
    """

    def __init__(self, func):
        super().__init__(func)
        self._ts = TimeSeries(func)

    @property
    def ts(self):
        return self._ts

    def mean(self, f: sf.StateFile, ns: Optional[Iterable]=None):
        props = f.get_props()
        self.func.model.set_props(props)

        return np.mean(self.ts(f, ns), axis=0)

    def std(self, f: sf.StateFile, ns: Optional[Iterable]=None):
        props = f.get_props()
        self.func.model.set_props(props)

        return np.std(self.ts(f, ns), axis=0)
