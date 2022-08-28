"""
Post-processing functionality

There are two main post-processing functions/classes
`BaseStateMeasure` : These take a tuple `(state, control, props)` and return
    the post-processed measure.
`BaseStateHistoryMeasure` : These take a history of states through a statefile
    `(f)` and return the post-processed measure.
"""

from typing import Optional, Iterable

import numpy as np

from blockarray import blockvec as bv

from femvf import statefile as sf
from femvf.models.transient.base import BaseTransientModel

class BaseStateMeasure():
    """
    Post-process an output from known `(state, control, props)`

    Parameters
    ----------
    model : BaseTransientModel
        The transient model to post-process
    kwargs :
        Optional keyword arguments for controlling the post-processing
        calculation
    """
    def __init__(self, model: BaseTransientModel, **kwargs):
        self._model = model

    def __call__(
            self,
            state: bv.BlockVector,
            control: bv.BlockVector,
            props: bv.BlockVector
        ):
        model = self.model
        model.set_props(props)
        model.set_fin_state(state)
        model.set_control(control)
        return self.assem(state, control, props)

    @property
    def model(self):
        return self._model

    def assem(
            self,
            state: bv.BlockVector,
            control: bv.BlockVector,
            props: bv.BlockVector
        ):
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


class BaseStateHistoryMeasure():
    """
    Post-process an output from state history `(f)`

    Parameters
    ----------
    model : BaseTransientModel
        The transient model to post-process
    kwargs :
        Optional keyword arguments for controlling the post-processing
        calculation
    """

    def __init__(self, model: BaseTransientModel, **kwargs):
        self._model = model

    def __call__(self, f: sf.StateFile, **kwargs):
        return self.assem(f, **kwargs)

    @property
    def model(self):
        return self._model

    def assem(self, f: sf.StateFile, **kwargs):
        raise NotImplementedError("Method must be implemented by subclasses")

class BaseDerivedStateHistoryMeasure(BaseStateHistoryMeasure):
    """
    Post-process an output from state history `(f)`

    The post-processing function is derived from `BaseStateMeasure` type
    function.

    Parameters
    ----------
    func : BaseStateMeasure
        The state post-processing function
    """

    def __init__(self, func: BaseStateMeasure):
        super().__init__(func.model)
        self._func = func

    @property
    def func(self):
        return self._func

    def assem(self, f: sf.StateFile, **kwargs):
        raise NotImplementedError("Method must be implemented by subclasses")


class TimeSeries(BaseDerivedStateHistoryMeasure):
    """
    Return a time series of a state measure

    Parameters
    ----------
    func : BaseStateMeasure
        The state post-processing function
    """

    def assem(self, f: sf.StateFile, ns: Optional[Iterable]=None):
        props = f.get_props()
        self.func.model.set_props(props)

        signals = [
            self.func(f.get_state(ii), f.get_control(ii), props)
            for ii in range(f.size)
        ]
        return np.array(signals)

class TimeSeriesStats(BaseDerivedStateHistoryMeasure):
    """
    Return statistics over the time series of a state measure

    Parameters
    ----------
    func : BaseStateMeasure
        The state post-processing function
    """

    def __init__(self, func):
        super().__init__(func)
        self._ts = TimeSeries(func)

    @property
    def ts(self):
        return self._ts

    def assem(self, f: sf.StateFile, ns: Optional[Iterable]=None):
        return self.mean(f, ns=ns)

    def mean(self, f: sf.StateFile, ns: Optional[Iterable]=None):
        props = f.get_props()
        self.func.model.set_props(props)

        return np.mean(self.ts(f, ns), axis=0)

    def std(self, f: sf.StateFile, ns: Optional[Iterable]=None):
        props = f.get_props()
        self.func.model.set_props(props)

        return np.std(self.ts(f, ns), axis=0)
