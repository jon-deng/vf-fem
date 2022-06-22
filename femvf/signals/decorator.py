"""
Contains the decorator used to transform function that return one signal at one
state to signals over all states in a file
"""

from typing import Optional, Iterable

import numpy as np

import femvf.statefile as sf

def transform_to_make_signals(make_signal):
    def make_signals(model, *args, **kwargs):

        def proc_signals(f):
            proc_signal = make_signal(model, *args, **kwargs)

            props = f.get_props()
            model.set_props(props)

            signals = [
                proc_signal(f.get_state(ii), f.get_control(ii), props)
                for ii in range(f.size)]
            return np.array(signals)
        return proc_signals
    return make_signals


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

class StateMeasure():
    """
    Returns data from a model state

    Parameters
    ----------
    model :
    """

    def __init__(self, model, *args, **kwargs):
        self._model = model

        self.__init_measure_context__(*args, **kwargs)

    def __init_measure_context__(self, *args, **kwargs):
        """
        Define any context variables needed to compute the post-processed thing
        """
        raise NotImplementedError("This function must be implemented by child classes")

    @property
    def model(self):
        return self._model

    def __call__(self, state, control, props):
        raise NotImplementedError("This function must be implemented by child classes")

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
        super().__init__(self, func)
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
