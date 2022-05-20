"""
Contains the decorator used to transform function that return one signal at one
state to signals over all states in a file
"""
import numpy as np

def transform_to_make_signals(make_signal):
    def make_signals(model):
        def proc_signals(f):
            proc_signal = make_signal(model)

            props = f.get_props()
            signals = [
                proc_signal(f.get_state(ii), f.get_control(ii), props)
                for ii in range(f.size)]
            return np.array(signals)
        return proc_signals
    return make_signals