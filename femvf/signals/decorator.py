"""
Contains the decorator used to transform function that return one signal at one
state to signals over all states in a file
"""
import numpy as np

def transform_to_proc_signals(make_signal):
    def proc_signals(model, f):
        proc_signal = make_signal(model)
        
        props = model.get_properties()
        signals = [
            proc_signal(model, f.get_state(ii), f.get_control(ii), props)
            for ii in range(f.size)]
        return np.array(signals)
    return proc_signals