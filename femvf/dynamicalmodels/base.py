"""
Contains a basic nonlinear dynamical system class definition
"""

class DynamicalSystemResidual:

    # def __init__()

    def set_state(self, state):
        self.state[:] = state

    def set_statet(self, statet):
        self.statet[:] = statet

    def set_icontrol(self, icontrol):
        self.icontrol[:] = icontrol
        
    def set_properties(self, props):
        self.properties[:] = props