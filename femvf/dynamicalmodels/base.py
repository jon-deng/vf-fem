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


    def set_dstate(self, state):
        self.dstate[:] = state

    def set_dstatet(self, statet):
        self.dstatet[:] = statet

    def set_dicontrol(self, icontrol):
        self.dicontrol[:] = icontrol