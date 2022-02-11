"""
Contains a basic nonlinear dynamical system class definition
"""

class DynamicalSystem:

    # def __init__()

    def set_state(self, state):
        self.state[:] = state

    def set_statet(self, statet):
        self.statet[:] = statet

    def set_icontrol(self, icontrol):
        self.icontrol[:] = icontrol
        
    def set_properties(self, props):
        self.properties[:] = props


    def set_dstate(self, dstate):
        self.dstate[:] = dstate

    def set_dstatet(self, dstatet):
        self.dstatet[:] = dstatet

    def set_dicontrol(self, dicontrol):
        self.dicontrol[:] = dicontrol
        