"""
Transient fluid model definition

TODO: Change smoothing parameters to all be expressed in units of length
(all the smoothing parameters have a smoothing effect that occurs over a small length.
The smaller the length, the sharper the smoothing. )
"""

import numpy as np

from . import base
from femvf.parameters.properties import property_vecs
from femvf.constants import SI_DENSITY_TO_CGS
from blocktensor.linalg import BlockVector

from ..equations.fluid.bernoulli_sep_at_ratio import (
    bernoulli_qp
)

from ..equations.fluid import bernoulli_sep_at_min

## 1D Bernoulli approximation codes

class QuasiSteady1DFluid(base.Model):
    """
    This class represents a 1D fluid model

    Parameters
    ----------
    s: np.ndarray
        streamwise channel centreline coordinates
    area: np.ndarray
        channel cross-sectional areas
    """
    def __init__(self, s):
        NVERT = s.size
        # the 'mesh' (also the x coordinates in the reference configuration)
        self.s_vertices = s

        p0 = np.zeros(NVERT)
        q0 = np.zeros((1,))
        self.state0 = BlockVector((q0, p0), labels=(('q', 'p'),))

        p1 = np.zeros(NVERT)
        q1 = np.zeros(1)
        self.state1 = BlockVector((q1, p1), labels=(('q', 'p'),))

        # form type quantities associated with the mesh
        # displacement and velocity along the surface at state 0 and 1
        area = np.ones(NVERT)
        psub, psup = np.zeros(1), np.zeros(1)
        self.control = BlockVector((area, psub, psup), labels=(('area', 'psub', 'psup'),))

        self.properties = self.get_properties_vec(set_default=True)

        self._dt = 1.0

    @property
    def fluid(self):
        return self

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        self._dt = value

    ## Parameter setting functions
    def set_ini_state(self, state):
        """
        Set the initial fluid state
        """
        self.state0[:] = state

    def set_fin_state(self, state):
        """
        Set the final fluid state
        """
        self.state1[:] = state

    def set_control(self, control):
        """
        Set the final surface displacement and velocity
        """
        self.control[:] = control

    def set_properties(self, props):
        """
        Set the fluid properties
        """
        self.properties[:] = props

    ## Get empty vectors
    def get_state_vec(self):
        """
        Return empty flow speed and pressure state vectors
        """
        return self.state1.copy()

    def get_properties_vec(self, set_default=True):
        """
        Return a BlockVector representing the properties of the fluid
        """
        field_size = 1
        prop_defaults = None
        if set_default:
            prop_defaults = self.PROPERTY_DEFAULTS
        vecs, labels = property_vecs(field_size, self.PROPERTY_TYPES, prop_defaults)

        return BlockVector(vecs, labels=[labels])

    def get_control_vec(self):
        ret = self.control.copy()
        ret.set(0.0)
        return ret

    ## Residual and sensitivity methods
    # These are all specific to the Bernoulli model below?
    def apply_dres_dstate0(self, x):
        dres = self.get_state_vec()
        dres.set(0.0)
        return dres

class Bernoulli(QuasiSteady1DFluid):
    """
    Represents the Bernoulli fluid model

    TODO : Refactor this to behave similar to the Solid model. A mesh should be used, corresponding
    to the reference configuration of the fluid (conformal with reference configuration of solid?)
    One of the properties should be the mapping from the reference configuration to the current
    configuration that would be used in ALE.

    Note there are a total of 4 smoothing properties that are meant to approximate the
    ad-hoc separation criteria used in the literature.
        zeta_amin :
            Factor controlling the smoothness of the minimum area function.
            As zeta_amin->0, the exact minimum area is produced (Should behave similarly as a true
            min function)
        zeta_ainv :
            Approximates smoothing for an inverse area function; given an area, return the surface
            coordinate. As zeta_ainv->0, the function selects the closest surface coordinate at
            where the area matches some target area.
        zeta_sep :
            Controls the sharpness of the cutoff that models separation. As zeta_sep->0,
            the sharpness will approach an instantaneous jump from 1 to 0
        zeta_lb :
            Controls the smoothness of the lower bound function that restricts glottal area to be
            larger than a lower bound

    Properties
    ----------

    """
    PROPERTY_TYPES = {
        'a_sub': ('const', ()),
        'a_sup': ('const', ()),
        'rho_air': ('const', ()),
        'r_sep': ('const', ()),
        'zeta_min': ('const', ()),
        'zeta_sep': ('const', ()),
        'zeta_inv': ('const', ()),
        'zeta_lb': ('const', ()),
        'area_lb': ('const', ())}

    PROPERTY_DEFAULTS = {
        'a_sub': 100000,
        'a_sup': 0.6,
        'r_sep': 1.0,
        'rho_air': 1.225 * SI_DENSITY_TO_CGS,
        'zeta_min': 0.002/3,
        'zeta_sep': 0.002/3,
        'zeta_inv': 2.5*0.002,
        'zeta_lb': 0.002/3,
        'area_lb': 0.001}

    ## Model res sensitivity interface
    def res(self):
        return self.state1 - self.solve_state1(self.state0)[0]

    def solve_state1(self, state1):
        """
        Return the final flow state
        """
        qp, info = bernoulli_qp(self.s_vertices, *self.control.vecs, self.properties)
        ret_state1 = self.state1.copy()
        ret_state1['q'][0] = qp[0]
        ret_state1['p'][:] = qp[1]
        return ret_state1, info

    def solve_dres_dstate1(self, b):
        return b

    def solve_dres_dstate1_adj(self, x):
        return x

    def apply_dres_dp_adj(self, x):
        b = self.get_properties_vec()
        b.set(0.0)
        return b

    def apply_dres_dcontrol(self, x):
        raise NotImplementedError()

    def apply_dres_dcontrol_adj(self, x):
        raise NotImplementedError()

class BernoulliMinimumSeparation(QuasiSteady1DFluid):
    """
    Bernoulli fluid model with separation at the minimum

    """
    PROPERTY_TYPES = {
        'a_sub': ('const', ()),
        'a_sup': ('const', ()),
        'rho_air': ('const', ()),
        'r_sep': ('const', ()),
        'zeta_min': ('const', ()),
        'zeta_sep': ('const', ()),
        'zeta_inv': ('const', ()),
        'zeta_lb': ('const', ()),
        'area_lb': ('const', ())}

    PROPERTY_DEFAULTS = {
        'a_sub': 100000,
        'a_sup': 0.6,
        'r_sep': 1.0,
        'rho_air': 1.225 * SI_DENSITY_TO_CGS,
        'zeta_min': 0.002/3,
        'zeta_sep': 0.002/3,
        'zeta_inv': 2.5*0.002,
        'zeta_lb': 0.002/3,
        'area_lb': 0.001}

    ## Model res sensitivity interface
    def res(self):
        return self.state1 - self.solve_state1(self.state0)[0]

    def solve_state1(self, state1):
        """
        Return the final flow state
        """
        s = self.s_vertices
        area = self.control['area']
        psub = self.control['psub'][0]
        psup = self.control['psup'][0]

        rho = self.properties['rho_air'][0]
        zeta_min = self.properties['zeta_sep'][0]
        zeta_sep = self.properties['zeta_min'][0]

        return bernoulli_sep_at_min.bernoulli_qp(
            area, s, psub, psup, rho, zeta_min, zeta_sep
            )
