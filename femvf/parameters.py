"""
Contains definitions of parametrizations. These objects should provide a mapping from their specific parameters to standardized parameters of the forward model, as well as the derivative of the map.
"""

from . import constants

import dolfin as dfn
from petsc4py import PETSc

class AbstractParametrization:
    """
    A parametrization is a map from one set of parameters to the 'standard' parameters of the
    forward model.

    Parameters
    ----------
    n : int
        The size of the parameter vector
    """

    def __init__(self, n, model):
        self.vector = dfn.PETScVector(PETSc.COMM_SELF, n)

        self.scalar_function_space = model.scalar_function_space
        self.vector_function_space = model.vector_function_space

    def convert(self):
        """
        Return the equivalent standardized parameters accepted by the forward model.

        Returns
        -------
        solid_props : dict
            A dictionary of solid properties
        fluid_props : dict
            A dictionary of solid properties
        """
        return NotImplementedError

    def dconvert(self, dg_solid_props, dg_fluid_props):
        """
        Return the equivalent standardized parameters accepted by the forward model.

        Parameters
        ----------
        dg_solid_props : dict
            The sensitivity of a functional with respect each property in solid_props
        dg_fluid_props:
            The sensitivity of a functional with respect each property in fluid_props

        Returns
        -------
        solid_props : dict
            A dictionary of derivatives of solid properties
        fluid_props : dict
            A dictionary of derivatives of fluid properties
        """
        return NotImplementedError

class NodalElasticModuli(AbstractParametrization):
    """
    A parametrization with variable elastic moduli at all nodes and default remaining parameter
    values.
    """

    def __init__(self, n, model):
        super(NodalElasticModuli, self).__init__(n, model)

        self.default_fluid_props = constants.DEFAULT_FLUID_PROPERTIES.copy()
        self.default_solid_props = constants.DEFAULT_SOLID_PROPERTIES.copy()

    def convert(self):
        out = dfn.Function(self.scalar_function_space)
        out.vector()[:] = self.vector

        solid_props = self.default_solid_props.copy()
        fluid_props = self.default_fluid_props.copy()

        solid_props['elastic_modulus'] = out

        return solid_props, fluid_props

    def dconvert(self, dg_solid_props, dg_fluid_props):
        pass

    def set_constant(self):
