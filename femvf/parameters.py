"""
Contains definitions of parametrizations. These objects should provide a mapping from their specific parameters to standardized parameters of the forward model, as well as the derivative of the map.
"""

from collections import OrderedDict
from . import properties as props
from . import constants

import dolfin as dfn
from petsc4py import PETSc

import numpy as np

class AbstractParameterization:
    """
    A parameterization is a mapping from a set of parameters to those of the forward model.

    Parameters
    ----------
    x : array_like
        The parameter vector of the parameterization
    kwargs : optional
        Additional keyword arguments needed to specify the parameterization.
        These will vary depending on the specific parameterization.

    Attributes
    ----------
    SHAPES : dict(tuple(str, tuple), ...)
        A dictionary storing the shape of each labeled parameter in the parameterization

    Returns
    -------
    solid_props : properties.SolidProperties
    fluid_props : properties.FluidProperties
    """
    SHAPES = OrderedDict(
        {'abstract_parameters': ('field', ())}
        )

    # def __new__(cls, model, **kwargs):
    #     return super().__new__(cls)

    def __init__(self, model, **kwargs):
        self.model = model

        self.parameters = {}
        for key, shape in self.SHAPES.items():
            if shape[0] == 'field':
                self.parameters[key] = np.zeros((model.scalar_function_space.dim(), *shape[1]))
            else:
                self.parameters[key] = np.zeros((*shape[1]))

        # self.scalar_function_space = model.scalar_function_space
        # self.vector_function_space = model.vector_function_space

    def __getitem__(self, key):
        """
        Gives dictionary like behaviour.

        Raises an errors if the key does not exist.
        """
        if key not in self.TYPES:
            raise KeyError(f"`{key}` is not a valid property")
        else:
            return self.data[key]

    def __setitem__(self, key, value):
        """
        Gives dictionary like behaviour.

        Raises an errors if the key does not exist.
        """
        if key not in self.TYPES:
            raise KeyError(f"`{key}` is not a valid property")
        else:
            self.data[key] = value

    def __str__(self):
        return self.data.__str__()

    def __repr__(self):
        return self.data.__repr__()

    @property
    def vector(self):
        """
        Returns the flattened parameter vector
        """
        parameter_vectors = []
        for _, item in self.parameters:
           parameter_vectors.append(item.flat)

        return np.concatenate(parameter_vectors, axis=-1)

    @property
    def size(self):
        """
        Return the size of the parameter vector
        """
        n = 0
        for key, item in self.parameters:
            n += item.size
        return n

    def convert(self):
        """
        Return the solid/fluid properties for the forward model.

        Returns
        -------
        solid_props : properties.SolidProperties
            A collection of solid properties
        fluid_props : properties.FluidProperties
            A collection of fluid properties
        """
        return NotImplementedError

    def dconvert(self):
        """
        Return the sensitivity of the solid/fluid properties to the parameter vector.

        Parameters
        ----------
        dg_solid_props : dict
            The sensitivity of a functional with respect each property in solid_props
        dg_fluid_props:
            The sensitivity of a functional with respect each property in fluid_props

        Returns
        -------
        solid_props : properties.SolidProperties
            A dictionary of derivatives of solid properties
        fluid_props : properties.FluidProperties
            A dictionary of derivatives of fluid properties
        """
        return NotImplementedError

class NodalElasticModuli(AbstractParameterization):
    """
    A parameterization consisting of nodal values of elastic moduli with defaults for the remaining parameters.
    """
    SHAPES = OrderedDict(
        {'elastic_moduli': ('field', ())}
    )

    def __init__(self, model, **kwargs):
        super(NodalElasticModuli, self).__init__(model, **kwargs)

        # Store the default values as a copy of the pass default properties
        self.default_fluid_props = props.SolidProperties(kwargs['default_fluid_props'])
        self.default_solid_props = props.FluidProperties(kwargs['default_solid_props'])

    def convert(self):
        solid_props = props.SolidProperties(self.default_solid_props)
        fluid_props = props.FluidProperties(self.default_fluid_props)

        solid_props['elastic_modulus'] = self.parameters['elastic_moduli']

        return solid_props, fluid_props

    def dconvert(self):
        """
        Returns the sensitivity of the elastic modulus with respect to parameters

        # TODO: This should return a dict or something that has the sensitivity
        # of all properties wrt parameters
        """
        # solid_props = props.SolidProperties(self.default_solid_props)
        # fluid_props = props.FluidProperties(self.default_fluid_props)

        # solid_props['elastic_modulus'] = self.parameters['elastic_moduli']

        return 1.0

class LagrangeConstrainedNodalElasticModuli(NodalElasticModuli):
    """
    A parameterization consisting of nodal values of elastic moduli with defaults for the remaining parameters.
    """
    SHAPES = OrderedDict(
        {'elastic_moduli': ('field', ()),
         'lagrange_multiplier': ('const', ())}
    )

    # def __init__(self, model, **kwargs):
    #     super(LagrangeConstrainedNodalElasticModuli, self).__init__(model, **kwargs)

    # def convert(self):
    #     solid_props = props.SolidProperties(self.default_solid_props)
    #     fluid_props = props.FluidProperties(self.default_fluid_props)

    #     solid_props['elastic_modulus'] = self.parameters['elastic_moduli']

    #     return solid_props, fluid_props

    # def dconvert(self):
    #     """
    #     Returns the sensitivity of the elastic modulus with respect to parameters

    #     # TODO: This should return a dict or something that has the sensitivity
    #     # of all properties wrt parameters
    #     """
    #     # solid_props = props.SolidProperties(self.default_solid_props)
    #     # fluid_props = props.FluidProperties(self.default_fluid_props)

    #     # solid_props['elastic_modulus'] = self.parameters['elastic_moduli']

    #     return 1.0
