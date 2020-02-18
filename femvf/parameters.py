"""
Contains definitions of parametrizations. These objects should provide a mapping from their specific parameters to standardized parameters of the forward model, as well as the derivative of the map.
"""

from collections import OrderedDict
from . import properties as props
from . import constants

import dolfin as dfn
from petsc4py import PETSc

import numpy as np

class Parameterization:
    """
    A parameterization is a mapping from a set of parameters to those of the forward model.

    The parameters are stored in a dictionary mapping `{parameter_label: parameter_array}`, where
    the parameterization can consist of multiple parameters labels. For example, these could be
    `{'body_elastic_modulus': 2.0,
      'cover_elastic_modulus': 1.0,
      'interior_elastic_moduli': [3, 2, 3, ... , 5.2]}`
    This is, hopefully, a readable way to store potentially unrelated parameters. Methods are
    provided to convert this to a single vector which is needed for optimization routines.

    Parameters
    ----------
    model : forms.ForwardModel
    constants : dict
        A dictionary of labelled constants mapping needed labels to constant values
        used in the parameterization.
    parameters : dict, optional
        A mapping of labeled parameters to values to initialize the parameterization
    kwargs : optional
        Additional keyword arguments needed to specify the parameterization.
        These will vary depending on the specific parameterization.

    Attributes
    ----------
    model


    TYPES : dict(tuple(str, tuple), ...)
        A dictionary storing the shape of each labeled parameter in the parameterization
    """
    TYPES = OrderedDict(
        {'abstract_parameters': ('field', ())}
        )

    # def __new__(cls, model, **kwargs):
    #     return super().__new__(cls)

    def __init__(self, model, constants, parameters=None):
        self._constants = constants
        self.model = model
        self.data = OrderedDict()

        if parameters is None:
            parameters = {}

        N_DOF = model.scalar_function_space.dim()
        for key, parameter_type in self.TYPES.items():
            shape = None
            if parameter_type[0] == 'field':
                shape = (N_DOF, *parameter_type[1])
            else:
                shape = (*parameter_type[1], )

            self.data[key] = np.zeros(shape)
            if key in parameters:
                self.data[key][:] = parameters[key]

    def __contains__(self, key):
        return key in self.TYPES

    def __getitem__(self, key):
        """
        Gives dictionary like behaviour.

        Raises an errors if the key does not exist.
        """
        if key not in self:
            raise KeyError(f"`{key}` is not a valid property")
        else:
            return self.data[key]

    def __setitem__(self, key, value):
        """
        Gives dictionary like behaviour.

        Note that this always tries to assign to the while parameter array for the specific key.

        Raises an errors if the key does not exist.
        """
        if key not in self:
            raise KeyError(f"`{key}` is not a valid property")
        else:
            self.data[key][:] = value

    def __iter__(self):
        """
        Copy dictionary iter behaviour
        """
        return self.data.__iter__()

    def __str__(self):
        return self.data.__str__()

    def __repr__(self):
        return self.data.__repr__()

    def copy(self):
        return type(self)(self.model, self._constants, parameters=self.data)

    @property
    def constants(self):
        """
        Returns constant values associated with the parameterization
        """
        return self._constants

    @property
    def vector(self):
        """
        Returns the flattened parameter vector
        """
        parameter_vectors = []
        for param_label in self:
            parameter_vectors.append(self[param_label].flat)

        return np.concatenate(parameter_vectors, axis=-1)

    @property
    def size(self):
        """
        Return the size of the parameter vector
        """
        n = 0
        for key, item in self.data.items():
            n += item.size
        return n

    def get_vector(self):
        return self.vector

    def set_vector(self, value):
        """
        Assigns a parameter vector to the parameterization
        """
        idx_start = 0
        for param_label in self:
            idx_end = idx_start + self[param_label].size
            self[param_label][:] = value[idx_start:idx_end]

            idx_start = idx_end

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

    def dconvert(self, demod):
        """
        Return the sensitivity of the solid/fluid properties to the parameter vector.

        Parameters
        ----------
        demod : array_like
            The sensitivity of a functional wrt. the elastic moduli
        dg_solid_props : dict
            The sensitivity of a functional with respect each property in solid_props
        dg_fluid_props:
            The sensitivity of a functional with respect each property in fluid_props

        Returns
        -------
        array_like
            The sensitvity of the functional wrt. the parameterization
        """
        return NotImplementedError

class NodalElasticModuli(Parameterization):
    """
    A parameterization consisting of nodal values of elastic moduli with defaults for the remaining parameters.
    """
    TYPES = OrderedDict(
        {'elastic_moduli': ('field', ())}
    )

    CONSTANTS_LABELS = ('default_solid_props', 
                        'default_fluid_props', 
                        'default_timing_props')
    def __init__(self, model, constants, parameters=None):
        
        for label in self.CONSTANTS_LABELS:
            if label not in constants:
                raise ValueError(f"{label} was not found as a constant value")

        super(NodalElasticModuli, self).__init__(model, constants, parameters)

    def convert(self):
        solid_props = props.SolidProperties(self.model, self.constants['default_solid_props'])
        fluid_props = props.FluidProperties(self.model, self.constants['default_fluid_props'])
        timing_props = self.constants['default_timing_props']

        solid_props['elastic_modulus'] = self.data['elastic_moduli']

        return solid_props, fluid_props, timing_props

    def dconvert(self):
        """
        Returns the sensitivity of the elastic modulus with respect to parameters

        # TODO: This should return a dict or something that has the sensitivity
        # of all properties wrt parameters
        """
        # solid_props = props.SolidProperties(model, self.default_solid_props)
        # fluid_props = props.FluidProperties(model, self.default_fluid_props)

        # solid_props['elastic_modulus'] = self.parameters['elastic_moduli']

        return 1.0

class LagrangeConstrainedNodalElasticModuli(NodalElasticModuli):
    """
    A parameterization consisting of nodal values of elastic moduli with defaults for the remaining parameters.
    """
    TYPES = OrderedDict(
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

