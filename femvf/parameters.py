"""
Contains definitions of parametrizations. These objects should provide a mapping from their specific parameters to standardized parameters of the forward model, as well as the derivative of the map.
"""

import math
# from functools import reduce
from collections import OrderedDict
from . import properties as props
from . import constants

import dolfin as dfn
from petsc4py import PETSc

import numpy as np

class Parameterization:
    """
    A parameterization is a mapping from a set of parameters to those of the forward model.

    The parameters are stored in a dictionary mapping `{param_label: param_values}`. The
    the parameterization can consist of multiple parameters labels, for example,
    `{'body_elastic_modulus': 2.0,
      'cover_elastic_modulus': 1.0,
      'interior_elastic_moduli': np.array([3, 2, 3, ... , 5.2])}`

    The underlying data is stored in a single vector but can be accessed either using the label or
    the vector. This is, hopefully, a readable way to store potentially unrelated parameters. Methods are
    provided to convert this to a single vector which is needed for optimization routines.

    Parameters
    ----------
    model : forms.ForwardModel
    constants : dict
        A dictionary of labelled constants mapping labels to constant values
        used in the parameterization.
    parameters : dict, optional
        A mapping of labeled parameters to values to initialize the parameterization
    kwargs : optional
        Additional keyword arguments needed to specify the parameterization.
        These will vary depending on the specific parameterization.

    Attributes
    ----------
    model : femvf.forms.ForwardModel
    constants : dict({str: value})
        A dictionary of labeled constants to values
    vector : np.ndarray
        The parameter vector
    PARAM_TYPES : OrderedDict(tuple( 'field'|'const' , tuple), ...)
        A dictionary storing the shape of each labeled parameter in the parameterization
    _PARAM_SHAPES : dict({param_label: shape_of_param_array})
    _PARAM_OFFSETS : dict({param_label: idx_offset_in_vector})
    """

    PARAM_TYPES = OrderedDict(
        {'abstract_parameters': ('field', ())}
        )
    CONSTANT_LABELS = {'foo': 2, 'bar': 99}

    def __new__(cls, model, constants, parameters=None):
        # Raise an error if one of the required constant labels is missing, then proceed with
        # initializing
        for label in cls.CONSTANT_LABELS:
            if label not in constants:
                raise ValueError(f"Could not find constant `{label}` in supplied constants")

        return super().__new__(cls)

    def __init__(self, model, constants, parameters=None):
        self._constants = constants
        self.model = model

        # Calculate the shape of the array for each labeled parameter and its offset in the vector
        self._PARAM_SHAPES = dict()
        self._PARAM_OFFSETS = dict()
        offset = 0
        N_DOF = model.scalar_function_space.dim()
        for key, param_type in self.PARAM_TYPES.items():
            shape = None
            if param_type[0] == 'field':
                shape = (N_DOF, *param_type[1])
            elif param_type[0] == 'const':
                shape = (*param_type[1], )
            else:
                raise ValueError("I haven't though about what to do here yet.")
            self._PARAM_SHAPES[key] = shape
            self._PARAM_OFFSETS[key] = offset
            offset += np.prod(shape, dtype=int, initial=1)

        # Initialize the size of the containing vector (the final calculated offset)
        self._vector = np.zeros(offset, dtype=float)

        # Assign the parameter values supplied in the initialization
        if parameters is None:
            parameters = {}

        for label in self:
            offset = self._PARAM_OFFSETS[label]
            size = np.prod(self._PARAM_SHAPES[label])

            if label in parameters:
                self._vector[offset:offset+size] = parameters[label]

    def __contains__(self, key):
        return key in self.PARAM_TYPES

    def __getitem__(self, key):
        """
        Returns the slice corresponding to the labelled parameter

        Parameters
        ----------
        key : str
            A parameter label
        """
        label = key

        if label not in self:
            raise KeyError(f"`{label}` is not a valid parameter label")
        else:
            # Get the parameter label from the key

            offset = self._PARAM_OFFSETS[label]
            shape = self._PARAM_SHAPES[label]
            size = np.prod(shape, dtype=int, initial=1)

            return self.vector[offset:offset+size].reshape(shape)

    # def __setitem__(self, key, value):
    #     """
    #     Sets the slice corresponding to the labelled parameter in the parameter vector

    #     Parameters
    #     ----------
    #     key : str
    #         A parameter label
    #     """
    #     # Get the parameter label from the key
    #     label = key

    #     offset = self._PARAM_OFFSETS[label]
    #     shape = self._PARAM_SHAPES[label]
    #     size = np.prod(shape, dtype=int, initial=1)

    #     if key not in self:
    #         raise KeyError(f"`{key}` is not a valid parameter label")
    #     else:
    #         self.vector[offset:offset+size].reshape(shape)[:]

    def __iter__(self):
        """
        Copy dictionary iter behaviour
        """
        return self.PARAM_TYPES.__iter__()

    def __str__(self):
        return self.PARAM_TYPES.__str__()

    def __repr__(self):
        return self.PARAM_TYPES.__repr__()

    def copy(self):
        out = type(self)(self.model, self.constants)
        out.vector[:] = self.vector
        return out

    @property
    def constants(self):
        """
        Return constant values associated with the parameterization
        """
        return self._constants

    @property
    def vector(self):
        """
        Return the flattened parameter vector
        """
        return self._vector

    @property
    def size(self):
        """
        Return the size of the parameter vector
        """
        return self.vector.size

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
    PARAM_TYPES = OrderedDict(
        {'elastic_moduli': ('field', ())}
    )

    CONSTANT_LABELS = ('default_solid_props',
                       'default_fluid_props',
                       'default_timing_props')

    def convert(self):
        solid_props = props.SolidProperties(self.model, self.constants['default_solid_props'])
        fluid_props = props.FluidProperties(self.model, self.constants['default_fluid_props'])
        timing_props = self.constants['default_timing_props']

        solid_props['elastic_modulus'] = self['elastic_moduli']

        return solid_props, fluid_props, timing_props

    def dconvert(self, demod):
        """
        Returns the sensitivity of the elastic modulus with respect to parameters

        # TODO: This should return a dict or something that has the sensitivity
        # of all properties wrt parameters
        """
        # solid_props = props.SolidProperties(model, self.default_solid_props)
        # fluid_props = props.FluidProperties(model, self.default_fluid_props)

        # solid_props['elastic_modulus'] = self.parameters['elastic_moduli']

        return 1.0*demod

class LagrangeConstrainedNodalElasticModuli(NodalElasticModuli):
    """
    A parameterization consisting of nodal values of elastic moduli with defaults for the remaining parameters.
    """
    PARAM_TYPES = OrderedDict(
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
