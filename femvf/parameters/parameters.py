"""
Contains definitions of parametrizations. These objects should provide a mapping from their specific parameters to standardized parameters of the forward model, as well as the derivative of the map.
"""

import math
# from functools import reduce
from collections import OrderedDict
from . import properties as props
from . import constants
from .forward import DEFAULT_NEWTON_SOLVER_PRM

import dolfin as dfn
from petsc4py import PETSc

import numpy as np

class Parameterization:
    """
    A parameterization is a mapping from a set of parameters to those of the forward model.

    Parameter values are stored in a single array. The slice corresponding to a specific parameter
    can be accessed through a label based index i.e. `self[param_label]`

    Each parameterization has to have `convert` and `dconvert` methods. `convert` transforms the 
    parameterization to a standard parameterization for the forward model and `dconvert` transforms
    gradients wrt. standard parameters, to gradients wrt. the parameterization.

    Parameters
    ----------
    model : model.ForwardModel
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
    model : femvf.model.ForwardModel
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
        N_DOF = model.solid.scalar_fspace.dim()
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

class Rayleigh(Parameterization):
    pass

class KelvinVoigt(Parameterization):
    pass

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
        solid_props = props.LinearElasticRayleigh(self.model, self.constants['default_solid_props'])
        fluid_props = props.FluidProperties(self.model, self.constants['default_fluid_props'])
        timing_props = self.constants['default_timing_props']

        solid_props['emod'] = self['elastic_moduli']

        return (0, 0, 0), solid_props, fluid_props, timing_props

    def dconvert(self, grad):
        """
        Returns the sensitivity of the elastic modulus with respect to parameters

        # TODO: This should return a dict or something that has the sensitivity
        # of all properties wrt parameters
        """
        # solid_props = props.SolidProperties(model, self.default_solid_props)
        # fluid_props = props.FluidProperties(model, self.default_fluid_props)

        # solid_props['elastic_modulus'] = self.parameters['elastic_moduli']
        out = self.copy()
        out.vector[:] = 0.0
        out['elastic_moduli'][:] = grad['emod']

        return out

class KelvinVoigtNodalConstants(Parameterization):
    """
    A parameterization consisting of nodal values of elastic moduli and damping constants
    for the Kelvin-Voigt constitutive model.
    """
    PARAM_TYPES = OrderedDict({
        'elastic_moduli': ('field', ()),
        'eta': ('field', ())
        })

    CONSTANT_LABELS = ('default_solid_props',
                       'default_fluid_props',
                       'default_timing_props')

    def convert(self):
        solid_props = props.LinearElasticRayleigh(self.model, self.constants['default_solid_props'])
        fluid_props = props.FluidProperties(self.model, self.constants['default_fluid_props'])
        timing_props = self.constants['default_timing_props']

        solid_props['elastic_modulus'] = self['elastic_moduli']

        return (0, 0, 0), solid_props, fluid_props, timing_props

    def dconvert(self, demod):
        """
        Returns the sensitivity of the elastic modulus with respect to parameters

        # TODO: This should return a dict or something that has the sensitivity
        # of all properties wrt parameters
        """
        # solid_props = props.SolidProperties(model, self.default_solid_props)
        # fluid_props = props.FluidProperties(model, self.default_fluid_props)

        # solid_props['elastic_modulus'] = self.parameters['elastic_moduli']
        out = self.copy()
        out.vector[:] = 0.0
        out['elastic_moduli'][:] = 1.0*demod
        out['eta'][:] = 0.0
        
        return out

class PeriodicKelvinVoigt(Parameterization):
    """
    A parameterization defining a periodic Kelvin-Voigt model
    """
    PARAM_TYPES = OrderedDict(
        {'u0': ('field', (2,)),
         'v0': ('field', (2,)),
        #  'elastic_moduli': ('field', ()),
        #  'eta': ('field', ()),
         'period': ('const', ())})

    CONSTANT_LABELS = ('default_solid_props',
                       'default_fluid_props',
                       'NUM_STATES_PER_PERIOD')

    def convert(self):
        solid_props = props.KelvinVoigt(self.model, self.constants['default_solid_props'])
        fluid_props = props.FluidProperties(self.model, self.constants['default_fluid_props'])

        N = self.constants['NUM_STATES_PER_PERIOD']
        dt = self['period']/(N-1)
        timing_props = {'t0': 0.0, 'dt_max': dt, 'tmeas': dt*np.arange(N)}
        
        ## Convert initial states
        u0 = self['u0'].reshape(-1)
        v0 = self['v0'].reshape(-1)
        a0 = dfn.Function(self.model.solid.vector_fspace).vector()

        ## Set parameters for the state
        self.model.set_params(uva0=(u0, v0, 0), solid_props=solid_props, fluid_props=fluid_props)

        q0, p0, _= self.model.get_pressure()
        self.model.set_params(qp0=(q0, p0))

        # Solve for the acceleration
        newton_solver_prm = DEFAULT_NEWTON_SOLVER_PRM
        dfn.solve(self.model.forms['form.un.f0'] == 0, self.model.a0, 
                  bcs=self.model.bc_base, J=self.model.forms['form.bi.df0_da0'],
                  solver_parameters={"newton_solver": newton_solver_prm})
        a0[:] = self.model.a0.vector()

        uva = (u0, v0, a0)

        return uva, solid_props, fluid_props, timing_props

    def dconvert(self, grad):
        """
        Returns the sensitivity of the elastic modulus with respect to parameters

        # TODO: This should return a dict or something that has the sensitivity
        # of all properties wrt. parameters
        """
        out = self.copy()
        out.vector[:] = 0.0

        # Set parameters before assembling to make sure entries are correct
        uva, solid_props, fluid_props, timing_props = self.convert()
        self.model.set_params(uva0=uva, solid_props=solid_props, fluid_props=fluid_props)

        q0, p0, _= self.model.get_pressure()
        self.model.set_params(qp0=(q0, p0))

        # Go through the calculations to convert the acceleration gradient to disp and vel
        # components
        grad_u0_part = grad['u0']
        grad_v0_part = grad['v0']
        grad_a0 = grad['a0']

        df0_du0_adj = dfn.assemble(self.model.forms['form.bi.df0_du0_adj']) 
        df0_dv0_adj = dfn.assemble(self.model.forms['form.bi.df0_dv0_adj']) 
        df0_da0_adj = dfn.assemble(self.model.forms['form.bi.df0_da0_adj'])
        adj_a0 = dfn.Function(self.model.solid.vector_fspace).vector()

        dfn.solve(df0_da0_adj, adj_a0, grad_a0, 'petsc')
        grad_u0 = grad_u0_part - df0_du0_adj*adj_a0
        grad_v0 = grad_v0_part - df0_dv0_adj*adj_a0

        out['u0'].flat[:] = grad['u0']
        out['v0'].flat[:] = grad['v0']

        N = self.constants['NUM_STATES_PER_PERIOD']
        out['period'][()] = np.sum(grad['dt']) * 1/(N-1)

        return out

class FixedPeriodKelvinVoigt(PeriodicKelvinVoigt):
    """
    A parameterization defining a periodic Kelvin-Voigt model
    """
    PARAM_TYPES = OrderedDict(
        {'u0': ('field', (2,)),
         'v0': ('field', (2,)),
         'elastic_moduli': ('field', ())
        })

    CONSTANT_LABELS = ('default_solid_props',
                       'default_fluid_props',
                       'NUM_STATES_PER_PERIOD',
                       'period')

    def convert(self):
        ## Convert solid properties
        solid_props = props.KelvinVoigt(self.model, self.constants['default_solid_props'])
        solid_props['emod'][:] = self['elastic_moduli']
        
        ## Convert fluid properties
        fluid_props = props.FluidProperties(self.model, self.constants['default_fluid_props'])

        ## Convert timing properties
        N = self.constants['NUM_STATES_PER_PERIOD']
        dt = self.constants['period']/(N-1)
        timing_props = {'t0': 0.0, 'dt_max': dt, 'tmeas': dt*np.arange(N)}
        
        ## Convert initial states
        u0 = self['u0'].reshape(-1)
        v0 = self['v0'].reshape(-1)
        a0 = dfn.Function(self.model.solid.vector_fspace).vector()

        # Set parameters for the state
        self.model.set_params(uva0=(u0, v0, 0), solid_props=solid_props, fluid_props=fluid_props)

        q0, p0, _= self.model.get_pressure()
        self.model.set_params(qp0=(q0, p0))

        # Solve for the acceleration
        newton_solver_prm = DEFAULT_NEWTON_SOLVER_PRM
        dfn.solve(self.model.forms['form.un.f0'] == 0, self.model.a0, 
                  bcs=self.model.bc_base, J=self.model.forms['form.bi.df0_da0'],
                  solver_parameters={"newton_solver": newton_solver_prm})
        a0[:] = self.model.a0.vector()

        uva = (u0, v0, a0)

        return uva, solid_props, fluid_props, timing_props

    def dconvert(self, grad):
        """
        Returns the sensitivity of the elastic modulus with respect to parameters

        # TODO: This should return a dict or something that has the sensitivity
        # of all properties wrt. parameters
        """
        out = self.copy()
        out.vector[:] = 0.0

        ## Elastic modulus
        out['elastic_moduli'][:] = grad['emod']

        ## Initial state
        # Set parameters before assembling to make sure entries are correct
        uva, solid_props, fluid_props, timing_props = self.convert()
        self.model.set_params(uva0=uva, solid_props=solid_props, fluid_props=fluid_props)

        q0, p0, _= self.model.get_pressure()
        self.model.set_params(qp0=(q0, p0))

        # Go through the calculations to convert the acceleration gradient to disp and vel
        # components
        grad_u0_part = grad['u0']
        grad_v0_part = grad['v0']
        grad_a0 = grad['a0']

        df0_du0_adj = dfn.assemble(self.model.forms['form.bi.df0_du0_adj']) 
        df0_dv0_adj = dfn.assemble(self.model.forms['form.bi.df0_dv0_adj']) 
        df0_da0_adj = dfn.assemble(self.model.forms['form.bi.df0_da0_adj'])
        adj_a0 = dfn.Function(self.model.solid.vector_fspace).vector()

        dfn.solve(df0_da0_adj, adj_a0, grad_a0, 'petsc')
        grad_u0 = grad_u0_part - df0_du0_adj*adj_a0
        grad_v0 = grad_v0_part - df0_dv0_adj*adj_a0

        out['u0'].flat[:] = grad['u0']
        out['v0'].flat[:] = grad['v0']

        return out
