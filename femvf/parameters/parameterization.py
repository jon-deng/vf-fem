"""
Contains definitions of parametrizations. These objects should provide a mapping from their specific parameters to standardized parameters of the forward model, as well as the derivative of the map.
"""

import math

from collections import OrderedDict

from blocklinalg.linalg import BlockVec
from . import properties as props
from .. import constants, linalg
from ..solverconst import DEFAULT_NEWTON_SOLVER_PRM

import dolfin as dfn
import ufl
from petsc4py import PETSc

import numpy as np

class Parameterization:
    """
    A parameterization is a mapping from one set of parameters to the basic parameters for the
    forward model.

    Each parameterization has `convert` and `dconvert` methods tha perform the mapping and
    calculate it's sensitivity. `convert` transforms the parameterization to a standard 
    parameterization for the forward model and `dconvert` transforms
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
    constants : tuple
        A dictionary of labeled constants to values
    bvector : np.ndarray
        The BlockVector representation of the parameterization
    """

    def __init__(self, model, *constants):
        self.model = model
        self.constants = constants

        self._bvector = None

    def __str__(self):
        return self.bvector.__str__()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.bvector == other.bvector
        elif isinstance(other, np.ndarray):
            return self.bvector.to_ndarray() == other
        else:
            raise TypeError(f"Cannot compare type {type(self)} <-- {type(other)}")

    def __ne__(self, other):
        return not self.__eq__(other)

    def copy(self):
        """
        Return a copy of parameterization
        """
        out = type(self)(self.model, *self.constants)
        out.bvector[:] = self.bvector
        return out

    ## Implement the array-like interface from the block vector
    @property
    def bvector(self):
        return self._bvector

    @property
    def size(self):
        return self.bvector.size

    ## Implement the dict-like interface coming from the blockvector
    def __contains__(self, key):
        return key in self.bvector

    def __getitem__(self, key):
        return self.bvector[key]

    def __iter__(self):
        return self.bvector.__iter__()

    def keys(self):
        return self.bvector.keys

    ## Subclasses must implement these two methods
    def init_vector(self):
        raise NotImplementedError

    def convert(self):
        """
        Return the solid/fluid properties for the forward model.

        Returns
        -------
        uva : tuple
            Initial state
        solid_props : BlockVec
            A collection of solid properties
        fluid_props : BlockVec
            A collection of fluid properties
        timing_props :
        """
        return NotImplementedError

    def dconvert(self, grad_uva, grad_solid, grad_fluid, grad_times):
        """
        Return the sensitivity of the solid/fluid properties to the parameter vector.

        Parameters
        ----------
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

class FullParameterization(Parameterization):
    def __init__(self, model, ntime):
        super().__init__(model, ntime)
        ini_state = model.get_state_vec()
        control = model.get_control_vec()
        props = model.get_properties_vec()
        times = linalg.BlockVec((np.zeros(ntime),), ('times',))
        self._bvector = linalg.concatenate_vec([ini_state, control, props, times]) #join_standard_to_bvector(ini_state, control, props, times)

    def convert(self):
        model = self.model
        subvec_bsizes = [bvec.bsize for bvec in (model.state0, model.control, model.properties)] + [1]
        ini_state, control, props, times = split_bvector_to_standard(model, self.bvector)

        return ini_state, [control], props, times

    def dconvert(self, grad_state, grad_controls, grad_props, grad_times):
        """
        Returns the sensitivity of the elastic modulus with respect to parameters

        # TODO: This should return a dict or something that has the sensitivity
        # of all properties wrt parameters
        """
        return join_standard_to_bvector(grad_state, grad_controls[0], grad_props, grad_times)

class SubsetParameterization(Parameterization):
    def __init__(self, model, ntime, subset_keys):
        super().__init__(model, ntime, subset_keys)
        ini_state = model.get_state_vec()
        control = model.get_control_vec()
        props = model.get_properties_vec()
        times = linalg.BlockVec((np.zeros(ntime),), ('times',))
        self._bvector = join_standard_to_bvector(ini_state, control, props, times)
        self.subset_keys = subset_keys
    
    def copy(self):
        """
        Return a copy of parameterization
        """
        out = type(self)(self.model, *self.constants)
        out.full_bvector[:] = self.full_bvector
        out.bvector[:] = self.bvector
        return out

    @property
    def full_bvector(self):
        return self._bvector

    @property
    def bvector(self):
        keys = self.subset_keys
        vecs = [self.full_bvector[key] for key in keys]
        return linalg.BlockVec(vecs, keys)

    def convert(self):
        model = self.model
        
        ini_state, control, props, times = split_bvector_to_standard(model, self.full_bvector)

        return ini_state, [control], props, times

    def dconvert(self, grad_state, grad_controls, grad_props, grad_times):
        full_grad = join_standard_to_bvector(grad_state, grad_controls[0], grad_props, grad_times)
        keys = self.subset_keys
        vecs = [full_grad[key] for key in keys]
        return linalg.BlockVec(vecs, keys)

class NodalElasticModuli(Parameterization):
    """
    A parameterization consisting of nodal values of elastic moduli with defaults for the remaining parameters.
    """
    def __init__(self, model, *constants):
        super().__init__(model, *constants)
        
        self._bvector = self.init_vector()

    def init_vector(self):
        vecs = [self.model.properties['emod'].copy()]
        labels = ['emod']
        return linalg.BlockVec(vecs, labels)

    def convert(self):
        ini_state = self.model.get_state_vec()
        ini_state.set(0.0)

        controls = [self.model.control.copy()]
        props = self.model.properties.copy()

        props['emod'] = self.bvector['emod']
        times = self.constants[0]['times']

        return ini_state, controls, props, times

    def dconvert(self, grad_state, grad_controls, grad_props, grad_times):
        """
        Returns the sensitivity of the elastic modulus with respect to parameters

        # TODO: This should return a dict or something that has the sensitivity
        # of all properties wrt parameters
        """
        out = self.bvector.copy()
        out.set(0.0)
        out['emod'][:] = grad_props['emod']

        return out

class ElasticModuliAndInitialState(Parameterization):
    """
    A parameterization consisting of nodal values of elastic moduli with defaults for the remaining parameters.
    """
    def init_vector(self):
        vecs = [self.model.state0.copy(), self.model.properties['emod'].copy()]
        labels = [*self.model.state0.keys, 'emod']
        return linalg.BlockVec(vecs, labels)

    def convert(self):
        ini_state = self.model.get_state_vec()
        ini_state[:] = self.bvector[:-1] 

        controls = [self.model.control.copy()]
        props = self.model.properties.copy()

        props['emod'] = self.bvector['emod']
        times = self.constants['times']

        return ini_state, controls, props, times

    def dconvert(self, grad_state, grad_controls, grad_props, grad_times):
        """
        Returns the sensitivity of the elastic modulus with respect to parameters

        # TODO: This should return a dict or something that has the sensitivity
        # of all properties wrt parameters
        """
        out = self.bvector.copy()
        out[:] = 0.0
        out['emod'] = grad_props['emod']
        out[:-1] = grad_state

        return out

def split_bvector_to_standard(model, full_bvec):
    subvec_bsizes = [bvec.bsize for bvec in (model.state0, model.control, model.properties)] + [1]
    subvec_idxs = np.cumsum([0] + subvec_bsizes)

    ini_state, control, props, times = [full_bvec[ii:jj] for ii, jj in zip(subvec_idxs[:-1], subvec_idxs[1:])]

    return ini_state, control, props, times

def join_standard_to_bvector(state, control, props, times):
    labels = [f'{key}'
              for bvec in (state, control, props, times)
              for key in bvec.keys]
    vecs = [*state.vecs, *control.vecs, *props.vecs, *times.vecs]

    # breakpoint()
    return linalg.BlockVec(vecs, labels)

## These need to be fixed since they use an old version of the code
class KelvinVoigtNodalConstants(Parameterization):
    """
    A parameterization consisting of nodal values of elastic moduli and damping constants
    for the Kelvin-Voigt constitutive model.
    """
    PARAM_TYPES = OrderedDict({
        'emod': ('field', ()),
        'eta': ('field', ())
        })

    CONSTANT_LABELS = ('default_solid_props',
                       'default_fluid_props',
                       'default_timing_props')

    def convert(self):
        solid_props = self.constants['default_solid_props'].copy()
        fluid_props = self.constants['default_fluid_props'].copy()
        timing_props = self.constants['default_timing_props'].copy()

        solid_props['emod'][:] = self['emod']
        solid_props['eta'][:] = self['eta']

        return (0, 0, 0), solid_props, fluid_props, timing_props

    def dconvert(self, grad_uva, grad_solid, grad_fluid, grad_times):
        """
        Returns the sensitivity of the elastic modulus with respect to parameters

        # TODO: This should return a dict or something that has the sensitivity
        # of all properties wrt parameters
        """
        out = self.copy()
        out.bvector[:] = 0.0
        out['emod'][:] = 1.0*grad_solid['emod']
        out['eta'][:] = grad_solid['eta']

        return out

class PeriodicKelvinVoigt(Parameterization):
    """
    A parameterization defining a periodic Kelvin-Voigt model
    """
    PARAM_TYPES = OrderedDict(
        {'u0': ('field', (2,)),
         'v0': ('field', (2,)),
        #  'emod': ('field', ()),
        #  'eta': ('field', ()),
         'period': ('const', ())})

    CONSTANT_LABELS = ('default_solid_props',
                       'default_fluid_props',
                       'NUM_STATES_PER_PERIOD')

    def convert(self):
        solid_props = self.constants['default_solid_props'].copy()
        fluid_props = self.constants['default_fluid_props'].copy()

        ## Convert to integration times
        N = self.constants['NUM_STATES_PER_PERIOD']
        times = np.linspace(0.0, self['period'], N)

        ## Convert to initial states
        u0, v0 = self['u0'].reshape(-1), self['v0'].reshape(-1)
        uva = convert_uv0(self.model, (u0, v0), solid_props, fluid_props)

        return uva, solid_props, fluid_props, times

    def dconvert(self, grad_uva, grad_solid, grad_fluid, grad_times):
        """
        Returns the sensitivity of the elastic modulus with respect to parameters

        # TODO: This should return a dict or something that has the sensitivity
        # of all properties wrt. parameters
        """
        out = self.copy()
        out.vector[:] = 0.0

        ## Convert gradients of uva
        uva0, solid_props, fluid_props, _ = self.convert()
        grad_u0, grad_v0, adj_a0 = dconvert_uv0(self.model, grad_uva, uva0, solid_props, fluid_props)

        out['u0'].flat[:] = grad_u0
        out['v0'].flat[:] = grad_v0
        # out['u0'].flat[:] = grad['u0']
        # out['v0'].flat[:] = grad['v0']

        ## Convert gradients of integration times to T (period)
        N = self.constants['NUM_STATES_PER_PERIOD']
        # This conversion rule is because the integration times are distributed evenly
        # using N points over the period
        out['period'][()] = np.dot(np.linspace(0, 1.0, N), grad_times)

        return out

class FixedPeriodKelvinVoigt(Parameterization):
    """
    A parameterization defining a periodic Kelvin-Voigt model
    """
    PARAM_TYPES = OrderedDict(
        {'u0': ('field', (2,)),
         'v0': ('field', (2,)),
         'emod': ('field', ())
        })

    CONSTANT_LABELS = ('default_solid_props',
                       'default_fluid_props',
                       'NUM_STATES_PER_PERIOD',
                       'period')

    def convert(self):
        ## Convert solid properties
        solid_props = self.constants['default_solid_props'].copy()
        solid_props['emod'][:] = self['emod']

        ## Convert fluid properties
        fluid_props = self.constants['default_fluid_props'].copy()

        ## Convert timing properties
        N = self.constants['NUM_STATES_PER_PERIOD']
        dt = self.constants['period']/(N-1)
        # timing_props = {'t0': 0.0, 'dt_max': dt, 'tmeas': dt*np.arange(N)}
        timing_props = dt*np.arange(N)

        ## Convert initial states
        u0 = self['u0'].reshape(-1)
        v0 = self['v0'].reshape(-1)
        uva = convert_uv0(self.model, (u0, v0), solid_props, fluid_props)

        return uva, solid_props, fluid_props, timing_props

    def dconvert(self, grad_uva, grad_solid, grad_fluid, grad_times):
        """
        Returns the sensitivity of the elastic modulus with respect to parameters

        # TODO: This should return a dict or something that has the sensitivity
        # of all properties wrt. parameters
        """
        out = self.copy()
        out.vector[:] = 0.0

        ## Convert initial states
        uva0, solid_props, fluid_props, _ = self.convert()
        grad_u0, grad_v0, adj_a0 = dconvert_uv0(self.model, grad_uva, uva0, solid_props, fluid_props)

        out['u0'].flat[:] = grad_u0
        out['v0'].flat[:] = grad_v0

        ## Convert elastic moduli
        scalar_trial = self.model.solid.scalar_trial
        forms = self.model.solid.forms
        df0_demod_adj_frm = dfn.adjoint(ufl.derivative(forms['form.un.f0'], forms['coeff.prop.emod'], scalar_trial))
        # df0_deta_adj_frm = dfn.adjoint(ufl.derivative(forms['form.un.f0'], forms['coeff.prop.eta'], scalar_trial))
        # df0_deta_adj = dfn.assemble(df0_deta_adj_frm)

        df0_demod_adj = dfn.assemble(df0_demod_adj_frm)
        out['emod'][:] = grad_solid['emod'] - df0_demod_adj*adj_a0

        return out

class FixedPeriodKelvinVoigtwithDamping(Parameterization):
    """
    A parameterization defining a periodic Kelvin-Voigt model
    """
    PARAM_TYPES = OrderedDict({
        'u0': ('field', (2,)),
        'v0': ('field', (2,)),
        'emod': ('field', ()),
        'eta': ('field', ())
        })

    CONSTANT_LABELS = ('default_solid_props',
                       'default_fluid_props',
                       'NUM_STATES_PER_PERIOD',
                       'period')

    def convert(self):
        ## Convert solid properties
        solid_props = self.constants['default_solid_props'].copy()
        solid_props['emod'][:] = self['emod']
        solid_props['eta'][:] = self['eta']

        ## Convert fluid properties
        fluid_props = self.constants['default_fluid_props'].copy()

        ## Convert timing properties
        N = self.constants['NUM_STATES_PER_PERIOD']
        dt = self.constants['period']/(N-1)
        # timing_props = {'t0': 0.0, 'dt_max': dt, 'tmeas': dt*np.arange(N)}
        timing_props = dt*np.arange(N)

        ## Convert initial states
        u0 = self['u0'].reshape(-1)
        v0 = self['v0'].reshape(-1)
        uva = convert_uv0(self.model, (u0, v0), solid_props, fluid_props)

        return uva, solid_props, fluid_props, timing_props

    def dconvert(self, grad_uva, grad_solid, grad_fluid, grad_times):
        """
        Returns the sensitivity of the elastic modulus with respect to parameters

        # TODO: This should return a dict or something that has the sensitivity
        # of all properties wrt. parameters
        """
        out = self.copy()
        out.vector[:] = 0.0

        ## Convert initial state
        uva0, solid_props, fluid_props, _ = self.convert()
        grad_u0, grad_v0, adj_a0 = dconvert_uv0(self.model, grad_uva, uva0, solid_props, fluid_props)

        out['u0'].flat[:] = grad_u0
        out['v0'].flat[:] = grad_v0

        ## Convert elastic moduli and damping
        forms = self.model.solid.forms
        scalar_trial = self.model.solid.scalar_trial
        df0_demod_adj_frm = dfn.adjoint(ufl.derivative(forms['form.un.f0'], forms['coeff.prop.emod'], scalar_trial))
        df0_deta_adj_frm = dfn.adjoint(ufl.derivative(forms['form.un.f0'], forms['coeff.prop.eta'], scalar_trial))

        df0_demod_adj = dfn.assemble(df0_demod_adj_frm)
        df0_deta_adj = dfn.assemble(df0_deta_adj_frm)
        out['emod'][:] = grad_solid['emod'] - df0_demod_adj*adj_a0
        out['eta'][:] = grad_solid['eta'] - df0_deta_adj*adj_a0

        return out

def convert_uv0(model, uv0, solid_props, fluid_props):
    """
    Convert a dis/vel to dis/vel/acc.

    This is needed because the displacement form newmark scheme approximates time derivatives using
    dis/vel/acc at an initial time, however, given any two of those you can find the third from
    the governing equations of motion at the initial time.

    Parameters
    ----------

    """
    ## Set parameters for the state
    model.set_ini_params(uva0=(*uv0, 0), solid_props=solid_props, fluid_props=fluid_props)
    q0, p0, _ = model.solve_qp0()
    model.set_ini_params(qp0=(q0, p0))

    ## Solve for the acceleration
    newton_solver_prm = DEFAULT_NEWTON_SOLVER_PRM
    dfn.solve(model.solid.forms['form.un.f0'] == 0, model.solid.a0, bcs=model.solid.bc_base,
              J=model.solid.forms['form.bi.df0_da0'],
              solver_parameters={"newton_solver": newton_solver_prm})

    u0 = model.solid.u0.vector().copy()
    v0 = model.solid.v0.vector().copy()
    a0 = model.solid.a0.vector().copy()

    uva0 = (u0, v0, a0)
    return uva0

def dconvert_uv0(model, grad_uva0, uva0, solid_props, fluid_props):
    """
    Convert sensitivities df/d(u,v,a)0 -> df/d(u,v)0

    This assumes that you parameterize the intiial dis/vel/acc by an initial dis/vel.
    """
    grad_u0_in = grad_uva0[0]
    grad_v0_in = grad_uva0[1]
    grad_a0_in = grad_uva0[2]

    ## Set the model parameters to correctly assemble needed matrices
    model.set_ini_params(uva0=uva0, solid_props=solid_props, fluid_props=fluid_props)

    q0, p0, _ = model.solve_qp0()
    # dpressure_du0 = dfn.PETScMatrix(model.get_flow_sensitivity()[1])
    _, dp0_du0 = model.solve_dqp0_du0_solid(adjoint=True)
    model.set_ini_params(qp0=(q0, p0))

    ## Assemble needed adjoint matrices
    df0_du0_adj = dfn.assemble(model.solid.forms['form.bi.df0_du0_adj'])
    df0_dv0_adj = dfn.assemble(model.solid.forms['form.bi.df0_dv0_adj'])
    df0_da0_adj = dfn.assemble(model.solid.forms['form.bi.df0_da0_adj'])
    df0_dp0_adj = dfn.as_backend_type(dfn.assemble(model.solid.forms['form.bi.df0_dp0_adj'])).mat()

    # map dfu0_dp0 to have p on the fluid domain
    solid_dofs, fluid_dofs = model.get_fsi_scalar_dofs()
    df0_dp0_adj = linalg.reorder_mat_rows(df0_dp0_adj, solid_dofs, fluid_dofs, model.fluid.p1.size)

    # TODO: I think this should be right to do? (It might not have an apparent effect if the BC is
    # zero)
    model.solid.bc_base.apply(df0_du0_adj)
    model.solid.bc_base.apply(df0_dv0_adj)
    model.solid.bc_base.apply(df0_da0_adj)

    ## Convert `grad_a0_in` to its effect on `grad_u0` and `grad_v0`
    adj_a0 = dfn.Function(model.solid.vector_fspace).vector()
    dfn.solve(df0_da0_adj, adj_a0, grad_a0_in, 'petsc')

    grad_u0 = grad_u0_in - df0_du0_adj*adj_a0
    grad_v0 = grad_v0_in - df0_dv0_adj*adj_a0

    ## Correct for the gradient of u0, since f is sensitive to pressure, and pressure depends on u0
    # grad_u0_correction = dfn.Function(model.solid.vector_fspace).vector()
    # dpressure_du0.transpmult(df0_dp0_adj*adj_a0, grad_u0_correction)
    grad_u0 = grad_u0 - dfn.PETScVector(dp0_du0*df0_dp0_adj*adj_a0.vec())
    return grad_u0, grad_v0, adj_a0
