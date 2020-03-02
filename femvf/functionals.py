"""
This module contains definitions of various functionals.

A functional should take in the entire time history of states from a forward model run and return a
real number. Each functional has the signature

functional(model, f, **kwargs) -> float, dict

, where `model` is a `ForwardModel` instance, `f` is a `StateFile` instance, and **kwargs are
keyword arguments specific to the functional. The functional returns its value as the first argument
and a dictionary of additional info as a second argument. The dictionary of additional info
can also be fed into the the sensitivity function of the functional to speed up the calculation.

For computing the sensitivity of the functional through the discrete adjoint method, you also need
the sensitivity of the functional with respect to the n'th state. This function has the signature

dfunctional_du(model, n, f, **kwargs) -> float, dict

, where the parameters have the same uses as defined previously. Parameter `n` is the state to
compute the sensitivity with respect to.
"""

import numpy as np
import dolfin as dfn
import ufl

import scipy.signal as sig

# class TypeFunctional(type):
#     """
#     This is a metaclass for the Functional that could (will?) be used to implement special methods.

#     This could be used to allow creating new `Functional` classes by performing arithmetic
#     operations on existing functionals. This could make it easy to express a penalty function, for
#     example as:
#     `PenaltyFunctional = Functional + (1/2)*penalty*ConstraintFunctional**2`
#     """
#     # Left binary ops
#     # funcs = {'':
#     #          '': }
#     def __new__(self, model, f, **kwargs):
#         pass

#     def __add__(self, other):
#         class SumOfFuncs(Functional):
#             def eval(self):
#                 return
#             def du(self, n, p0, p1):
#             def dv(n, p0, p1):
#             def da(n, p0, p1):
#             def dp(n, p0, p1):

#         return SumOfFuncs

#     def __sub__(self, other):
#         class DiffOfFuncs(Functional):
#             def eval(self):
#                 return
#             def du(self, n, p0, p1):
#             def dv(n, p0, p1):
#             def da(n, p0, p1):
#             def dp(n, p0, p1):

#     def __mul__(self, other):
#         class ProductOfFuncs(Functional):
#             def eval(self):
#                 return
#             def du(self, n, p0, p1):
#             def dv(n, p0, p1):
#             def da(n, p0, p1):
#             def dp(n, p0, p1):

#     def __truediv__(self, other):
#         class QuotientOfFuncs(Functional):
#             def eval(self):
#                 return
#             def du(self, n, p0, p1):
#             def dv(n, p0, p1):
#             def da(n, p0, p1):
#             def dp(n, p0, p1):

#     def __pow__(self, other):
#         class PowerOfFuncs(Functional):
#             def eval(self):
#                 return
#             def du(self, n, p0, p1):
#             def dv(n, p0, p1):
#             def da(n, p0, p1):
#             def dp(n, p0, p1):

#     # Right binary ops
#     def __radd__(self, other):
#         return NotImplementedError("")

#     def __rsub__(self, other):
#         return NotImplementedError("")

#     def __rmul__(self, other):
#         return NotImplementedError("")

#     def __rtruediv__(self, other):
#         return NotImplementedError("")

#     # Unary ops
#     def __neg__(self):
#         class IdentityOfFunc(Functional):
#             def eval(self):
#                 return
#             def du(self, n, p0, p1):
#             def dv(n, p0, p1):
#             def da(n, p0, p1):
#             def dp(n, p0, p1):

#     def __pos__(self):
#         class NegationOfFunc(Functional):
#             def eval(self):
#                 return
#             def du(self, n, p0, p1):
#             def dv(n, p0, p1):
#             def da(n, p0, p1):
#             def dp(n, p0, p1):

class Functional:
    """
    This class represents a functional computed over the solution history of a forward model run.

    After creating an instance of the Functional, the value of the functional will basically be a
    constant. After it's computed the first time, it's cached in _value.

    Parameters
    ----------
    model : forms.ForwardModel
        The forward model instance
    f : StateFile
        A file object containing the solution history of the model run.
    kwargs : optional
        Additional arguments that specify functional specific options of how it should be computed

    Attributes
    ----------
    f : statefile.StateFile
    model : forms.ForwardModel
    kwargs : dict
        A dictionary of additional options of how to compute the functional
    value : float
        The value of the functional. This should be a constant any time it's computed because
        the instance of the functional is always tied to the same file.
    funcs : dict of callable
        Dictionary of any member functionals which are used in computing the functional.
    cache : dict
        A dictionary of cached values. These are specific to the functional and values are likely
        to be cached if they are needed to compute sensitivities and are expensive to compute.
    """
    def __init__(self, model, f, **kwargs):
        self.model = model
        self.f = f
        self.kwargs = kwargs

        self.funcs = dict()
        self.cache = dict()
        self._value = None

    def __call__(self):
        if self._value is None:
            self._value = self.eval()

        return self._value

    def eval(self):
        """
        Return the value of the functional.
        """
        raise NotImplementedError("You have to implement this you goofball")

    def du(self, n, iter_params0, iter_params1):
        """
        Return the sensitivity of the functional with respect to the the `n`th state.

        Parameters
        ----------
        n : int
            Index of state
        iter_params0, iter_params1 : dict
            Dictionary of parameters that specify iteration n. These are parameters fed into
            `forms.ForwardModel.set_iter_params` with signature:
            `(x0=None, dt=None, u1=None, solid_props=None, fluid_props=None)`

            `iter_params0` specifies the parameters needed to map the states at `n-1` to the states
            at `n+0`.
        """
        return dfn.Function(self.model.vector_function_space).vector()

    def dv(self, n, iter_params0, iter_params1):
        return dfn.Function(self.model.vector_function_space).vector()

    def da(self, n, iter_params0, iter_params1):
        return dfn.Function(self.model.vector_function_space).vector()

    def dp(self):
        """
        Return the sensitivity of the functional with respect to the parameters.
        """
        return NotImplementedError("You have to implement this")

class Null(Functional):
    """
    The null functional that always returns 0
    """
    def eval(self):
        return 0.0

    def du(self, n, iter_params0, iter_params1):
        return dfn.Function(self.model.vector_function_space).vector()

    def dv(self, n, iter_params0, iter_params1):
        return dfn.Function(self.model.vector_function_space).vector()

    def da(self, n, iter_params0, iter_params1):
        return dfn.Function(self.model.vector_function_space).vector()

    def dp(self):
        return dfn.Function(self.model.scalar_function_space).vector()

class FinalDisplacementNorm(Functional):
    r"""
    Return the l2 norm of displacement at the final time

    This returns :math:`\sum{||\vec{u}||}_2`.
    """

    # def __init__(self, model, f, **kwargs):
    #     super(FinalDisplacementNorm, self).__init__(model, f, **kwargs)

    def eval(self):
        u = dfn.Function(self.model.vector_function_space).vector()

        _u, _, _ = self.f.get_state(self.f.size-1)

        u[:] = _u
        res = u.norm('l2')

        return res

    def du(self, n, iter_params0, iter_params1):
        res = None

        if n == self.f.size-1:
            _u = iter_params1['x0'][0]

            u = dfn.Function(self.model.vector_function_space).vector()
            u[:] = _u

            u_norm = u.norm('l2')

            res = None
            if u_norm == 0:
                res = dfn.Function(self.model.vector_function_space).vector()
            else:
                res = 1/u_norm * u
        else:
            res = dfn.Function(self.model.vector_function_space).vector()

        return res

    def dp(self):
        return dfn.Function(self.model.scalar_function_space).vector()

class FinalVelocityNorm(Functional):
    r"""
    Return the l2 norm of velocity at the final time

    This returns :math:`\sum{||\vec{u}||}_2`.
    """

    # def __init__(self, model, f, **kwargs):
    #     super(FinalDisplacementNorm, self).__init__(model, f, **kwargs)

    def eval(self):
        v = dfn.Function(self.model.vector_function_space).vector()

        _, _v, _ = self.f.get_state(self.f.size-1)
        v[:] = _v

        return v.norm('l2')

    def dv(self, n, iter_params0, iter_params1):
        res = None

        if n == self.f.size-1:
            _v = iter_params1['x0'][1]

            v = dfn.Function(self.model.vector_function_space).vector()
            v[:] = _v

            v_norm = v.norm('l2')

            res = None
            if v_norm == 0:
                res = dfn.Function(self.model.vector_function_space).vector()
            else:
                res = 1/v_norm * v
        else:
            res = dfn.Function(self.model.vector_function_space).vector()

        return res

class DisplacementNorm(Functional):
    r"""
    Represents the sum over time of l2 norms of displacements.

    :math:`\sum{||\vec{u}||}_2`
    """

    def __init__(self, model, f, **kwargs):
        super(DisplacementNorm, self).__init__(model, f, **kwargs)

        self.kwargs.setdefault('use_meas_indices', False)
        self.kwargs.setdefault('m_start', 0)
        self.kwargs.setdefault('m_final', self.f.size)

    def eval(self):
        # N_STATE = self.f.size

        res = 0
        u = dfn.Function(self.model.vector_function_space).vector()
        for ii in range(self.kwargs['m_start'], self.kwargs['m_final']):
            # Set form coefficients to represent the model form index ii -> ii+1
            _u, _, _ = self.f.get_state(ii)

            u[:] = _u
            res += u.norm('l2')

        return res

    def du(self, n, iter_params0, iter_params1):
        # res = dfn.Function(self.model.vector_function_space)

        N_START = self.kwargs['m_start']
        N_FINAL = self.kwargs['m_final']
        if n >= N_START and n < N_FINAL:
            _u = iter_params1['x0'][0]

            u = dfn.Function(self.model.vector_function_space).vector()
            u[:] = _u

            u_norm = u.norm('l2')

            res = None
            if u_norm == 0:
                res = u
            else:
                res = 1/u_norm * u

        return res

class VelocityNorm(Functional):
    r"""
    Represents the sum over all solution states of l2 norms of velocities.

    :math:`\sum{||\vec{v}||}_2`
    """

    def __init__(self, model, f, **kwargs):
        super(VelocityNorm, self).__init__(model, f, **kwargs)

        self.kwargs.setdefault('use_meas_indices', False)

    def eval(self):
        N_STATE = self.f.size
        v = dfn.Function(self.model.vector_function_space).vector()

        res = 0
        for ii in range(N_STATE):
            # Set form coefficients to represent the model form index ii -> ii+1
            _, _v, _ = self.f.get_state(ii)

            v[:] = _v
            res += v.norm('l2')

        return res

    def dv(self, n, iter_params0, iter_params1):
        # res = dfn.Function(self.model.vector_function_space)

        _v = iter_params1['x0'][1]
        # _u, _v, _ = self.f.get_state(n)

        v = dfn.Function(self.model.vector_function_space).vector()
        v[:] = _v

        v_norm = v.norm('l2')

        res = None
        if v_norm == 0:
            res = v
        else:
            res = 1/v_norm * v

        return res

class StrainWork(Functional):
    """
    Represent the strain work dissipated in the tissue due to damping
    """
    def __init__(self, model, f, **kwargs):
        super(StrainWork, self).__init__(model, f, **kwargs)

        from .forms import biform_m, biform_k

        vector_trial = model.forms['trial.vector']
        scalar_trial = model.forms['trial.scalar']
        ray_m = model.forms['coeff.rayleigh_m']
        ray_k = model.forms['coeff.rayleigh_k']
        rho = model.forms['coeff.rho']
        emod = model.forms['coeff.emod']
        nu = model.forms['coeff.nu']

        v0 = model.forms['coeff.v0']

        self.damping_power = ray_m*biform_m(v0, v0, rho) + ray_k*biform_k(v0, v0, emod, nu)

        self.ddamping_power_dv = ufl.derivative(self.damping_power, v0, vector_trial)
        self.ddamping_power_demod = ufl.derivative(self.damping_power, emod, scalar_trial)

        self.kwargs.setdefault('m_start', 0)

    def eval(self):
        N_START = self.kwargs['m_start']
        N_STATE = self.f.get_num_states()

        res = 0
        for ii in range(N_START, N_STATE-1):
            # Set form coefficients to represent the equation from state ii to ii+1
            self.model.set_iter_params_fromfile(self.f, ii+1)

            res += dfn.assemble(self.damping_power) * self.model.dt.values()[0]

        return res

    def dv(self, n, iter_params0, iter_params1):
        # breakpoint()
        self.model.set_iter_params(**iter_params1)

        return dfn.assemble(self.ddamping_power_dv) * self.model.dt.values()[0]

    def dp(self):
        return dfn.assemble(self.ddamping_power_demod) * self.model.dt.values()[0]

class TransferWork(Functional):
    """
    Return work done by the fluid on the vocal folds.

    Parameters
    ----------
    n_start : int, optional
        Starting index to compute the functional over
    """
    def __init__(self, model, f, **kwargs):
        super(TransferWork, self).__init__(model, f, **kwargs)

        self.kwargs.setdefault('m_start', 0)

        # Define the form needed to compute the work transferred from fluid to solid
        mesh = self.model.mesh
        ds = dfn.Measure('ds', domain=mesh, subdomain_data=self.model.facet_function)
        vector_test = self.model.forms['test.vector']
        scalar_test = self.model.forms['test.scalar']
        facet_labels = self.model.facet_labels
        pressure = self.model.forms['pressure']

        u1 = self.model.forms['u1']
        u0 = self.model.forms['u0']
        deformation_gradient = ufl.grad(u0) + ufl.Identity(2)
        deformation_cofactor = ufl.det(deformation_gradient) * ufl.inv(deformation_gradient).T
        fluid_force = -pressure*deformation_cofactor*dfn.FacetNormal(mesh)
        self.fluid_work = ufl.dot(fluid_force, u1-u0) * ds(facet_labels['pressure'])
        self.dfluid_work_du0 = ufl.derivative(fluid_work, u0, vector_test)
        self.dfluid_work_du1 = ufl.derivative(fluid_work, u1, vector_test)
        self.dfluid_work_dpressure = ufl.derivative(fluid_work, pressure, scalar_test)

    def eval(self):
        N_START = self.kwargs['m_start']
        N_STATE = self.f.get_num_states()

        res = 0
        for ii in range(N_START, N_STATE-1):
            # Set parameters for the iteration
            self.model.set_iter_params_fromfile(self.f, ii+1)
            res += dfn.assemble(self.model.fluid_work)

        return res

    def du(self, n, iter_params0, iter_params1):
        out = 0

        N_START = self.kwargs['m_start']
        N_STATE = self.f.get_num_states()

        if n < N_START:
            out += dfn.Function(self.model.vector_function_space).vector()
        else:
            # The sensitivity of the functional to state [n] generally contains two components
            # since the work is summed as
            # ... + work(state[n-1], state[n]) + work(state[n], state[n+1]) + ...
            # The below code adds the sensitivities of the two components only if n is not
            # at the final or first time index.

            if n > N_START:
                # self.model.set_iter_params_fromfile(self.f, n)
                self.model.set_iter_params(**iter_params0)

                out += dfn.assemble(self.dfluid_work_du1)

            if n < N_STATE-1:
                # self.model.set_iter_params_fromfile(self.f, n+1)
                self.model.set_iter_params(**iter_params1)
                dp_du, _ = self.model.get_flow_sensitivity()

                out += dfn.assemble(self.dfluid_work_du0)

                # Correct dfluidwork_du0 since pressure depends on u0
                dfluidwork_dp = dfn.assemble(self.dfluid_work_dpressure,
                                             tensor=dfn.PETScVector()).vec()

                dfluidwork_du0_correction = dfn.as_backend_type(out).vec().copy()
                dp_du.multTranspose(dfluidwork_dp, dfluidwork_du0_correction)

                out += dfn.Vector(dfn.PETScVector(dfluidwork_du0_correction))

        return out

    def dv(self, n, iter_params0, iter_params1):
        return dfn.Function(self.model.vector_function_space).vector()

    def da(self, n, iter_params0, iter_params1):
        return dfn.Function(self.model.vector_function_space).vector()

    def dp(self):
        return None

class F0WeightedTransferPower(Functional):
    """
    Return work done by the fluid on the vocal folds.

    # TODO: This is a little bit tricky because the total work done is
    :math: \int_{time} \int_{surface} p \cdot v_n ds dt = <p, v_n>
    To apply parseval's theorem to get power spectral density, you have to decompose p and v_n along
    the surface and time dimensions with a fft (a 2d one). The power spectral density is the product 
    of the two ffts and not the fft of p times vn! I'm not sure if this can be done in fenics easily

    Parameters
    ----------
    n_start : int, optional
        Starting index to compute the functional over
    """
    def __init__(self, model, f, **kwargs):
        super(F0WeightedTransferPower, self).__init__(model, f, **kwargs)

        self.kwargs.setdefault('tukey_alpha', 0.05)

        # Define the form needed to compute the work transferred from fluid to solid
        mesh = self.model.mesh
        ds = dfn.Measure('ds', domain=mesh, subdomain_data=self.model.facet_function)
        vector_test = self.model.forms['test.vector']
        scalar_test = self.model.forms['test.scalar']
        facet_labels = self.model.facet_labels
        pressure = self.model.forms['pressure']

        u0 = self.model.forms['u0']
        v0 = self.model.forms['v0']
        deformation_gradient = ufl.grad(u0) + ufl.Identity(2)
        deformation_cofactor = ufl.det(deformation_gradient) * ufl.inv(deformation_gradient).T
        fluid_force = -pressure*deformation_cofactor*dfn.FacetNormal(mesh)
        self.fluid_power = ufl.dot(fluid_force, v0) * ds(facet_labels['pressure'])
        self.dfluid_power_du0 = ufl.derivative(self.fluid_power, u0, vector_test)
        self.dfluid_power_dv0 = ufl.derivative(self.fluid_power, v0, vector_test)
        self.dfluid_power_dpressure = ufl.derivative(self.fluid_power, pressure, scalar_test)

    def eval(self):
        meas_ind = self.f.get_meas_indices()
        meas_times = self.f.get_solution_times()[meas_ind]
        M = meas_ind.size

        # Calculate the instantaenous fluid power at every measured time index
        fluid_power = []
        for m in range(M):
            # Set parameters for the iteration
            self.model.set_params_fromfile(self.f, meas_ind[m])
            fluid_power.append(dfn.assemble(self.fluid_power))

        fluid_power = np.array(fluid_power)

        # Convert the fluid power to the frequency domain with a tukey window
        tukey_window = sig.tukey(M, alpha=self.kwargs['tukey_alpha'])

        dft_fluid_power_tukey = np.fft.fft(fluid_power * tukey_window, n=M)
        dft_freq = np.fft.fftfreq(M, d=meas_times[1]-meas_times[0])

        return res

    def du(self, n, iter_params0, iter_params1):
        out = 0

        N_START = self.kwargs['m_start']
        N_STATE = self.f.get_num_states()

        if n < N_START:
            out += dfn.Function(self.model.vector_function_space).vector()
        else:
            # The sensitivity of the functional to state [n] generally contains two components
            # since the work is summed as
            # ... + work(state[n-1], state[n]) + work(state[n], state[n+1]) + ...
            # The below code adds the sensitivities of the two components only if n is not
            # at the final or first time index.

            if n > N_START:
                # self.model.set_iter_params_fromfile(self.f, n)
                self.model.set_iter_params(**iter_params0)

                out += dfn.assemble(self.dfluid_work_du1)

            if n < N_STATE-1:
                # self.model.set_iter_params_fromfile(self.f, n+1)
                self.model.set_iter_params(**iter_params1)
                dp_du, _ = self.model.get_flow_sensitivity()

                out += dfn.assemble(self.dfluid_work_du0)

                # Correct dfluidwork_du0 since pressure depends on u0
                dfluidwork_dp = dfn.assemble(self.dfluid_work_dpressure,
                                             tensor=dfn.PETScVector()).vec()

                dfluidwork_du0_correction = dfn.as_backend_type(out).vec().copy()
                dp_du.multTranspose(dfluidwork_dp, dfluidwork_du0_correction)

                out += dfn.Vector(dfn.PETScVector(dfluidwork_du0_correction))

        return out

    def dv(self, n, iter_params0, iter_params1):
        return dfn.Function(self.model.vector_function_space).vector()

    def da(self, n, iter_params0, iter_params1):
        return dfn.Function(self.model.vector_function_space).vector()

    def dp(self):
        return None

class VolumeFlow(Functional):
    """
    Return the total volume of fluid that flowed through the vocal folds

    Parameters
    ----------
    n_start : int, optional
        Starting index to compute the functional over
    """
    def __init__(self, model, f, **kwargs):
        super(VolumeFlow, self).__init__(model, f, **kwargs)

        self.kwargs.setdefault('m_start', 0)
        self.kwargs.setdefault('tukey_alpha', 0.0)

    def eval(self):
        N_STATE = self.f.get_num_states()
        N_START = self.kwargs['m_start']

        totalflow = 0
        for ii in range(N_START, N_STATE-1):
            fluid_info, _ = self.model.set_iter_params_fromfile(self.f, ii+1)

            totalflow += fluid_info['flow_rate'] * self.model.dt.values()[0]

        return totalflow

    def du(self, n, iter_params0, iter_params1):
        dtotalflow_dun = None
        N_START = self.kwargs['m_start']

        num_states = self.f.get_num_states()
        if n < N_START or n == num_states-1:
            dtotalflow_dun = dfn.Function(self.model.vector_function_space).vector()
        else:
            # self.model.set_iter_params_fromfile(self.f, n+1)
            self.model.set_iter_params(**iter_params1)
            _, dq_dun = self.model.get_flow_sensitivity()
            dtotalflow_dun = dq_dun * self.model.dt.values()[0]

        return dtotalflow_dun

    def dp(self):
        return None

class SubglottalWork(Functional):
    """
    Return the total work input into the fluid from the lungs (subglottal).
    """
    def __init__(self, model, f, **kwargs):
        super(SubglottalWork, self).__init__(model, f, **kwargs)

        self.kwargs.setdefault('m_start', 0)

    def eval(self):
        meas_ind = self.f.get_meas_indices()
        N_START = meas_ind[self.kwargs['m_start']]
        N_STATE = self.f.get_num_states()

        ret = 0
        for ii in range(N_START, N_STATE-1):
            # Set form coefficients to represent the equation mapping state ii->ii+1
            fluid_info, _ = self.model.set_iter_params_fromfile(self.f, ii+1)

            ret += self.model.dt.values()[0]*fluid_info['flow_rate']*self.model.fluid_props['p_sub']

        self.cache.update({'m_start': N_START, 'N_STATE': N_STATE})

        return ret

    def du(self, n, iter_params0, iter_params1):
        ret = dfn.Function(self.model.vector_function_space).vector()

        N_START = self.cache['m_start']
        N_STATE = self.cache['N_STATE']

        if n >= N_START and n < N_STATE-1:
            # fluid_props = iter_params1['fluid_props']
            fluid_props = self.model.fluid_props
            self.model.set_iter_params(**iter_params1)
            _, dq_du = self.model.get_flow_sensitivity()

            ret += self.model.dt.values()[0] * fluid_props['p_sub'] * dq_du
        else:
            pass

        return ret

    def dp(self):
        return None

class TransferEfficiency(Functional):
    """
    Returns the total vocal efficiency.

    This is the ratio of the total work done by the fluid on the folds to the total input work on
    the fluid.
    """
    def __init__(self, model, f, **kwargs):
        super(TransferEfficiency, self).__init__(model, f, **kwargs)

        self.kwargs.setdefault('m_start', 0)

        self.funcs['FluidWork'] = TransferWork(model, f, **kwargs)
        self.funcs['SubglottalWork'] = SubglottalWork(model, f, **kwargs)

    def eval(self):
        totalfluidwork = self.funcs['FluidWork']()
        totalinputwork = self.funcs['SubglottalWork']()

        res = totalfluidwork/totalinputwork

        self.cache.update({'totalfluidwork': totalfluidwork, 'totalinputwork': totalinputwork})
        return res

    def du(self, n, iter_params0, iter_params1):
        # TODO : Is there something slightly wrong with this one? Seems slightly wrong from
        # comparing with FD. The error is small but it is not propto step size?
        N_START = self.kwargs['m_start']

        tfluidwork = self.cache.get('totalfluidwork', None)
        tinputwork = self.cache.get('totalinputwork', None)

        dtotalfluidwork_dun = self.funcs['FluidWork'].du(n, iter_params0, iter_params1)
        dtotalinputwork_dun = self.funcs['SubglottalWork'].du(n, iter_params0, iter_params1)

        if n < N_START:
            return dfn.Function(self.model.vector_function_space).vector()
        else:
            return dtotalfluidwork_dun/tinputwork - tfluidwork/tinputwork**2*dtotalinputwork_dun

    def dp(self):
        return None

class MFDR(Functional):
    """
    Return the maximum flow declination rate.
    """
    def __init__(self, model, f, **kwargs):
        super(MFDR, self).__init__(model, f, **kwargs)

        self.kwargs.setdefault('m_start', 0)

    def eval(self):
        flow_rate = []
        info = {}

        num_states = self.f.get_num_states()
        for ii in range(num_states-1):
            # Set form coefficients to represent the equation at state ii
            info, _ = self.model.set_iter_params_fromfile(self.f, ii+1)

            flow_rate.append(info['flow_rate'])
        flow_rate = np.array(flow_rate)

        times = self.f.get_solution_times()[:-1]
        dflow_rate_dt = (flow_rate[1:]-flow_rate[:-1]) / (times[1:] - times[:-1])

        N_START = self.kwargs['m_start']
        idx_min = np.argmin(dflow_rate_dt[N_START:]) + N_START

        res = dflow_rate_dt[idx_min]

        self.cache.update({'idx_mfdr': idx_min})

        return res

    # TODO: Pretty sure this is wrong so you should fix it if you are going to use it
    def du(self, n, iter_params0, iter_params1):
        res = None

        idx_mfdr = self.cache.get('idx_mfdr', None)

        if n == idx_mfdr or n == idx_mfdr+1:
            # First calculate flow rates at n and n+1
            # fluid_info, _ = self.model.set_iter_params_fromfile(f, n+2)

            # q1 = fluid_info['flow_rate']
            dq1_du = self.model.get_flow_sensitivity()[1]
            t1 = self.f.get_time(n+1)

            # fluid_info, _ = self.model.set_iter_params_fromfile(f, n+1)

            # q0 = fluid_info['flow_rate']
            dq0_du = self.model.get_flow_sensitivity()[1]
            t0 = self.f.get_time(n)

            dfdr_du0 = -dq0_du / (t1-t0)
            dfdr_du1 = dq1_du / (t1-t0)

            if n == idx_mfdr:
                res = dfdr_du0
            elif n == idx_mfdr+1:
                res = dfdr_du1
        else:
            res = dfn.Function(self.model.vector_function_space).vector()

        return res

    def dp(self):
        return None

class WSSGlottalWidth(Functional):
    """
    Return the weighted sum of squared glottal widths.

    # TODO: This class uses the 'hard maximum' approach to measuring minimum area which would lead
    # non-smoothness in the functional. You should change this to use the smooth minimum used in the
    # Bernoulli fluids model.
    """
    def __init__(self, model, f, **kwargs):
        super(WSSGlottalWidth, self).__init__(model, f, **kwargs)

        # Set default values of kwargs if they were not passed
        N_STATE = self.f.get_num_states()
        self.kwargs.setdefault('weights', np.ones(N_STATE) / N_STATE)
        self.kwargs.setdefault('meas_indices', np.arange(N_STATE))
        self.kwargs.setdefault('meas_glottal_widths', np.zeros(N_STATE))

        assert kwargs['meas_indices'].size == kwargs['meas_glottal_widths'].size

    def eval(self):
        wss = 0

        u = dfn.Function(self.model.vector_function_space)
        v = dfn.Function(self.model.vector_function_space)
        a = dfn.Function(self.model.vector_function_space)

        weights = self.kwargs['weights']
        meas_indices = self.kwargs['meas_indices']
        meas_glottal_widths = self.kwargs['meas_glottal_widths']

        # Loop through every state
        for ii, gw_meas, weight in zip(meas_indices, meas_glottal_widths, weights):

            u, v, a = self.f.get_state(ii, self.model.vector_function_space)
            self.model.set_initial_state(u, v, a)

            # Find the maximum y coordinate on the surface
            cur_surface = self.model.get_surface_state()[0]
            idx_surface = np.argmax(cur_surface[:, 1])

            # Find the maximum y coordinate on the surface
            fluid_props = self.f.get_fluid_props(0)
            gw_modl = 2 * (fluid_props['y_midline'] - cur_surface[idx_surface, 1])

            wss += weight * (gw_modl - gw_meas)**2

        return wss

    def du(self, n, iter_params0, iter_params1):
        dwss_du = dfn.Function(self.model.vector_function_space).vector()

        weights = self.kwargs['weights']
        meas_indices = self.kwargs['meas_indices']
        meas_glottal_widths = self.kwargs['meas_glottal_widths']

        # The sensitivity is only non-zero if n corresponds to a measurement index
        if n in set(meas_indices):
            weight = weights[n]
            gw_meas = meas_glottal_widths[n]

            self.model.set_iter_params(**iter_params1)

            # u, v, a = self.f.get_state(n, self.model.vector_function_space)
            # self.model.set_initial_state(u, v, a)

            # Find the surface vertex corresponding to where the glottal width is measured
            # This is numbered according to the 'local' numbering scheme of the surface vertices
            # (from downstream to upstream)
            cur_surface = self.model.get_surface_state()[0]
            idx_surface = np.argmax(cur_surface[:, 1])

            # Find the maximum y coordinate on the surface
            fluid_props = self.f.get_fluid_props(0)
            gw_modl = 2 * (fluid_props['y_midline'] - cur_surface[idx_surface, 1])
            dgw_modl_du_width = -2

            # Find the vertex number according to the mesh vertex numbering scheme
            global_idx_surface = self.model.surface_vertices[idx_surface]

            # Finally convert it to the u-DOF number that actually influences glottal width
            dof_width = self.model.vert_to_vdof[global_idx_surface, 1]

            # wss = weight * (gw_modl - gw_meas)**2
            dwss_du[dof_width] = 2*weight*(gw_modl - gw_meas)*dgw_modl_du_width
        else:
            # In this case the derivative is simply 0 so the default value is right
            pass

        return dwss_du

    def dp(self):
        """
        Returns the sensitivity of the thing wrt to the starting time.
        """
        dwss_dt = 0

        weights = self.kwargs['weights']
        meas_indices = self.kwargs['meas_indices']
        meas_glottal_widths = self.kwargs['meas_glottal_widths']

        assert meas_indices.size == meas_glottal_widths.size

        # Loop through every state
        for ii, gw_meas, weight in zip(meas_indices, meas_glottal_widths, weights):
            u, v, a = self.f.get_state(ii, self.model.vector_function_space)
            self.model.set_initial_state(u, v, a)

            cur_surface = self.model.get_surface_state()[0]

            # Find the maximum y coordinate on the surface
            idx_surface = np.argmax(cur_surface[:, 1])

            # Find the vertex number according to the mesh vertex numbering scheme
            idx_body = self.model.surface_vertices[idx_surface]

            # Finally convert it to the u-DOF number that actually influences glottal width
            dof_width = self.model.vert_to_vdof[idx_body, 1]

            # Find the maximum y coordinate on the surface
            gw_modl = 2 * (self.model.y_midline - cur_surface[idx_surface, 1])
            dgw_modl_dt = -2 * v[dof_width]

            wss += weight * (gw_modl - gw_meas)**2
            dwss_dt += weight * 2 * (gw_modl - gw_meas) * dgw_modl_dt

        return dwss_dt

class SampledMeanFlowRate(Functional):
    def __init__(self, model, f, **kwargs):
        super(SampledMeanFlowRate, self).__init__(model, f, **kwargs)

        self.kwargs.setdefault('tukey_alpha', 0.0)

    def eval(self):
        meas_ind = self.f.get_meas_indices()
        tukey_window = sig.tukey(meas_ind.size, alpha=self.kwargs['tukey_alpha'])

        # Note that we loop through measured indices
        # you should space these at equal time intervals for a valid DFT result
        sum_flow_rate = 0
        for m, n in enumerate(meas_ind):
            self.model.set_params_fromfile(self.f, n)
            sum_flow_rate += self.model.get_pressure()['flow_rate'] * tukey_window[m]

        return sum_flow_rate / meas_ind.size

    def du(self, n, iter_params0, iter_params1):
        meas_ind = self.f.get_meas_indices()
        tukey_window = sig.tukey(meas_ind.size, alpha=self.kwargs['tukey_alpha'])

        dtotalflow_dun = None

        m = np.where(meas_ind == n)[0]
        assert m.size == 0 or m.size == 1
        if m.size == 1:
            self.model.set_iter_params(**iter_params1)
            _, dq_dun = self.model.get_flow_sensitivity()
            dtotalflow_dun = dq_dun * tukey_window[m] / meas_ind.size
        else:
            dtotalflow_dun = dfn.Function(self.model.vector_function_space).vector()

        return dtotalflow_dun

    def dp(self):
        return dfn.Function(self.model.scalar_function_space).vector()

def gaussian_f0_comb(dft_freq, f0=1.0, df=1):
    """
    Return a 'comb' of gaussians at multiples of f0
    """
    harmonic_dft_freq = dft_freq - np.floor(dft_freq/f0) * f0
    return np.exp(-0.5*((harmonic_dft_freq-f0)/df)^2)

# TODO: Previously had a lagrangian regularization term here but accidentally
# deleted that code... need to make it again.
