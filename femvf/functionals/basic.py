"""
This module contains definitions of various functionals.

A functional is mapping that accepts the time history of all states
.. ::math {(u, v, a, q, p; t, params)_n for n in {0, 1, ..., N-1}}
and returns a real number.

A functional should take in the entire time history of states from a forward model run and return a
real number.

For computing the sensitivity of the functional through the discrete adjoint method, you also need
the sensitivity of the functional with respect to the n'th state. This function has the signature

dfunctional_du(model, n, f, ....) -> float, dict

, where the parameters have the same uses as defined previously. Parameter `n` is the state to
compute the sensitivity with respect to.

TODO: __call__ should implement caching behaviour based on whether the statefile instance is the
same as was passed on the last call
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

import dolfin as dfn
import ufl

from ..fluids import smooth_minimum, dsmooth_minimum_df

from .abstract import AbstractFunctional

class Functional(AbstractFunctional):
    """
    This is a class that interfaces manual `Functional` definitions to the `AbstractFunctional`

    It gives all `Functionals` a simple init call `__init__(self, model)` and a standard way to
    make new manually defined functionals.

    To define a new functional, create class attributes `func_types` and `default_constants`. The
    `func_types` should be a tuple of `Functional` classes that are needed to calculate the
    functional, and will be added to the `funcs` instance attribute as functional objects where
    you can access them. Then you need to implement the `eval`, `eval_du` ... `eval_dp`,
    `form_definitions` methods.
    """
    ## To subclass, implement the following
    # default_constants
    # func_types
    # def eval(self, f):
    # def eval_duva(self, f, n, iter_params0, iter_params1):
    # def eval_dp(self, f, n, iter_params0, iter_params1):

    def __init__(self, model):
        funcs = tuple(Func(model) for Func in type(self).func_types)
        super().__init__(model, *funcs)

class Constant(Functional):
    """
    Functional that always evaluates to a constant
    """
    func_types = ()
    default_constants = {
        'value': 0.0
    }

    def eval(self, f):
        return self.constants['value']

    def eval_duva(self, f, n, iter_params0, iter_params1):
        return (0.0, 0.0, 0.0)

    def eval_dp(self, f):
        return None

class PeriodicError(Functional):
    """
    Functional that measures the periodicity of a simulation

    This returns
    .. math:: \int_\Omega ||u(T)-u(0)||_2^2 + ||v(T)-v(0)||_2^2 \, dx ,
    where :math:T is the period.
    """
    func_types = ()
    default_constants = {}

    @staticmethod
    def form_definitions(model):
        forms = {}
        forms['u_0'] = dfn.Function(model.solid.vector_fspace)
        forms['u_N'] = dfn.Function(model.solid.vector_fspace)
        forms['v_0'] = dfn.Function(model.solid.vector_fspace)
        forms['v_N'] = dfn.Function(model.solid.vector_fspace)

        res_u = forms['u_N']-forms['u_0']
        res_v = forms['v_N']-forms['v_0']

        forms['res'] = (ufl.inner(res_u, res_u) + ufl.inner(res_v, res_v)) * ufl.dx

        forms['dres_du_0'] = ufl.derivative(forms['res'], forms['u_0'], model.solid.vector_trial)
        forms['dres_dv_0'] = ufl.derivative(forms['res'], forms['v_0'], model.solid.vector_trial)

        forms['dres_du_N'] = ufl.derivative(forms['res'], forms['u_N'], model.solid.vector_trial)
        forms['dres_dv_N'] = ufl.derivative(forms['res'], forms['v_N'], model.solid.vector_trial)
        return forms

    def eval(self, f):
        self.forms['u_0'].vector()[:], self.forms['v_0'].vector()[:], _ = f.get_state(0)
        self.forms['u_N'].vector()[:], self.forms['v_N'].vector()[:], _ = f.get_state(f.size-1)

        # TODO: Values should be cached in self.u_..... anyway so don't have to cahce anything
        # self.cache['u_0'] = self.u_0.vector()
        # self.cache['v_0'] = self.v_0.vector()
        # self.cache['u_N'] = self.u_N.vector()
        # self.cache['v_N'] = self.v_N.vector()

        return dfn.assemble(self.forms['res'])

    def eval_duva(self, f, n, iter_params0, iter_params1):
        du = dfn.Function(self.model.solid.vector_fspace).vector()
        dv = dfn.Function(self.model.solid.vector_fspace).vector()
        if n == 0:
            du[:] = dfn.assemble(self.forms['dres_du_0'])
            dv[:] = dfn.assemble(self.forms['dres_dv_0'])
        elif n == f.size-1:
            du[:] = dfn.assemble(self.forms['dres_du_N'])
            dv[:] = dfn.assemble(self.forms['dres_dv_N'])
        return du, dv, 0.0

    def eval_dp(self, f):
        return None

class FinalDisplacementNorm(Functional):
    r"""
    Return the l2 norm of displacement at the final time

    This returns :math:`\sum{||\vec{u}||}_2`.
    """
    func_types = ()
    default_constants = {}

    @staticmethod
    def form_definitions(model):
        forms = {}
        forms['u'] = dfn.Function(model.solid.vector_fspace)
        forms['res'] = ufl.inner(forms['u'], forms['u']) * ufl.dx

        forms['dres_du'] = ufl.derivative(forms['res'], forms['u'], model.solid.vector_trial)
        return forms

    def eval(self, f):
        u = dfn.Function(self.model.solid.vector_fspace).vector()
        self.forms['u'].vector()[:] = f.get_state(f.size-1)[0]

        return dfn.assemble(self.forms['res'])

    def eval_duva(self, f, n, iter_params0, iter_params1):
        du = dfn.Function(self.model.solid.vector_fspace).vector()

        if n == f.size-1:
            self.forms['u'].vector()[:] = iter_params1['uva0'][0]
            du = dfn.assemble(self.forms['dres_du'])

        return du, 0.0, 0.0

    def eval_dp(self, f):
        return None

class FinalVelocityNorm(Functional):
    r"""
    Return the l2 norm of velocity at the final time

    This returns :math:`\sum{||\vec{u}||}_2`.
    """
    func_types = ()
    default_constants = {}

    @staticmethod
    def form_definitions(model):
        forms = {}
        forms['v'] = dfn.Function(model.solid.vector_fspace)
        forms['res'] = ufl.inner(forms['v'], forms['v']) * ufl.dx

        forms['dres_dv'] = ufl.derivative(forms['res'], forms['v'], model.solid.vector_trial)
        return forms

    def eval(self, f):
        u = dfn.Function(self.model.solid.vector_fspace).vector()
        self.forms['v'].vector()[:] = f.get_state(f.size-1)[1]

        return dfn.assemble(self.forms['res'])

    def eval_duva(self, f, n, iter_params0, iter_params1):
        dv = dfn.Function(self.model.solid.vector_fspace).vector()

        if n == f.size-1:
            self.forms['v'].vector()[:] = iter_params1['uva0'][1]
            dv = dfn.assemble(self.forms['dres_dv'])

        return 0.0, dv, 0.0

    def eval_dp(self, f):
        return None

class FinalSurfaceDisplacementNorm(Functional):
    r"""
    Return the l2 norm of displacement at the final time

    This returns :math:`\sum{||\vec{u}||}_2`.
    """
    func_types = ()
    default_constants = {}

    @staticmethod
    def form_definitions(model):
        solid = model.solid
        forms = {}
        forms['u'] = dfn.Function(model.solid.vector_fspace)
        forms['res'] = ufl.inner(forms['u'], forms['u']) * solid.ds(solid.facet_labels['pressure'])

        forms['dres_du'] = ufl.derivative(forms['res'], forms['u'], model.solid.vector_trial)
        return forms

    def eval(self, f):
        self.forms['u'].vector()[:] = f.get_state(f.size-1)[0]

        return dfn.assemble(self.forms['res'])

    def eval_duva(self, f, n, iter_params0, iter_params1):
        du = dfn.Function(self.model.solid.vector_fspace).vector()

        if n == f.size-1:
            self.forms['u'].vector()[:] = iter_params1['uva0'][0]
            du = dfn.assemble(self.forms['dres_du'])

        return du, 0.0, 0.0

    def eval_dp(self, f):
        return None

class FinalSurfacePressureNorm(Functional):
    r"""
    Return the l2 norm of pressure at the final time

    This returns :math:`\sum{||\vec{u}||}_2`.
    """
    func_types = ()
    default_constants = {}

    @staticmethod
    def form_definitions(model):
        solid = model.solid
        forms = {}
        forms['pressure'] = model.solid.forms['coeff.fsi.pressure']
        forms['res'] = forms['pressure']**2 * solid.ds(solid.facet_labels['pressure'])

        forms['dres_dpressure'] = ufl.derivative(forms['res'], forms['pressure'], model.solid.scalar_trial)
        return forms

    def eval(self, f):
        self.model.set_params_fromfile(f, f.size-1)
        # self.forms['pressure'].vector()[:] = f.get_fluid_state(f.size-1)[0]

        return dfn.assemble(self.forms['res'])

    def eval_duva(self, f, n, iter_params0, iter_params1):
        du = dfn.Function(self.model.solid.vector_fspace).vector()

        if n == f.size-1:
            self.model.set_params_fromfile(f, f.size-1)
            dp_du, _ = self.model.get_flow_sensitivity()

            # Correct dfluidwork_du0 since pressure depends on u0 too
            dres_dpressure = dfn.assemble(self.forms['dres_dpressure'],
                                          tensor=dfn.PETScVector()).vec()

            dres_du = dfn.as_backend_type(du).vec().copy()
            dp_du.multTranspose(dres_dpressure, dres_du)

            du = dfn.Vector(dfn.PETScVector(dres_du))

        return du, 0.0, 0.0

    def eval_dp(self, f):
        return None

class FinalSurfacePower(Functional):
    """
    Return instantaneous power of the fluid on the vocal folds at the final time.
    """
    func_types = ()
    default_constants = {}

    @staticmethod
    def form_definitions(model):
        solid = model.solid
        mesh = solid.mesh
        ds = solid.ds

        vector_trial = solid.forms['trial.vector']
        scalar_trial = solid.forms['trial.scalar']

        pressure = solid.forms['coeff.fsi.pressure']
        u0 = solid.forms['coeff.state.u0']
        v0 = solid.forms['coeff.state.v0']

        deformation_gradient = ufl.grad(u0) + ufl.Identity(2)
        deformation_cofactor = ufl.det(deformation_gradient) * ufl.inv(deformation_gradient).T
        fluid_force = -pressure*deformation_cofactor*dfn.FacetNormal(mesh)

        forms = {}
        forms['fluid_power'] = ufl.inner(fluid_force, v0) * ds(solid.facet_labels['pressure'])
        forms['dfluid_power_du'] = ufl.derivative(forms['fluid_power'], u0, vector_trial)
        forms['dfluid_power_dv'] = ufl.derivative(forms['fluid_power'], v0, vector_trial)
        forms['dfluid_power_dpressure'] = ufl.derivative(forms['fluid_power'], pressure, scalar_trial)
        return forms

    def eval(self, f):
        N_STATE = f.size

        self.model.set_params_fromfile(f, N_STATE-1)
        fluid_power = dfn.assemble(self.forms['fluid_power'])

        return fluid_power

    def eval_duva(self, f, n, iter_params0, iter_params1):
        # The work terms that involve state n are given by
        # ... + 1/2*(power[n-1]+power[n])*(t[n]-t[n-1]) + 1/2*(power[n]+power[n+1])*(t[n+1]-t[n]) + ...

        du = dfn.Function(self.model.solid.vector_fspace).vector()
        dv = dfn.Function(self.model.solid.vector_fspace).vector()
        N_STATE = f.size

        if n == N_STATE-1:
            self.model.set_params_fromfile(f, n)
            dp_du, _ = self.model.get_flow_sensitivity()
            dfluid_power_dun = dfn.assemble(self.forms['dfluid_power_du'])
            dfluid_power_dvn = dfn.assemble(self.forms['dfluid_power_dv'])

            # Correct dfluid_power_n_du since pressure depends on u0 too
            dfluidpower_dp = dfn.assemble(self.forms['dfluid_power_dpressure'],
                                          tensor=dfn.PETScVector()).vec()

            dfluidpower_du_correction = dfn.as_backend_type(dfluid_power_dun).vec().copy()
            dp_du.multTranspose(dfluidpower_dp, dfluidpower_du_correction)

            dfluid_power_dun += dfn.Vector(dfn.PETScVector(dfluidpower_du_correction))

            du = dfluid_power_dun
            dv = dfluid_power_dvn

        return du, dv, 0.0

    def eval_dp(self, f):
        return None

class KVDampingWork(Functional):
    """
    Returns the work dissipated in the tissue due to damping
    """
    func_types = ()
    default_constants = {}

    @staticmethod
    def form_definitions(model):
        solid = model.solid

        # Load some ufl forms from the solid model
        vector_trial = solid.forms['trial.vector']
        scalar_trial = solid.forms['trial.scalar']
        v0 = solid.forms['coeff.state.v0']
        eta = solid.forms['coeff.prop.eta']

        v0 = solid.forms['coeff.state.v0']

        forms = {}
        from solids import strain
        forms['damping_power'] = ufl.inner(eta*strain(v0), strain(v0)) * ufl.dx

        forms['ddamping_power_dv'] = ufl.derivative(forms['damping_power'], v0, vector_trial)
        forms['ddamping_power_deta'] = ufl.derivative(forms['damping_power'], eta, scalar_trial)
        return forms

    def eval(self, f):
        N_START = 0
        N_STATE = f.get_num_states()

        res = 0
        # Calculate total damped work by the trapezoidal rule
        time = f.get_times()
        self.model.set_params_fromfile(f, 0)
        power_left = dfn.assemble(self.forms['damping_power'])
        for ii in range(N_START+1, N_STATE):
            # Set form coefficients to represent the equation from state ii to ii+1
            self.model.set_params_fromfile(f, ii)
            power_right = dfn.assemble(self.forms['damping_power'])
            res += (power_left+power_right)/2 * (time[ii]-time[ii-1])

        return res

    def eval_duva(self, f, n, iter_params0, iter_params1):
        N_START = 0
        N_STATE = f.get_num_states()
        time = f.get_times()

        dv = dfn.Function(self.model.solid.vector_fspace)
        if n >= N_START:
            self.model.set_params_fromfile(f, n)
            dpower_dvn = dfn.assemble(self.forms['ddamping_power_dv'])

            if n > N_START:
                # Add the sensitivity to `v` from the left intervals right integration point
                dv += 0.5*dpower_dvn*(time[n] - time[n-1])
            if n < N_STATE-1:
                # Add the sensitivity to `v` from the right intervals left integration point
                dv += 0.5*dpower_dvn*(time[n+1] - time[n])

        return 0.0, dv, 0.0

    def eval_dp(self, f):
        N_START = 0
        N_STATE = f.get_num_states()

        dwork_deta = dfn.Function(self.model.solid.scalar_fspace)
        # Calculate total damped work by the trapezoidal rule
        time = f.get_times()
        self.model.set_params_fromfile(f, 0)
        dpower_left_deta = dfn.assemble(self.forms['ddamping_power_demod'])
        for ii in range(N_START+1, N_STATE):
            # Set form coefficients to represent the equation from state ii to ii+1
            self.model.set_params_fromfile(f, ii)
            dpower_right_deta = dfn.assemble(self.forms['ddamping_power_demod'])
            dwork_deta += (dpower_left_deta + dpower_right_deta)/2 * (time[ii]-time[ii-1])

        return dwork_deta

# TODO: Pretty sure this one below does not work
class RayleighDampingWork(Functional):
    """
    Represent the strain work dissipated in the tissue due to damping
    """
    func_types = ()
    default_constants = {}

    @staticmethod
    def form_definitions(model):
        solid = model.solid

        # Load some ufl forms from the solid model
        vector_trial = solid.forms['trial.vector']
        scalar_trial = solid.forms['trial.scalar']
        ray_m = solid.forms['coeff.prop.rayleigh_m']
        ray_k = solid.forms['coeff.prop.rayleigh_k']
        rho = solid.forms['coeff.prop.rho']
        emod = solid.forms['coeff.prop.emod']
        nu = solid.forms['coeff.prop.nu']

        v0 = solid.forms['coeff.state.v0']

        from ..solids import biform_m, biform_k
        forms = {}
        forms['damping_power'] = ray_m*biform_m(v0, v0, rho) + ray_k*biform_k(v0, v0, emod, nu)

        forms['ddamping_power_dv'] = ufl.derivative(forms['damping_power'], v0, vector_trial)
        forms['ddamping_power_demod'] = ufl.derivative(forms['damping_power'], emod, scalar_trial)
        return forms

    def eval(self, f):
        N_START = 0
        N_STATE = f.get_num_states()

        res = 0
        for ii in range(N_START, N_STATE-1):
            # Set form coefficients to represent the equation from state ii to ii+1
            self.model.set_iter_params_fromfile(f, ii+1)
            res += dfn.assemble(self.forms['damping_power']) * self.model.solid.dt.vector()[0]

        return res

    def eval_duva(self, f, n, iter_params0, iter_params1):
        self.model.set_iter_params(**iter_params1)

        dv = dfn.assemble(self.forms['ddamping_power_dv']) * self.model.solid.dt.vector()[0]
        return 0.0, dv, 0.0

    def eval_dp(self, f):
        return dfn.assemble(self.forms['ddamping_power_demod']) * self.model.solid.dt.vector()[0]

class TransferWorkbyVelocity(Functional):
    """
    Return work done by the fluid on the vocal folds by integrating power on the surface over time.

    Parameters
    ----------
    n_start : int, optional
        Starting index to compute the functional over
    """
    func_types = ()
    default_constants = {
        'n_start': 10
    }

    @staticmethod
    def form_definitions(model):
        solid = model.solid
        mesh = solid.mesh
        ds = solid.ds

        vector_trial = solid.forms['trial.vector']
        scalar_trial = solid.forms['trial.scalar']

        pressure = solid.forms['coeff.fsi.pressure']
        u0 = solid.forms['coeff.state.u0']
        v0 = solid.forms['coeff.state.v0']

        deformation_gradient = ufl.grad(u0) + ufl.Identity(2)
        deformation_cofactor = ufl.det(deformation_gradient) * ufl.inv(deformation_gradient).T
        fluid_force = -pressure*deformation_cofactor*dfn.FacetNormal(mesh)

        forms = {}
        forms['fluid_power'] = ufl.inner(fluid_force, v0) * ds(solid.facet_labels['pressure'])
        forms['dfluid_power_du'] = ufl.derivative(forms['fluid_power'], u0, vector_trial)
        forms['dfluid_power_dv'] = ufl.derivative(forms['fluid_power'], v0, vector_trial)
        forms['dfluid_power_dpressure'] = ufl.derivative(forms['fluid_power'], pressure, scalar_trial)
        return forms

    def eval(self, f):
        N_START = self.constants['n_start']
        N_STATE = f.size

        # Calculate the power at `ii` and `ii+1` then use trapezoidal rule to integrate
        # the power over that time increment to get the work done
        work = 0
        self.model.set_params_fromfile(f, N_START)
        fluid_power0 = dfn.assemble(self.forms['fluid_power'])
        for ii in range(N_START, N_STATE-1):
            self.model.set_params_fromfile(f, ii+1)
            fluid_power1 = dfn.assemble(self.forms['fluid_power'])

            ts = f['time'][ii:ii+2]
            dt = ts[1] - ts[0]
            work += 1/2*(fluid_power0 + fluid_power1)*dt

            fluid_power0 = fluid_power1

        return work

    def eval_duva(self, f, n, iter_params0, iter_params1):
        # The work terms that involve state n are given by
        # ... + 1/2*(power[n-1]+power[n])*(t[n]-t[n-1]) + 1/2*(power[n]+power[n+1])*(t[n+1]-t[n]) + ...
        N_START = self.constants['n_start']
        N_STATE = f.size

        self.model.set_params_fromfile(f, n)
        dp_du, _ = self.model.get_flow_sensitivity()
        dfluid_power_dun = dfn.assemble(self.forms['dfluid_power_du'])
        dfluid_power_dvn = dfn.assemble(self.forms['dfluid_power_dv'])

        # Correct dfluid_power_n_du since pressure depends on u0 too
        dfluidpower_dp = dfn.assemble(self.forms['dfluid_power_dpressure'],
                                      tensor=dfn.PETScVector()).vec()

        dfluidpower_du0_correction = dfn.as_backend_type(dfluid_power_dun).vec().copy()
        dp_du.multTranspose(dfluidpower_dp, dfluidpower_du0_correction)

        dfluid_power_dun += dfn.Vector(dfn.PETScVector(dfluidpower_du0_correction))

        # Here, add sensitivity to state `n` based on the 'left' and 'right' quadrature intervals
        # Note that if `n == 0` / `n == N_STATE-1` there is not left/right quadrature interval,
        # respectively
        dt_left = 0
        dt_right = 0
        if n >= N_START:
            if n != N_START:
                ts = f['time'][n-1:n+1]
                dt_left = ts[1] - ts[0]
            if n != N_STATE-1:
                ts = f['time'][n:n+2]
                dt_right = ts[1] - ts[0]

        du = 0.5 * dfluid_power_dun * (dt_left + dt_right)
        dv = 0.5 * dfluid_power_dvn * (dt_left + dt_right)

        return du, dv, 0.0

    def eval_dt(self, f, n):
        # TODO: This hasn't been tested yet or called in the adjoint code
        N_STATE = f.size

        # Add sensitivity to state `n` based on the 'left' quadrature interval
        fluid_power0 = 0.0
        if n != 0:
            self.model.set_params_fromfile(f, n-1)
            fluid_power0 = dfn.assemble(self.forms['fluid_power'])

        fluid_power2 = 0.0
        if n != N_STATE-1:
            self.model.set_params_fromfile(f, n+1)
            fluid_power2 = dfn.assemble(self.forms['fluid_power'])

        # dt_n = 0.5*(fluid_power1+fluid_power0) - 0.5*(fluid_power2+fluid_power1)
        dfluid_power_dtn = 0.5*(fluid_power0 - fluid_power2)

        return dfluid_power_dtn

    def eval_dp(self, f):
        return None

class TransferWorkbyDisplacementIncrement(Functional):
    """
    Return work done by the fluid on the vocal folds.

    Parameters
    ----------
    n_start : int, optional
        Starting index to compute the functional over
    """
    func_types = ()
    default_constants = {
        'n_start': 0,
        'tukey_alpha': 0.0
    }

    @staticmethod
    def form_definitions(model):
        # Define the form needed to compute the work transferred from fluid to solid
        solid = model.solid
        mesh = solid.mesh
        ds = solid.ds

        vector_trial = solid.forms['trial.vector']
        scalar_trial = solid.forms['trial.scalar']

        pressure = solid.forms['coeff.fsi.pressure']
        u1 = solid.forms['coeff.state.u1']
        u0 = solid.forms['coeff.state.u0']

        deformation_gradient = ufl.grad(u0) + ufl.Identity(2)
        deformation_cofactor = ufl.det(deformation_gradient) * ufl.inv(deformation_gradient).T
        fluid_force = -pressure*deformation_cofactor*dfn.FacetNormal(mesh)

        forms = {}
        forms['fluid_work'] = ufl.inner(fluid_force, u1-u0) * ds(solid.facet_labels['pressure'])
        forms['dfluid_work_du0'] = ufl.derivative(forms['fluid_work'], u0, vector_trial)
        forms['dfluid_work_du1'] = ufl.derivative(forms['fluid_work'], u1, vector_trial)
        forms['dfluid_work_dpressure'] = ufl.derivative(forms['fluid_work'], pressure, scalar_trial)
        return forms

    def eval(self, f):
        N_START = self.constants['n_start']
        N_STATE = f.size

        tukey_window = sig.tukey(N_STATE-N_START, self.constants['tukey_alpha'])

        res = 0
        for ii in range(N_START, N_STATE-1):
            # Set parameters for the iteration
            self.model.set_iter_params_fromfile(f, ii+1)
            incremental_work = dfn.assemble(self.forms['fluid_work'])*tukey_window[ii-N_START]
            res += incremental_work

        return res

    def eval_duva(self, f, n, iter_params0, iter_params1):

        N_START = self.constants['n_start']
        N_STATE = f.get_num_states()

        du = dfn.Function(self.model.solid.vector_fspace).vector()
        if n >= N_START:
            if n != N_START:
                self.model.set_iter_params_fromfile(f, n)
                # self.model.set_iter_params(**iter_params0)

                du += dfn.assemble(self.forms['dfluid_work_du1'])

            if n != N_STATE-1:
                self.model.set_iter_params_fromfile(f, n+1)
                # self.model.set_iter_params(**iter_params1)
                dp_du, _ = self.model.get_flow_sensitivity()

                du += dfn.assemble(self.forms['dfluid_work_du0'])

                # Correct dfluidwork_du0 since pressure depends on u0 too
                dfluidwork_dp = dfn.assemble(self.forms['dfluid_work_dpressure'],
                                                tensor=dfn.PETScVector()).vec()

                dfluidwork_du0_correction = dfn.as_backend_type(du).vec().copy()
                dp_du.multTranspose(dfluidwork_dp, dfluidwork_du0_correction)

                du += dfn.Vector(dfn.PETScVector(dfluidwork_du0_correction))

        return du, 0.0, 0.0

    def eval_dp(self, f):
        return None

class VolumeFlow(Functional):
    """
    Return the total volume of fluid that flowed through the vocal folds

    Parameters
    ----------
    n_start : int, optional
        Starting index to compute the functional over
    """
    func_types = ()
    default_constants = {
        'm_start': 0,
        'tukey_alpha': 0.0}

    def eval(self, f):
        N_STATE = f.get_num_states()
        N_START = self.constants['m_start']

        totalflow = 0
        for ii in range(N_START, N_STATE-1):
            fluid_info, _ = self.model.set_iter_params_fromfile(f, ii+1)

            totalflow += fluid_info['flow_rate'] * self.model.dt.values()[0]

        return totalflow

    def eval_duva(self, f, n, iter_params0, iter_params1):
        N_START = self.constants['m_start']

        num_states = f.get_num_states()
        du = None
        if n < N_START or n == num_states-1:
            du = dfn.Function(self.model.solid.vector_fspace).vector()
        else:
            # self.model.set_iter_params_fromfile(f, n+1)
            self.model.set_iter_params(**iter_params1)
            _, dq_dun = self.model.get_flow_sensitivity()
            du = dq_dun * self.model.dt.values()[0]

        return du, 0.0, 0.0

    def eval_dp(self, f):
        return None

class SubglottalWork(Functional):
    """
    Return the total work input into the fluid from the lungs (subglottal).
    """
    func_types = ()
    default_constants = {
        'm_start': 0}

    def eval(self, f):
        meas_ind = f.get_meas_indices()
        N_START = meas_ind[self.constants['m_start']]
        N_STATE = f.get_num_states()

        ret = 0
        for ii in range(N_START, N_STATE-1):
            # Set form coefficients to represent the equation mapping state ii->ii+1
            fluid_info, _ = self.model.set_iter_params_fromfile(f, ii+1)

            ret += self.model.dt.values()[0]*fluid_info['flow_rate']*self.model.fluid_props['p_sub']

        self.cache.update({'m_start': N_START, 'N_STATE': N_STATE})

        return ret

    def eval_duva(self, f, n, iter_params0, iter_params1):
        du = dfn.Function(self.model.solid.vector_fspace).vector()

        N_START = self.cache['m_start']
        N_STATE = self.cache['N_STATE']

        if n >= N_START and n < N_STATE-1:
            # fluid_props = iter_params1['fluid_props']
            fluid_props = self.model.fluid_props
            self.model.set_iter_params(**iter_params1)
            _, dq_du = self.model.get_flow_sensitivity()

            du += self.model.dt.values()[0] * fluid_props['p_sub'] * dq_du
        else:
            pass

        return du, 0.0, 0.0

    def eval_dp(self, f):
        return None

class TransferEfficiency(Functional):
    """
    Returns the total vocal efficiency.

    This is the ratio of the total work done by the fluid on the folds to the total input work on
    the fluid.
    """
    func_types = (TransferWorkbyDisplacementIncrement, SubglottalWork)
    default_constants = {}

    def eval(self, f):
        totalfluidwork = self.funcs[0]()
        totalinputwork = self.funcs[1]()

        res = totalfluidwork/totalinputwork

        self.cache.update({'totalfluidwork': totalfluidwork, 'totalinputwork': totalinputwork})
        return res

    def eval_duva(self, f, n, iter_params0, iter_params1):
        # TODO : Is there something slightly wrong with this one? Seems slightly wrong from
        # comparing with FD. The error is small but it is not propto step size?
        N_START = self.constants['m_start']

        tfluidwork = self.cache.get('totalfluidwork', None)
        tinputwork = self.cache.get('totalinputwork', None)

        dtotalfluidwork_dun = self.funcs[0].eval_du(f, n, iter_params0, iter_params1)
        dtotalinputwork_dun = self.funcs[1].eval_du(f, n, iter_params0, iter_params1)

        du = None
        if n < N_START:
            du = dfn.Function(self.model.solid.vector_fspace).vector()
        else:
            du = dtotalfluidwork_dun/tinputwork - tfluidwork/tinputwork**2*dtotalinputwork_dun

        return du, 0.0, 0.0

    def eval_dp(self, f):
        return None

class MFDR(Functional):
    """
    Return the maximum flow declination rate.
    """
    func_types = ()
    default_constants = {
        'm_start': 0
    }

    def eval(self, f):
        flow_rate = []
        info = {}

        num_states = f.get_num_states()
        for ii in range(num_states-1):
            # Set form coefficients to represent the equation at state ii
            info, _ = self.model.set_iter_params_fromfile(f, ii+1)

            flow_rate.append(info['flow_rate'])
        flow_rate = np.array(flow_rate)

        times = f.get_solution_times()[:-1]
        dflow_rate_dt = (flow_rate[1:]-flow_rate[:-1]) / (times[1:] - times[:-1])

        N_START = self.constants['m_start']
        idx_min = np.argmin(dflow_rate_dt[N_START:]) + N_START

        res = dflow_rate_dt[idx_min]

        self.cache.update({'idx_mfdr': idx_min})

        return res

    # TODO: Pretty sure this is wrong so you should fix it if you are going to use it
    def eval_duva(self, f, n, iter_params0, iter_params1):
        du = None

        idx_mfdr = self.cache.get('idx_mfdr', None)

        if n == idx_mfdr or n == idx_mfdr+1:
            # First calculate flow rates at n and n+1
            # fluid_info, _ = self.model.set_iter_params_fromfile(f, n+2)

            # q1 = fluid_info['flow_rate']
            dq1_du = self.model.get_flow_sensitivity()[1]
            t1 = f.get_time(n+1)

            # fluid_info, _ = self.model.set_iter_params_fromfile(f, n+1)

            # q0 = fluid_info['flow_rate']
            dq0_du = self.model.get_flow_sensitivity()[1]
            t0 = f.get_time(n)

            dfdr_du0 = -dq0_du / (t1-t0)
            dfdr_du1 = dq1_du / (t1-t0)

            if n == idx_mfdr:
                du = dfdr_du0
            elif n == idx_mfdr+1:
                du = dfdr_du1
        else:
            du = dfn.Function(self.model.solid.vector_fspace).vector()

        return du, 0.0, 0.0

    def eval_dp(self, f):
        return None

class SampledMeanFlowRate(Functional):
    func_types = ()
    default_constants = {
        'tukey_alpha': 0.0
    }

    def eval(self, f):
        meas_ind = f.get_meas_indices()
        tukey_window = sig.tukey(meas_ind.size, alpha=self.constants['tukey_alpha'])

        # Note that we loop through measured indices
        # you should space these at equal time intervals for a valid DFT result
        sum_flow_rate = 0
        for m, n in enumerate(meas_ind):
            self.model.set_params_fromfile(f, n)
            sum_flow_rate += self.model.get_pressure()['flow_rate'] * tukey_window[m]

        return sum_flow_rate / meas_ind.size

    def eval_duva(self, f, n, iter_params0, iter_params1):
        meas_ind = f.get_meas_indices()
        tukey_window = sig.tukey(meas_ind.size, alpha=self.constants['tukey_alpha'])

        du = None

        m = np.where(meas_ind == n)[0]
        assert m.size == 0 or m.size == 1
        if m.size == 1:
            self.model.set_iter_params(**iter_params1)
            _, dq_dun = self.model.get_flow_sensitivity()
            du = dq_dun * tukey_window[m] / meas_ind.size
        else:
            du = dfn.Function(self.model.solid.vector_fspace).vector()

        return du, 0.0, 0.0

    def eval_dp(self, f):
        return dfn.Function(self.model.solid.scalar_fspace).vector()

class GlottalWidthErrorNorm(Functional):
    """
    Represents the difference between a measured vs model glottal width
    """
    func_types = ()
    default_constants = {
        'gw_meas': 0.0,
        'smooth_min_alpha': -2000.0
    }

    def eval(self, f):
        model = self.model

        # Get the initial locations of the nodes
        X_REF = model.get_ref_config().reshape(-1)
        DOF_SURFACE = model.solid.vert_to_vdof[model.surface_vertices].reshape(-1)
        X_REF_SURFACE = X_REF[DOF_SURFACE]

        # Calculate the glottal width at every node
        gw_model = []
        idx_meas = f.get_meas_indices()
        for n in idx_meas:
            u = f.get_state(n)[0]
            xy_surf = X_REF_SURFACE + u[DOF_SURFACE]
            y_surf = xy_surf[1::2]
            gw = smooth_minimum(y_surf, alpha=self.constants['smooth_min_alpha'])
            gw_model.append(gw)

        return np.sum((np.array(gw_model) - self.constants['gw_meas'])**2)

    def eval_duva(self, f, n, iter_params0, iter_params1):
        model = self.model

        # Get the initial locations of the nodes
        X_REF = model.get_ref_config().reshape(-1)
        DOF_SURFACE = model.solid.vert_to_vdof[model.surface_vertices].reshape(-1)
        Y_DOF = DOF_SURFACE[1::2]
        X_REF_SURFACE = X_REF[DOF_SURFACE]

        # Set up a map from state index to measured index
        N = f.size
        idx_meas = f.get_meas_indices()
        M = idx_meas.size

        n_to_m = {n: -1 for n in range(N)}
        for m, n in enumerate(idx_meas):
            n_to_m[n] = m

        du = dfn.Function(model.solid.vector_fspace).vector()
        if n_to_m[n] != -1:
            u = iter_params0['uva0'][0]
            xy_surf = X_REF_SURFACE + u[DOF_SURFACE]
            y_surf = xy_surf[1::2]

            du[Y_DOF] = dsmooth_minimum_df(y_surf, self.model.fluid.s_vertices, alpha=self.constants['smooth_min_alpha'])

        return du, 0.0, 0.0

    def eval_dp(self, f):
        return dfn.Function(self.model.solid.scalar_fspace).vector()

class DFTGlottalWidthErrorNorm(Functional):
    """
    Represents the difference between a measured vs model glottal width using DFT coefficients
    """
    func_types = ()
    default_constants = {
        'gw_meas': 0.0,
        'tukey_alpha': 0.05,
        'smooth_min_alpha': -2000.0
    }

    def eval(self, f):
        model = self.model

        # Get the initial locations of the nodes
        X_REF = model.get_ref_config()
        DOF_SURFACE = model.solid.vert_to_vdof[model.surface_vertices].reshape(-1)
        X_REF_SURFACE = X_REF[DOF_SURFACE]

        # Calculate the glottal width at every node
        gw_model = []
        idx_meas = f.get_meas_indices()
        for n in idx_meas:
            u = f.get_state(n)[0]
            xy_surf = X_REF_SURFACE + u[DOF_SURFACE]
            y_surf = xy_surf[1::2]
            gw = smooth_minimum(y_surf, alpha=self.constants['alpha_min'])
            gw_model.append(gw)

        dft_gw_model = np.fft.rfft(gw_model)
        dft_gw_meas = np.fft.rfft(self.constants['gw_meas'])

        err = dft_gw_model - dft_gw_meas

        self.cache['gw_model'] = gw_model
        self.cache['dft_gw_model'] = dft_gw_model
        self.cache['dft_gw_meas'] = dft_gw_meas
        return np.sum(np.abs(err)**2)

    def eval_duva(self, f, n, iter_params0, iter_params1):
        model = self.model

        # Get the initial locations of the nodes
        X_REF = model.get_ref_config()
        DOF_SURFACE = model.solid.vert_to_vdof[model.surface_vertices].reshape(-1)
        Y_DOF = DOF_SURFACE[1::2]
        X_REF_SURFACE = X_REF[DOF_SURFACE]

        # Set up a map from state index to measured index
        N = f.size
        idx_meas = f.get_meas_indices()
        M = idx_meas.size

        n_to_m = {n: -1 for n in range(N)}
        for m, n in enumerate(idx_meas):
            n_to_m[n] = m

        #
        gw_model = self.cache['gw_model']

        dft_gw_model = self.cache['dft_gw_model']
        m_meas = n_to_m[n]
        dft_gw_model_dgw_n = np.exp(1j*2*np.pi*m_meas*np.arange(M)/M)

        dft_gw_meas = self.cache['dft_gw_meas']
        raise NotImplementedError("You need to fix this")

    def eval_dp(self, f):
        return None

class InitialGlottalWidth(Functional):
    default_constants = {}
    func_types = ()

    def eval(self, f):
        model = self.model
        params = model.set_iter_params_fromfile(f, 1)
        fluid_props = params['fluid_props']

        x, _, _ = self.model.get_surface_state()
        y = x[:, 1]

        area = 2 * (fluid_props['y_midline'] - y)

        amin = smooth_minimum(area, model.fluid.s_vertices, fluid_props['alpha'])

        return amin

    def eval_duva(self, f, n, iter_params0, iter_params1):
        model = self.model

        params = model.set_iter_params_fromfile(f, 0)
        fluid_props = params['fluid_props']

        x, v, a = self.model.get_surface_state()
        y = x[:, 1]

        dy_du = np.zeros(x.shape); dy_du[:, 1] = 1.0

        area = 2 * (fluid_props['y_midline'] - y)
        darea_du = 2 * -dy_du

        amin = smooth_minimum(area, model.fluid.s_vertices, fluid_props['alpha'])
        damin_darea = dsmooth_minimum_df(area, model.fluid.s_vertices, fluid_props['alpha'])
        damin_du = damin_darea[:, None] * darea_du

        ## Map the sensitivity to the solid DOFS
        # Get a list of pressure DOFS
        du = dfn.Function(self.model.solid.vector_fspace).vector()

        pressure_vertices = model.surface_vertices
        pressure_dofs = model.solid.vert_to_sdof[pressure_vertices]
        vector_dofs = np.stack([pressure_dofs, pressure_dofs+1], axis=-1)
        du[vector_dofs.reshape(-1)] = damin_du.reshape(-1)

        return du, 0.0, 0.0

    def dp(self, f):
        return None

class InitialGlottalWidthVelocity(Functional):
    def eval(self, f):
        model = self.model
        params = model.set_iter_params_fromfile(f, 0)
        fluid_props = params['fluid_props']

        x, v, a = self.model.get_surface_state()
        y, dy_dt = x[1::2], v[1::2]

        area = 2 * (fluid_props['y_midline'] - y)
        darea_dt = 2 * -dy_dt

        amin = smooth_minimum(area, model.fluid.s_vertices, fluid_props['alpha'])
        damin_darea = dsmooth_minimum_df(area, model.fluid.s_vertices, fluid_props['alpha'])

        damin_dt = np.dot(damin_darea, darea_dt)

        return damin_dt

    def eval_duva(self, f, n, iter_params0, iter_params1):
        model = self.model
        params = model.set_iter_params_fromfile(f, 0)
        fluid_props = params['fluid_props']

        x, v, a = self.model.get_surface_state()
        y, dy_dt = x[1::2], v[1::2]

        dy_du = np.zeros(x.size); dy_du[1::2] = 1.0
        ddy_dt_dv = np.zeros(x.size); ddy_dt_dv[1::2] = 1.0

        area = 2 * (fluid_props['y_midline'] - y)
        darea_dt = 2 * -dy_dt

        darea_du = 2 * -dy_du
        ddarea_dt_dv = 2*-ddy_dt_dv

        amin = smooth_minimum(area, model.fluid.s_vertices, fluid_props['alpha'])
        damin_darea = dsmooth_minimum_df(area, model.fluid.s_vertices, fluid_props['alpha'])

        damin_darea

        damin_dt = np.dot(damin_darea, darea_dt)

        return damin_dt

    def dp(self, f):
        return None

def gaussian_f0_comb(dft_freq, f0=1.0, df=1):
    """
    Return a 'comb' of gaussians at multiples of f0

    Parameters
    ----------
    dft_freq : array_like
        DFT discrete frequencies output by the FFT function
    """
    # harmonic_dft_freq = dft_freq - np.floor(dft_freq/f0) * f0
    # comb = np.exp(-0.5*((harmonic_dft_freq-f0)/df)**2)

    # Build the comb by adding up each gaussian 'tooth' for every harmonic
    comb = np.zeros(dft_freq.size)
    n = 0
    fn = f0
    while fn < np.max(dft_freq):
        # We need to add 'teeth' for bothe positive and negative frequencies
        comb += np.exp(-0.5*((dft_freq-fn)/df)**2)

        comb += np.exp(-0.5*((dft_freq+fn)/df)**2)

        n += 1
        fn = (n+1)*f0

    return comb

# TODO: Previously had a lagrangian regularization term here but accidentally
# deleted that code... need to make it again.
