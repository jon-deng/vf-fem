"""
This module contains definitions of various functionals.

A functional is mapping from the time history of all states and parameters to a real number
.. ::math {(u, v, a, q, p; t, params)_n for n in {0, 1, ..., N-1}} .

A functional should take in the entire time history of states from a forward model run and return a
real number.

For computing the sensitivity of the functional through the discrete adjoint method, you also need
the sensitivity of the functional with respect to the n'th state. This function has the signature

dfunctional_du(model, n, f, ....) -> float, dict

, where the parameters have the same uses as defined previously. Parameter `n` is the state to
compute the sensitivity with respect to.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

import dolfin as dfn
import ufl

from .abstract import AbstractFunctional
from ..fluids import smooth_minimum, dsmooth_minimum_df
from ..solids import strain

class Functional(AbstractFunctional):
    """
    This class provides an interface/method to define basic functionals

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

class PeriodicError(Functional):
    """
    Functional that measures the periodicity of a simulation

    This returns
    .. math:: \int_\Omega ||u(T)-u(0)||_2^2 + ||v(T)-v(0)||_2^2 \, dx ,
    where :math:T is the period.
    """
    func_types = ()
    default_constants = {'alpha': 1e3}

    @staticmethod
    def form_definitions(model):
        forms = {}
        forms['u_0'] = dfn.Function(model.solid.vector_fspace)
        forms['u_N'] = dfn.Function(model.solid.vector_fspace)
        forms['v_0'] = dfn.Function(model.solid.vector_fspace)
        forms['v_N'] = dfn.Function(model.solid.vector_fspace)

        res_u = forms['u_N']-forms['u_0']
        res_v = forms['v_N']-forms['v_0']

        # forms['alpha'] = dfn.Constant(1.0)

        forms['resu'] = ufl.inner(res_u, res_u) * ufl.dx
        forms['resv'] = ufl.inner(res_v, res_v) * ufl.dx

        forms['dresu_du_0'] = ufl.derivative(forms['resu'], forms['u_0'], model.solid.vector_trial)
        forms['dresu_du_N'] = ufl.derivative(forms['resu'], forms['u_N'], model.solid.vector_trial)

        forms['dresv_dv_0'] = ufl.derivative(forms['resv'], forms['v_0'], model.solid.vector_trial)
        forms['dresv_dv_N'] = ufl.derivative(forms['resv'], forms['v_N'], model.solid.vector_trial)
        return forms

    def eval(self, f):
        alphau = self.constants['alpha']
        self.forms['u_0'].vector()[:], self.forms['v_0'].vector()[:], _ = f.get_state(0)
        self.forms['u_N'].vector()[:], self.forms['v_N'].vector()[:], _ = f.get_state(f.size-1)
        erru = dfn.assemble(self.forms['resu'])
        errv = dfn.assemble(self.forms['resv'])
        print(alphau**2*erru, errv)
        return alphau**2*erru + errv

    def eval_duva(self, f, n, iter_params0, iter_params1):
        alphau = self.constants['alpha']
        du = dfn.Function(self.model.solid.vector_fspace).vector()
        dv = dfn.Function(self.model.solid.vector_fspace).vector()
        if n == 0:
            du[:] = alphau**2*dfn.assemble(self.forms['dresu_du_0'])
            dv[:] = dfn.assemble(self.forms['dresv_dv_0'])
        elif n == f.size-1:
            du[:] = alphau**2*dfn.assemble(self.forms['dresu_du_N'])
            dv[:] = dfn.assemble(self.forms['dresv_dv_N'])
        return du, dv, 0.0

    def eval_dqp(self, f, n, iter_params0, iter_params1):
        dq, dp = self.model.fluid.get_state_vecs()
        dq[:], dp[:] = 0, 0
        return dq, dp

    def eval_dsolid(self, f):
        dsolid = self.model.solid.get_properties()
        dsolid.vector[:] = 0.0
        return dsolid

    def eval_dfluid(self, f):
        dfluid = self.model.fluid.get_properties()
        dfluid.vector[:] = 0.0
        return dfluid

    def eval_dt0(self, f, n):
        return 0.0

    def eval_ddt(self, f, n):
        return 0.0

class PeriodicEnergyError(Functional):
    """
    Functional that measures the periodicity of a simulation

    This returns
    .. math:: \int_\Omega ||u(T)-u(0)||_K^2 + ||v(T)-v(0)||_M^2 \, dx ,
    where :math:T is the period.
    """
    func_types = ()
    default_constants = {'alpha': 1.0}

    @staticmethod
    def form_definitions(model):
        from ..solids import biform_m, biform_k
        forms = {}
        forms['u_0'] = dfn.Function(model.solid.vector_fspace)
        forms['u_N'] = dfn.Function(model.solid.vector_fspace)
        forms['v_0'] = dfn.Function(model.solid.vector_fspace)
        forms['v_N'] = dfn.Function(model.solid.vector_fspace)

        res_u = forms['u_N']-forms['u_0']
        res_v = forms['v_N']-forms['v_0']
        rho = model.solid.forms['coeff.prop.rho']
        emod = model.solid.forms['coeff.prop.emod']
        nu = model.solid.forms['coeff.prop.nu']

        forms['resu'] = biform_k(res_u, res_u, emod, nu)
        forms['resv'] = biform_m(res_v, res_v, rho)

        forms['dresu_du_0'] = ufl.derivative(forms['resu'], forms['u_0'], model.solid.vector_trial)
        forms['dresu_du_N'] = ufl.derivative(forms['resu'], forms['u_N'], model.solid.vector_trial)
        forms['dresu_demod'] = ufl.derivative(forms['resu'], emod, model.solid.scalar_trial)

        forms['dresv_dv_0'] = ufl.derivative(forms['resv'], forms['v_0'], model.solid.vector_trial)
        forms['dresv_dv_N'] = ufl.derivative(forms['resv'], forms['v_N'], model.solid.vector_trial)
        return forms

    def eval(self, f):
        alphau = self.constants['alpha']

        self.model.set_solid_props(f.get_solid_props())
        self.forms['u_0'].vector()[:], self.forms['v_0'].vector()[:], _ = f.get_state(0)
        self.forms['u_N'].vector()[:], self.forms['v_N'].vector()[:], _ = f.get_state(f.size-1)

        erru = dfn.assemble(self.forms['resu'])
        errv = dfn.assemble(self.forms['resv'])
        print(alphau**2*erru, errv)
        return alphau**2*erru + errv

    def eval_duva(self, f, n, iter_params0, iter_params1):
        alphau = self.constants['alpha']
        du = dfn.Function(self.model.solid.vector_fspace).vector()
        dv = dfn.Function(self.model.solid.vector_fspace).vector()
        if n == 0:
            du[:] = alphau**2*dfn.assemble(self.forms['dresu_du_0'])
            dv[:] = dfn.assemble(self.forms['dresv_dv_0'])
        elif n == f.size-1:
            du[:] = alphau**2*dfn.assemble(self.forms['dresu_du_N'])
            dv[:] = dfn.assemble(self.forms['dresv_dv_N'])
        return du, dv, 0.0

    def eval_dqp(self, f, n, iter_params0, iter_params1):
        dq, dp = self.model.fluid.get_state_vecs()
        dq[:], dp[:] = 0, 0
        return dq, dp

    def eval_dsolid(self, f):
        dsolid = self.model.solid.get_properties()
        dsolid.vector[:] = 0.0

        alphau = self.constants['alpha']

        self.model.set_solid_props(f.get_solid_props())
        self.forms['u_0'].vector()[:], self.forms['v_0'].vector()[:], _ = f.get_state(0)
        self.forms['u_N'].vector()[:], self.forms['v_N'].vector()[:], _ = f.get_state(f.size-1)

        derru_demod = dfn.assemble(self.forms['dresu_demod'])
        dsolid['emod'][:] = alphau**2*derru_demod

        return dsolid

    def eval_dfluid(self, f):
        dfluid = self.model.fluid.get_properties()
        dfluid.vector[:] = 0.0
        return dfluid

    def eval_dt0(self, f, n):
        return 0.0

    def eval_ddt(self, f, n):
        return 0.0

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

    def eval_dqp(self, f, n, iter_params0, iter_params1):
        dq, dp = self.model.fluid.get_state_vecs()
        return dq, dp

    def eval_dsolid(self, f):
        dsolid = self.model.solid.get_properties()
        dsolid.vector[:] = 0.0
        return dsolid

    def eval_dfluid(self, f):
        dfluid = self.model.fluid.get_properties()
        dfluid.vector[:] = 0.0
        return dfluid

    def eval_dt0(self, f, n):
        return 0.0

    def eval_ddt(self, f, n):
        return 0.0

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

    def eval_dqp(self, f, n, iter_params0, iter_params1):
        dq, dp = self.model.fluid.get_state_vecs()
        dq[:], dp[:] = 0, 0
        return dq, dp

    def eval_dsolid(self, f):
        dsolid = self.model.solid.get_properties()
        dsolid.vector[:] = 0.0
        return dsolid

    def eval_dfluid(self, f):
        dfluid = self.model.fluid.get_properties()
        dfluid.vector[:] = 0.0
        return dfluid

    def eval_dt0(self, f, n):
        return 0.0

    def eval_ddt(self, f, n):
        return 0.0

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

    def eval_dsolid(self, f):
        dsolid = self.model.solid.get_properties()
        dsolid.vector[:] = 0.0
        return dsolid

    def eval_dfluid(self, f):
        dfluid = self.model.fluid.get_properties()
        dfluid.vector[:] = 0.0
        return dfluid

    def eval_dt0(self, f, n):
        return 0.0

    def eval_ddt(self, f, n):
        return 0.0

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

    def eval_dsolid(self, f):
        dsolid = self.model.solid.get_properties()
        dsolid.vector[:] = 0.0
        return dsolid

    def eval_dfluid(self, f):
        dfluid = self.model.fluid.get_properties()
        dfluid.vector[:] = 0.0
        return dfluid

    def eval_dt0(self, f, n):
        return 0.0

    def eval_ddt(self, f, n):
        return 0.0

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

    def eval_dsolid(self, f):
        dsolid = self.model.solid.get_properties()
        dsolid.vector[:] = 0.0
        return dsolid

    def eval_dfluid(self, f):
        dfluid = self.model.fluid.get_properties()
        dfluid.vector[:] = 0.0
        return dfluid

    def eval_dt0(self, f, n):
        return 0.0

    def eval_ddt(self, f, n):
        return 0.0

class FinalFlowRateNorm(Functional):
    func_types = ()

    def eval(self, f):
        qp = f.get_fluid_state(f.size-1)

        return qp[0]**2

    def eval_duva(self, f, n, iter_params0, iter_params1):
        return (0.0, 0.0, 0.0)

    def eval_dqp(self, f, n, iter_params0, iter_params1):
        dq, dp = self.model.fluid.get_state_vecs()

        if n == f.size-1:
            qp = f.get_fluid_state(n)
            dq[:] = 2*qp[0]

        return dq, dp

    def eval_dsolid(self, f):
        dsolid = self.model.solid.get_properties()
        dsolid.vector[:] = 0.0
        return dsolid

    def eval_dfluid(self, f):
        dfluid = self.model.fluid.get_properties()
        dfluid.vector[:] = 0.0
        return dfluid

    def eval_dt0(self, f, n):
        return 0.0

    def eval_ddt(self, f, n):
        return 0.0

## Energy functionals
class ElasticEnergyDifference(Functional):
    func_types = ()
    default_constants = {}

    @staticmethod
    def form_definitions(model):
        from ..solids import biform_k
        forms = {}

        emod = model.solid.forms['coeff.prop.emod']
        nu = model.solid.forms['coeff.prop.nu']

        u_ini = dfn.Function(model.solid.vector_fspace)
        u_fin = dfn.Function(model.solid.vector_fspace)

        en_elastic_ini = biform_k(u_ini, u_ini, emod, nu)
        en_elastic_fin = biform_k(u_fin, u_fin, emod, nu)

        forms['u_ini'] = u_ini
        forms['u_fin'] = u_fin
        forms['en_elastic_ini'] = en_elastic_ini
        forms['en_elastic_fin'] = en_elastic_fin
        forms['den_elastic_ini_du'] = dfn.derivative(en_elastic_ini, u_ini, model.solid.vector_trial)
        forms['den_elastic_fin_du'] = dfn.derivative(en_elastic_fin, u_fin, model.solid.vector_trial)

        forms['den_elastic_ini_demod'] = dfn.derivative(en_elastic_ini, emod)
        forms['den_elastic_fin_demod'] = dfn.derivative(en_elastic_fin, emod)
        return forms

    def eval(self, f):
        self.model.set_solid_props(f.get_solid_props())
        u_ini_vec = f.get_state(0)[0]
        u_fin_vec = f.get_state(-1)[0]

        self.forms['u_ini'].vector()[:] = u_ini_vec
        self.forms['u_fin'].vector()[:] = u_fin_vec

        en_elastic_ini = dfn.assemble(self.forms['en_elastic_ini'])
        en_elastic_fin = dfn.assemble(self.forms['en_elastic_fin'])
        return (en_elastic_fin - en_elastic_ini)**2

    def eval_duva(self, f, n, iter_params0, iter_params1):
        du, dv, da = 0.0, 0.0, 0.0
        if n == 0 or n == f.size-1:
            u_ini_vec = f.get_state(0)[0]
            u_fin_vec = f.get_state(-1)[0]

            self.forms['u_ini'].vector()[:] = u_ini_vec
            self.forms['u_fin'].vector()[:] = u_fin_vec
            self.model.set_solid_props(f.get_solid_props())

            en_elastic_ini = dfn.assemble(self.forms['en_elastic_ini'])
            en_elastic_fin = dfn.assemble(self.forms['en_elastic_fin'])

            den_elastic_ini_du = dfn.assemble(self.forms['den_elastic_ini_du'])
            den_elastic_fin_du = dfn.assemble(self.forms['den_elastic_fin_du'])

            # out = (en_elastic_fin - en_elastic_ini)**2
            if n == f.size-1:
                du = 2*(en_elastic_fin - en_elastic_ini) * den_elastic_fin_du
            elif n == 0:
                du = 2*(en_elastic_fin - en_elastic_ini) * -den_elastic_ini_du
        else:
            du = 0

        return du, dv, da

    def eval_dqp(self, f, n, iter_params0, iter_params1):
        dq, dp = self.model.fluid.get_state_vecs()
        return dq, dp

    def eval_dsolid(self, f):
        dsolid = self.model.solid.get_properties()
        dsolid.vector[:] = 0.0

        self.model.set_solid_props(f.get_solid_props())
        u_ini_vec = f.get_state(0)[0]
        u_fin_vec = f.get_state(-1)[0]

        self.forms['u_ini'].vector()[:] = u_ini_vec
        self.forms['u_fin'].vector()[:] = u_fin_vec

        en_elastic_ini = dfn.assemble(self.forms['en_elastic_ini'])
        en_elastic_fin = dfn.assemble(self.forms['en_elastic_fin'])

        den_elastic_ini_demod = dfn.assemble(self.forms['den_elastic_ini_demod'])
        den_elastic_fin_demod = dfn.assemble(self.forms['den_elastic_fin_demod'])
        # out = (en_elastic_fin - en_elastic_ini)**2
        demod = 2*(en_elastic_fin - en_elastic_ini)*(den_elastic_fin_demod - den_elastic_ini_demod)
        dsolid['emod'][:] = demod

        return dsolid

    def eval_dfluid(self, f):
        dfluid = self.model.fluid.get_properties()
        dfluid.vector[:] = 0.0
        return dfluid

    def eval_dt0(self, f, n):
        return 0.0

    def eval_ddt(self, f, n):
        return 0.0

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

        forms = {}
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
            power_left = power_right

        return res

    def eval_duva(self, f, n, iter_params0, iter_params1):
        N_START = 0
        N_STATE = f.get_num_states()
        time = f.get_times()

        dv = dfn.Function(self.model.solid.vector_fspace).vector()
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

    def eval_dqp(self, f, n, iter_params0, iter_params1):
        dq, dp = self.model.fluid.get_state_vecs()
        return dq, dp

    def eval_dsolid(self, f):
        dsolid = self.model.solid.get_properties()
        dsolid.vector[:] = 0.0

        N_START = 0
        N_STATE = f.get_num_states()

        dwork_deta = dfn.Function(self.model.solid.scalar_fspace).vector()
        # Calculate total damped work by the trapezoidal rule
        time = f.get_times()
        self.model.set_params_fromfile(f, 0)
        dpower_left_deta = dfn.assemble(self.forms['ddamping_power_deta'])
        for ii in range(N_START+1, N_STATE):
            # Set form coefficients to represent the equation from state ii to ii+1
            self.model.set_params_fromfile(f, ii)
            dpower_right_deta = dfn.assemble(self.forms['ddamping_power_deta'])
            dwork_deta += (dpower_left_deta + dpower_right_deta)/2 * (time[ii]-time[ii-1])

        dsolid['eta'][:] = dwork_deta
        return dsolid

    def eval_dfluid(self, f):
        dfluid = self.model.fluid.get_properties()
        dfluid.vector[:] = 0.0
        return dfluid

    def eval_dt0(self, f, n):
        dt0 = 0
        # N_START = 0
        # N_STATE = f.get_num_states()

        # # Add dt0 due to work increment to the 'left' of dt0
        # if n >= N_START+1:
        #     # Calculate damped work over n-1 -> n
        #     self.model.set_params_fromfile(f, n-1)
        #     power_left = dfn.assemble(self.forms['damping_power'])

        #     self.model.set_params_fromfile(f, n)
        #     power_right = dfn.assemble(self.forms['damping_power'])
        #     # res += (power_left+power_right)/2 * (time[n]-time[n-1])
        #     dt0 += (power_left+power_right)/2

        # if n < N_STATE-1:
        #     # Calculate damped work over n -> n + 1
        #     self.model.set_params_fromfile(f, n)
        #     power_left = dfn.assemble(self.forms['damping_power'])

        #     self.model.set_params_fromfile(f, n+1)
        #     power_right = dfn.assemble(self.forms['damping_power'])
        #     # res += (power_left+power_right)/2 * (time[n+1]-time[n])
        #     dt0 += -(power_left+power_right)/2

        return dt0

    def eval_ddt(self, f, n):
        ddt = 0
        N_START = 0
        N_STATE = f.get_num_states()
        if n >= N_START+1:
            # Calculate damped work over n-1 -> n
            time = f.get_times()

            self.model.set_params_fromfile(f, n-1)
            power_left = dfn.assemble(self.forms['damping_power'])

            self.model.set_params_fromfile(f, n)
            power_right = dfn.assemble(self.forms['damping_power'])
            # res += (power_left+power_right)/2 * (time[n]-time[n-1])
            ddt = (power_left+power_right)/2

        return ddt

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

    def eval_dqp(self, f, n, iter_params0, iter_params1):
        dq, dp = self.model.fluid.get_state_vecs()
        return dq, dp

    def eval_dsolid(self, f):
        dsolid = self.model.solid.get_properties()
        dsolid.vector[:] = 0.0

        dsolid['emod'][:] = dfn.assemble(self.forms['ddamping_power_demod']) * self.model.solid.dt.vector()[0]
        return dsolid

    def eval_dfluid(self, f):
        dfluid = self.model.fluid.get_properties()
        dfluid.vector[:] = 0.0
        return dfluid

    def eval_dt0(self, f, n):
        return 0.0

    def eval_ddt(self, f, n):
        ddt = 0
        N_START = 0
        # N_STATE = f.get_num_states()

        res = 0
        # for ii in range(N_START, N_STATE-1):
        if n > N_START:
            # Set form coefficients to represent the equation from state ii to ii+1
            self.model.set_iter_params_fromfile(f, n)
            # res += dfn.assemble(self.forms['damping_power']) * self.model.solid.dt.vector()[0]
            ddt += dfn.assemble(self.forms['damping_power'])

        return ddt

class TransferWorkbyVelocity(Functional):
    """
    Return work done by the fluid on the vocal folds by integrating power over the surface over time.

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

        pressure = solid.forms['coeff.fsi.p0']
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
        dfluid_power_dun = dfn.assemble(self.forms['dfluid_power_du'])
        dfluid_power_dvn = dfn.assemble(self.forms['dfluid_power_dv'])

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

    def eval_dqp(self, f, n, iter_params0, iter_params1):
        dq, dp = self.model.fluid.get_state_vecs()

        # The work terms that involve state n are given by
        # ... + 1/2*(power[n-1]+power[n])*(t[n]-t[n-1]) + 1/2*(power[n]+power[n+1])*(t[n+1]-t[n]) + ...
        N_START = self.constants['n_start']
        N_STATE = f.size

        self.model.set_params_fromfile(f, n)
        dfluidpower_dp = dfn.assemble(self.forms['dfluid_power_dpressure'])
        dfluidpower_dp = self.model.map_fsi_scalar_from_solid_to_fluid(dfluidpower_dp)

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

        dp = 0.5 * dfluidpower_dp * (dt_left + dt_right)

        return dq, dp

    def eval_dsolid(self, f):
        dsolid = self.model.solid.get_properties()
        dsolid.vector[:] = 0.0
        return dsolid

    def eval_dfluid(self, f):
        dfluid = self.model.fluid.get_properties()
        dfluid.vector[:] = 0.0
        return dfluid

    def eval_dt0(self, f, n):
        dt0 = 0.0

        # N_START = self.constants['n_start']
        # N_STATE = f.size

        # # t = f['time'][:]
        # def power(n):
        #     self.model.set_params_fromfile(f, n)
        #     return dfn.assemble(self.forms['fluid_power'])

        # # derivative due to the left interval
        # if n > N_START:
        #     # work += 1/2*(power(n-1) + power(n))*(t[n]-t[n-1])
        #     dt0 += 1/2*(power(n-1) + power(n))

        # # derivative due to right interval
        # if n < N_STATE-1:
        #     # work += 1/2*(power(n+1) + power(n))*(t[n+1]-t[n])
        #     dt0 += -1/2*(power(n+1) + power(n))

        return dt0

    def eval_ddt(self, f, n):
        ddt = 0.0

        N_START = self.constants['n_start']

        # Add sensitivity to state `n` based on the 'left' quadrature interval
        if n > N_START:
            self.model.set_params_fromfile(f, n-1)
            fluid_power0 = dfn.assemble(self.forms['fluid_power'])

            self.model.set_params_fromfile(f, n)
            fluid_power1 = dfn.assemble(self.forms['fluid_power'])
            ddt += 0.5 * (fluid_power0 + fluid_power1)

        return ddt

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

        pressure = solid.forms['coeff.fsi.p1']
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

                du[:] += dfn.assemble(self.forms['dfluid_work_du1'])

            if n != N_STATE-1:
                self.model.set_iter_params_fromfile(f, n+1)

                du[:] += dfn.assemble(self.forms['dfluid_work_du0'])

        return du, 0.0, 0.0

    def eval_dqp(self, f, n, iter_params0, iter_params1):
        N_START = self.constants['n_start']
        N_STATE = f.get_num_states()

        dq, dp = self.model.fluid.get_state_vecs()

        if n >= N_START:
            if n != N_START:
                # self.model.set_iter_params_fromfile(f, n)
                self.model.set_iter_params_fromfile(f, n)
                dfluidwork_dp = dfn.assemble(self.forms['dfluid_work_dpressure'])
                dp[:] += self.model.map_fsi_scalar_from_solid_to_fluid(dfluidwork_dp)

        return dq, dp

    def eval_dsolid(self, f):
        dsolid = self.model.solid.get_properties()
        dsolid.vector[:] = 0.0
        return dsolid

    def eval_dfluid(self, f):
        dfluid = self.model.fluid.get_properties()
        dfluid.vector[:] = 0.0
        return dfluid

    def eval_dt0(self, f, n):
        return 0.0

    def eval_ddt(self, f, n):
        return 0.0

class SubglottalWork(Functional):
    """
    Return the total work input into the fluid from the lungs.
    """
    func_types = ()
    default_constants = {'n_start': 0}

    def eval(self, f):
        # meas_ind = f.get_meas_indices()
        N_START = self.constants['n_start']
        N_STATE = f.get_num_states()

        ret = 0
        fluid_props = f.get_fluid_props(0)
        for ii in range(N_START+1, N_STATE):
            # Set form coefficients to represent the equation mapping state ii->ii+1
            qp0 = f.get_fluid_state(ii-1)
            qp1 = f.get_fluid_state(ii)
            dt = f.get_time(ii) - f.get_time(ii-1)

            ret += 0.5*(qp0[0]+qp1[0])*fluid_props['p_sub']*dt

        self.cache.update({'N_STATE': N_STATE})

        return ret

    def eval_duva(self, f, n, iter_params0, iter_params1):
        return 0.0, 0.0, 0.0

    def eval_dqp(self, f, n, iter_params0, iter_params1):
        dq, dp = self.model.fluid.get_state_vecs()

        N_START = self.constants['n_start']
        N_STATE = self.cache['N_STATE']

        fluid_props = f.get_fluid_props(0)
        # derivative from left quadrature interval
        if n > N_START:
            dt = f.get_time(n) - f.get_time(n-1)
            dq[:] += 0.5*fluid_props['p_sub']*dt

        # derivative from right quadrature interval
        if n < N_STATE-1:
            dt = f.get_time(n+1) - f.get_time(n)
            dq[:] += 0.5*fluid_props['p_sub']*dt

        return dq, dp

    def eval_dsolid(self, f):
        dsolid = self.model.solid.get_properties()
        dsolid.vector[:] = 0.0
        return dsolid

    def eval_dfluid(self, f):
        dfluid = self.model.fluid.get_properties()
        dfluid.vector[:] = 0.0
        return dfluid

    def eval_dt0(self, f, n):
        return 0.0

    def eval_ddt(self, f, n):
        ddt = 0.0

        N_START = self.constants['n_start']

        fluid_props = f.get_fluid_props(0)
        def power(n):
            qp = f.get_fluid_state(n)
            return qp[0]*fluid_props['p_sub']

        if n > N_START:
            # dt = f.get_time(n) - f.get_time(n-1)
            # work = 0.5*(power(n-1)+power(n))*dt
            ddt += 0.5*(power(n-1)+power(n))

        return ddt

class TransferEfficiency(Functional):
    """
    Returns the total vocal efficiency.

    This is the ratio of the total work done by the fluid on the folds to the total input work on
    the fluid.
    """
    func_types = (TransferWorkbyDisplacementIncrement, SubglottalWork)
    default_constants = {}

    def eval(self, f):
        totalfluidwork = self.funcs[0](f)
        totalinputwork = self.funcs[1](f)

        res = totalfluidwork/totalinputwork

        self.cache.update({'totalfluidwork': totalfluidwork, 'totalinputwork': totalinputwork})
        return res

    def eval_duva(self, f, n, iter_params0, iter_params1):
        # TODO : Is there something slightly wrong with this one? Seems slightly wrong from
        # comparing with FD. The error is small but it is not propto step size?
        # N_START = self.constants['m_start']

        tfluidwork = self.funcs[0](f)
        tinputwork = self.funcs[1](f)

        dtotalfluidwork_dun = self.funcs[0].du(f, n, iter_params0, iter_params1)
        dtotalinputwork_dun = self.funcs[1].du(f, n, iter_params0, iter_params1)

        # du = None
        # if n < N_START:
        #     du = dfn.Function(self.model.solid.vector_fspace).vector()
        # else:
        du = dtotalfluidwork_dun/tinputwork - tfluidwork/tinputwork**2*dtotalinputwork_dun

        return du, 0.0, 0.0

    def eval_dqp(self, f, n, iter_params0, iter_params1):
        dq, dp = self.model.fluid.get_state_vecs()
        return dq, dp

    def eval_dsolid(self, f):
        dsolid = self.model.solid.get_properties()
        dsolid.vector[:] = 0.0
        return dsolid

    def eval_dfluid(self, f):
        dfluid = self.model.fluid.get_properties()
        dfluid.vector[:] = 0.0
        return dfluid

    def eval_dt0(self, f, n):
        return 0.0

    def eval_ddt(self, f, n):
        tfluidwork = self.funcs[0](f)
        tinputwork = self.funcs[1](f)

        dtotalfluidwork_ddt = self.funcs[0].ddt(f, n)
        dtotalinputwork_ddt = self.funcs[1].ddt(f, n)
        return dtotalfluidwork_ddt/tinputwork - tfluidwork/tinputwork**2*dtotalinputwork_ddt

## Glottal width functionals
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

    def eval_dsolid(self, f):
        dsolid = self.model.solid.get_properties()
        dsolid.vector[:] = 0.0
        return dsolid

    def eval_dfluid(self, f):
        dfluid = self.model.fluid.get_properties()
        dfluid.vector[:] = 0.0
        return dfluid

    def eval_dt0(self, f, n):
        return 0.0

    def eval_ddt(self, f, n):
        return 0.0

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
