"""
This module contains definitions of various functionals over the solid state.

A SolidFunctional is mapping from the time history of all states and parameters to a real number
.. ::math {(u, v, a, q, p; t, params)_n for n in {0, 1, ..., N-1}} .

A SolidFunctional should take in the entire time history of states from a forward model run and return a
real number.

For computing the sensitivity of the SolidFunctional through the discrete adjoint method, you also need
the sensitivity of the SolidFunctional with respect to the n'th state. This function has the signature

dfunctional_du(model, n, f, ....) -> float, dict

, where the parameters have the same uses as defined previously. Parameter `n` is the state to
compute the sensitivity with respect to.
"""

import numpy as np
# import matplotlib.pyplot as plt
# import scipy.signal as sig

import dolfin as dfn
import ufl

from .. import linalg
from .base import AbstractFunctional
from ..models.solid import form_inf_strain, Solid

class SolidFunctional(AbstractFunctional):
    """
    This class provides an interface/method to define basic solid functionals

    To define a new SolidFunctional, create class attributes `func_types` and `default_constants`. The
    `func_types` should be a tuple of `SolidFunctional` classes that are needed to calculate the
    SolidFunctional, and will be added to the `funcs` instance attribute as SolidFunctional objects where
    you can access them. Then you need to implement the `eval`, `eval_du` ... `eval_dp`,
    `form_definitions` methods.
    """
    def __init__(self, model):
        super().__init__(model, ())

        if isinstance(model, Solid):
            self.solid = model
            self.coupled_model = False
        else:
            self.coupled_model = True
            self.solid = model.solid
        
        self._forms = self.form_definitions(self.solid)

    @staticmethod
    def form_definitions(model):
        return None

    @property
    def forms(self):
        """
        Return a dictionary of UFL variational forms.
        """
        return self._forms

    # These are written to handle the case where you have a coupled model input
    # then the provided eval_dsl_state only supplies the solid portion and needs to be 
    # extended
    def eval_dstate(self, f, n):
        vecs = [self.eval_dsl_state(f, n)]
        for attr in ('fluid', 'acoustic'):
            if hasattr(self.model, attr):
                vecs.append(getattr(self.model, attr).get_state_vec())
        return linalg.concatenate(*vecs)

    def eval_dprops(self, f):
        vecs = [self.eval_dsl_props(f)]
        for attr in ('fluid', 'acoustic'):
            if hasattr(self.model, attr):
                vecs.append(getattr(self.model, attr).get_properties_vec())
        return linalg.concatenate(*vecs)

class PeriodicError(SolidFunctional):
    r"""
    SolidFunctional that measures the periodicity of a simulation

    This returns
    .. math:: \int_\Omega ||u(T)-u(0)||_2^2 + ||v(T)-v(0)||_2^2 \, dx ,
    where :math:T is the period.
    """
    func_types = ()
    default_constants = {'alpha': 1e3}

    @staticmethod
    def form_definitions(solid):
        forms = {}
        forms['u_0'] = dfn.Function(solid.vector_fspace)
        forms['u_N'] = dfn.Function(solid.vector_fspace)
        forms['v_0'] = dfn.Function(solid.vector_fspace)
        forms['v_N'] = dfn.Function(solid.vector_fspace)
        forms['a_0'] = dfn.Function(solid.vector_fspace)
        forms['a_N'] = dfn.Function(solid.vector_fspace)

        res_u = forms['u_N']-forms['u_0']
        res_v = forms['v_N']-forms['v_0']
        res_a = forms['a_N']-forms['a_0']

        # forms['alpha'] = dfn.Constant(1.0)

        forms['resu'] = ufl.inner(res_u, res_u) * ufl.dx
        forms['resv'] = ufl.inner(res_v, res_v) * ufl.dx
        forms['resa'] = ufl.inner(res_a, res_a) * ufl.dx

        forms['dresu_du_0'] = ufl.derivative(forms['resu'], forms['u_0'], solid.vector_trial)
        forms['dresu_du_N'] = ufl.derivative(forms['resu'], forms['u_N'], solid.vector_trial)

        forms['dresv_dv_0'] = ufl.derivative(forms['resv'], forms['v_0'], solid.vector_trial)
        forms['dresv_dv_N'] = ufl.derivative(forms['resv'], forms['v_N'], solid.vector_trial)

        forms['dresa_da_0'] = ufl.derivative(forms['resa'], forms['a_0'], solid.vector_trial)
        forms['dresa_da_N'] = ufl.derivative(forms['resa'], forms['a_N'], solid.vector_trial)
        return forms

    def eval(self, f):
        alphau = self.constants['alpha']
        self.forms['u_0'].vector()[:], self.forms['v_0'].vector()[:], self.forms['a_0'].vector()[:] = f.get_state(0)[:3]
        self.forms['u_N'].vector()[:], self.forms['v_N'].vector()[:], self.forms['a_N'].vector()[:] = f.get_state(f.size-1)[:3]
        erru = dfn.assemble(self.forms['resu'])
        errv = dfn.assemble(self.forms['resv'])
        erra = dfn.assemble(self.forms['resa'])
        # print(alphau**2*erru, errv)
        return alphau**2*erru + errv + erra

    def eval_dsl_state(self, f, n):
        duva = self.solid.get_state_vec()
        alphau = self.constants['alpha']
        if n == 0:
            duva[0][:] = alphau**2*dfn.assemble(self.forms['dresu_du_0'])
            duva[1][:] = dfn.assemble(self.forms['dresv_dv_0'])
            duva[2][:] = dfn.assemble(self.forms['dresa_da_0'])
        elif n == f.size-1:
            duva[0][:] = alphau**2*dfn.assemble(self.forms['dresu_du_N'])
            duva[1][:] = dfn.assemble(self.forms['dresv_dv_N'])
            duva[2][:] = dfn.assemble(self.forms['dresa_da_N'])
        return duva

    def eval_dsl_props(self, f):
        dsolid = self.solid.get_properties_vec()
        dsolid.set(0.0)
        return dsolid

    def eval_dt0(self, f, n):
        return 0.0

    def eval_ddt(self, f, n):
        return 0.0

class ComponentPeriodicError(SolidFunctional):
    r"""
    SolidFunctional that measures the periodicity of a single state component

    This returns
    .. math:: \int_\Omega ||u(T)-u(0)||_2^2 + ||v(T)-v(0)||_2^2 \, dx ,
    where :math:T is the period.
    """
    IDX_COMP = -1
    @staticmethod
    def form_definitions(solid):
        forms = {}
        forms['u_0'] = dfn.Function(solid.vector_fspace)
        forms['u_N'] = dfn.Function(solid.vector_fspace)

        res_u = forms['u_N']-forms['u_0']

        forms['resu'] = ufl.inner(res_u, res_u) * ufl.dx

        forms['dresu_du_0'] = ufl.derivative(forms['resu'], forms['u_0'], solid.vector_trial)
        forms['dresu_du_N'] = ufl.derivative(forms['resu'], forms['u_N'], solid.vector_trial)
        return forms

    def eval(self, f):
        self.forms['u_0'].vector()[:]= f.get_state(0)[self.IDX_COMP]
        self.forms['u_N'].vector()[:]= f.get_state(f.size-1)[self.IDX_COMP]
        erru = dfn.assemble(self.forms['resu'])
        return erru

    def eval_dsl_state(self, f, n):
        duva = self.solid.get_state_vec()
        if n == 0:
            duva[self.IDX_COMP][:] = dfn.assemble(self.forms['dresu_du_0'])
        elif n == f.size-1:
            duva[self.IDX_COMP][:] = dfn.assemble(self.forms['dresu_du_N'])
        return duva

    def eval_dsl_props(self, f):
        dsolid = self.solid.get_properties_vec()
        dsolid.set(0.0)
        return dsolid

    def eval_dt0(self, f, n):
        return 0.0

    def eval_ddt(self, f, n):
        return 0.0

class UPeriodicError(ComponentPeriodicError):
    IDX_COMP = 0

class VPeriodicError(ComponentPeriodicError):
    IDX_COMP = 1

class APeriodicError(ComponentPeriodicError):
    IDX_COMP = 2

class PeriodicEnergyError(SolidFunctional):
    r"""
    SolidFunctional that measures the periodicity of a simulation

    This returns
    .. math:: \int_\Omega ||u(T)-u(0)||_K^2 + ||v(T)-v(0)||_M^2 \, dx ,
    where :math:T is the period.
    """
    func_types = ()
    default_constants = {'alpha': 1.0}

    @staticmethod
    def form_definitions(solid):
        from ..solid import biform_m, biform_k
        forms = {}
        forms['u_0'] = dfn.Function(solid.vector_fspace)
        forms['u_N'] = dfn.Function(solid.vector_fspace)
        forms['v_0'] = dfn.Function(solid.vector_fspace)
        forms['v_N'] = dfn.Function(solid.vector_fspace)

        res_u = forms['u_N']-forms['u_0']
        res_v = forms['v_N']-forms['v_0']
        rho = solid.forms['coeff.prop.rho']
        emod = solid.forms['coeff.prop.emod']
        nu = solid.forms['coeff.prop.nu']

        forms['resu'] = biform_k(res_u, res_u, emod, nu)
        forms['resv'] = biform_m(res_v, res_v, rho)

        forms['dresu_du_0'] = ufl.derivative(forms['resu'], forms['u_0'], solid.vector_trial)
        forms['dresu_du_N'] = ufl.derivative(forms['resu'], forms['u_N'], solid.vector_trial)
        forms['dresu_demod'] = ufl.derivative(forms['resu'], emod, solid.scalar_trial)

        forms['dresv_dv_0'] = ufl.derivative(forms['resv'], forms['v_0'], solid.vector_trial)
        forms['dresv_dv_N'] = ufl.derivative(forms['resv'], forms['v_N'], solid.vector_trial)
        return forms

    def eval(self, f):
        alphau = self.constants['alpha']

        self.model.set_solid_props(f.get_solid_props(0))
        self.forms['u_0'].vector()[:], self.forms['v_0'].vector()[:], _ = f.get_state(0)
        self.forms['u_N'].vector()[:], self.forms['v_N'].vector()[:], _ = f.get_state(f.size-1)

        erru = dfn.assemble(self.forms['resu'])
        errv = dfn.assemble(self.forms['resv'])
        print(alphau**2*erru, errv)
        return alphau**2*erru + errv

    def eval_dsl_state(self, f, n):
        duva = self.solid.get_state_vec()

        alphau = self.constants['alpha']
        if n == 0:
            duva[0][:] = alphau**2*dfn.assemble(self.forms['dresu_du_0'])
            duva[1][:] = dfn.assemble(self.forms['dresv_dv_0'])
        elif n == f.size-1:
            duva[0][:] = alphau**2*dfn.assemble(self.forms['dresu_du_N'])
            duva[1][:] = dfn.assemble(self.forms['dresv_dv_N'])
        return duva

    def eval_dsl_props(self, f):
        dsolid = self.solid.get_properties_vec()
        dsolid.set(0.0)

        alphau = self.constants['alpha']

        self.model.set_solid_props(f.get_solid_props(0))
        self.forms['u_0'].vector()[:], self.forms['v_0'].vector()[:], _ = f.get_state(0)
        self.forms['u_N'].vector()[:], self.forms['v_N'].vector()[:], _ = f.get_state(f.size-1)

        derru_demod = dfn.assemble(self.forms['dresu_demod'])
        dsolid['emod'][:] = alphau**2*derru_demod

        return dsolid

    def eval_dt0(self, f, n):
        return 0.0

    def eval_ddt(self, f, n):
        return 0.0

class FinalDisplacementNorm(SolidFunctional):
    r"""
    Return the l2 norm of displacement at the final time

    This returns :math:`\sum{||\vec{u}||}_2`.
    """
    func_types = ()
    default_constants = {}

    @staticmethod
    def form_definitions(solid):
        forms = {}
        forms['u'] = dfn.Function(solid.vector_fspace)
        forms['res'] = ufl.inner(forms['u'], forms['u']) * ufl.dx

        forms['dres_du'] = ufl.derivative(forms['res'], forms['u'], solid.vector_trial)
        return forms

    def eval(self, f):
        self.forms['u'].vector()[:] = f.get_state(f.size-1)[0]

        return dfn.assemble(self.forms['res'])

    def eval_dsl_state(self, f, n):
        duva = self.solid.get_state_vec()

        if n == f.size-1:
            self.forms['u'].vector()[:] = f.get_state(n)[0]
            duva['u'][:] = dfn.assemble(self.forms['dres_du'])

        return duva

    def eval_dsl_props(self, f):
        dsolid = self.solid.get_properties_vec()
        dsolid.set(0.0)
        return dsolid

    def eval_dt0(self, f, n):
        return 0.0

    def eval_ddt(self, f, n):
        return 0.0

class FinalVelocityNorm(SolidFunctional):
    r"""
    Return the l2 norm of velocity at the final time

    This returns :math:`\sum{||\vec{u}||}_2`.
    """
    func_types = ()
    default_constants = {}

    @staticmethod
    def form_definitions(solid):
        forms = {}
        forms['v'] = dfn.Function(solid.vector_fspace)
        forms['res'] = ufl.inner(forms['v'], forms['v']) * ufl.dx

        forms['dres_dv'] = ufl.derivative(forms['res'], forms['v'], solid.vector_trial)
        return forms

    def eval(self, f):
        self.forms['v'].vector()[:] = f.get_state(f.size-1)[1]

        return dfn.assemble(self.forms['res'])

    def eval_dsl_state(self, f, n):
        duva = self.solid.get_state_vec()

        if n == f.size-1:
            self.forms['v'].vector()[:] = f.get_state(n)[1]
            duva['v'][:] = dfn.assemble(self.forms['dres_dv'])

        return duva

    def eval_dsl_props(self, f):
        dsolid = self.solid.get_properties_vec()
        dsolid.set(0.0)
        return dsolid

    def eval_dt0(self, f, n):
        return 0.0

    def eval_ddt(self, f, n):
        return 0.0

class FinalSurfaceDisplacementNorm(SolidFunctional):
    r"""
    Return the l2 norm of displacement at the final time

    This returns :math:`\sum{||\vec{u}||}_2`.
    """
    func_types = ()
    default_constants = {}

    @staticmethod
    def form_definitions(solid):
        solid = solid
        forms = {}
        forms['u'] = dfn.Function(solid.vector_fspace)
        forms['res'] = ufl.inner(forms['u'], forms['u']) * solid.ds(solid.facet_label_to_id['pressure'])

        forms['dres_du'] = ufl.derivative(forms['res'], forms['u'], solid.vector_trial)
        return forms

    def eval(self, f):
        self.forms['u'].vector()[:] = f.get_state(f.size-1)[0]

        return dfn.assemble(self.forms['res'])

    def eval_dsl_state(self, f, n):
        duva = self.solid.get_state_vec()

        if n == f.size-1:
            self.forms['u'].vector()[:] = f.get_state(n)[0]
            duva['u'][:] = dfn.assemble(self.forms['dres_du'])

        return duva

    def eval_dsl_props(self, f):
        dsolid = self.solid.get_properties_vec()
        dsolid.set(0.0)
        return dsolid

    def eval_dt0(self, f, n):
        return 0.0

    def eval_ddt(self, f, n):
        return 0.0

## Energy functionals
class ElasticEnergyDifference(SolidFunctional):
    func_types = ()
    default_constants = {}

    @staticmethod
    def form_definitions(solid):
        from ..solid import biform_k
        forms = {}

        emod = solid.forms['coeff.prop.emod']
        nu = solid.forms['coeff.prop.nu']

        u_ini = dfn.Function(solid.vector_fspace)
        u_fin = dfn.Function(solid.vector_fspace)

        en_elastic_ini = biform_k(u_ini, u_ini, emod, nu)
        en_elastic_fin = biform_k(u_fin, u_fin, emod, nu)

        forms['u_ini'] = u_ini
        forms['u_fin'] = u_fin
        forms['en_elastic_ini'] = en_elastic_ini
        forms['en_elastic_fin'] = en_elastic_fin
        forms['den_elastic_ini_du'] = dfn.derivative(en_elastic_ini, u_ini, solid.vector_trial)
        forms['den_elastic_fin_du'] = dfn.derivative(en_elastic_fin, u_fin, solid.vector_trial)

        forms['den_elastic_ini_demod'] = dfn.derivative(en_elastic_ini, emod)
        forms['den_elastic_fin_demod'] = dfn.derivative(en_elastic_fin, emod)
        return forms

    def eval(self, f):
        self.model.set_solid_props(f.get_solid_props(0))
        u_ini_vec = f.get_state(0)[0]
        u_fin_vec = f.get_state(-1)[0]

        self.forms['u_ini'].vector()[:] = u_ini_vec
        self.forms['u_fin'].vector()[:] = u_fin_vec

        en_elastic_ini = dfn.assemble(self.forms['en_elastic_ini'])
        en_elastic_fin = dfn.assemble(self.forms['en_elastic_fin'])
        return (en_elastic_fin - en_elastic_ini)**2

    def eval_dsl_state(self, f, n):
        duva = self.solid.get_state_vec()
        if n == 0 or n == f.size-1:
            u_ini_vec = f.get_state(0)[0]
            u_fin_vec = f.get_state(-1)[0]

            self.forms['u_ini'].vector()[:] = u_ini_vec
            self.forms['u_fin'].vector()[:] = u_fin_vec
            self.model.set_solid_props(f.get_solid_props(0))

            en_elastic_ini = dfn.assemble(self.forms['en_elastic_ini'])
            en_elastic_fin = dfn.assemble(self.forms['en_elastic_fin'])

            den_elastic_ini_du = dfn.assemble(self.forms['den_elastic_ini_du'])
            den_elastic_fin_du = dfn.assemble(self.forms['den_elastic_fin_du'])

            # out = (en_elastic_fin - en_elastic_ini)**2
            if n == f.size-1:
                duva[0][:] = 2*(en_elastic_fin - en_elastic_ini) * den_elastic_fin_du
            elif n == 0:
                duva[0][:] = 2*(en_elastic_fin - en_elastic_ini)*-1*den_elastic_ini_du

        return duva

    def eval_dsl_props(self, f):
        dsolid = self.solid.get_properties_vec()
        dsolid.set(0.0)

        self.model.set_solid_props(f.get_solid_props(0))
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

    def eval_dt0(self, f, n):
        return 0.0

    def eval_ddt(self, f, n):
        return 0.0

class KV3DDampingWork(SolidFunctional):
    """
    Returns the work dissipated in the tissue due to damping
    """
    func_types = ()
    default_constants = {
        'n_start': 0
    }

    @staticmethod
    def form_definitions(solid):
        solid = solid

        # Load some ufl forms from the solid model
        v1 = solid.forms['coeff.state.v1']
        eta = solid.forms['coeff.prop.eta']
        uant, upos = dfn.Function(solid.vector_fspace), dfn.Function(solid.vector_fspace)

        d2v_dz2 = (uant - 2*v1 + upos) / solid.forms['coeff.prop.length']**2

        forms = {}
        forms['damping_power'] = (ufl.inner(eta*form_inf_strain(v1), form_inf_strain(v1)) 
                                  + ufl.inner(-0.5*eta*d2v_dz2, v1)) * ufl.dx
        forms['ddamping_power_dv'] = dfn.derivative(forms['damping_power'], v1)
        forms['ddamping_power_deta'] = dfn.derivative(forms['damping_power'], eta)
        return forms

    def eval(self, f):
        solid = self.model.solid
        self.model.set_properties(f.get_properties())
        N_START = self.constants['n_start']
        N_STATE = f.size

        res = 0
        # Calculate total damped work by the trapezoidal rule
        time = f.get_times()
        self.model.set_fin_state(f.get_state(N_START))
        power_left = dfn.assemble(self.forms['damping_power'])
        for ii in range(N_START+1, N_STATE):
            # Set form coefficients to represent the equation from state ii to ii+1
            self.model.set_fin_state(f.get_state(ii))

            power_right = dfn.assemble(self.forms['damping_power'])
            res += (power_left+power_right)/2 * (time[ii]-time[ii-1])
            power_left = power_right

        return res

    def eval_dsl_state(self, f, n):
        duva = self.solid.get_state_vec()
        N_START = self.constants['n_start']
        N_STATE = f.size
        time = f.get_times()

        if n >= N_START:
            # self.model.set_params_fromfile(f, n)
            self.model.set_ini_state(f.get_state(n))
            dpower_dvn = dfn.assemble(self.forms['ddamping_power_dv'])

            if n > N_START:
                # Add the sensitivity to `v` from the left intervals right integration point
                duva['v'][:] += 0.5*dpower_dvn*(time[n] - time[n-1])
            if n < N_STATE-1:
                # Add the sensitivity to `v` from the right intervals left integration point
                duva['v'][:] += 0.5*dpower_dvn*(time[n+1] - time[n])

        return duva

    def eval_dsl_props(self, f):
        dsolid = self.solid.get_properties_vec()
        dsolid.set(0.0)

        N_START = N_START = self.constants['n_start']
        N_STATE = f.size

        dwork_deta = dfn.Function(self.solid.scalar_fspace).vector()
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

    def eval_dt0(self, f, n):
        dt0 = 0

        return dt0

    def eval_ddt(self, f, n):
        ddt = 0
        N_START = N_START = self.constants['n_start']
        N_STATE = f.size
        if n >= N_START+1:
            # Calculate damped work over n-1 -> n
            # time = f.get_times()

            self.model.set_params_fromfile(f, n-1)
            power_left = dfn.assemble(self.forms['damping_power'])

            self.model.set_params_fromfile(f, n)
            power_right = dfn.assemble(self.forms['damping_power'])
            # res += (power_left+power_right)/2 * (time[n]-time[n-1])
            ddt = (power_left+power_right)/2

        return ddt

class KVDampingWork(SolidFunctional):
    """
    Returns the work dissipated in the tissue due to damping
    """
    func_types = ()
    default_constants = {
        'n_start': 0
    }

    @staticmethod
    def form_definitions(solid):
        solid = solid

        # Load some ufl forms from the solid model
        v1 = solid.forms['coeff.state.v1']
        eta = solid.forms['coeff.prop.eta']

        forms = {}
        forms['damping_power'] = ufl.inner(eta*form_inf_strain(v1), form_inf_strain(v1)) * ufl.dx
        forms['ddamping_power_dv'] = dfn.derivative(forms['damping_power'], v1)
        forms['ddamping_power_deta'] = dfn.derivative(forms['damping_power'], eta)
        return forms

    def eval(self, f):
        solid = self.model.solid
        self.model.set_properties(f.get_properties())
        N_START = self.constants['n_start']
        N_STATE = f.size

        res = 0
        # Calculate total damped work by the trapezoidal rule
        time = f.get_times()
        
        self.model.set_fin_state(f.get_state(N_START))
        power_left = dfn.assemble(self.forms['damping_power'])
        for ii in range(N_START+1, N_STATE):
            # Set form coefficients to represent the equation from state ii to ii+1
            self.model.set_fin_state(f.get_state(ii))

            power_right = dfn.assemble(self.forms['damping_power'])
            res += (power_left+power_right)/2 * (time[ii]-time[ii-1])
            power_left = power_right

        return res

    def eval_dsl_state(self, f, n):
        duva = self.solid.get_state_vec()
        N_START = self.constants['n_start']
        N_STATE = f.size
        time = f.get_times()

        if n >= N_START:
            self.model.set_fin_state(f.get_state(n))
            dpower_dvn = dfn.assemble(self.forms['ddamping_power_dv'])

            if n > N_START:
                # Add the sensitivity to `v` from the left intervals right integration point
                duva['v'][:] += 0.5*dpower_dvn*(time[n] - time[n-1])
            if n < N_STATE-1:
                # Add the sensitivity to `v` from the right intervals left integration point
                duva['v'][:] += 0.5*dpower_dvn*(time[n+1] - time[n])

        return duva

    def eval_dsl_props(self, f):
        dsolid = self.solid.get_properties_vec()
        dsolid.set(0.0)

        N_START = N_START = self.constants['n_start']
        N_STATE = f.size

        dwork_deta = dfn.Function(self.solid.scalar_fspace).vector()
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

    def eval_dt0(self, f, n):
        dt0 = 0

        return dt0

    def eval_ddt(self, f, n):
        ddt = 0
        N_START = N_START = self.constants['n_start']
        N_STATE = f.size
        if n >= N_START+1:
            # Calculate damped work over n-1 -> n
            # time = f.get_times()

            self.model.set_params_fromfile(f, n-1)
            power_left = dfn.assemble(self.forms['damping_power'])

            self.model.set_params_fromfile(f, n)
            power_right = dfn.assemble(self.forms['damping_power'])
            # res += (power_left+power_right)/2 * (time[n]-time[n-1])
            ddt = (power_left+power_right)/2

        return ddt

class RayleighDampingWork(SolidFunctional):
    """
    Represent the strain work dissipated in the tissue due to damping
    """
    func_types = ()
    default_constants = {}

    @staticmethod
    def form_definitions(solid):
        solid = solid

        # Load some ufl forms from the solid model
        vector_trial = solid.forms['trial.vector']
        scalar_trial = solid.forms['trial.scalar']
        ray_m = solid.forms['coeff.prop.rayleigh_m']
        ray_k = solid.forms['coeff.prop.rayleigh_k']
        rho = solid.forms['coeff.prop.rho']
        emod = solid.forms['coeff.prop.emod']
        nu = solid.forms['coeff.prop.nu']

        v0 = solid.forms['coeff.state.v0']

        from ..solid import biform_m, biform_k
        forms = {}
        forms['damping_power'] = ray_m*biform_m(v0, v0, rho) + ray_k*biform_k(v0, v0, emod, nu)

        forms['ddamping_power_dv'] = ufl.derivative(forms['damping_power'], v0, vector_trial)
        forms['ddamping_power_demod'] = ufl.derivative(forms['damping_power'], emod, scalar_trial)
        return forms

    def eval(self, f):
        N_START = 0
        N_STATE = f.size

        res = 0
        for ii in range(N_START, N_STATE-1):
            # Set form coefficients to represent the equation from state ii to ii+1
            self.model.set_iter_params_fromfile(f, ii+1)
            res += dfn.assemble(self.forms['damping_power']) * self.solid.dt

        return res

    def eval_dsl_state(self, f, n):
        duva = self.solid.get_state_vec()

        duva['v'][:] = dfn.assemble(self.forms['ddamping_power_dv']) * self.solid.dt
        return duva

    def eval_dsl_props(self, f):
        dsolid = self.solid.get_properties_vec()
        dsolid.set(0.0)

        dsolid['emod'][:] = dfn.assemble(self.forms['ddamping_power_demod']) * self.solid.dt
        return dsolid

    def eval_dt0(self, f, n):
        return 0.0

    def eval_ddt(self, f, n):
        ddt = 0
        N_START = 0
        # N_STATE = f.size

        # res = 0
        # for ii in range(N_START, N_STATE-1):
        if n > N_START:
            # Set form coefficients to represent the equation from state ii to ii+1
            self.model.set_iter_params_fromfile(f, n)
            # res += dfn.assemble(self.forms['damping_power']) * self.solid.dt
            ddt += dfn.assemble(self.forms['damping_power'])

        return ddt

## Glottal width functionals
class GlottalWidthErrorNorm(SolidFunctional):
    """
    Represents the difference between a measured vs model glottal width
    """
    func_types = ()
    default_constants = {
        'gw_meas': 0.0,
        'smooth_min_alpha': -2000.0
    }

    def eval(self, f):
        raise NotImplementedError("Need to fix this")
        model = self.model

        # Get the initial locations of the nodes
        X_REF = model.get_ref_config().reshape(-1)
        DOF_SURFACE = solid.vert_to_vdof[model.fsi_verts].reshape(-1)
        X_REF_SURFACE = X_REF[DOF_SURFACE]

        # Calculate the glottal width at every node
        gw_model = []
        idx_meas = f.get_meas_indices()
        for n in idx_meas:
            u = f.get_state(n)[0]
            xy_surf = X_REF_SURFACE + u[DOF_SURFACE]
            y_surf = xy_surf[1::2]
            # TODO: Fix this if you're going to use it
            # gw = smoothmin(y_surf, alpha=self.constants['smooth_min_alpha'])
            gw = 0.0
            gw_model.append(gw)

        return np.sum((np.array(gw_model) - self.constants['gw_meas'])**2)

    def eval_dsl_state(self, f, n):
        duva = self.solid.get_state_vec()
        model = self.model

        # Get the initial locations of the nodes
        X_REF = model.get_ref_config().reshape(-1)
        DOF_SURFACE = solid.vert_to_vdof[model.fsi_verts].reshape(-1)
        Y_DOF = DOF_SURFACE[1::2]
        X_REF_SURFACE = X_REF[DOF_SURFACE]

        # Set up a map from state index to measured index
        N = f.size
        idx_meas = f.get_meas_indices()
        M = idx_meas.size

        n_to_m = {n: -1 for n in range(N)}
        for m, n in enumerate(idx_meas):
            n_to_m[n] = m

        if n_to_m[n] != -1:
            u = f.get_state(n)[0]
            xy_surf = X_REF_SURFACE + u[DOF_SURFACE]
            y_surf = xy_surf[1::2]

            duva['u'][Y_DOF] = dsmoothmin_df(y_surf, self.model.fluid.s_vertices, alpha=self.constants['smooth_min_alpha'])

        return duva

    def eval_dsl_props(self, f):
        dsolid = self.solid.get_properties_vec()
        dsolid.set(0.0)
        return dsolid

    def eval_dt0(self, f, n):
        return 0.0

    def eval_ddt(self, f, n):
        return 0.0
