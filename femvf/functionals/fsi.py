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
# import matplotlib.pyplot as plt
import scipy.signal as sig

import dolfin as dfn
import ufl

from .solid import SolidFunctional
from ..models.fluid import smoothmin, dsmoothmin_df
from ..models.solid import strain

class FSIFunctional(SolidFunctional):
    def __init__(self, model):
        super().__init__()
        self.fluid = model.fluid

    # These are written to handle the case where you have a coupled model input
    # then the provided eval_dsl_state only supplies the solid portion and needs to be 
    # extended
    def eval_dstate(self, f, n):
        raise NotImplementedError("")

    def eval_dprops(self, f):
        vecs = [self.eval_dsl_props]
        keys = self.properties.keys
        for attr in ('fluid', 'acoustic'):
            if hasattr(self.model, attr):
                vecs.append(getattr(self.model, attr).get_properties_vec())
        return linalg.concatenate(vecs, keys)

class TransferWorkbyVelocity(FSIFunctional):
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

    def eval_duva(self, f, n):
        duva = self.model.solid.get_state_vec()
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

        duva['u'][:] = 0.5 * dfluid_power_dun * (dt_left + dt_right)
        duva['v'][:] = 0.5 * dfluid_power_dvn * (dt_left + dt_right)

        return duva

    def eval_dqp(self, f, n):
        dqp = self.model.fluid.get_state_vec()

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

        dqp['p'][:] = 0.5 * dfluidpower_dp * (dt_left + dt_right)

        return dqp

    def eval_dsolid(self, f):
        dsolid = self.model.solid.get_properties_vec()
        dsolid.set(0.0)
        return dsolid

    def eval_dfluid(self, f):
        dfluid = self.model.fluid.get_properties_vec()
        dfluid.set(0.0)
        return dfluid

    def eval_dt0(self, f, n):
        dt0 = 0.0

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

class TransferWorkbyDisplacementIncrement(FSIFunctional):
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

    def eval_duva(self, f, n):
        duva = self.model.solid.get_state_vec()
        N_START = self.constants['n_start']
        N_STATE = f.size

        if n >= N_START:
            if n != N_START:
                self.model.set_iter_params_fromfile(f, n)
                # self.model.set_iter_params(**iter_params0)

                duva[0][:] += dfn.assemble(self.forms['dfluid_work_du1'])

            if n != N_STATE-1:
                self.model.set_iter_params_fromfile(f, n+1)

                duva[0][:] += dfn.assemble(self.forms['dfluid_work_du0'])

        return duva

    def eval_dqp(self, f, n):
        dqp = self.model.fluid.get_state_vec()
        N_START = self.constants['n_start']
        # N_STATE = f.size

        if n >= N_START:
            if n != N_START:
                # self.model.set_iter_params_fromfile(f, n)
                self.model.set_iter_params_fromfile(f, n)
                dfluidwork_dp = dfn.assemble(self.forms['dfluid_work_dpressure'])
                dqp['p'][:] += self.model.map_fsi_scalar_from_solid_to_fluid(dfluidwork_dp)

        return dqp

    def eval_dsolid(self, f):
        dsolid = self.model.solid.get_properties_vec()
        dsolid.set(0.0)
        return dsolid

    def eval_dfluid(self, f):
        dfluid = self.model.fluid.get_properties_vec()
        dfluid.set(0.0)
        return dfluid

    def eval_dt0(self, f, n):
        return 0.0

    def eval_ddt(self, f, n):
        return 0.0
