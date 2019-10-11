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

import os.path as path

import numpy as np
import dolfin as dfn
import ufl

from petsc4py import PETSc

# from . import statefileutils as sfu
from . import forms


class AbstractFunctional():
    """
    Represents a functional over the solution history of a forward model run.

    Parameters
    ----------
    model : ForwardModel
        The forward model instance
    f : StateFile
        A file object containing the solution history of the model run.
    kwargs : optional
        Additional arguments that specify how the functional should be computed

    Attributes
    ----------
    f : statefile.StateFile
    model : forms.ForwardModel
    kwargs : dict
        A dictionary of additional options of how to compute the functional
    funcs : dict of callable
        Dictionary containing sub-functionals instances that are used in computing the functional
    """
    def __init__(self, model, f, **kwargs):
        self.model = model
        self.f = f
        self.kwargs = kwargs

        self.funcs = dict()
        self.cache = dict()

    def __call__(self):
        """
        Return the value of the functional.
        """
        raise NotImplementedError("Method not implemented")

    def du(self, n):
        """
        Return the sensitivity of the functional with respect to the the `n`th state.
        """
        raise NotImplementedError("Method not implemented")

    def dparam(self):
        """
        Return the sensitivity of the functional with respect to the parameters.
        """
        raise NotImplementedError("Method not implemented")

class FluidWork(AbstractFunctional):
    """
    Returns the work done by the fluid on the vocal folds.

    Parameters
    ----------
    n_start : int, optional
        Starting index to compute the functional over
    """
    # TODO: fluid_work is implemented in the ForwardModel class, but you could (should) shift it
    # into here.
    def __init__(self, model, f, **kwargs):
        super(FluidWork, self).__init__(model, f, **kwargs)

        self.kwargs.setdefault('n_start', 0)

    def __call__(self):
        N_START = self.kwargs['n_start']
        N_STATE = self.f.get_num_states()

        res = 0
        for ii in range(N_START, N_STATE-1):
            # Set form coefficients to represent the equation from state ii to ii+1
            self.model.set_iteration_fromfile(self.f, ii+1)
            res += dfn.assemble(self.model.fluid_work)

        return res

    def du(self, n):
        out = 0

        N_START = self.kwargs['n_start']
        N_STATE = self.f.get_num_states()

        if n < N_START:
            out += dfn.Function(self.model.vector_function_space).vector()
        else:
            # Add the sensitivity component due to work from n to n+1
            if n < N_STATE-1:
                self.model.set_iteration_fromfile(self.f, n+1)
                dp_du, _ = self.model.get_flow_sensitivity()

                out += dfn.assemble(self.model.dfluid_work_du0)

                # Correct dfluidwork_du0 since pressure depends on u0
                dfluidwork_dp = dfn.assemble(self.model.dfluid_work_dp,
                                             tensor=dfn.PETScVector()).vec()

                dfluidwork_du0_correction = dfn.as_backend_type(out).vec().copy()
                dp_du.multTranspose(dfluidwork_dp, dfluidwork_du0_correction)

                out += dfn.Vector(dfn.PETScVector(dfluidwork_du0_correction))

            # Add the sensitviity component due to work from n-1 to n
            if n > N_START:
                self.model.set_iteration_fromfile(self.f, n)

                out += dfn.assemble(self.model.dfluid_work_du1)

        return out

    def dparam(self):
        return None

class VolumeFlow(AbstractFunctional):
    """
    Returns the volume of fluid that flowed through the vocal folds.

    Parameters
    ----------
    n_start : int, optional
        Starting index to compute the functional over
    """
    def __init__(self, model, f, **kwargs):
        super(VolumeFlow, self).__init__(model, f, **kwargs)

        self.kwargs.setdefault('n_start', 0)

    def __call__(self):
        N_STATE = self.f.get_num_states()
        N_START = self.kwargs['n_start']

        totalflow = 0
        for ii in range(N_START, N_STATE-1):
            fluid_info, _ = self.model.set_iteration_fromfile(self.f, ii+1)

            totalflow += fluid_info['flow_rate'] * self.model.dt.values()[0]

        return totalflow

    def du(self, n):
        dtotalflow_dun = None
        N_START = self.kwargs['n_start']

        num_states = self.f.get_num_states()
        if n < N_START or n == num_states-1:
            dtotalflow_dun = dfn.Function(self.model.vector_function_space).vector()
        else:
            self.model.set_iteration_fromfile(self.f, n+1)
            _, dq_dun = self.model.get_flow_sensitivity()
            dtotalflow_dun = dq_dun * self.model.dt.values()[0]

        return dtotalflow_dun

    def dparam(self):
        return None

class SubglottalWork(AbstractFunctional):
    """
    Returns the total work input into the fluid from the subglottal region (lungs).
    """
    def __init__(self, model, f, **kwargs):
        super(SubglottalWork, self).__init__(model, f, **kwargs)

        self.kwargs.setdefault('n_start', 0)

    def __call__(self):
        N_START = self.kwargs['n_start']
        N_STATE = self.f.get_num_states()

        ret = 0
        for ii in range(N_START, N_STATE-1):
            # Set form coefficients to represent the equation mapping state ii->ii+1
            fluid_info, fluid_props = self.model.set_iteration_fromfile(self.f, ii+1)

            ret += self.model.dt.values()[0]*fluid_info['flow_rate']*fluid_props['p_sub']

        return ret

    def du(self, n):
        ret = dfn.Function(self.model.vector_function_space).vector()

        N_START = self.kwargs['n_start']
        N_STATE = self.f.get_num_states()

        if n >= N_START and n < N_STATE-1:
            _, fluid_props = self.model.set_iteration_fromfile(self.f, n+1)
            _, dq_du = self.model.get_flow_sensitivity()

            ret += self.model.dt.values()[0] * fluid_props['p_sub'] * dq_du
        else:
            pass

        return ret

    def dparam(self):
        return None

class VocalEfficiency(AbstractFunctional):
    """
    Returns the total vocal efficiency.

    This is the ratio of the total work done by the fluid on the folds to the total input work on
    the fluid.
    """
    def __init__(self, model, f, **kwargs):
        super(VocalEfficiency, self).__init__(model, f, **kwargs)

        self.kwargs.setdefault('n_start', 0)

        self.funcs['FluidWork'] = FluidWork(model, f, **kwargs)
        self.funcs['SubglottalWork'] = SubglottalWork(model, f, **kwargs)

    def __call__(self):
        totalfluidwork = self.funcs['FluidWork']()
        totalinputwork = self.funcs['SubglottalWork']()

        res = totalfluidwork/totalinputwork

        self.cache.update({'totalfluidwork': totalfluidwork, 'totalinputwork': totalinputwork})
        return res

    def du(self, n):
        # TODO : Is there something slightly wrong with this one? Seems slightly wrong from
        # comparing with FD. The error is small but it is not propto step size?
        N_START = self.kwargs['n_start']

        tfluidwork = self.cache.get('totalfluidwork', None)
        tinputwork = self.cache.get('totalinputwork', None)

        dtotalfluidwork_dun = self.funcs['FluidWork'].du(n)
        dtotalinputwork_dun = self.funcs['SubglottalWork'].du(n)

        if n < N_START:
            return dfn.Function(self.model.vector_function_space).vector()
        else:
            return dtotalfluidwork_dun/tinputwork - tfluidwork/tinputwork**2*dtotalinputwork_dun

    def dparam(self):
        return None

class MFDR(AbstractFunctional):
    """
    Return the maximum flow declination rate.
    """
    def __init__(self, model, f, **kwargs):
        super(MFDR, self).__init__(model, f, **kwargs)

        self.kwargs.setdefault('n_start', 0)

    def __call__(self):
        flow_rate = []
        info = {}

        num_states = self.f.get_num_states()
        for ii in range(num_states-1):
            # Set form coefficients to represent the equation at state ii
            info, _ = self.model.set_iteration_fromfile(self.f, ii+1)

            flow_rate.append(info['flow_rate'])
        flow_rate = np.array(flow_rate)

        times = self.f.get_solution_times()[:-1]
        dflow_rate_dt = (flow_rate[1:]-flow_rate[:-1]) / (times[1:] - times[:-1])

        N_START = self.kwargs['n_start']
        idx_min = np.argmin(dflow_rate_dt[N_START:]) + N_START

        res = dflow_rate_dt[idx_min]

        self.cache.update({'idx_mfdr': idx_min})

        return res

    def du(self, n):
        res = None

        idx_mfdr = self.cache.get('idx_mfdr', None)

        if n == idx_mfdr or n == idx_mfdr+1:
            # First calculate flow rates at n and n+1
            # fluid_info, _ = model.set_iteration_fromfile(f, n+2)

            # q1 = fluid_info['flow_rate']
            dq1_du = self.model.get_flow_sensitivity()[1]
            t1 = self.f.get_time(n+1)

            # fluid_info, _ = model.set_iteration_fromfile(f, n+1)

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

    def dparam(self):
        return None

class WSSGlottalWidth(AbstractFunctional):
    """
    Returns the weighted sum of squared glottal widths.
    """
    def __init__(self, model, f, **kwargs):
        super(WSSGlottalWidth, self).__init__(model, f, **kwargs)

        # Set default values of kwargs if they were not passed
        N_STATE = self.f.get_num_states()
        self.kwargs.setdefault('weights', np.ones(N_STATE) / N_STATE)
        self.kwargs.setdefault('meas_indices', np.arange(N_STATE))
        self.kwargs.setdefault('meas_glottal_widths', np.zeros(N_STATE))

        assert kwargs['meas_indices'].size == kwargs['meas_glottal_widths'].size

    def __call__(self):
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
            fluid_props = self.f.get_fluid_properties(0)
            gw_modl = 2 * (fluid_props['y_midline'] - cur_surface[idx_surface, 1])

            wss += weight * (gw_modl - gw_meas)**2

        return wss

    def du(self, n):
        dwss_du = dfn.Function(self.model.vector_function_space).vector()

        weights = self.kwargs['weights']
        meas_indices = self.kwargs['meas_indices']
        meas_glottal_widths = self.kwargs['meas_glottal_widths']

        # The sensitivity is only non-zero if n corresponds to a measurement index
        if n in set(meas_indices):
            weight = weights[n]
            gw_meas = meas_glottal_widths[n]

            u, v, a = self.f.get_state(n, self.model.vector_function_space)
            self.model.set_initial_state(u, v, a)

            # Find the surface vertex corresponding to where the glottal width is measured
            # This is numbered according to the 'local' numbering scheme of the surface vertices
            # (from downstream to upstream)
            cur_surface = self.model.get_surface_state()[0]
            idx_surface = np.argmax(cur_surface[:, 1])

            # Find the maximum y coordinate on the surface
            fluid_props = self.f.get_fluid_properties(0)
            gw_modl = 2 * (fluid_props['y_midline'] - cur_surface[idx_surface, 1])
            dgw_modl_du_width = -2

            # Find the vertex number according to the mesh vertex numbering scheme
            idx_body = self.model.surface_vertices[idx_surface]

            # Finally convert it to the u-DOF number that actually influences glottal width
            dof_width = self.model.vert_to_vdof[idx_body, 1]

            # wss = weight * (gw_modl - gw_meas)**2
            dwss_du[dof_width] = 2*weight*(gw_modl - gw_meas)*dgw_modl_du_width
        else:
            # In this case the derivative is simply 0 so the default value is right
            pass

        return dwss_du

    def dparam(self):
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

# TODO: Previously had a lagrangian regularization term here but accidentally
# deleted that code... need to make it again.
