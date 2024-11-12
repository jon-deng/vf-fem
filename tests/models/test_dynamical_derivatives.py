"""
Test correctness of dynamical model derivatives

Correctness is tested by comparing finite differences against
implemented derivatives along specified perturbations.
"""

from os import path

import pytest
import numpy as np
import dolfin as dfn

from blockarray import linalg as bla, blockvec as bv
from femvf.models.dynamical import (
    solid as dynsl,
    fluid as dynfl,
    coupled as dynco,
    base as dynbase,
)
from femvf import load

from petsc4py import PETSc

# pylint: disable=redefined-outer-name

# warnings.filterwarnings('error', 'RuntimeWarning')
# np.seterr(invalid='raise')

Model = dynbase.BaseDynamicalModel
LinModel = dynbase.BaseLinearizedDynamicalModel
BVec = bv.BlockVector


def _set_dirichlet_bvec(dirichlet_bc, bvec: bv.BlockVector):
    for label in ['u', 'v']:
        if label in bvec:
            subvec = bvec.sub[label]
            if isinstance(subvec, PETSc.Vec):
                subvec = dfn.PETScVector(subvec)
            dirichlet_bc.apply(subvec)
    return bvec


def split_model_components(model):
    """
    Return model split into fluid/solid/coupled parts
    """
    # Determine whether the model has fluid/solid components
    if isinstance(model, dynco.BaseDynamicalFSIModel):
        model_solid = model.solid
        model_fluids = model.fluids
        model_coupl = model
    elif isinstance(model, dynsl.Model):
        model_solid = model
        model_fluids = None
        model_coupl = None
    elif isinstance(model, dynfl.BaseDynamicalModel):
        model_solid = None
        model_fluids = model
        model_coupl = None
    return model_solid, model_fluids, model_coupl


def set_linearization(model: dynbase.BaseDynamicalModel, state, statet, control, prop):
    """
    Set the model linearization point
    """
    model.set_state(state)
    model.set_statet(statet)
    model.set_control(control)
    model.set_prop(prop)


def set_and_assemble(x, set_x, assem):
    set_x(x)
    # A copy is needed because the assembler functions often return the same matrix/vector object
    # As a result, not creating copies will keep overwriting 'previous' instances of an assembled
    # tensor
    return assem().copy()


def _test_taylor(x0, dx, res, jac):
    """
    Test that the Taylor convergence order is 2
    """
    alphas = 2 ** np.arange(4)[::-1]  # start with the largest step and move to original
    res_ns = [res(x0 + float(alpha) * dx) for alpha in alphas]
    res_0 = res(x0)

    dres_exacts = [res_n - res_0 for res_n in res_ns]
    dres_linear = bla.mult_mat_vec(jac(x0), dx)

    errs = [
        (dres_exact - float(alpha) * dres_linear).norm()
        for dres_exact, alpha in zip(dres_exacts, alphas)
    ]
    magnitudes = [
        1 / 2 * (dres_exact + float(alpha) * dres_linear).norm()
        for dres_exact, alpha in zip(dres_exacts, alphas)
    ]
    with np.errstate(invalid='ignore'):
        conv_rates = [
            np.log(err_0 / err_1) / np.log(alpha_0 / alpha_1)
            for err_0, err_1, alpha_0, alpha_1 in zip(
                errs[:-1], errs[1:], alphas[:-1], alphas[1:]
            )
        ]
        rel_errs = np.array(errs) / np.array(magnitudes) * 100

    print("")
    print(
        f"||dres_linear||, ||dres_exact|| = {dres_linear.norm()}, {dres_exacts[-1].norm()}"
    )
    print("Relative errors: ", rel_errs)
    print("Convergence rates: ", np.array(conv_rates))


class _TestDerivative:
    """
    Test correctness of model derivatives
    """

    def test_assem_dres_dstate(
        self,
        model: Model,
        state: BVec,
        statet: BVec,
        control: BVec,
        prop: BVec,
        dstate: BVec,
    ):
        """
        Test `model.assem_dres_dstate`
        """
        set_linearization(model, state, statet, control, prop)
        res = lambda state: set_and_assemble(state, model.set_state, model.assem_res)
        jac = lambda state: set_and_assemble(
            state, model.set_state, model.assem_dres_dstate
        )

        _test_taylor(state, dstate, res, jac)

    def test_assem_dres_dstatet(
        self,
        model: Model,
        state: BVec,
        statet: BVec,
        control: BVec,
        prop: BVec,
        dstatet: BVec,
    ):
        """
        Test `model.assem_dres_dstatet`
        """
        set_linearization(model, state, statet, control, prop)
        res = lambda state: set_and_assemble(state, model.set_statet, model.assem_res)
        jac = lambda state: set_and_assemble(
            state, model.set_statet, model.assem_dres_dstatet
        )

        _test_taylor(statet, dstatet, res, jac)

    def test_assem_dres_dcontrol(
        self,
        model: Model,
        state: BVec,
        statet: BVec,
        control: BVec,
        prop: BVec,
        dcontrol: BVec,
    ):
        """
        Test `model.assem_dres_dcontrol`
        """
        set_linearization(model, state, statet, control, prop)
        res = lambda state: set_and_assemble(state, model.set_control, model.assem_res)
        jac = lambda state: set_and_assemble(
            state, model.set_control, model.assem_dres_dcontrol
        )

        _test_taylor(control, dcontrol, res, jac)

    def test_assem_dres_dprop(
        self,
        model: Model,
        state: BVec,
        statet: BVec,
        control: BVec,
        prop: BVec,
        dprop: BVec,
    ):
        """
        Test `model.assem_dres_dprop`
        """
        set_linearization(model, state, statet, control, prop)
        res = lambda state: set_and_assemble(state, model.set_prop, model.assem_res)
        jac = lambda state: set_and_assemble(
            state, model.set_prop, model.assem_dres_dprop
        )

        _test_taylor(prop, dprop, res, jac)

    def test_dres_dstate_vs_dres_state(
        self,
        model: Model,
        model_linear: LinModel,
        state: BVec,
        statet: BVec,
        control: BVec,
        prop: BVec,
        dstate: BVec,
    ):
        """
        Test consistency between `model` and `model_linear_state`

        `model` represents a residual F(...)
        `model_linear_state` represents the linearized residual (dF/dstate * del_state)(...)
        This test checks that:
            dF/dstate(...) * del_state    (computed from `model`)
            is equal to
            (dF/dstate * del_state)(...)  (computed from `model_linear_state`)
        """
        set_linearization(model, state, statet, control, prop)
        set_linearization(model_linear, state, statet, control, prop)

        # compute the linearized residual from `model`
        dres_dstate = set_and_assemble(state, model.set_state, model.assem_dres_dstate)
        dres_state_a = bla.mult_mat_vec(dres_dstate, dstate)

        model_linear.set_dstate(dstate)
        _zero_del_xt = model_linear.dstatet.copy()
        _zero_del_xt[:] = 0
        model_linear.set_dstatet(_zero_del_xt)

        dres_state_b = set_and_assemble(
            state, model_linear.set_state, model_linear.assem_res
        )
        err = dres_state_a - dres_state_b

        for vec, name in zip(
            [dres_state_a, dres_state_b, err],
            ["from model", "from linear_state_model", "error"],
        ):
            print(f"\n{name}")
            for key, subvec in vec.sub_items():
                print(key, subvec.norm())

    def test_dres_dstatet_vs_dres_statet(
        self,
        model: Model,
        model_linear: LinModel,
        state: BVec,
        statet: BVec,
        control: BVec,
        prop: BVec,
        dstatet: BVec,
    ):
        """
        Test consistency between `model` and `model_linear_state`

        `model` represents a residual F(...)
        `model_linear_state` represents the linearized residual (dF/dstate * del_state)(...)
        This test checks that:
            dF/dstate(...) * del_state    (computed from `model`)
            is equal to
            (dF/dstate * del_state)(...)  (computed from `model_linear_state`)
        """
        set_linearization(model, state, statet, control, prop)
        set_linearization(model_linear, state, statet, control, prop)

        # compute the linearized residual from `model`
        dres_dstatet = set_and_assemble(
            statet, model.set_state, model.assem_dres_dstatet
        )
        dres_statet_a = bla.mult_mat_vec(dres_dstatet, dstatet)

        model_linear.set_dstatet(dstatet)
        _zero_del_x = model_linear.dstate.copy()
        _zero_del_x[:] = 0
        model_linear.set_dstate(_zero_del_x)

        dres_statet_b = set_and_assemble(
            statet, model_linear.set_state, model_linear.assem_res
        )
        err = dres_statet_a - dres_statet_b

        for vec, name in zip(
            [dres_statet_a, dres_statet_b, err],
            ["from model", "from linear_state_model", "error"],
        ):
            print(f"\n{name}")
            for key, subvec in vec.sub_items():
                print(key, subvec.norm())


class GenericFixtureMixin:
    """
    A set of generic fixtures for `_TestDerivative`
    """

    @pytest.fixture(params=[(dynsl.KelvinVoigt, dynsl.LinearizedKelvinVoigt, {})])
    def SolidModelPair(self, request):
        """
        Return a tuple of non-linear and linearized models, and kwargs
        """
        return request.param

    @pytest.fixture(
        params=[
            # (dynfl.BernoulliSmoothMinSep, dynfl.LinearizedBernoulliSmoothMinSep, {}),
            (
                dynfl.BernoulliFixedSep,
                dynfl.LinearizedBernoulliFixedSep,
                {'separation_vertex_label': 'separation-inf'},
            ),
            # (dynfl.BernoulliFlowFixedSep, dynfl.LinearizedBernoulliFlowFixedSep, {'separation_vertex_label': 'separation-inf'}),
        ]
    )
    def FluidModelPair(self, request):
        """
        Return a tuple of non-linear and linearized models, and kwargs
        """
        return request.param

    @pytest.fixture(params=['M5_BC--GA0.00--DZ0.00.msh'])
    def mesh_path(self, request):
        mesh_name = request.param
        mesh_path = path.join('../../meshes', mesh_name)

        return mesh_path

    @pytest.fixture()
    def model(self, mesh_path, SolidModelPair, FluidModelPair):
        """
        Return a dynamical system model residual
        """
        solid_mesh = mesh_path
        fluid_mesh = None

        SolidType, LinSolidType, solid_kwargs = SolidModelPair
        FluidType, LinFluidType, fluid_kwargs = FluidModelPair
        if SolidType is not None and FluidType is not None:
            model = load.load_dynamical_fsi_model(
                solid_mesh,
                fluid_mesh,
                SolidType,
                FluidType,
                fsi_facet_labels=('pressure',),
                fixed_facet_labels=('fixed',),
                **solid_kwargs,
                **fluid_kwargs,
            )
        elif SolidType is not None and FluidType is None:
            model = load.load_solid_model(
                solid_mesh,
                SolidType,
                pressure_facet_labels=('pressure',),
                fixed_facet_labels=('fixed',),
                **solid_kwargs,
            )
        elif SolidType is None and FluidType is not None:
            model = load.load_fluid_model(fluid_mesh, FluidType, **fluid_kwargs)
        else:
            assert False

        return model

    @pytest.fixture()
    def model_linear(self, mesh_path, SolidModelPair, FluidModelPair):
        """
        Return a linearized dynamical system model residual
        """
        solid_mesh = mesh_path
        fluid_mesh = None

        SolidType, LinSolidType, solid_kwargs = SolidModelPair
        FluidType, LinFluidType, fluid_kwargs = FluidModelPair

        model_coupled_linear = load.load_dynamical_fsi_model(
            solid_mesh,
            fluid_mesh,
            LinSolidType,
            LinFluidType,
            fsi_facet_labels=('pressure',),
            fixed_facet_labels=('fixed',),
            **solid_kwargs,
            **fluid_kwargs,
        )

        return model_coupled_linear

    @pytest.fixture()
    def state(self, model: Model):
        model_solid, model_fluids, model_coupl = split_model_components(model)

        ## Model state
        state0 = model.state.copy()

        if model_solid is not None:
            # Make the initial displacement a pure shear motion
            dim = model_solid.residual.mesh().topology().dim()
            xref = model_solid.XREF.copy()
            xx = xref[:-1:dim]
            yy = xref[1::dim]
            _u = np.zeros(state0['u'].shape)
            _u[:-1:dim] = 0.01 * (yy / yy.max())
            _u[1::dim] = 0.0 * yy
            state0['u'] = _u

        if model_fluids is not None:
            for n in range(len(model_fluids)):
                state0[f'fluid{n}.q'] = 1
                state0[f'fluid{n}.p'] = 1e4
        return state0

    @pytest.fixture()
    def statet(self, model: Model):
        statet0 = model.state.copy()

        return statet0

    @pytest.fixture()
    def prop(self, model: Model):
        model_solid, model_fluids, model_coupl = split_model_components(model)

        props0 = model.prop.copy()
        if model_solid is not None:
            props0['emod'] = 5e3 * 10
            props0['rho'] = 1.0

        if model_coupl is not None:
            dim = model_solid.residual.mesh().topology().dim()
            ymax = np.max(model_coupl.solid.XREF[1::dim])
            ygap = 0.01  # gap between VF and symmetry plane
            ymid = ymax + ygap
            ycontact = ymid - 0.1 * ygap
            props0['ycontact'] = ycontact

            model_coupl.ymid = ymid

        if model_fluids is not None:
            prop_values = {'zeta_sep': 1e-4, 'zeta_min': 1e-4, 'rho_air': 1.2e-3}
            for n in range(len(model_fluids)):
                for key, value in prop_values.items():
                    _key = f'fluid{n}.{key}'
                    if _key in props0:
                        props0[_key] = value
        return props0

    @pytest.fixture()
    def control(self, model: Model):
        model_solid, model_fluids, model_coupl = split_model_components(model)

        control0 = model.control.copy()
        control0[:] = 1.0

        if model_fluids is not None:
            control_values = {'qsub': 100, 'psub': 800 * 10, 'psup': 0}
            for n in range(len(model_fluids)):
                for key, value in control_values.items():
                    _key = f'fluid{n}.{key}'
                    if _key in control0:
                        control0[_key] = value
        return control0

    @pytest.fixture()
    def dstate(self, model: Model):
        """Return a state perturbation"""

        model_solid, model_fluids, model_coupl = split_model_components(model)

        dstate = model.state.copy()

        if model_solid is not None:
            dxu = model_solid.state['u'].copy()
            dxu[:] = 1e-3 * np.arange(dxu[:].size)
            dxu[:] = 1e-8
            # dxu[:] = 0
            # model_solid.forms['bc.dirichlet'].apply(dxu)
            dstate['u'] = dxu

            dxv = model_solid.state['v'].copy()
            dxv[:] = 1e-8
            # model_solid.forms['bc.dirichlet'].apply(dxv)
            dstate['v'] = dxv

            for bc in model_solid.residual.dirichlet_bcs:
                _set_dirichlet_bvec(bc, dstate)

        if model_fluids is not None:
            values = {'q': 1e-3, 'p': 1e-3}
            for n in range(len(model_fluids)):
                for key, value in values.items():
                    _key = f'fluid{n}.{key}'
                    if _key in dstate:
                        dstate[_key] = value

        return dstate

    @pytest.fixture()
    def dstatet(self, model: Model):
        """Return a state derivative perturbation"""

        model_solid, model_fluids, model_coupl = split_model_components(model)

        dstatet = model.state.copy()

        dstatet[:] = 1e-6
        if model_solid is not None:
            for bc in model_solid.residual.dirichlet_bcs:
                _set_dirichlet_bvec(bc, dstatet)

        return dstatet

    @pytest.fixture()
    def dcontrol(self, model: Model):
        """Return a control perturbation"""

        dcontrol = model.control.copy()
        dcontrol[:] = 1e0

        return dcontrol

    @pytest.fixture()
    def dprop(self, model: Model):
        """Return a properties perturbation"""

        model_solid, model_fluids, model_coupl = split_model_components(model)

        dprop = model.prop.copy()
        dprop[:] = 0

        if model_solid is not None:
            dprop['emod'] = 1.0

            if 'umesh' in dprop:
                # Use a uniaxial y stretching motion
                fspace = model_solid.residual.form['coeff.state.u1'].function_space()
                VDOF_TO_VERT = dfn.dof_to_vertex_map(fspace)
                coords = np.array(model_solid.XREF[:]).copy().reshape(-1, 2)
                umesh = coords.copy()
                umesh[:, 0] = 0
                umesh[:, 1] = 1e-5 * coords[:, 1] / coords[:, 1].max()
                dprop['umesh'] = umesh.reshape(-1)[VDOF_TO_VERT]
                # dprop['umesh'] = 0
        return dprop


class TestShapeModel(_TestDerivative, GenericFixtureMixin):

    @pytest.fixture(
        params=[(dynsl.KelvinVoigtWShape, dynsl.LinearizedKelvinVoigtWShape, {})]
    )
    def SolidModelPair(self, request):
        """
        Return a tuple of non-linear and linearized models, and kwargs
        """
        return request.param

    @pytest.fixture(
        params=[
            (None, None, {}),
            (
                dynfl.BernoulliFixedSep,
                dynfl.LinearizedBernoulliFixedSep,
                {'separation_vertex_label': 'separation-inf'},
            ),
        ]
    )
    def FluidModelPair(self, request):
        """
        Return a tuple of non-linear and linearized models, and kwargs
        """
        return request.param

    @pytest.fixture()
    def control(self, model: Model):
        model_solid, model_fluids, model_coupl = split_model_components(model)

        control0 = model.control.copy()
        control0[:] = 0.0

        if model_fluids is not None:
            control_values = {'qsub': 100, 'psub': 1e-8 * 10, 'psup': 0 * 10}
            for n in range(len(model_fluids)):
                for key, value in control_values.items():
                    _key = f'fluid{n}.{key}'
                    if _key in control0:
                        control0[_key] = value
        return control0

    @pytest.fixture()
    def prop(self, model: Model):
        model_solid, model_fluids, model_coupl = split_model_components(model)

        props0 = model.prop.copy()
        if model_solid is not None:
            props0['emod'] = 5e3 * 10
            props0['rho'] = 1.0

        if model_coupl is not None:
            dim = model_solid.residual.mesh().topology().dim()
            ymax = np.max(model_coupl.solid.XREF[1::dim])
            ygap = 0.01  # gap between VF and symmetry plane
            ymid = ymax + ygap
            ycontact = ymid - 0.1 * ygap
            props0['ycontact'] = ycontact

            model_coupl.ymid = ymid

        if model_fluids is not None:
            prop_values = {'zeta_sep': 1e-4, 'zeta_min': 1e-4, 'rho_air': 1.2e-3}
            for n in range(len(model_fluids)):
                for key, value in prop_values.items():
                    _key = f'fluid{n}.{key}'
                    if _key in props0:
                        props0[_key] = value
        return props0

    @pytest.fixture()
    def dprop(self, model: Model):
        """Return a properties perturbation"""

        model_solid, model_fluids, model_coupl = split_model_components(model)

        dprop = model.prop.copy()
        dprop[:] = 0

        if model_solid is not None:
            # Test mesh motion along a uniaxial y-direction stretching motion
            fspace = model_solid.residual.form['coeff.state.u1'].function_space()
            VDOF_TO_VERT = dfn.dof_to_vertex_map(fspace)
            coords = np.array(model_solid.XREF[:]).copy().reshape(-1, 2)
            umesh = coords.copy()
            umesh[:, 0] = 0
            umesh[:, 1] = 1e-5 * coords[:, 1] / coords[:, 1].max()
            dprop['umesh'] = umesh.reshape(-1)[VDOF_TO_VERT]
            # dprop['umesh'] = 0
        return dprop

    @pytest.mark.skip()
    def test_assem_dres_dstate(
        self,
        model: Model,
        state: BVec,
        statet: BVec,
        control: BVec,
        prop: BVec,
        dstate: BVec,
    ):
        """
        Test `model.assem_dres_dstate`
        """
        super().test_assem_dres_dstate(model, state, statet, control, prop, dstate)

    @pytest.mark.skip()
    def test_assem_dres_dstatet(
        self,
        model: Model,
        state: BVec,
        statet: BVec,
        control: BVec,
        prop: BVec,
        dstatet: BVec,
    ):
        super().test_assem_dres_dstatet(model, state, statet, control, prop, dstatet)

    @pytest.mark.skip()
    def test_assem_dres_dcontrol(
        self,
        model: Model,
        state: BVec,
        statet: BVec,
        control: BVec,
        prop: BVec,
        dcontrol: BVec,
    ):
        """
        Test `model.assem_dres_dcontrol`
        """
        super().test_assem_dres_dcontrol(model, state, statet, control, prop, dcontrol)

    def test_assem_dres_dprop(
        self,
        model: Model,
        state: BVec,
        statet: BVec,
        control: BVec,
        prop: BVec,
        dprop: BVec,
    ):
        super().test_assem_dres_dprop(model, state, statet, control, prop, dprop)

    @pytest.mark.skip()
    def test_dres_dstate_vs_dres_state(
        self,
        model: Model,
        model_linear: LinModel,
        state: BVec,
        statet: BVec,
        control: BVec,
        prop: BVec,
        dstate: BVec,
    ):
        super().test_dres_dstate_vs_dres_state(
            model, model_linear, state, statet, control, prop, dstate
        )

    @pytest.mark.skip()
    def test_dres_dstatet_vs_dres_statet(
        self,
        model: Model,
        model_linear: LinModel,
        state: BVec,
        statet: BVec,
        control: BVec,
        prop: BVec,
        dstatet: BVec,
    ):
        super().test_dres_dstatet_vs_dres_statet(
            model, model_linear, state, statet, control, prop, dstatet
        )


class TestNoShapeModel(_TestDerivative, GenericFixtureMixin):
    pass
