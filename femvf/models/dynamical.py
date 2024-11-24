"""
Nonlinear dynamical systems models

These models can be written in the form:
F(x, d/dt x; g, ...)
where x is the state vector, d/dt x the time derivative, and g are any additional parameters


Base Dynamical Model
--------------------

The class `BaseDynamicalModel` mainly specifies the attributes/methods that
specific dynamical system classes must contain but does not contain any
implementation details.


Fenics Dynamical Model
-----------------------

The nonlinear dynamical systems here are defined in "FEniCS" and then augmented
a bit manually. The basic dynamical system is represented by a nonlinear
residual with the block form:
    - `F(x, xt, g) = [Fu(x, xt, g, p), Fv(x, xt)]`
    - `x = [u, v]`
    - `xt = [ut, vt]`
where `F` denotes the nonlinear residual and `x` and `xt` are the 'state' and
'state time derivative', respectively. The variables `u` and `v` stand for
'position' and 'velocity'. The final two parameters `g` and `p` denote some
arbitrary collections of control and model parameters, respectively.

The two blocks of `F` are defined by:
    - `Fu(x, xt, g)` is defined symbolically in "FEniCS" with with the `ufl`
    form language.
    - `Fv(x, xt, g)` is defined by `v-ut` through the derivative trick used to
    convert second order ODEs to first order.

The classes represent either the residual `F` or its linearization `F`, and each
class gives methods to evaluate the residual and its derivatives w.r.t the
parameters `x`, `xt`, `g`, `p`. This is done in the classes
`SolidDynamicalSystem` and `LinearizedSolidDynamicalSystem` below.

JAX Dynamical Model
-------------------

The nonlinear dynamical systems here are defined in jax/numpy and augmented a
bit manually. The basic dynamical system residual has a block form
F(x, xt, g) = [Fq(x, xt, g), Fp(x, xt)]
x = [q, p]
xt = [qt, pt]
and where q and p stand for flow and pressure for a 1D fluid model

FSI Dynamical model
--------------------

"""

from typing import TypeVar, Union, Callable
from numpy.typing import NDArray

import numpy as np
import dolfin as dfn
from petsc4py import PETSc as PETSc

from blockarray import subops
from blockarray import blockvec as bv, blockmat as bm, linalg as bla

from femvf.equations import form
from femvf.residuals import solid, fluid


### Generic dynamical model
T = TypeVar('T')
Vector = Union[subops.DfnVector, subops.GenericSubarray, subops.PETScVector]
Matrix = Union[subops.DfnMatrix, subops.GenericSubarray, subops.PETScMatrix]

BlockVec = bv.BlockVector[Vector]
BlockMat = bm.BlockMatrix[Matrix]


class BaseDynamicalModel:

    def set_state(self, state: BlockVec):
        raise NotImplementedError()

    def set_statet(self, statet: BlockVec):
        raise NotImplementedError()

    def set_control(self, control: BlockVec):
        raise NotImplementedError()

    def set_prop(self, prop: BlockVec):
        raise NotImplementedError()

    def assem_res(self) -> BlockVec:
        raise NotImplementedError()

    def assem_dres_dstate(self) -> BlockVec:
        raise NotImplementedError()

    def assem_dres_dstatet(self) -> BlockMat:
        raise NotImplementedError()

    def assem_dres_dcontrol(self) -> BlockMat:
        raise NotImplementedError()

    def assem_dres_dprop(self) -> BlockMat:
        raise NotImplementedError()


class BaseLinearizedDynamicalModel(BaseDynamicalModel):

    def set_dstate(self, dstate: BlockVec):
        raise NotImplementedError()

    def set_dstatet(self, dstatet: BlockVec):
        raise NotImplementedError()

    def set_dcontrol(self, dcontrol: BlockVec):
        raise NotImplementedError()


### FENICS MODELS

from .assemblyutils import CachedFormAssembler
from .transient import (
    properties_bvec_from_forms,
    depack_form_coefficient_function,
)

# pylint: disable=abstract-method

def cast_output_bmat_to_petsc(func):
    def wrapped_func(*args, **kwargs):
        mat = func(*args, **kwargs)
        return bm.convert_subtype_to_petsc(mat)

    return wrapped_func


def cast_output_bvec_to_petsc(func):
    def wrapped_func(*args, **kwargs):
        vec = func(*args, **kwargs)
        return bv.convert_subtype_to_petsc(vec)

    return wrapped_func


class BaseDynamicalFenicsModel:

    def __init__(self, residual: solid.FenicsResidual):

        self._residual = residual

        # assert isinstance(fsi_facet_labels, (list, tuple))
        # assert isinstance(fixed_facet_labels, (list, tuple))
        # self.residual_form_name = residual_form_name
        # self._forms = self.form_definitions(mesh, mesh_funcs, mesh_entities_label_to_value, fsi_facet_labels, fixed_facet_labels)
        # bilinear_forms = gen_residual_bilinear_forms(self.residual.form)
        hopf_forms = form.gen_jac_state_forms(self.residual.form)
        prop_jac_forms = form.gen_jac_property_forms(self.residual.form)
        forms = {**hopf_forms, **prop_jac_forms}

        self.u = self.residual.form['coeff.state.u1']
        self.v = self.residual.form['coeff.state.v1']
        self.state = bv.BlockVector(
            (self.u.vector(), self.v.vector()), labels=[('u', 'v')]
        )
        self.state = bv.convert_subtype_to_petsc(self.state)

        self.ut = dfn.Function(self.residual.form['coeff.state.u1'].function_space())
        self.vt = self.residual.form['coeff.state.a1']
        self.statet = bv.BlockVector(
            (self.ut.vector(), self.vt.vector()), labels=[('u', 'v')]
        )
        self.statet = bv.convert_subtype_to_petsc(self.statet)

        self.control = bv.BlockVector(
            (self.residual.form['coeff.fsi.p1'].vector(),), labels=[('p',)]
        )
        self.control = bv.convert_subtype_to_petsc(self.control)

        self.prop = properties_bvec_from_forms(self.residual.form)
        self.prop = bv.convert_subtype_to_petsc(self.prop)
        self.set_prop(self.prop)

        self.cached_form_assemblers = {
            key: CachedFormAssembler(form)
            for key, form in forms.items()
            if ('form.' in key and form is not None)
        }

        self.cached_form_assemblers['form.un.res'] = CachedFormAssembler(
            self.residual.form.ufl_forms
        )

    @property
    def residual(self) -> solid.FenicsResidual:
        return self._residual

    def set_state(self, state):
        self.state[:] = state

    def set_statet(self, statet):
        self.statet[:] = statet

    def set_control(self, control):
        self.control[:] = control

    def set_prop(self, prop):
        for key in prop.labels[0]:
            # TODO: Check types to make sure the input property is compatible with the solid type
            coefficient = depack_form_coefficient_function(
                self.residual.form['coeff.prop.' + key]
            )

            # If the property is a field variable, values have to be assigned to every spot in
            # the vector
            if isinstance(coefficient, dfn.function.constant.Constant):
                coefficient.assign(dfn.Constant(np.squeeze(prop[key])))
            else:
                coefficient.vector()[:] = prop[key]

        # If a shape parameter exists, it needs special handling to update the mesh coordinates
        if 'coeff.prop.umesh' in self.residual.form:
            u_mesh_coeff = self.residual.form['coeff.prop.umesh']

            mesh = self.residual.mesh()
            fspace = self.residual.form['coeff.state.u1'].function_space()
            ref_mesh_coord = self.residual.ref_mesh_coords
            VERT_TO_VDOF = dfn.vertex_to_dof_map(fspace)
            dmesh_coords = np.array(u_mesh_coeff.vector()[VERT_TO_VDOF]).reshape(
                ref_mesh_coord.shape
            )
            mesh_coord = ref_mesh_coord + dmesh_coords
            mesh.coordinates()[:] = mesh_coord

    # Convenience methods
    @property
    def XREF(self) -> dfn.Function:
        xref = self.state.sub[0].copy()
        function_space = self.residual.form['coeff.state.u1'].function_space()
        n_subspace = function_space.num_sub_spaces()

        xref[:] = (
            function_space.tabulate_dof_coordinates()[::n_subspace, :]
            .reshape(-1)
            .copy()
        )
        return xref


class FenicsModel(BaseDynamicalFenicsModel, BaseDynamicalModel):

    @cast_output_bvec_to_petsc
    def assem_res(self):
        resu = self.cached_form_assemblers['form.un.res'].assemble()
        resv = self.v.vector() - self.ut.vector()
        return bv.BlockVector([resu, resv], labels=[['u', 'v']])

    @cast_output_bmat_to_petsc
    def assem_dres_dstate(self):
        dresu_du = self.cached_form_assemblers['form.bi.dres_du1'].assemble()
        dresu_dv = self.cached_form_assemblers['form.bi.dres_dv1'].assemble()

        n = self.v.vector().size()
        dresv_du = dfn.PETScMatrix(subops.zero_mat(n, n))
        dresv_dv = dfn.PETScMatrix(subops.ident_mat(n))

        mats = [[dresu_du, dresu_dv], [dresv_du, dresv_dv]]
        return bm.BlockMatrix(mats, labels=(['u', 'v'], ['u', 'v']))

    @cast_output_bmat_to_petsc
    def assem_dres_dstatet(self):
        n = self.u.vector().size()
        dresu_dut = dfn.PETScMatrix(subops.diag_mat(n, diag=0))
        dresu_dvt = self.cached_form_assemblers['form.bi.dres_da1'].assemble()

        dresv_dut = dfn.PETScMatrix(-1 * subops.ident_mat(n))
        dresv_dvt = dfn.PETScMatrix(subops.diag_mat(n, diag=0))

        mats = [[dresu_dut, dresu_dvt], [dresv_dut, dresv_dvt]]
        return bm.BlockMatrix(mats, labels=(['u', 'v'], ['u', 'v']))

    @cast_output_bmat_to_petsc
    def assem_dres_dcontrol(self):
        n = self.u.vector().size()
        dresu_dcontrol = self.cached_form_assemblers['form.bi.dres_dp1'].assemble()

        dresv_dcontrol = dfn.PETScMatrix(
            subops.zero_mat(self.state['v'].size, self.control['p'].size)
        )

        mats = [[dresu_dcontrol], [dresv_dcontrol]]
        return bm.BlockMatrix(mats, labels=self.state.labels + self.control.labels)

    @cast_output_bmat_to_petsc
    def assem_dres_dprop(self):
        nu, nv = self.state['u'].size, self.state['v'].size
        mats = [
            [subops.zero_mat(nu, prop_subvec.size) for prop_subvec in self.prop],
            [subops.zero_mat(nv, prop_subvec.size) for prop_subvec in self.prop],
        ]

        j_emod = self.prop.labels[0].index('emod')
        mats[0][j_emod] = self.cached_form_assemblers['form.bi.dres_demod'].assemble()

        if 'umesh' in self.prop:
            j_shape = self.prop.labels[0].index('umesh')
            mats[0][j_shape] = self.cached_form_assemblers[
                'form.bi.dres_dumesh'
            ].assemble()

        return bm.BlockMatrix(mats, labels=(self.state.labels[0], self.prop.labels[0]))


class LinearizedFenicsModel(BaseDynamicalFenicsModel, BaseLinearizedDynamicalModel):

    def __init__(self, residual: solid.FenicsResidual):

        new_form = form.modify_unary_linearized_forms(residual.form)
        new_residual = solid.FenicsResidual(
            new_form,
            residual.mesh(),
            residual._mesh_functions,
            residual._mesh_subdomains
        )
        super().__init__(new_residual)

        self.du = self.residual.form['coeff.dstate.u1']
        self.dv = self.residual.form['coeff.dstate.v1']
        self.dstate = bv.BlockVector(
            (self.du.vector(), self.dv.vector()), labels=[('u', 'v')]
        )
        self.dstate = bv.convert_subtype_to_petsc(self.dstate)

        self.dut = dfn.Function(self.residual.form['coeff.dstate.u1'].function_space())
        self.dvt = self.residual.form['coeff.dstate.a1']
        self.dstatet = bv.BlockVector(
            (self.dut.vector(), self.dvt.vector()), labels=[('u', 'v')]
        )
        self.dstatet = bv.convert_subtype_to_petsc(self.dstatet)

        # self.p = self.forms['coeff.dfsi.p1']
        self.dcontrol = bv.BlockVector(
            (self.residual.form['coeff.dfsi.p1'].vector(),), labels=[('p',)]
        )
        self.dcontrol = bv.convert_subtype_to_petsc(self.dcontrol)

    def set_dstate(self, dstate):
        self.dstate[:] = dstate

    def set_dstatet(self, dstatet):
        self.dstatet[:] = dstatet

    def set_dcontrol(self, dcontrol):
        self.dcontrol[:] = dcontrol

    @cast_output_bvec_to_petsc
    def assem_res(self):
        resu = self.cached_form_assemblers['form.un.res'].assemble()
        resv = self.dv.vector() - self.dut.vector()
        return bv.BlockVector([resu, resv], labels=[['u', 'v']])

    @cast_output_bmat_to_petsc
    def assem_dres_dstate(self):
        dresu_du = self.cached_form_assemblers['form.bi.dres_du1'].assemble()
        dresu_dv = self.cached_form_assemblers['form.bi.dres_dv1'].assemble()

        n = self.u.vector().size()
        dresv_du = dfn.PETScMatrix(subops.zero_mat(n, n))
        dresv_dv = dfn.PETScMatrix(subops.zero_mat(n, n))

        mats = [[dresu_du, dresu_dv], [dresv_du, dresv_dv]]
        return bm.BlockMatrix(mats, labels=(['u', 'v'], ['u', 'v']))

    @cast_output_bmat_to_petsc
    def assem_dres_dstatet(self):
        n = self.u.vector().size()
        dresu_dut = dfn.PETScMatrix(subops.zero_mat(n, n))
        dresu_dvt = self.cached_form_assemblers['form.bi.dres_da1'].assemble()

        dresv_dut = dfn.PETScMatrix(subops.zero_mat(n, n))
        dresv_dvt = dfn.PETScMatrix(subops.zero_mat(n, n))

        mats = [[dresu_dut, dresu_dvt], [dresv_dut, dresv_dvt]]
        return bm.BlockMatrix(mats, labels=(['u', 'v'], ['u', 'v']))

    @cast_output_bmat_to_petsc
    def assem_dres_dcontrol(self):
        n = self.u.vector().size()
        m = self.control['p'].size
        dresu_dg = self.cached_form_assemblers['form.bi.dres_dp1'].assemble()

        dresv_dg = dfn.PETScMatrix(subops.zero_mat(n, m))

        mats = [[dresu_dg], [dresv_dg]]
        return bm.BlockMatrix(mats, labels=(['u', 'v'], ['g']))

    @cast_output_bmat_to_petsc
    def assem_dres_dprop(self):
        nu, nv = self.state['u'].size, self.state['v'].size
        mats = [
            [subops.zero_mat(nu, subvec.size) for subvec in self.prop],
            [subops.zero_mat(nv, subvec.size) for subvec in self.prop],
        ]

        j_emod = self.prop.labels[0].index('emod')
        mats[0][j_emod] = self.cached_form_assemblers['form.bi.dres_demod'].assemble()

        if 'umesh' in self.prop:
            j_shape = self.prop.labels[0].index('umesh')
            mats[0][j_shape] = self.cached_form_assemblers[
                'form.bi.dres_dumesh'
            ].assemble()

        return bm.BlockMatrix(mats, labels=self.state.labels + self.prop.labels)


### JAX MODELS

import jax
from .jaxutils import blockvec_to_dict, flatten_nested_dict

# pylint: disable=missing-docstring
DictVec = dict[str, NDArray]

JaxResidualArgs = tuple[DictVec, DictVec, DictVec]
JaxLinearizedResidualArgs = tuple[
    DictVec, DictVec, DictVec, tuple[DictVec, DictVec, DictVec]
]

JaxResidualFunction = Callable[[JaxResidualArgs], DictVec]
JaxLinearizedResidualFunction = Callable[[JaxLinearizedResidualArgs], DictVec]

Residual = tuple[
    NDArray, tuple[bv.BlockVector, bv.BlockVector, bv.BlockVector], Callable
]

class BaseDynamicalJaxModel:
    _res: Union[JaxResidualFunction, JaxLinearizedResidualFunction]
    _res_args: Union[JaxResidualArgs, JaxLinearizedResidualArgs]

    def __init__(self, residual: fluid.JaxResidual):

        self._residual = residual

        (state, control, prop) = residual.res_args

        self.state = bv.BlockVector(list(state.values()), labels=[list(state.keys())])

        self.statet = self.state.copy()

        self.control = bv.BlockVector(
            list(control.values()), labels=[list(control.keys())]
        )

        self.prop = bv.BlockVector(list(prop.values()), labels=[list(prop.keys())])

    @property
    def residual(self) -> Union[JaxResidualFunction, JaxLinearizedResidualFunction]:
        return self._residual

    @property
    def residual_args(self) -> Union[JaxResidualArgs, JaxLinearizedResidualArgs]:
        return self._res_args

    def set_state(self, state):
        self.state[:] = state

    def set_statet(self, statet):
        self.statet[:] = statet

    def set_control(self, control):
        self.control[:] = control

    def set_prop(self, prop):
        self.prop[:] = prop

    def assem_res(self):
        submats = self._res(*self.residual_args)
        labels = self.state.labels
        submats, shape = flatten_nested_dict(submats, labels)
        return bv.BlockVector(submats, shape, labels)

    def assem_dres_dstate(self):
        submats = jax.jacfwd(self._res, argnums=0)(*self.residual_args)
        labels = self.state.labels + self.state.labels
        submats, shape = flatten_nested_dict(submats, labels)
        return bv.BlockMatrix(submats, shape, labels)

    def assem_dres_dstatet(self):
        dresq_dq = np.diag(np.zeros(self.state['q'].size))
        dresq_dp = np.zeros((self.state['q'].size, self.state['p'].size))

        dresp_dp = np.diag(np.zeros(self.state['p'].size))
        dresp_dq = np.zeros((self.state['p'].size, self.state['q'].size))
        mats = [[dresq_dq, dresq_dp], [dresp_dq, dresp_dp]]
        labels = self.state.labels + self.state.labels
        return bv.BlockMatrix(mats, labels=labels)

    def assem_dres_dcontrol(self):
        submats = jax.jacfwd(self._res, argnums=1)(*self.residual_args)
        labels = self.state.labels + self.control.labels
        submats, shape = flatten_nested_dict(submats, labels)
        return bv.BlockMatrix(submats, shape, labels)

    def assem_dres_dprop(self):
        submats = jax.jacfwd(self._res, argnums=2)(*self.residual_args)
        labels = self.state.labels + self.prop.labels
        submats, shape = flatten_nested_dict(submats, labels)
        return bv.BlockMatrix(submats, shape, labels)


# NOTE: `JaxModel` and `LinearizedJaxModel` are very similar except for
# the residual functions and arguments (the latter is linearized)
class JaxModel(BaseDynamicalJaxModel, BaseDynamicalModel):
    """
    Representation of a dynamical system model
    """

    def __init__(self, residual: fluid.JaxResidual):
        super().__init__(residual)

        self._res = jax.jit(residual.res)
        # self._res = residual.res
        self._res_args = (
            blockvec_to_dict(self.state),
            blockvec_to_dict(self.control),
            blockvec_to_dict(self.prop),
        )


class LinearizedJaxModel(BaseDynamicalJaxModel, BaseLinearizedDynamicalModel):
    """
    Representation of a linearized dynamical system model
    """

    def __init__(self, residual: Residual):

        super().__init__(residual)

        self.dstate = self.state.copy()
        self.dstatet = self.statet.copy()
        self.dcontrol = self.control.copy()
        self.dprop = self.prop.copy()

        self.dstate[:] = 0.0
        self.dstatet[:] = 0.0
        self.dcontrol[:] = 0.0
        self.dprop[:] = 0.0

        primals = (
            blockvec_to_dict(self.state),
            blockvec_to_dict(self.control),
            blockvec_to_dict(self.prop),
        )
        tangents = (
            blockvec_to_dict(self.dstate),
            blockvec_to_dict(self.dcontrol),
            blockvec_to_dict(self.dprop),
        )

        self._res = lambda state, control, prop, tangents: jax.jvp(
            jax.jit(residual.res), (state, control, prop), tangents
        )[1]
        self._res_args = (*primals, tangents)

    def set_dstate(self, dstate):
        self.dstate[:] = dstate

    def set_dstatet(self, dstatet):
        self.dstatet[:] = dstatet

    def set_dcontrol(self, dcontrol):
        self.dcontrol[:] = dcontrol

    def set_dprop(self, dprop):
        self.dprop[:] = dprop


### FSI dynamical models

from . import fsi

# pylint: disable=missing-function-docstring


class FSIModel(BaseDynamicalModel):
    """
    Class representing a fluid-solid coupled dynamical system
    """

    def __init__(
        self,
        solid: FenicsModel,
        fluids: Union[list[FenicsModel], FenicsModel],
        solid_fsi_dofs: Union[list[NDArray], NDArray],
        fluid_fsi_dofs: Union[list[NDArray], NDArray],
    ):
        if isinstance(fluids, list):
            fluids = tuple(fluids)
            fluid_fsi_dofs = tuple(fluid_fsi_dofs)
            solid_fsi_dofs = tuple(solid_fsi_dofs)
        elif isinstance(fluids, tuple):
            pass
        else:
            fluids = (fluids,)
            fluid_fsi_dofs = (fluid_fsi_dofs,)
            solid_fsi_dofs = (solid_fsi_dofs,)
        self.solid = solid
        self.fluids = fluids

        self._models = (self.solid,) + tuple(self.fluids)

        _to_petsc = bv.convert_subtype_to_petsc
        self._fl_state = bv.concatenate_with_prefix(
            [_to_petsc(fluid.state) for fluid in fluids], 'fluid'
        )
        self.state = bv.concatenate([_to_petsc(solid.state), self._fl_state])

        self._fl_statet = bv.concatenate_with_prefix(
            [_to_petsc(fluid.statet) for fluid in fluids], 'fluid'
        )
        self.statet = bv.concatenate([_to_petsc(solid.statet), self._fl_statet])

        # This selects only psub and psup from the fluid control
        self._fl_control = bv.concatenate_with_prefix(
            [_to_petsc(fluid.control[['psub', 'psup']]) for fluid in fluids],
            prefix='fluid',
        )
        self.control = self._fl_control

        _ymid_props = bv.BlockVector([np.array([1.0])], labels=[['ymid']])
        self._fl_prop = bv.concatenate_with_prefix(
            [_to_petsc(fluid.prop) for fluid in fluids], 'fluid'
        )
        self.prop = bv.concatenate(
            [
                _to_petsc(solid.prop),
                self._fl_prop,
                bv.convert_subtype_to_petsc(_ymid_props),
            ]
        )

        ## -- FSI --
        (
            fsimaps,
            solid_area,
            dflcontrol_dslstate,
            dslcontrol_dflstate,
            dflcontrol_dslprops,
        ) = fsi.make_coupling_stuff(solid, fluids, solid_fsi_dofs, fluid_fsi_dofs)
        self._fsimaps = fsimaps
        self._solid_area = solid_area
        self._dflcontrol_dslstate = dflcontrol_dslstate
        self._dslcontrol_dflstate = dslcontrol_dflstate
        self._dflcontrol_dslprops = dflcontrol_dslprops

        # Make null BlockMats relating fluid/solid states
        # Make null `BlockMatrix`s relating fluid/solid states
        mats = [
            [subops.zero_mat(slvec.size, flvec.size) for flvec in self._fl_state.blocks]
            for slvec in self.solid.state.blocks
        ]
        self._null_dslstate_dflstate = bm.BlockMatrix(mats)
        mats = [
            [
                subops.zero_mat(flvec.size, slvec.size)
                for slvec in self.solid.state.blocks
            ]
            for flvec in self._fl_state.blocks
        ]
        self._null_dflstate_dslstate = bm.BlockMatrix(mats)

    def set_state(self, state):
        self.state[:] = state
        block_sizes = [model.state.size for model in self._models]
        sub_states = bv.chunk(state, block_sizes)
        for model, sub_state in zip(self._models, sub_states):
            model.set_state(sub_state)

        self._transfer_solid_to_fluid()
        self._transfer_fluid_to_solid()

    def _transfer_solid_to_fluid(self):
        """
        Update fluid controls from the solid state
        """
        ## The below are needed to communicate FSI interactions
        # Set solid_area
        dim = self.solid.residual.mesh().topology().dim()
        self._solid_area[:] = 2 * (
            self.prop['ymid'][0] - (self.solid.XREF + self.solid.state.sub['u'])[1::dim]
        )

        # map solid_area to fluid area
        for fsimap, fluid in zip(self._fsimaps, self.fluids):
            control = fluid.control.copy()
            fsimap.map_solid_to_fluid(self._solid_area, control['area'])
            fluid.set_control(control)

    def _transfer_fluid_to_solid(self):
        """
        Update solid controls from the fluid state
        """
        # map fluid pressure to solid pressure
        control = self.solid.control.copy()
        for fsimap, fluid in zip(self._fsimaps, self.fluids):
            fsimap.map_fluid_to_solid(fluid.state['p'], control['p'])
        self.solid.set_control(control)

    # Since the fluid has no time dependence there should be no need to set FSI interactions here
    # for the specialized 1D Bernoulli model so I've left it empty for now
    def set_statet(self, statet):
        self.statet[:] = statet
        block_sizes = [model.statet.size for model in self._models]
        sub_states = bv.chunk(statet, block_sizes)
        for model, sub_state in zip(self._models, sub_states):
            model.set_statet(sub_state)

    def set_control(self, control):
        self.control[:] = control
        for n, fluid in enumerate(self.fluids):
            fl_control = fluid.control.copy()
            # Set psub/psup of the coupled model to the fluid model control
            keys = ['psub', 'psup']
            _keys = [f'fluid{n}.{key}' for key in keys]
            fl_control[keys][:] = control[_keys]
            fluid.set_control(fl_control)

    def set_prop(self, prop):
        self.prop[:] = prop
        # You have to add a final size of 1 block to account for the final ymid
        # property block
        block_sizes = [model.prop.size for model in self._models] + [1]
        sub_props = bv.chunk(prop, block_sizes)
        for model, sub_prop in zip(self._models, sub_props):
            model.set_prop(sub_prop)

        # NOTE: You have to update the fluid control on a property due to shape
        # changes
        self._transfer_solid_to_fluid()

    def assem_res(self):
        return bv.concatenate(
            [bv.convert_subtype_to_petsc(model.assem_res()) for model in self._models]
        )

    def assem_dres_dstate(self):
        dslres_dslx = bm.convert_subtype_to_petsc(self.solid.assem_dres_dstate())
        dslres_dflx = bla.mult_mat_mat(
            bm.convert_subtype_to_petsc(self.solid.assem_dres_dcontrol()),
            self._dslcontrol_dflstate,
        )

        # TODO: This probably won't work in 3D since there would be multiple fluid models
        dflres_dflx = bm.convert_subtype_to_petsc(self._models[1].assem_dres_dstate())
        dflres_dslx = bla.mult_mat_mat(
            bm.convert_subtype_to_petsc(self._models[1].assem_dres_dcontrol()),
            self._dflcontrol_dslstate,
        )
        bmats = [[dslres_dslx, dslres_dflx], [dflres_dslx, dflres_dflx]]
        return bm.concatenate(bmats)

    def assem_dres_dstatet(self):
        # Because the fluid models is quasi-steady, there are no time varying FSI quantities
        # As a result, the off-diagonal block terms here are just zero
        dslres_dslx = bm.convert_subtype_to_petsc(self._models[0].assem_dres_dstatet())
        # dfsolid_dxfluid = self._models[0].assem_dres_dcontrolt() * self.dslcontrolt_dflstatet
        dslres_dflx = bm.convert_subtype_to_petsc(self._null_dslstate_dflstate)

        dflres_dflx = bm.convert_subtype_to_petsc(self._models[1].assem_dres_dstatet())
        # dffluid_dxsolid = self._models[1].assem_dres_dcontrolt() * self.dflcontrolt_dslstatet
        dflres_dslx = bm.convert_subtype_to_petsc(self._null_dflstate_dslstate)
        bmats = [[dslres_dslx, dslres_dflx], [dflres_dslx, dflres_dflx]]
        return bm.concatenate(
            bmats, labels=(self.state.labels[0], self.state.labels[0])
        )

    def assem_dres_dprop(self):

        ## Solid residual sensitivities
        dslres_dslprops = bm.convert_subtype_to_petsc(self.solid.assem_dres_dprop())

        submats = [
            subops.zero_mat(slsubvec.size, propsubvec.size)
            for slsubvec in self.solid.state
            for propsubvec in self._fl_prop
        ]
        dslres_dflprops = bm.BlockMatrix(
            submats,
            shape=self.solid.state.f_shape + self._fl_prop.f_shape,
            labels=self.solid.state.labels + self._fl_prop.labels,
        )

        submats = [
            subops.zero_mat(slsubvec.size, self.prop['ymid'].size)
            for slsubvec in self.solid.state
        ]
        dslres_dymid = bm.BlockMatrix(
            submats,
            shape=self.solid.state.shape + (1,),
            labels=self.solid.state.labels + (('ymid',),),
        )

        ## Fluid residual sensitivities
        dflres_dflcontrol = bm.concatenate_diag(
            [fluid.assem_dres_dcontrol() for fluid in self.fluids]
        )
        dflres_dslprops = bla.mult_mat_mat(
            bm.convert_subtype_to_petsc(dflres_dflcontrol), self._dflcontrol_dslprops
        )

        dflres_dflprops = bm.convert_subtype_to_petsc(
            bm.concatenate_diag([fluid.assem_dres_dprop() for fluid in self.fluids])
        )

        submats = [
            subops.zero_mat(flsubvec.size, self.prop['ymid'].size)
            for flsubvec in self._fl_state
        ]
        dflres_dymid = bm.BlockMatrix(
            submats,
            shape=self._fl_state.f_shape + (1,),
            labels=self._fl_state.labels + (('ymid',),),
        )

        bmats = [
            [dslres_dslprops, dslres_dflprops, dslres_dymid],
            [dflres_dslprops, dflres_dflprops, dflres_dymid],
        ]
        return bm.concatenate(bmats)

    def assem_dres_dcontrol(self):
        _mats = [
            [subops.zero_mat(m, n) for n in self.control.bshape[0]]
            for m in self.solid.state.bshape[0]
        ]
        dslres_dg = bm.BlockMatrix(
            _mats, labels=self.solid.state.labels + self.control.labels
        )

        # dflres_dflg = bm.convert_subtype_to_petsc(self.fluid.assem_dres_dcontrol())
        dflres_dflg = bm.concatenate_diag(
            [fluid.assem_dres_dcontrol() for fluid in self.fluids]
        )
        _mats = [
            [row[kk] for kk in range(1, dflres_dflg.shape[1])] for row in dflres_dflg
        ]
        # breakpoint()
        dflres_dg = bm.convert_subtype_to_petsc(
            bm.BlockMatrix(_mats, labels=self._fl_state.labels + self.control.labels)
        )
        return bm.concatenate([[dslres_dg], [dflres_dg]])


class LinearizedFSIModel(
    BaseLinearizedDynamicalModel, FSIModel
):
    """
    Class representing a fluid-solid coupled dynamical system
    """

    def __init__(
        self,
        solid: LinearizedFenicsModel,
        fluids: list[LinearizedFenicsModel],
        solid_fsi_dofs,
        fluid_fsi_dofs,
    ):
        super().__init__(solid, fluids, solid_fsi_dofs, fluid_fsi_dofs)

        self.dstate = bv.concatenate(
            [
                bv.convert_subtype_to_petsc(self.solid.dstate),
                bv.concatenate_with_prefix(
                    [
                        bv.convert_subtype_to_petsc(fluid.dstate)
                        for fluid in self.fluids
                    ],
                    prefix='fluid',
                ),
            ]
        )
        self.dstatet = bv.concatenate(
            [
                bv.convert_subtype_to_petsc(self.solid.dstatet),
                bv.concatenate_with_prefix(
                    [
                        bv.convert_subtype_to_petsc(fluid.dstatet)
                        for fluid in self.fluids
                    ],
                    prefix='fluid',
                ),
            ]
        )

        self._dsolid_area = dfn.Function(
            self.solid.residual.form['coeff.fsi.p1'].function_space()
        ).vector()

    def set_dstate(self, dstate):
        self.dstate[:] = dstate
        block_sizes = [model.dstate.size for model in self._models]
        sub_states = bv.chunk(dstate, block_sizes)
        for model, sub_state in zip(self._models, sub_states):
            model.set_dstate(sub_state)

        self._transfer_linearized_solid_to_fluid()
        self._transfer_linearized_fluid_to_solid()

    def set_dstatet(self, dstatet):
        self.dstatet[:] = dstatet
        block_sizes = [model.dstatet.size for model in self._models]
        sub_states = bv.chunk(dstatet, block_sizes)
        for model, sub_state in zip(self._models, sub_states):
            model.set_dstatet(sub_state)

    def _transfer_linearized_solid_to_fluid(self):
        """
        Update fluid controls from the solid state
        """
        ## The below are needed to communicate FSI interactions
        # map linearized state to linearized solid area
        dim = self.solid.residual.mesh().topology().dim()
        self._dsolid_area[:] = -2 * (self.dstate['u'][1::dim])

        # map linearized solid area to fluid area
        for fsimap, fluid in zip(self._fsimaps, self.fluids):
            dfl_control = fluid.dcontrol.copy()
            dfl_control['area'][:] = subops.mult_mat_vec(
                fsimap.dfluid_dsolid, subops.convert_vec_to_petsc(self._dsolid_area)
            )
            fluid.set_dcontrol(dfl_control)

    def _transfer_linearized_fluid_to_solid(self):
        """
        Update fluid controls from the solid state
        """
        # map linearized fluid pressure to solid pressure
        dsolid_control = self.solid.control.copy()
        for fsimap, fluid in zip(self._fsimaps, self.fluids):
            dsolid_control['p'][:] = subops.mult_mat_vec(
                fsimap.dsolid_dfluid, subops.convert_vec_to_petsc(fluid.dstate['p'])
            )
        self.solid.set_dcontrol(dsolid_control)

    def set_dcontrol(self, dcontrol):
        raise NotImplementedError()
