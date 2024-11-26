"""
This module defines the basic interface for a transient model.
"""

from typing import TypeVar, Union, Optional, Any
from numpy.typing import NDArray

from petsc4py import PETSc
import jax
import numpy as np
import dolfin as dfn

from blockarray import subops, linalg
from blockarray import blockvec as bv, blockmat as bm
from blockarray.subops import diag_mat, zero_mat

import functools

from femvf.residuals import solid as slr, fluid as flr
from femvf.solverconst import DEFAULT_NEWTON_SOLVER_PRM

from nonlineq import newton_solve, iterative_solve

T = TypeVar('T')
Vector = Union[subops.DfnVector, subops.GenericSubarray, subops.PETScVector]
Matrix = Union[subops.DfnMatrix, subops.GenericSubarray, subops.PETScMatrix]

BlockVec = bv.BlockVector[Vector]
BlockMat = bm.BlockMatrix[Matrix]


class BaseTransientModel:
    """
    This object represents the equations defining a system over one time step.

    The residual represents the error in the equations for a time step given by:
        F(u1, u0, g, p, dt)
        where
        u1, u0 : final/initial states of the system
        g : control vector at the current time (i.e. index 1)
        p : properties (constant in time)
        dt : time step

    Derivatives of F w.r.t u1, u0, g, p, dt and adjoint of those operators should all be
    defined.
    """

    ## Parameter setting functions

    @property
    def dt(self):
        """
        Return/set the time step
        """
        raise NotImplementedError(f"Subclass {type(self)} must implement this function")

    def set_ini_state(self, state0: BlockVec):
        """
        Set the initial state (`self.state0`)

        Parameters
        ----------
        state0: BlockVec
            The state to set
        """
        raise NotImplementedError(f"Subclass {type(self)} must implement this function")

    def set_fin_state(self, state1: BlockVec):
        """
        Set the final state (`self.state1`)

        Parameters
        ----------
        state1: BlockVec
            The state to set
        """
        raise NotImplementedError(f"Subclass {type(self)} must implement this function")

    def set_control(self, control: BlockVec):
        """
        Set the control (`self.control`)

        Parameters
        ----------
        control: BlockVec
            The controls to set
        """
        raise NotImplementedError(f"Subclass {type(self)} must implement this function")

    def set_prop(self, prop: BlockVec):
        """
        Set the properties (`self.prop`)

        Parameters
        ----------
        prop: BlockVec
            The properties to set
        """
        raise NotImplementedError(f"Subclass {type(self)} must implement this function")

    ## Residual and sensitivity methods
    def assem_res(self) -> BlockVec:
        """
        Return the residual of the current time step
        """
        raise NotImplementedError(f"Subclass {type(self)} must implement this function")

    def assem_dres_dstate0(self) -> BlockMat:
        """
        Return the residual sensitivity to the initial state for the time step
        """
        raise NotImplementedError(f"Subclass {type(self)} must implement this function")

    def assem_dres_dstate1(self) -> BlockMat:
        """
        Return the residual sensitivity to the final state for the time step
        """
        raise NotImplementedError(f"Subclass {type(self)} must implement this function")

    def assem_dres_dcontrol(self) -> BlockMat:
        """
        Return the residual sensitivity to the control for the time step
        """
        raise NotImplementedError(f"Subclass {type(self)} must implement this function")

    def assem_dres_dprops(self) -> BlockMat:
        """
        Return the residual sensitivity to the properties for the time step
        """
        raise NotImplementedError(f"Subclass {type(self)} must implement this function")

    ## Solver methods
    def solve_state1(
        self, state1: BlockVec, options: Optional[dict[str, Any]]
    ) -> tuple[BlockVec, dict[str, Any]]:
        """
        Solve for the final state for the time step

        Parameters
        ----------
        state1: BlockVec
            An initial guess for the final state. For nonlinear models, this
            serves as the initial guess for an iterative procedure.

        Returns
        -------
        BlockVec
            The final state at the end of the time step
        dict
            A dictionary of information about the solution. This depends on the
            solver but usually includes information like: the number of
            iterations, residual error, etc.
        """
        raise NotImplementedError(f"Subclass {type(self)} must implement this function")

    # NOTE: If you want to take derivatives of Transient models, you will need
    # to implement solvers for the jacobian and its adjoint
    # (`solve_dres_dstate1` and `solve_dres_dstate1_adj`).
    # In addition you'll need adjoints of all the `assem_*` functions
    # Currently some of these functions are left over from a previous implementation
    # but no longer work due to code changes.


## Solid type models

from femvf.equations import newmark
from femvf.equations import form
from .assemblyutils import FormAssembler


def depack_form_coefficient_function(form_coefficient):
    """
    Return the coefficient `Function` instance from a `forms` dict value

    This function mainly handles tuple coefficents which occurs
    for the shape parameter
    """
    #
    if isinstance(form_coefficient, tuple):
        # tuple coefficients consist of a `(function, ufl_object)` tuple
        coefficient, _ = form_coefficient
    else:
        coefficient = form_coefficient
    return coefficient


def properties_bvec_from_forms(forms, defaults=None):
    defaults = {} if defaults is None else defaults
    prop_labels = [
        form_name.split('.')[-1]
        for form_name in forms.keys()
        if 'coeff.prop' == form_name[: len('coeff.prop')]
    ]
    vecs = []
    for prop_label in prop_labels:
        coefficient = depack_form_coefficient_function(
            forms['coeff.prop.' + prop_label]
        )

        # Generally the size of the vector comes directly from the property;
        # i.e. constants are size 1, scalar fields have size matching number of dofs, etc.
        # The time step `dt` is a special case since it is size 1 but is specificied as a field as
        # a workaround in order to take derivatives
        if (
            isinstance(coefficient, dfn.function.constant.Constant)
            or prop_label == 'dt'
        ):
            vec = np.ones(coefficient.values().size)
            vec[:] = coefficient.values()
        else:
            vec = coefficient.vector().copy()

        if prop_label in defaults:
            vec[:] = defaults[prop_label]

        vecs.append(vec)

    return bv.BlockVector(vecs, labels=[prop_labels])


class FenicsModel(BaseTransientModel):
    """
    Class representing the discretized governing equations of a solid
    """

    def __init__(self, residual: slr.FenicsResidual):

        # Modify the input residual + form with newmark time-discretization

        new_form = form.modify_newmark_time_discretization(residual.form)
        residual = slr.FenicsResidual(
            new_form,
            residual._mesh,
            residual._mesh_functions,
            residual._mesh_subdomains
        )

        self._residual = residual

        ## Define the state/controls/properties
        u0 = self.residual.form['coeff.state.u0']
        v0 = self.residual.form['coeff.state.v0']
        a0 = self.residual.form['coeff.state.a0']
        u1 = self.residual.form['coeff.state.u1']
        v1 = self.residual.form['coeff.state.v1']
        a1 = self.residual.form['coeff.state.a1']

        self.state0 = bv.BlockVector(
            (u0.vector(), v0.vector(), a0.vector()), labels=[('u', 'v', 'a')]
        )
        self.state1 = bv.BlockVector(
            (u1.vector(), v1.vector(), a1.vector()), labels=[('u', 'v', 'a')]
        )
        self.control = bv.BlockVector(
            (self.residual.form['coeff.fsi.p1'].vector(),), labels=[('p',)]
        )
        self.prop = properties_bvec_from_forms(self.residual.form)
        self.set_prop(self.prop)

        # TODO: Refactor handling of multiple `ufl_forms`
        # This is super unclear right now
        self._assembler = FormAssembler(new_form)

    @property
    def assembler(self):
        return self._assembler

    @property
    def residual(self) -> slr.FenicsResidual:
        return self._residual

    @property
    def XREF(self) -> dfn.Function:
        xref = self.state0.sub[0].copy()
        function_space = self.residual.form['coeff.state.u1'].function_space()
        n_subspace = function_space.num_sub_spaces()

        xref[:] = (
            function_space.tabulate_dof_coordinates()[::n_subspace, :]
            .reshape(-1)
            .copy()
        )
        return xref

    @property
    def solid(self) -> 'FenicsModel':
        return self

    ## Parameter setting functions
    @property
    def dt(self):
        return self.residual.form['coeff.time.dt'].vector()[0]

    @dt.setter
    def dt(self, value):
        self.residual.form['coeff.time.dt'].vector()[:] = value

    def set_ini_state(self, state):
        """
        Sets the initial state variables, (u, v, a)

        Parameters
        ----------
        u0, v0, a0 : array_like
        """
        self.state0[:] = state

    def set_fin_state(self, state):
        """
        Sets the final state variables.

        Note that the solid forms are in displacement form so only the displacement is needed as an
        initial guess to solve the solid equations. The state `v1` and `a1` are specified explicitly
        by the Newmark relations once you solve for `u1`.

        Parameters
        ----------
        u1, v1, a1 : array_like
        """
        self.state1[:] = state

    def set_control(self, p1):
        self.control[:] = p1

    def set_prop(self, prop):
        """
        Sets the properties of the solid

        Properties are essentially all settable values that are not states. In UFL, all things that
        can have set values are `coefficients`, but this ditinguishes between state coefficients,
        and other property coefficients.

        Parameters
        ----------
        prop : Property / dict-like
        """
        for key, value in prop.sub_items():
            # TODO: Check types to make sure the input property is compatible with the solid type
            coefficient = self.residual.form['coeff.prop.' + key]

            # If the property is a field variable, values have to be assigned to every spot in
            # the vector
            if isinstance(coefficient, dfn.function.constant.Constant):
                coefficient.assign(dfn.Constant(np.squeeze(value)))
            else:
                coefficient.vector()[:] = value

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

    ## Residual and sensitivity functions
    def assem_res(self):
        dt = self.dt
        u1, v1, a1 = self.state1.sub_blocks.flat

        res = self.state1.copy()
        values = [
            self.assembler.assemble('u'),
            v1 - newmark.newmark_v(u1, *self.state0.sub_blocks, dt),
            a1 - newmark.newmark_a(u1, *self.state0.sub_blocks, dt),
        ]
        res[:] = values
        for bc in self.residual.dirichlet_bcs['coeff.state.u1']:
            bc.apply(res.sub['u'])
        return res

    @functools.cached_property
    def _const_assem_dres_dstate1(self):
        # These are constant matrices that only have to be computed once
        N = self.state1.bshape[0][0]
        # dfu_du = dfn.PETScMatrix(diag_mat(N, 1))
        dfu_du = None
        dfu_dv = dfn.PETScMatrix(zero_mat(N, N))
        dfu_da = dfn.PETScMatrix(zero_mat(N, N))

        dfv_du = dfn.PETScMatrix(diag_mat(N, 1))
        dfv_dv = dfn.PETScMatrix(diag_mat(N, 1))
        dfv_da = dfn.PETScMatrix(zero_mat(N, N))

        dfa_du = dfn.PETScMatrix(diag_mat(N, 1))
        dfa_dv = dfn.PETScMatrix(zero_mat(N, N))
        dfa_da = dfn.PETScMatrix(diag_mat(N, 1))
        return (dfu_du, dfu_dv, dfu_da, dfv_du, dfv_dv, dfv_da, dfa_du, dfa_dv, dfa_da)

    def assem_dres_dstate1(self):
        # BUG: Applying BCs to a tensor (`dfn.PETScMatrix()`) then
        # trying to reassemble into that tensor seems to cause problems.
        # This is done with the `cached_form_assembler` since it caches the
        # tensor it applies on
        dfu_du = self.assembler.assemble_derivative('u', 'coeff.state.u1')
        for bc in self.residual.dirichlet_bcs['coeff.state.u1']:
            bc.apply(dfu_du)

        (_, dfu_dv, dfu_da, dfv_du, dfv_dv, dfv_da, dfa_du, dfa_dv, dfa_da) = (
            self._const_assem_dres_dstate1
        )
        dfv_du = -newmark.newmark_v_du1(self.dt) * dfv_du
        dfa_du = -newmark.newmark_a_du1(self.dt) * dfa_du

        submats = [
            dfu_du,
            dfu_dv,
            dfu_da,
            dfv_du,
            dfv_dv,
            dfv_da,
            dfa_du,
            dfa_dv,
            dfa_da,
        ]
        return bm.BlockMatrix(
            submats, shape=(3, 3), labels=2 * self.state1.labels, check_bshape=False
        )

    def assem_dres_dstate0(self):
        assert len(self.state1.bshape) == 1
        N = self.state1.bshape[0][0]

        dfu_du = self.assembler.assemble_derivative('u', 'coeff.state.u0')
        dfu_dv = self.assembler.assemble_derivative('u', 'coeff.state.v0')
        dfu_da = self.assembler.assemble_derivative('u', 'coeff.state.a0')
        for mat in (dfu_du, dfu_dv, dfu_da):
            for bc in self.residual.dirichlet_bcs['coeff.state.u1']:
                bc.apply(mat)

        dfv_du = dfn.PETScMatrix(diag_mat(N, 0 - newmark.newmark_v_du0(self.dt)))
        dfv_dv = dfn.PETScMatrix(diag_mat(N, 0 - newmark.newmark_v_dv0(self.dt)))
        dfv_da = dfn.PETScMatrix(diag_mat(N, 0 - newmark.newmark_v_da0(self.dt)))

        dfa_du = dfn.PETScMatrix(diag_mat(N, 0 - newmark.newmark_a_du0(self.dt)))
        dfa_dv = dfn.PETScMatrix(diag_mat(N, 0 - newmark.newmark_a_dv0(self.dt)))
        dfa_da = dfn.PETScMatrix(diag_mat(N, 0 - newmark.newmark_a_da0(self.dt)))

        submats = [
            dfu_du,
            dfu_dv,
            dfu_da,
            dfv_du,
            dfv_dv,
            dfv_da,
            dfa_du,
            dfa_dv,
            dfa_da,
        ]
        return bm.BlockMatrix(submats, shape=(3, 3), labels=2 * self.state1.labels)

    def assem_dres_dcontrol(self):
        N = self.state1.bshape[0][0]
        M = self.control.bshape[0][0]

        # It should be hardcoded that the control is just the surface pressure
        assert self.control.shape[0] == 1
        dfu_dcontrol = self.assembler.assemble_derivative('u', 'coeff.fsi.p1')
        dfv_dcontrol = dfn.PETScMatrix(zero_mat(N, M))

        submats = [[dfu_dcontrol], [dfv_dcontrol]]
        return bm.BlockMatrix(submats, labels=self.state1.labels + self.control.labels)

    def assem_dres_dprops(self):
        raise NotImplementedError("Not implemented yet!")

    ## Solver functions
    def solve_state1(self, state1, options=None):
        if options is None:
            options = DEFAULT_NEWTON_SOLVER_PRM

        x = state1.copy()

        def linearized_subproblem(state):
            """
            Return a solver and residual corresponding to the linearized subproblem

            Returns
            -------
            assem_res : callable() -> type(state)
            solver : callable(type(state)) -> type(state)
            """
            self.set_fin_state(state)
            assem_res = self.assem_res

            def solve(res):
                dres_dstate1 = self.assem_dres_dstate1()
                return self.solve_dres_dstate1(dres_dstate1, x, res)

            return assem_res, solve

        state_n, solve_info = newton_solve(
            state1, linearized_subproblem, params=options
        )
        return state_n, solve_info

    def solve_dres_dstate1(self, dres_dstate1, x, b):
        """
        Solve the linearized residual problem

        This solution has a special format due to the Newmark time discretization.
        As a result, the below code only has to do one matrix solve (for the 'u'
        residual).
        """
        # dres_dstate1 = self.assem_dres_dstate1()

        dfu1_du1 = dres_dstate1.sub['u', 'u']
        dfv1_du1 = dres_dstate1.sub['v', 'u']
        dfa1_du1 = dres_dstate1.sub['a', 'u']

        bu, bv, ba = b.sub_blocks

        xu = x.sub['u']
        dfn.solve(dfu1_du1, xu, bu, 'petsc')
        x['v'][:] = bv - dfv1_du1 * xu
        x['a'][:] = ba - dfa1_du1 * xu

        return x

    def solve_dres_dstate1_adj(self, dres_dstate1_adj, x, b):
        """
        Solve the linearized adjoint residual problem

        This solution has a special format due to the Newmark time discretization.
        As a result, the below code only has to do one matrix solve (for the 'u'
        residual).
        """
        # Form key matrices
        dfu_du = dres_dstate1_adj.sub['u', 'u']
        dfv_du = dres_dstate1_adj.sub['v', 'u']
        dfa_du = dres_dstate1_adj.sub['a', 'u']

        # Solve A^T b = x
        bu, bv, ba = b.sub_blocks
        x.sub['a'][:] = ba
        x.sub['v'][:] = bv

        rhs_u = bu - (dfv_du * b['v'] + dfa_du * b['a'])
        dfn.solve(dfu_du, x['u'], rhs_u, 'petsc')
        return x


class NodalContactModel(FenicsModel):
    """
    This class modifies the default behaviour of the solid to implement contact
    pressures interpolated with the displacement function space. This involves
    manual modification of the matrices generated by FEniCS.
    """

    def set_fin_state(self, state):
        # This sets the 'standard' state variables u/v/a
        super().set_fin_state(state)

        self.residual.form['coeff.state.manual.tcontact'].vector()[:] = (
            self._contact_traction(state.sub['u'])
        )

    def assem_dres_dstate1(self):
        dres_dstate1 = super().assem_dres_dstate1()

        _ = dres_dstate1.sub['u', 'u']
        _ += self._assem_dresu_du_contact()
        return dres_dstate1

    def _contact_traction(self, u):
        # This computes the nodal values of the contact traction function
        XREF = self.XREF
        ycontact = self.residual.form['coeff.prop.ycontact'].values()[0]
        ncontact = self.residual.form['coeff.prop.ncontact'].values()
        kcontact = self.residual.form['coeff.prop.kcontact'].values()[0]

        ndim = self.residual.form['coeff.state.u0'].ufl_shape[0]
        gap = np.dot((XREF + u)[:].reshape(-1, ndim), ncontact) - ycontact
        tcontact = (
            (-form.pressure_contact_cubic_penalty(gap, kcontact)[:, None] * ncontact)
            .reshape(-1)
            .copy()
        )
        return tcontact

    def _assem_dresu_du_contact(self, adjoint=False):
        # Compute things needed to find sensitivities of contact pressure
        dfu2_dtcontact = self.assembler.assemble_derivative(
            'u', 'coeff.state.manual.tcontact', adjoint=adjoint
        )

        XREF = self.XREF
        kcontact = self.residual.form['coeff.prop.kcontact'].values()[0]
        ycontact = self.residual.form['coeff.prop.ycontact'].values()[0]
        u1 = self.residual.form['coeff.state.u1'].vector()
        ncontact = self.residual.form['coeff.prop.ncontact'].values()
        gap = (
            np.dot((XREF + u1)[:].reshape(-1, ncontact.shape[-1]), ncontact) - ycontact
        )
        dgap_du = ncontact

        # FIXME: This code below only works if n is aligned with the x/y axes.
        # for a general collision plane normal, the operation 'df_dtc*dtc_du' will
        # have to be represented by a block diagonal dtc_du (need to loop in python to do this). It
        # reduces to a diagonal if n is aligned with a coordinate axis.
        dtcontact_du2 = self.residual.form['coeff.state.u1'].vector().copy()
        dpcontact_dgap, _ = form.dform_cubic_penalty_pressure(gap, kcontact)
        dtcontact_du2[:] = np.array((-dpcontact_dgap[:, None] * dgap_du).reshape(-1))

        if adjoint:
            dfu2_dtcontact.mat().diagonalScale(dtcontact_du2.vec(), None)
        else:
            dfu2_dtcontact.mat().diagonalScale(None, dtcontact_du2.vec())
        dfu2_du2_contact = dfu2_dtcontact
        return dfu2_du2_contact


## Fluid type models

from .jaxutils import blockvec_to_dict, flatten_nested_dict

class JaxModel(BaseTransientModel):

    def __init__(self, residual: flr.JaxResidual):
        self._residual = residual

        res, (state, control, prop) = residual.res, residual.res_args

        self._res = jax.jit(res)
        self._dres = lambda state, control, prop, tangents: jax.jvp(
            res, (state, control, prop), tangents
        )[1]

        self.state0 = bv.BlockVector(list(state.values()), labels=[list(state.keys())])
        self.state1 = self.state0.copy()

        self.control = bv.BlockVector(
            list(control.values()), labels=[list(control.keys())]
        )

        self.prop = bv.BlockVector(list(prop.values()), labels=[list(prop.keys())])

        self.primals = (
            blockvec_to_dict(self.state1),
            blockvec_to_dict(self.control),
            blockvec_to_dict(self.prop),
        )

    @property
    def residual(self) -> flr.JaxResidual:
        return self._residual

    @property
    def fluid(self):
        return self

    ## Parameter setting functions
    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        self._dt = value

    def set_ini_state(self, state):
        """
        Set the initial fluid state
        """
        self.state0[:] = state

    def set_fin_state(self, state):
        """
        Set the final fluid state
        """
        self.state1[:] = state

    def set_control(self, control):
        """
        Set the final surface displacement and velocity
        """
        self.control[:] = control

    def set_prop(self, prop):
        """
        Set the fluid properties
        """
        self.prop[:] = prop

    ## Residual functions
    # TODO: Make remaining residual/solving functions
    def assem_res(self):
        labels = self.state1.labels
        subvecs = self._res(*self.primals)
        subvecs, shape = flatten_nested_dict(subvecs, labels)
        return bv.BlockVector(subvecs, shape, labels)

    ## Solver functions
    def solve_state1(self, state1, options=None):
        """
        Return the final flow state
        """
        info = {}
        return self.state1 - self.assem_res(), info

## Coupled models

from . import fsi

class BaseTransientFSIModel(BaseTransientModel):
    """
    Represents a coupled system of a solid and fluid(s) models

    Parameters
    ----------
    solid, fluid

    Attributes
    ----------
    solid : Solid
        A solid model object
    fluids : list[Fluid]
        A collection of 1D fluid model objects
    solid_fsi_dofs, fluid_fsi_dofs : Union[list[NDArray], NDArray]
        A collection of corresponding DOF arrays for fluid/structure interaction
        on the solid and fluid models, respectively.
        If there are `n` fluid models, then there must be `n` DOF arrays on both
        solid and fluid models.
        Note that while each DOF array
    """

    # TODO: Get rid of multiple fluid models
    def __init__(
        self,
        solid: FenicsModel,
        fluids: Union[list[JaxModel], JaxModel],
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

        ## Specify state, controls, and properties
        fluid_state0 = bv.concatenate_with_prefix(
            [fluid.state0 for fluid in fluids], 'fluid'
        )
        fluid_state1 = bv.concatenate_with_prefix(
            [fluid.state1 for fluid in fluids], 'fluid'
        )
        self.state0 = bv.concatenate([self.solid.state0, fluid_state0])
        self.state1 = bv.concatenate([self.solid.state1, fluid_state1])
        self.fluid_state0 = fluid_state0
        self.fluid_state1 = fluid_state1

        # The control is just the subglottal and supraglottal pressures
        self.control = bv.concatenate_with_prefix(
            [fluid.control[1:] for fluid in self.fluids], 'fluid'
        )

        _self_properties = bv.BlockVector((np.array([1.0]),), (1,), (('ymid',),))
        fluid_props = bv.concatenate_with_prefix(
            [fluid.prop for fluid in self.fluids], 'fluid'
        )
        self.prop = bv.concatenate([self.solid.prop, fluid_props, _self_properties])

        ## FSI related stuff
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

        # Make null `BlockMatrix`s relating fluid/solid states
        mats = [
            [subops.zero_mat(slvec.size, flvec.size) for flvec in fluid_state0.blocks]
            for slvec in self.solid.state0.blocks
        ]
        self._null_dslstate_dflstate = bm.BlockMatrix(mats)
        mats = [
            [
                subops.zero_mat(flvec.size, slvec.size)
                for slvec in self.solid.state0.blocks
            ]
            for flvec in fluid_state0.blocks
        ]
        self._null_dflstate_dslstate = bm.BlockMatrix(mats)

    @property
    def fsimaps(self):
        """
        Return the FSI map object
        """
        return self._fsimaps

    # These have to be defined to exchange data between fluid/solid domains
    # Explicit/implicit coupling methods may define these in different ways to
    # achieve the desired coupling
    def _set_ini_solid_state(self, uva0: bv.BlockVector):
        raise NotImplementedError("Subclasses must implement this method")

    def _set_fin_solid_state(self, uva1: bv.BlockVector):
        raise NotImplementedError("Subclasses must implement this method")

    def _set_ini_fluid_state(self, qp0: bv.BlockVector):
        raise NotImplementedError("Subclasses must implement this method")

    def _set_fin_fluid_state(self, qp1: bv.BlockVector):
        raise NotImplementedError("Subclasses must implement this method")

    ## Parameter setting methods
    @property
    def dt(self):
        return self.solid.dt

    @dt.setter
    def dt(self, value):
        self.solid.dt = value
        for fluid in self.fluids:
            fluid.dt = value

    def set_ini_state(self, state):
        sl_state, fl_state = bv.chunk(
            state, (self.solid.state0.size, self.fluid_state0.size)
        )

        self._set_ini_solid_state(sl_state)
        self._set_ini_fluid_state(fl_state)

    def set_fin_state(self, state):
        sl_state, fl_state = bv.chunk(
            state, (self.solid.state1.size, self.fluid_state1.size)
        )

        self._set_fin_solid_state(sl_state)
        self._set_fin_fluid_state(fl_state)

    def set_control(self, control):
        self.control[:] = control

        chunk_sizes = len(self.fluids) * [2]
        control_chunks = bv.chunk(self.control, chunk_sizes)

        for n, control in enumerate(control_chunks):
            for key, value in control.sub_items():
                key_postfix = '.'.join(key.split('.')[1:])
                self.fluids[n].control[key_postfix][:] = value

    def set_prop(self, prop):
        self.prop[:] = prop

        # The final `+ [1]` accounts for the 'ymid' property
        chunk_sizes = [model.prop.size for model in (self.solid,) + self.fluids] + [1]
        prop_chunks = bv.chunk(self.prop, chunk_sizes)[:-1]
        prop_setters = [self.solid.set_prop] + [fluid.set_prop for fluid in self.fluids]

        for set_prop, prop in zip(prop_setters, prop_chunks):
            set_prop(prop)

        # self.solid.set_prop(prop[:self.solid.prop.size])
        # self.fluid.set_prop(prop[self.solid.prop.size:-1])


# TODO: The `assem_*` type methods are incomplete as I haven't had to use them
class ExplicitFSIModel(BaseTransientFSIModel):

    # The functions below are set to represent an explicit (staggered) coupling
    # This means for a timestep:
    # - The fluid loading on the solid domain is based on the fluid loads
    #   at the beginning of the time step. i.e. the fluid loading is
    #   constant/known
    #     - Setting the initial fluid state changes the solid control
    # - The geometry of the fluid domain is based on the deformation of
    #   the solid in the current time step. i.e. the geometry of the fluid
    #   changes based on the computed deformation for the current time step
    #    - Setting the final solid state updates the fluid control
    def _set_ini_solid_state(self, uva0):
        self.solid.set_ini_state(uva0)

    def _set_fin_solid_state(self, uva1):
        self.solid.set_fin_state(uva1)

        # For explicit coupling, the final fluid area corresponds to the final solid deformation
        ndim = self.solid.residual.mesh().topology().dim()
        self._solid_area[:] = 2 * (
            self.prop['ymid'][0]
            - (self.solid.XREF + self.solid.state1.sub['u'])[1::ndim]
        )
        for n, (fluid, fsimap) in enumerate(zip(self.fluids, self.fsimaps)):
            fl_control = fluid.control.copy()
            fsimap.map_solid_to_fluid(self._solid_area, fl_control.sub['area'][:])
            fluid.set_control(fl_control)

    def _set_ini_fluid_state(self, qp0):
        # For explicit coupling, the final/current solid pressure corresponds to
        # the initial/previous fluid pressure
        sl_control = self.solid.control.copy()
        sl_control['p'] = 0
        qp0_parts = bv.chunk(qp0, tuple(fluid.state0.size for fluid in self.fluids))
        for fluid, fsimap, qp0_part in zip(self.fluids, self.fsimaps, qp0_parts):
            fluid.set_ini_state(qp0_part)
            fsimap.map_fluid_to_solid(qp0_part[1], sl_control.sub['p'])
        self.solid.set_control(sl_control)

    def _set_fin_fluid_state(self, qp1):
        """Set the final fluid state"""
        qp1_parts = bv.chunk(qp1, tuple(fluid.state1.size for fluid in self.fluids))
        for fluid, qp1_part in zip(self.fluids, qp1_parts):
            fluid.set_fin_state(qp1_part)

    ## Residual and derivative assembly functions
    def assem_res(self):
        """
        Return the residual
        """
        res_sl = self.solid.assem_res()
        res_fl = self.fluid.assem_res()
        return bv.concatenate((res_sl, res_fl))

    def assem_dres_dstate0(self):
        # TODO: Make this correct
        drsl_dxsl = self.solid.assem_dres_dstate0()
        drsl_dxfl = linalg.mult_mat_mat(
            self.solid.assem_dres_dcontrol(), self._dslcontrol_dflstate
        )
        drfl_dxfl = self.fluid.assem_dres_dstate0()
        drfl_dxsl = linalg.mult_mat_mat(
            self.fluid.assem_dres_dcontrol(), self._dflcontrol_dslstate
        )
        bmats = [[drsl_dxsl, drsl_dxfl], [drfl_dxsl, drfl_dxfl]]
        return bm.concatenate(bmats)

    def assem_dres_dstate1(self):
        drsl_dxsl = self.solid.assem_dres_dstate0()
        drsl_dxfl = linalg.mult_mat_mat(
            self.solid.assem_dres_dcontrol(), self._dslcontrol_dflstate
        )
        drfl_dxfl = self.fluid.assem_dres_dstate0()
        drfl_dxsl = linalg.mult_mat_mat(
            self.fluid.assem_dres_dcontrol(), self._dflcontrol_dslstate
        )
        bmats = [[drsl_dxsl, drsl_dxfl], [drfl_dxsl, drfl_dxfl]]
        return bm.concatenate(bmats)

    # Forward solver methods
    def solve_state1(self, ini_state, options=None):
        """
        Solve for the final state given an initial guess
        """
        # Set the initial guess for the final state
        self.set_fin_state(ini_state)

        uva1, solid_info = self.solid.solve_state1(ini_state[:3], options)

        self._set_fin_solid_state(uva1)

        fluids_solve_res = [
            fluid.solve_state1(ini_state[3:], options) for fluid in self.fluids
        ]
        qp1s = [solve_res[0] for solve_res in fluids_solve_res]
        fluids_solve_info = [solve_res[1] for solve_res in fluids_solve_res]

        step_info = solid_info
        step_info.update(
            {
                f'fluid{ii}_info': fluid_info
                for ii, fluid_info in enumerate(fluids_solve_info)
            }
        )

        return bv.concatenate([uva1] + qp1s, labels=self.state1.labels), step_info

    def solve_dres_dstate1(self, b):
        """
        Solve, dF/du x = f
        """
        x = self.state0.copy()

        x[:3] = self.solid.solve_dres_dstate1(b[:3])

        dq_du, dp_du = self.fluid.solve_dqp1_du1_solid(self, adjoint=False)
        dfq2_du2 = 0.0 - dq_du
        dfp2_du2 = 0.0 - dp_du

        x['q'][:] = b['q'] - dfq2_du2.inner(x['u'])
        x['p'][:] = b['p'] - dfp2_du2 * x['u'].vec()
        return x

    def solve_dres_dstate1_adj(self, x):
        """
        Solve, dF/du^T x = f
        """
        ## Assemble sensitivity matrices
        dq_du, dp_du = self.fluid.solve_dqp1_du1_solid(self, adjoint=True)
        dfq2_du2 = 0 - dq_du
        dfp2_du2 = 0 - dp_du

        ## Do the linear algebra that solves for x
        b = self.state0.copy()
        b_qp, b_uva = b[3:], b[:3]

        # This performs the fluid part of the solve
        b_qp[:] = x[3:]

        # This is the solid part of the
        bp_vec = dfp2_du2.getVecRight()
        bp_vec[:] = b_qp['p']
        rhs = x[:3].copy()
        rhs['u'] -= dfq2_du2 * b_qp['q'] + dfn.PETScVector(dfp2_du2 * bp_vec)
        b_uva[:] = self.solid.solve_dres_dstate1_adj(rhs)

        return b


class ImplicitFSIModel(BaseTransientFSIModel):
    ## These must be defined to properly exchange the forcing data between the solid and domains
    def _set_ini_fluid_state(self, qp0):
        qp0_parts = bv.chunk(qp0, tuple(fluid.state0.size for fluid in self.fluids))
        for fluid, qp0_part in zip(self.fluids, qp0_parts):
            fluid.set_ini_state(qp0_part)

    def _set_fin_fluid_state(self, qp1):
        sl_control = self.solid.control.copy()
        sl_control['p'] = 0
        qp1_parts = bv.chunk(qp1, tuple(fluid.state1.size for fluid in self.fluids))
        for fluid, fsimap, qp1_part in zip(self.fluids, self.fsimaps, qp1_parts):
            fluid.set_fin_state(qp1_part)
            fsimap.map_fluid_to_solid(qp1_part[1], sl_control.sub['p'])
        self.solid.set_control(sl_control)

    def _set_ini_solid_state(self, uva0):
        """Set the initial solid state"""
        self.solid.set_ini_state(uva0)

    def _set_fin_solid_state(self, uva1):
        self.solid.set_fin_state(uva1)

        # For both implicit/explicit coupling, the final fluid area corresponds to the final solid deformation
        ndim = self.solid.residual.mesh().topology().dim()
        self._solid_area[:] = 2 * (
            self.prop['ymid'][0] - (self.solid.XREF + self.solid.state1['u'])[1::ndim]
        )
        for n, (fluid, fsimap) in enumerate(zip(self.fluids, self.fsimaps)):
            fl_control = fluid.control
            fsimap.map_solid_to_fluid(self._solid_area, fl_control['area'][:])
            fluid.set_control(fl_control)

    ## Forward solver methods
    def assem_res(self):
        """
        Return the residual vector, F
        """
        res_sl = self.solid.assem_res()
        res_fl = self.fluid.assem_res()
        return bv.concatenate(res_sl, res_fl)

    def solve_state1(self, ini_state, options=None):
        """
        Solve for the final state given an initial guess

        This uses a fixed-point iteration where the solid is solved, then the fluid and so-on.
        """
        if options is None:
            options = DEFAULT_NEWTON_SOLVER_PRM

        def iterative_subproblem(x):
            uva1_0 = x[:3]
            qp1_0 = x[3:]

            def solve(res):
                # Solve the solid with the previous iteration's fluid pressures
                uva1, _ = self.solid.solve_state1(uva1_0, options)

                # Compute new fluid pressures for the updated solid position
                self._set_fin_solid_state(uva1)
                qp1, fluid_info = self.fluid.solve_state1(qp1_0)
                self._set_fin_fluid_state(qp1_0)
                return bv.concatenate((uva1, qp1))

            def assem_res():
                return self.assem_res()

            return assem_res, solve

        return iterative_solve(
            ini_state, iterative_subproblem, norm=bv.norm, params=options
        )

    def solve_dres_dstate1(self, b):
        """
        Solve, dF/du x = f
        """
        dt = self.solid.dt
        x = self.state0.copy()

        solid = self.solid

        dfu1_du1 = self.solid.assembler.assemble_derivative('u', 'coeff.state.u1')
        dfv2_du2 = 0 - newmark.newmark_v_du1(dt)
        dfa2_du2 = 0 - newmark.newmark_a_du1(dt)

        dq_du, dp_du = self.fluid.solve_dqp1_du1_solid(self, adjoint=False)
        dfq2_du2 = 0 - dq_du
        dfp2_du2 = 0 - dp_du

        for bc in self.solid.residual.dirichlet_bcs['coeff.state.u1']:
            bc.apply(dfu1_du1)
        dfn.solve(dfu1_du1, x['u'], b['u'], 'petsc')
        x['v'][:] = b['v'] - dfv2_du2 * x['u']
        x['a'][:] = b['a'] - dfa2_du2 * x['u']

        x['q'][:] = b['q'] - dfq2_du2.inner(x['u'])
        x['p'][:] = b['p'] - dfn.PETScVector(dfp2_du2 * x['u'].vec())
        return x

    def solve_dres_dstate1_adj(self, b):
        """
        Solve, dF/du^T x = f
        """
        ## Assemble sensitivity matrices
        # self.set_iter_params(**it_params)
        dt = self.solid.dt

        dfu2_du2 = self.solid.assembler.assemble_derivative('u', 'coeff.state.u1', adjoint=True)
        dfv2_du2 = 0 - newmark.newmark_v_du1(dt)
        dfa2_du2 = 0 - newmark.newmark_a_du1(dt)
        dfu2_dp2 = self.solid.assembler.assemble_derivative('u', 'coeff.fsi.p1', adjoint=True)

        # map dfu2_dp2 to have p on the fluid domain
        solid_dofs, fluid_dofs = self.get_fsi_scalar_dofs()
        dfu2_dp2 = dfn.as_backend_type(dfu2_dp2).mat()
        dfu2_dp2 = linalg.reorder_mat_rows(
            dfu2_dp2, solid_dofs, fluid_dofs, self.fluid.state1['p'].size
        )

        dq_du, dp_du = self.fluid.solve_dqp1_du1_solid(self, adjoint=True)
        dfq2_du2 = 0 - dq_du
        dfp2_du2 = 0 - dp_du

        ## Do the linear algebra that solves for the adjoint states
        adj_uva = self.solid.state0.copy()
        adj_qp = self.fluid.state0.copy()

        adj_u_rhs, adj_v_rhs, adj_a_rhs, adj_q_rhs, adj_p_rhs = b

        # adjoint states for v, a, and q are explicit so we can solve for them
        for bc in self.solid.residual.dirichlet_bcs['coeff.state.u1']:
            bc.apply(adj_v_rhs)
        adj_uva['v'][:] = adj_v_rhs

        for bc in self.solid.residual.dirichlet_bcs['coeff.state.u1']:
            bc.apply(adj_a_rhs)
        adj_uva['a'][:] = adj_a_rhs

        # TODO: how to apply fluid boundary conditions in a generic way?
        adj_qp['q'][:] = adj_q_rhs

        adj_u_rhs -= (
            dfv2_du2 * adj_uva['v'] + dfa2_du2 * adj_uva['a'] + dfq2_du2 * adj_qp['q']
        )

        bc_dofs = np.concatenate(
            [
                list(bc.get_boundary_values().keys())
                or bc in self.solid.residual.dirichlet_bcs['coeff.state.u1']
            ],
            dtype=np.int32
        )
        bc_dofs = np.unique(bc_dofs)

        for bc in self.solid.residual.dirichlet_bcs['coeff.state.u1']:
            bc.apply(dfu2_du2, adj_u_rhs)
        dfp2_du2.zeroRows(bc_dofs, diag=0.0)

        # solve the coupled system for pressure and displacement residuals
        dfu2_du2_mat = dfn.as_backend_type(dfu2_du2).mat()
        blocks = [[dfu2_du2_mat, dfp2_du2], [dfu2_dp2, 1.0]]

        dfup2_dup2 = linalg.form_block_matrix(blocks)
        adj_up, rhs = dfup2_dup2.getVecs()

        # calculate rhs vectors
        rhs[: adj_u_rhs.size()] = adj_u_rhs
        rhs[adj_u_rhs.size() :] = adj_p_rhs

        # Solve the block linear system with LU factorization
        ksp = PETSc.KSP().create()
        ksp.setType(ksp.Type.PREONLY)

        pc = ksp.getPC()
        pc.setType(pc.Type.LU)

        ksp.setOperators(dfup2_dup2)
        ksp.solve(rhs, adj_up)

        adj_uva['u'][:] = adj_up[: adj_u_rhs.size()]
        adj_qp['p'][:] = adj_up[adj_u_rhs.size() :]

        return bv.concatenate([adj_uva, adj_qp])

## NOTE: OLD ACOUSTIC MODELS CODE
# This definitely doesn't work anymore but I've kept the commented code here in
# case you want to updated it.
# A new version should define a acoustic residual which could then be loaded
# into a JAX model

# class Acoustic1D(base.BaseTransientModel):
#     def __init__(self, num_tube):
#         assert num_tube % 2 == 0

#         # self._dt = 0.0

#         # pinc (interlaced f1, b2 partial pressures) are incident pressures
#         # pref (interlaced b1, f2 partial pressures) are reflected pressures
#         pinc = np.zeros((num_tube // 2 + 1) * 2)
#         pref = np.zeros((num_tube // 2 + 1) * 2)
#         self.state0 = vec.BlockVector((pinc, pref), labels=[('pinc', 'pref')])

#         self.state1 = self.state0.copy()

#         # The control is the input flow rate
#         qin = np.zeros((1,))
#         self.control = vec.BlockVector((qin,), labels=[('qin',)])

#         length = np.ones(1)
#         area = np.ones(num_tube)
#         gamma = np.ones(num_tube)
#         rho = 1.225 * 1e-3 * np.ones(1)
#         c = 340 * 100 * np.ones(1)
#         rrad = np.ones(1)
#         lrad = np.ones(1)
#         self.prop = vec.BlockVector(
#             (length, area, gamma, rho, c, rrad, lrad),
#             labels=[
#                 ('length', 'area', 'proploss', 'rhoac', 'soundspeed', 'rrad', 'lrad')
#             ],
#         )

#     ## Setting parameters of the acoustic model
#     @property
#     def dt(self):
#         NTUBE = self.prop['area'].size
#         length = self.prop['length'][0]
#         C = self.prop['soundspeed'][0]
#         return (2 * length / NTUBE) / C

#     @dt.setter
#     def dt(self, value):
#         # Note the acoustic model can't change the time step without changing the length of the tract
#         # because the tract length and time step are linked throught the speed of sound
#         raise NotImplementedError("You can't set the time step of a WRAnalog tube")

#     def set_ini_state(self, state):
#         self.state0[:] = state

#     def set_fin_state(self, state):
#         self.state1[:] = state

#     def set_control(self, control):
#         self.control[:] = control

#     def set_prop(self, prop):
#         self.prop[:] = prop

#     ## Getting empty vectors
#     def get_state_vec(self):
#         ret = self.state0.copy()
#         ret[:] = 0.0
#         return ret

#     def get_control_vec(self):
#         ret = self.control.copy()
#         ret[:] = 0.0
#         return ret

#     def get_properties_vec(self, set_default=True):
#         ret = self.prop.copy()
#         if not set_default:
#             ret[:] = 0.0
#         return ret


# class WRAnalog(Acoustic1D):
#     @property
#     def z(self):
#         return self.prop['rhoac'] * self.prop['soundspeed'] / self.prop['area']

#     def set_prop(self, prop):
#         super().set_prop(prop)

#         # Reset the WRAnalog 'reflect' function when properties of the tract are updated
#         # The reflection function behaviour only changes if the properties are changed
#         # so must be reset here
#         self.init_wra()

#     def init_wra(self):
#         dt = self.dt
#         cspeed = self.prop['soundspeed'][0]
#         rho = self.prop['rhoac'][0]
#         area = self.prop['area'].copy()
#         gamma = self.prop['proploss'].copy()

#         ## Set radiation proeprties
#         # Ignore below for now?
#         R = self.prop['rrad'][0]
#         L = self.prop['lrad'][0]

#         # Formula given by Story and Flanagan (equations 2.103 and 2.104 from Story's thesis)
#         PISTON_RAD = np.sqrt(area[-1] / np.pi)
#         R = 128 / (9 * np.pi**2)
#         L = 16 / dt * PISTON_RAD / (3 * np.pi * cspeed)

#         NUM_TUBE = area.size

#         # 1, 2 represent the areas to the left and right of even junctions
#         # note that the number of junctions is 1+number of tubes so some of
#         # these are ficitious areas a1 @ junction 0 doesn't really exist
#         # the same for a2 @ final junction
#         a1 = np.concatenate([[1.0], area[1::2]])
#         a2 = np.concatenate([area[:-1:2], [1.0]])

#         gamma1 = np.concatenate([[1.0], gamma[1::2]])
#         gamma2 = np.concatenate([gamma[:-1:2], [1.0]])

#         self.reflect, self.reflect00, self.inputq = wra(
#             dt, a1, a2, gamma1, gamma2, NUM_TUBE, cspeed, rho, R=R, L=L
#         )

#     ## Solver functions
#     def solve_state1(self):
#         qin = self.control['qin'][0]
#         pinc, pref = self.state0.vecs
#         pinc_1, pref_1 = self.reflect(pinc, pref, qin)

#         state1 = vec.BlockVector((pinc_1, pref_1), labels=self.state1.labels)
#         info = {}
#         return state1, info

#     def assem_res(self):
#         return self.state1 - self.solve_state1()[0]

#     def solve_dres_dstate1_adj(self, x):
#         return x

#     def apply_dres_dstate0_adj(self, x):
#         args = (*self.state0.vecs, *self.control.vecs)
#         ATr = jax.linear_transpose(self.reflect, *args)

#         b_pinc, b_pref, b_qin = ATr(x.vecs)
#         bvecs = (np.asarray(b_pinc), np.asarray(b_pref))
#         return -vec.BlockVector(bvecs, labels=self.state0.labels)

#     def apply_dres_dcontrol(self, x):
#         args = (*self.state0.vecs, *self.control.vecs)
#         _, A = jax.linearize(self.reflect, *args)

#         x_ = vec.concatenate_vec([self.state0.copy(), x])
#         bvecs = [np.asarray(vec) for vec in A(*x_.vecs)]

#         return -vec.BlockVector(bvecs, labels=self.state1.labels)

#     def apply_dres_dp_adj(self, x):
#         b = self.prop.copy()
#         b[:] = 0.0
#         return b


# def wra(dt, a1, a2, gamma1, gamma2, N, C, RHO, R=1.0, L=1.0):
#     """ """
#     assert gamma1.size == N // 2 + 1
#     assert gamma2.size == N // 2 + 1

#     assert a1.size == N // 2 + 1
#     assert a2.size == N // 2 + 1

#     z1 = RHO * C / a1
#     z2 = RHO * C / a2

#     def inputq(q, pinc):
#         q = jnp.squeeze(q)
#         z = z2[0]
#         gamma = gamma2[0]

#         f1, b2 = pinc[0], pinc[1]
#         b2 = gamma * b2

#         f2 = z * q + b2
#         b1 = b2 + f2 - f1
#         return jnp.array([b1, f2])

#     def dinputq(q, bi, z, gamma):
#         dfr_dq = z
#         dfr_dbi = gamma * 1.0
#         return dfr_dq, dfr_dbi

#     def radiation(pinc, pinc_prev, pref_prev):
#         gamma = gamma1[-1]
#         f1prev = pinc_prev[0]
#         b1prev, f2prev = pref_prev[0], pref_prev[1]
#         f1 = pinc[0]

#         f1 = gamma * f1

#         _a1 = -R + L - R * L
#         _a2 = -R - L + R * L

#         _b1 = -R + L + R * L
#         _b2 = R + L + R * L

#         b1 = 1 / _b2 * (f1 * _a2 + f1prev * _a1 + b1prev * _b1)
#         f2 = 1 / _b2 * (f2prev * _b1 + f1 * (_b2 + _a2) + f1prev * (_a1 - _b1))
#         return jnp.array([b1, f2])

#     def dradiation(f1, f1prev, b1prev, f2prev, gamma):
#         _a1 = -R + L - R * L
#         _a2 = -R - L + R * L

#         _b1 = -R + L + R * L
#         _b2 = R + L + R * L

#         df2_df1 = gamma * (1 / _b2) * (_b2 + _a2)
#         df2_df1prev = gamma * (1 / _b2) * (_a1 - _b1)
#         df2_db1prev = 0.0
#         df2_df2prev = gamma * (1 / _b2) * (_b1)

#         db1_df1 = gamma * (1 / _b2) * (_a2)
#         db1_df1prev = gamma * (1 / _b2) * (_a1)
#         db1_db1prev = gamma * (1 / _b2) * (_b1)
#         db1_df2prev = 0.0
#         return (df2_df1, df2_df1prev, df2_db1prev, df2_df2prev), (
#             db1_df1,
#             db1_df1prev,
#             db1_db1prev,
#             db1_df2prev,
#         )

#     def reflect00(pinc, pinc_prev, pref_prev, q):
#         # Note that int, inp, rad refer to interior, input, and radiation junction
#         # locations, respectively
#         f1, b2 = pinc[:-1:2], pinc[1::2]

#         f1 = gamma1 * f1
#         b2 = gamma2 * b2

#         r1 = (z2 - z1) / (z2 + z1)

#         f2int = (f1 + (f1 - b2) * r1)[1:-1]
#         b1int = (b2 + (f1 - b2) * r1)[1:-1]
#         pref_int = jnp.stack([b1int, f2int], axis=-1).reshape(-1)

#         ## Input boundary
#         pinc_inp = pinc[:2]
#         pref_inp = inputq(q, pinc_inp) * np.ones(1)

#         ## Radiation boundary
#         pinc_rad = pinc[-2:]
#         pinc_rad_prev = pinc_prev[-2:]
#         pref_rad_prev = pref_prev[-2:]
#         pref_rad = radiation(pinc_rad, pinc_rad_prev, pref_rad_prev)

#         pref = jnp.concatenate([pref_inp, pref_int, pref_rad])
#         return pref

#     def dreflect00(f1, b2, f1prev, b2prev, b1prev, f2prev, q):
#         r1 = (z2 - z1) / (z2 + z1)

#         df2_df1 = gamma1 * (1 + r1)
#         df2_db2 = gamma2 * (-r1)

#         db1_df1 = gamma1 * (r1)
#         db1_db2 = gamma2 * (1 - r1)

#         ## Input boundary
#         df2_dq, df2_db2[0] = dinputq(q, b2[0], z2[0], gamma2[0])
#         db1_dq = 0

#         ## Radiation boundary
#         _df2, _db1 = dradiation(f1[-1], f1prev[-1], b1prev[-1], f2prev[-1], gamma1[-1])
#         df2_dfm, df2_dfmprev, df2_dbmprev, df2_dfradprev = _df2
#         db1_dfm, db1_dfmprev, db1_dbmprev, db1_dfradprev = _db1

#         df2_df1[-1] = df2_dfm
#         db1_df1[-1] = db1_dfm

#         df2_df1prev = np.zeros(f1prev.shape)
#         df2_db1prev = np.zeros(f1prev.shape)
#         df2_df2prev = np.zeros(f1prev.shape)
#         df2_db2prev = 0.0

#         db1_df1prev = np.zeros(f1prev.shape)
#         db1_db1prev = np.zeros(f1prev.shape)
#         db1_df2prev = np.zeros(f1prev.shape)
#         db1_db2prev = 0.0

#         db1_df1prev[-1] = db1_dfmprev
#         db1_db1prev[-1] = db1_dbmprev
#         db1_df2prev[-1] = db1_dfradprev

#         df2_df1prev[-1] = df2_dfmprev
#         df2_db1prev[-1] = df2_dbmprev
#         df2_df2prev[-1] = df2_dfradprev

#         df2 = (
#             df2_df1,
#             df2_db2,
#             df2_df1prev,
#             df2_db1prev,
#             df2_df2prev,
#             df2_db2prev,
#             df2_dq,
#         )
#         db1 = (
#             db1_df1,
#             db1_db2,
#             db1_df1prev,
#             db1_db1prev,
#             db1_df2prev,
#             db1_db2prev,
#             db1_dq,
#         )
#         return db1, df2

#     def reflect05(pinc):
#         z1_ = z2[:-1]
#         z2_ = z1[1:]

#         gamma1_ = gamma2[:-1]
#         gamma2_ = gamma1[1:]

#         f1 = pinc[:-1:2]
#         b2 = pinc[1::2]

#         f1 = gamma1_ * f1
#         b2 = gamma2_ * b2
#         r = (z2_ - z1_) / (z2_ + z1_)

#         b1 = b2 + (f1 - b2) * r
#         f2 = f1 + (f1 - b2) * r
#         pref = jnp.stack([b1, f2], axis=-1).reshape(-1)
#         return pref

#     def dreflect05(f1, b2, gamma1, gamma2, z1, z2):
#         r = (z2 - z1) / (z2 + z1)

#         df2_df1 = (1.0 + r) * gamma1
#         df2_db2 = (-r) * gamma2

#         db1_df1 = (r) * gamma1
#         db1_db2 = (1.0 - r) * gamma2
#         return (db1_df1, db1_db2), (df2_df1, df2_db2)

#     # @jax.jit
#     def reflect(pinc, pref, q):
#         f1, b2 = pinc[:-1:2], pinc[1::2]
#         b1, f2 = pref[:-1:2], pref[1::2]

#         # f2 and b1 (reflected @ 0.0) -> f1, b2 (incident @ 0.5)
#         f1_05 = f2[:-1]
#         b2_05 = b1[1:]
#         pinc_05 = jnp.stack([f1_05, b2_05], axis=-1).reshape(-1)

#         pref_05 = reflect05(pinc_05)
#         b1_05, f2_05 = pref_05[:-1:2], pref_05[1::2]

#         # f2_05 and b1_05 (reflected @ 0.5) -> f1, b2 (incident @ 1.0)
#         f1inp, b2rad = np.zeros(1), np.zeros(1)
#         f1_1 = jnp.concatenate([f1inp, f2_05])
#         b2_1 = jnp.concatenate([b1_05, b2rad])
#         pinc_1 = jnp.stack([f1_1, b2_1], axis=-1).reshape(-1)

#         pref_1 = reflect00(pinc_1, pinc, pref, q)
#         return pinc_1, pref_1

#     def dreflect(f1, b2, b1, f2, q):
#         df1_1_df1 = np.zeros(f1.shape)
#         df1_1_db2 = np.zeros(f1.shape)
#         df1_1_db1 = np.zeros(f1.shape)
#         df1_1_df2 = np.zeros(f1.shape)

#         db1_1_df1 = np.zeros(f1.shape)
#         db1_1_db2 = np.zeros(f1.shape)
#         db1_1_db1 = np.zeros(f1.shape)
#         db1_1_df2 = np.zeros(f1.shape)

#         df2_1_df1 = np.zeros(f1.shape)
#         df2_1_db2 = np.zeros(f1.shape)
#         df2_1_db1 = np.zeros(f1.shape)
#         df2_1_df2 = np.zeros(f1.shape)

#         db2_1_df1 = np.zeros(f1.shape)
#         db2_1_db2 = np.zeros(f1.shape)
#         db2_1_db1 = np.zeros(f1.shape)
#         db2_1_df2 = np.zeros(f1.shape)

#         z1_05 = z2[:-1]
#         z2_05 = z1[1:]

#         gamma1_05 = gamma2[:-1]
#         gamma2_05 = gamma1[1:]

#         f1_05 = f2[:-1]
#         b2_05 = b1[1:]

#         db1_05, df2_05 = dreflect05(f1_05, b2_05, gamma1_05, gamma2_05, z1_05, z2_05)

#         # f1_1 = np.concatenate([f1inp, f2_05])
#         # b2_1 = np.concatenate([b1_05, b2rad])
#         df1_1_df2[1:] = df2_05[0]
#         df1_1_db1[1:] = df2_05[1]

#         db2_1_df2[:-1] = db1_05[0]
#         db2_1_db1[:-1] = db1_05[1]
#         pass

#     return reflect, reflect00, inputq


# def input_and_output_impedance(model, n=2**12):
#     """
#     Return the input and output impedances
#     """
#     state0 = model.state0.copy()
#     state0[:] = 0.0

#     qinp_impulse = 1.0
#     state0['pref'][:2] = model.inputq(qinp_impulse, state0['pinc'][:2])
#     control = model.control.copy()
#     control[:] = 0.0

#     times = np.arange(0, n) * model.dt

#     qinp = np.zeros(n)
#     pinp, pout = np.zeros(n), np.zeros(n)
#     qinp[0] = qinp_impulse
#     pinp[0] = state0['pinc'][0] + state0['pref'][0]
#     pout[0] = state0['pinc'][-1] + state0['pref'][-1]
#     for n in range(1, times.size):
#         model.set_ini_state(state0)
#         model.set_control(control)

#         state1, _ = model.solve_state1()
#         pinp[n] = state1['pinc'][0] + state1['pref'][0]
#         pout[n] = state1['pinc'][-1] + state1['pref'][-1]

#         state0 = state1

#     zinp = np.fft.fft(pinp) / np.fft.fft(qinp)
#     zout = np.fft.fft(pout) / np.fft.fft(qinp)
#     return zinp, zout