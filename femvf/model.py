"""
Contains the Model class that couples fluid and solid behaviour
"""

from os import path

import numpy as np
import dolfin as dfn
from petsc4py import PETSc
# import ufl

from . import newmark
from . import solids, fluids
from . import meshutils
from . import linalg
from .solverconst import DEFAULT_NEWTON_SOLVER_PRM

def load_fsi_model(solid_mesh, fluid_mesh, Solid=solids.KelvinVoigt, Fluid=fluids.Bernoulli, 
                   fsi_facet_labels=('pressure',), fixed_facet_labels=('fixed',), coupling='explicit'):
    """
    Factory function that loads a model based on input solid/fluid meshes.

    The function will delegate to specific loading routines depending on how the mesh
    is input etc.

    Parameters
    ----------
    """
    solid, fluid, fsi_verts = process_fsi(solid_mesh, fluid_mesh, 
        Solid=Solid, Fluid=Fluid, fsi_facet_labels=fsi_facet_labels, fixed_facet_labels=fixed_facet_labels)
    
    if coupling == 'explicit':
        model = ExplicitFSIModel(solid, fluid, fsi_verts)
    elif coupling == 'implicit':
        model = ImplicitFSIModel(solid, fluid, fsi_verts)

    return model

def load_fsai_model(solid_mesh, fluid_mesh, acoustic, Solid=solids.KelvinVoigt, Fluid=fluids.Bernoulli, 
                    fsi_facet_labels=('pressure',), fixed_facet_labels=('fixed',), coupling='explicit'):
    solid, fluid, fsi_verts = process_fsi(solid_mesh, fluid_mesh, 
        Solid=Solid, Fluid=Fluid, fsi_facet_labels=fsi_facet_labels, fixed_facet_labels=fixed_facet_labels)

    return FSAIModel(solid, fluid, acoustic, fsi_verts)

def process_fsi(solid_mesh, fluid_mesh, Solid=solids.KelvinVoigt, Fluid=fluids.Bernoulli, 
                fsi_facet_labels=('pressure',), fixed_facet_labels=('fixed',), coupling='explicit'):
    ## Load the solid
    mesh, facet_func, cell_func, facet_labels, cell_labels = None, None, None, None, None
    if isinstance(solid_mesh, str):
        ext = path.splitext(solid_mesh)[1]
        # if no extension is supplied, assume it's a fenics xml mesh
        if ext == '':
            ext = '.xml'

        if ext.lower() == '.xml':
            # The solid mesh is an xml file
            mesh, facet_func, cell_func, facet_labels, cell_labels = meshutils.load_fenics_xmlmesh(solid_mesh)
        else:
            raise ValueError(f"Can't process mesh {solid_mesh} with extension {ext}")
    solid = Solid(mesh, facet_func, facet_labels, cell_func, cell_labels, 
                  fsi_facet_labels, fixed_facet_labels)

    ## Process the fsi surface vertices to set the coupling between solid and fluid
    # Find vertices corresponding to the fsi facets
    fsi_facet_ids = [solid.facet_labels[name] for name in solid.fsi_facet_labels]
    fsi_edges = np.array([nedge for nedge, fedge in enumerate(solid.facet_func.array()) 
                                if fedge in set(fsi_facet_ids)])
    fsi_verts = meshutils.vertices_from_edges(fsi_edges, solid.mesh)
    fsi_coordinates = solid.mesh.coordinates()[fsi_verts]

    # Sort the fsi vertices from inferior to superior
    # NOTE: This only works for a 1D fluid mesh and isn't guaranteed if the VF surface is shaped strangely
    idx_sort = meshutils.sort_vertices_by_nearest_neighbours(fsi_coordinates)
    fsi_verts = fsi_verts[idx_sort]

    # Load the fluid
    fluid = None
    if fluid_mesh is None and issubclass(Fluid, fluids.QuasiSteady1DFluid):
        xfluid, yfluid = meshutils.streamwise1dmesh_from_edges(mesh, facet_func, [facet_labels[label] for label in fsi_facet_labels])
        fluid = Fluid(xfluid, yfluid)
    elif fluid_mesh is None and not issubclass(Fluid, fluids.QuasiSteady1DFluid):
        raise ValueError(f"`fluid_mesh` cannot be `None` if the fluid is not 1D")
    else:
        raise ValueError(f"This function does not yet support input fluid meshes.")
    return solid, fluid, fsi_verts


class FSIModel:
    """
    Represents a coupled system of a solid and a fluid model

    Parameters
    ----------
    solid, fluid

    Attributes
    ----------
    solid : Solid
        A solid model object
    fluid : Fluid
        A fluid model object

    fsi_verts : array_like
        A list of vertex indices on the pressure driven surface (FSI). These must be are ordered in 
        increasing streamwise direction since the fluid mesh numbering is ordered like that too.
    fsi_coordinates
    """
    def __init__(self, solid, fluid, fsi_verts):
        self.solid = solid
        self.fluid = fluid

        ## Specify state, controls, and properties
        self.state0 = linalg.concatenate(self.solid.state0, self.fluid.state0)
        self.state1 = linalg.concatenate(self.solid.state1, self.fluid.state1)

        # The control is just the subglottal and supraglottal pressures
        self.control = self.fluid.control[2:].copy()
        self.properties = linalg.concatenate(self.solid.properties, self.fluid.properties)

        self.fsi_verts = fsi_verts
        self.fsi_coordinates = self.solid.mesh.coordinates()[fsi_verts]
    
    ## These have to be defined to exchange data between domains
    def set_ini_solid_state(self, uva0):
        """
        Sets the state variables u, v, and a at the start of the step.

        Parameters
        ----------
        u0, v0, a0 : array_like
        """
        self.solid.set_ini_state(uva0)

        fl_control = self.fluid.get_control_vec()
        
        fsi_ref_config = self.solid.mesh.coordinates()[self.fsi_verts].reshape(-1)
        fsi_vdofs = self.solid.vert_to_vdof.reshape(-1, 2)[self.fsi_verts].reshape(-1).copy()
        fl_control = sl_state_to_fl_control(uva0, fl_control, fsi_ref_config, fsi_vdofs)

    def set_fin_solid_state(self, uva1):
        """
        Sets the displacement at the end of the time step.

        This could be an initial guess in the case of non-linear governing equations, or a solved
        state so that the non-linear form can be linearized for the given state.

        Parameters
        ----------
        uva1 : tuple of array_like
        """
        self.solid.set_fin_state(uva1)

        fl_control = self.fluid.control.copy()
        
        fsi_ref_config = self.solid.mesh.coordinates()[self.fsi_verts].reshape(-1)
        fsi_vdofs = self.solid.vert_to_vdof.reshape(-1, 2)[self.fsi_verts].reshape(-1).copy()
        fl_control = sl_state_to_fl_control(uva1, fl_control, fsi_ref_config, fsi_vdofs)
        self.fluid.set_control(fl_control)

    ## Methods for settings parameters of the model
    @property
    def dt(self):
        return self.solid.dt

    @dt.setter
    def dt(self, value):
        self.solid.dt = value
        self.fluid.dt = value

    def set_ini_state(self, state):
        self.set_ini_solid_state(state[:3])
        self.set_ini_fluid_state(state[3:])

    def set_fin_state(self, state):
        self.set_fin_solid_state(state[:3])
        self.set_fin_fluid_state(state[3:])
    
    def set_control(self, control):
        for key, value in control.items():
            self.fluid.control[key][:] = value

    def set_properties(self, props):
        self.solid.set_properties(props[:len(self.solid.properties.size)])
        self.fluid.set_properties(props[len(self.solid.properties.size):])

    # Additional more friendly method for setting parameters (use the above defined methods)
    def set_iter_params(self, state0, state1, control1, dt):
        """
        Sets all parameter values needed to integrate the model over a time step.

        The parameter specified at the final time (index 1) are initial guesses for the solution.
        One can then use a Newton method to iteratively solve for the actual final states.

        Unspecified parameters will have unchanged values.

        Parameters
        ----------
        state : tuple of array_like
            Initial and final solid states
        dt : float
        """
        self.set_ini_params(state0, control0)
        self.set_fin_params(state1, control1)
        self.dt = dt

    def set_params_fromfile(self, f, n, update_props=True):
        """
        Set all parameters needed to integrate the model from a recorded value.

        Iteration `n` is the implicit relation
        :math:`f^{n}(u_n, u_{n-1}, p)`
        that gives the displacement at index `n`, given the state at `n-1` and all additional
        parameters.

        Parameters
        ----------
        f : statefileutils.StateFile
        n : int
            Index of iteration to set
        """
        # Get data from the state file
        if update_props:
            self.set_properties(f.get_properties())

        state = f.get_state(n)
        control = None
        if f.variable_controls:
            control = f.get_control(n)
        else:
            control = f.get_control(0)
        self.set_fin_state(state)
        self.set_control(control)

    def set_iter_params_fromfile(self, f, n, update_props=True):
        """
        Set all parameters needed to integrate the model and an initial guess, based on a recorded
        iteration.

        Iteration `n` is the implicit relation
        :math:`f^{n}(u_n, u_{n-1}, p)`
        that gives the displacement at index `n`, given the state at `n-1` and all additional
        parameters.

        Parameters
        ----------
        statefile : statefileutils.StateFile
        n : int
            Index of iteration to set
        """
        # Get data from the state file
        if update_props:
            self.set_properties(f.get_properties())

        state0 = f.get_state(n-1)
        state1 = f.get_state(n)

        control0, control1 = None, None
        if f.variable_controls:
            control0 = f.get_control(n-1)
            control1 = f.get_control(n)
        else:
            control0 = f.get_control(0)
            control1 = f.get_control(0)

        dt = f.get_time(n) - f.get_time(n-1)

        # Assign the values to the model
        self.set_iter_params(state0, control0, state1, control1, dt)

    ## Solid / fluid interfacing functions
    def get_fsi_scalar_dofs(self):
        """
        Return dofs of the FSI interface on the solid and fluid

        This is needed to pass information between the two domains using conformal interfaces
        between them. Currently this is specifically made to work for the 1D fluid, so if you want
        to do something else, you'll have to think of how to generalized it.
        """
        sdof_solid = self.solid.vert_to_sdof[self.fsi_verts]
        sdof_fluid = np.arange(self.fsi_verts.size)

        return sdof_solid, sdof_fluid

    def get_fsi_vector_dofs(self):
        """
        Return dofs of the FSI interface on the solid and fluid

        This is needed to pass information between the two domains using conformal interfaces
        between them. Currently this is specifically made to work for the 1D fluid, so if you want
        to do something else, you'll have to think of how to generalized it.
        """
        vdof_solid = self.solid.vert_to_vdof.reshape(-1, 2)[self.fsi_verts].reshape(-1).copy()
        vdof_fluid = np.arange(vdof_solid.size)

        return vdof_solid, vdof_fluid

    # @staticmethod
    def _ignore_nonvector(map_fsi_func):
        """Decorator so that floats are ignored in the map_ functions"""
        def wrapped_map_fsi_func(self, x):
            if isinstance(x, (float, int)):
                return x
            else:
                return map_fsi_func(self, x)

        return wrapped_map_fsi_func

    @_ignore_nonvector
    def map_fsi_scalar_from_solid_to_fluid(self, solid_scalar):
        sdof_solid, sdof_fluid = self.get_fsi_scalar_dofs()

        fluid_scalar = self.fluid.get_surf_scalar()
        fluid_scalar[sdof_fluid] = solid_scalar[sdof_solid]
        return fluid_scalar

    @_ignore_nonvector
    def map_fsi_vector_from_solid_to_fluid(self, solid_vector):
        vdof_solid, vdof_fluid = self.get_fsi_vector_dofs()

        fluid_vector = self.fluid.control['usurf'].copy()
        fluid_vector[:] = 0.0
        fluid_vector[vdof_fluid] = solid_vector[vdof_solid]
        return fluid_vector

    @_ignore_nonvector
    def map_fsi_scalar_from_fluid_to_solid(self, fluid_scalar):
        sdof_solid, sdof_fluid = self.get_fsi_scalar_dofs()

        solid_scalar = dfn.Function(self.solid.scalar_fspace).vector()
        solid_scalar[sdof_solid] = fluid_scalar[sdof_fluid]
        return solid_scalar

    @_ignore_nonvector
    def map_fsi_vector_from_fluid_to_solid(self, fluid_vector):
        vdof_solid, vdof_fluid = self.get_fsi_vector_dofs()

        solid_vector = dfn.Function(self.solid.vector_fspace).vector()
        solid_vector[vdof_solid] = fluid_vector[vdof_fluid]
        return solid_vector

    ## Mesh functions
    def get_ref_config(self):
        """
        Returns the current configuration of the body.

        Coordinates are in vertex order.

        Returns
        -------
        array_like
            An array of mesh coordinate point ordered with increasing vertices.
        """
        return self.solid.mesh.coordinates()

    def get_cur_config(self):
        """
        Returns the current configuration of the body.

        Coordinates are in vertex order.

        Returns
        -------
        array_like
            An array of mesh coordinate point ordered with increasing vertices.
        """
        displacement = self.solid.u0.vector()[self.solid.vert_to_vdof].reshape(-1, 2)
        return self.solid.mesh.coordinates() + displacement

    # Fluid 'residuals'
    # These are designed for quasi-steady, Bernoulli fluids where you don't need any iterative
    # methods to solve.
    def solve_dqp1_du1_solid(self, adjoint=False):
        return self.fluid.solve_dqp1_du1_solid(self, adjoint)

    def solve_dqp0_du0_solid(self, adjoint=False):
        return self.fluid.solve_dqp0_du0_solid(self, adjoint)

    ## Convenience functions
    def get_triangulation(self, config='ref'):
        from matplotlib.tri import Triangulation
        coords = None
        if config == 'ref':
            coords = self.get_ref_config()
        elif config == 'cur':
            coords = self.get_cur_config()
        else:
            raise ValueError(f"`config` must be one of 'ref' or 'cur'.")

        cells = self.solid.mesh.cells()
        return Triangulation(coords[:, 0], coords[:, 1], triangles=cells)

    ## Methods for getting vectors
    def get_state_vec(self):
        return linalg.concatenate(self.solid.get_state_vec(), self.fluid.get_state_vec())

    def get_control_vec(self):
        ret = self.control.copy()
        ret.set(0.0)
        return ret

    def get_properties_vec(self):
        return linalg.concatenate(self.solid.get_properties_vec(), self.fluid.get_properties_vec())

    ## Solver functions
    # Specific models must implement these
    def res(self):
        raise NotImplementedError
    
    # Forward solver methods
    def solve_state1(self, ini_state, newton_solver_prm=None):
        """
        Solve for the final state given an initial guess
        """
        raise NotImplementedError

    def solve_dres_dstate1(self, b):
        """
        Solve, dF/du x = f
        """
        raise NotImplementedError

    # Adjoint solver methods
    def solve_dres_dstate1_adj(self, b):
        """
        Solve, dF/du^T x = f
        """
        raise NotImplementedError

    def apply_dres_dstate0_adj(self, x):
        raise NotImplementedError

    def apply_dres_dp_adj(self, x):
        raise NotImplementedError

    def apply_dres_dcontrol_adj(self, x):
        raise NotImplementedError

class ExplicitFSIModel(FSIModel):
    ## These setting functions ensure explicit coupling between the domains
    def set_ini_fluid_state(self, qp0):
        self.fluid.set_ini_state(qp0)

        # Note that setting the initial fluid state sets the final driving pressure
        # for explicit coupling
        p0_solid = self.map_fsi_scalar_from_fluid_to_solid(qp0[1])
        control = linalg.BlockVec((p0_solid,), self.solid.control.keys)
        self.solid.set_control(control)

    def set_fin_fluid_state(self, qp1):
        self.fluid.set_fin_state(qp1)

    ## Solver functions
    def res(self):
        """
        Return the residual vector, F
        """
        dt = self.solid.dt
        u1, v1, a1 = self.solid.u1.vector(), self.solid.v1.vector(), self.solid.a1.vector()
        u0, v0, a0 = self.solid.u0.vector(), self.solid.v0.vector(), self.solid.a0.vector()
        res = self.get_state_vec()

        res['u'][:] = dfn.assemble(self.solid.forms['form.un.f1'])
        self.solid.bc_base.apply(res['u'])
        res['v'][:] = v1 - newmark.newmark_v(u1, u0, v0, a0, dt)
        res['a'][:] = a1 - newmark.newmark_a(u1, u0, v0, a0, dt)

        qp, *_ = self.fluid.solve_qp1()
        res['q'][:] = self.fluid.state1['q'] - qp['q']
        res['p'][:] = self.fluid.state1['p'] - qp['p']
        return res
    
    # Forward solver methods
    def solve_state1(self, ini_state, newton_solver_prm=None):
        """
        Solve for the final state given an initial guess
        """
        # Update form coefficients and initial guess
        self.set_fin_state(ini_state)

        if newton_solver_prm is None:
            newton_solver_prm = DEFAULT_NEWTON_SOLVER_PRM

        uva1, _ = self.solid.solve_state1(newton_solver_prm)

        self.set_fin_solid_state(uva1)
        qp1, fluid_info = self.fluid.solve_qp1()

        step_info = {'fluid_info': fluid_info}

        return linalg.concatenate(uva1, qp1), step_info

    def solve_dres_dstate1(self, b):
        """
        Solve, dF/du x = f
        """
        dt = self.solid.dt
        x = self.get_state_vec()

        solid = self.solid

        dfu1_du1 = dfn.assemble(solid.df1_du1)
        dfv2_du2 = 0 - newmark.newmark_v_du1(dt)
        dfa2_du2 = 0 - newmark.newmark_a_du1(dt)

        dq_du, dp_du = self.solve_dqp1_du1_solid(adjoint=False)
        dfq2_du2 = 0 - dq_du
        dfp2_du2 = 0 - dp_du

        self.solid.bc_base.apply(dfu1_du1)
        dfn.solve(dfu1_du1, x['u'], b['u'], 'petsc')
        x['v'][:] = b['v'] - dfv2_du2*x['u']
        x['a'][:] = b['a'] - dfa2_du2*x['u']

        # qp1, fluid_info = self.fluid.solve_qp1()

        x['q'][:] = b['q'] - dfq2_du2.inner(x['u'])
        x['p'][:] = b['p'] - dfn.PETScVector(dfp2_du2*x['u'].vec())
        return x

    # Adjoint solver methods
    def solve_dres_dstate1_adj(self, b):
        """
        Solve, dF/du^T x = f
        """
        ## Assemble sensitivity matrices
        dq_du, dp_du = self.solve_dqp1_du1_solid(adjoint=True)
        dfq2_du2 = 0 - dq_du
        dfp2_du2 = 0 - dp_du

        ## Do the linear algebra that solves for x

        x = self.get_state_vec()
        x_qp = x[3:]
        x_qp[:] = b[-2:]

        x_uva = x[:3]

        _adj_p = dfp2_du2.getVecRight()
        _adj_p[:] = x_qp['p']

        b_uva = b[:3].copy()
        b_uva['u'] -= (dfq2_du2*x_qp['q'] + dfn.PETScVector(dfp2_du2*_adj_p))
        x_uva[:] = self.solid.solve_dres_dstate1_adj(b_uva)

        return x

    def apply_dres_dstate0_adj(self, x):
        dfu2_dp1 = dfn.assemble(self.solid.forms['form.bi.df1_dp1_adj'])

        solid_dofs, fluid_dofs = self.get_fsi_scalar_dofs()
        dfu2_dp1 = dfn.as_backend_type(dfu2_dp1).mat()
        dfu2_dp1 = linalg.reorder_mat_rows(dfu2_dp1, solid_dofs, fluid_dofs, fluid_dofs.size)
        matvec_adj_p_rhs = dfu2_dp1*dfn.as_backend_type(x['u']).vec()

        b = self.get_state_vec()
        # Set uva blocks of b
        b[:3] = self.solid.apply_dres_dstate0_adj(x[:3])
        b['q'][:] = 0.0
        b['p'][:] = matvec_adj_p_rhs
        return b
    
    # @profile
    def apply_dres_dp_adj(self, x):
        # bsolid = self.solid.get_properties_vec()
        # bfluid = self.fluid.get_properties_vec()
        bsolid = self.solid.apply_dres_dp_adj(x[:3])
        bfluid = self.fluid.apply_dres_dp_adj(x[3:])
        return linalg.concatenate(bsolid, bfluid)

    def apply_dres_dcontrol_adj(self, x):
        # b = self.get_properties_vec()
        pass

class ImplicitFSIModel(FSIModel):
    ## These must be defined to properly exchange the forcing data between the solid and domains
    def set_ini_fluid_state(self, qp0):
        self.fluid.set_ini_state(qp0)

        p0_solid = self.map_fsi_scalar_from_fluid_to_solid(qp0[1])
        control = linalg.BlockVec((p0_solid,), self.solid.control.keys)

    def set_fin_fluid_state(self, qp1):
        self.fluid.set_fin_state(qp1)

        p1_solid = self.map_fsi_scalar_from_fluid_to_solid(qp1[1])
        control = linalg.BlockVec((p1_solid,), self.solid.control.keys)
        self.solid.set_control(control)

    ## Forward solver methods
    def res(self):
        """
        Return the residual vector, F
        """
        dt = self.solid.dt
        u1, v1, a1 = self.solid.u1.vector(), self.solid.v1.vector(), self.solid.a1.vector()
        u0, v0, a0 = self.solid.u0.vector(), self.solid.v0.vector(), self.solid.a0.vector()
        res = self.get_state_vec()

        res['u'][:] = dfn.assemble(self.solid.forms['form.un.f1'])
        self.solid.bc_base.apply(res['u'])
        res['v'][:] = v1 - newmark.newmark_v(u1, u0, v0, a0, dt)
        res['a'][:] = a1 - newmark.newmark_a(u1, u0, v0, a0, dt)

        qp, *_ = self.fluid.solve_qp1()
        res['q'][:] = self.fluid.state1['q'] - qp['q']
        res['p'][:] = self.fluid.state1['p'] - qp['p']
        return res
    
    def solve_state1(self, ini_state, newton_solver_prm=None):
        """
        Solve for the final state given an initial guess 
        
        This uses a fixed-point iteration where the solid is solved, then the fluid and so-on.
        """
        if newton_solver_prm is None:
            newton_solver_prm = DEFAULT_NEWTON_SOLVER_PRM

        # Set initial guesses for the states at the next time
        # uva1 = solid.get_state_vec()
        uva1 = ini_state[:3].copy()
        qp1 = ini_state[3:].copy()


        # Calculate the initial residual
        self.set_fin_solid_state(uva1)
        self.set_fin_fluid_state(qp1)
        res0 = dfn.assemble(self.solid.f1)
        self.solid.bc_base.apply(res0)

        # Set tolerances for the fixed point iterations
        fluid_info = None
        nit, max_nit = 0, 10
        abs_tol, rel_tol = newton_solver_prm['absolute_tolerance'], newton_solver_prm['relative_tolerance']
        abs_err_prev, abs_err, rel_err = 1.0, np.inf, np.inf
        while abs_err > abs_tol and rel_err > rel_tol and nit < max_nit:
            # Solve the solid with the previous iteration's fluid pressures
            uva1, _ = self.solid.solve_state1(newton_solver_prm)

            # Compute new fluid pressures for the updated solid position
            self.set_fin_solid_state(uva1)
            qp1, fluid_info = self.fluid.solve_qp1()
            self.set_fin_fluid_state(qp1)

            # Calculate the error in the solid residual with the updated pressures
            res = dfn.assemble(self.solid.f1)
            self.solid.bc_base.apply(res)

            abs_err = res.norm('l2')
            rel_err = abs_err/abs_err_prev

            nit += 1
        res = dfn.assemble(self.solid.forms['form.un.f1'])
        self.solid.bc_base.apply(res)

        step_info = {'fluid_info': fluid_info,
                     'nit': nit, 'abs_err': abs_err, 'rel_err': rel_err}

        return linalg.concatenate(uva1, qp1), step_info

    def solve_dres_dstate1(self, b):
        """
        Solve, dF/du x = f
        """
        dt = self.solid.dt
        x = self.get_state_vec()

        solid = self.solid

        dfu1_du1 = dfn.assemble(solid.df1_du1)
        dfv2_du2 = 0 - newmark.newmark_v_du1(dt)
        dfa2_du2 = 0 - newmark.newmark_a_du1(dt)

        dq_du, dp_du = self.solve_dqp1_du1_solid(adjoint=False)
        dfq2_du2 = 0 - dq_du
        dfp2_du2 = 0 - dp_du

        self.solid.bc_base.apply(dfu1_du1)
        dfn.solve(dfu1_du1, x['u'], b['u'], 'petsc')
        x['v'][:] = b['v'] - dfv2_du2*x['u']
        x['a'][:] = b['a'] - dfa2_du2*x['u']

        # qp1, fluid_info = self.fluid.solve_qp1()
        x['q'][:] = b['q'] - dfq2_du2.inner(x['u'])
        x['p'][:] = b['p'] - dfn.PETScVector(dfp2_du2*x['u'].vec())
        return x

    ## Adjoint solver methods
    def solve_dres_dstate1_adj(self, b):
        """
        Solve, dF/du^T x = f
        """
        ## Assemble sensitivity matrices
        # self.set_iter_params(**it_params)
        dt = self.solid.dt

        dfu2_du2 = self.solid.cached_form_assemblers['bilin.df1_du1_adj'].assemble()
        dfv2_du2 = 0 - newmark.newmark_v_du1(dt)
        dfa2_du2 = 0 - newmark.newmark_a_du1(dt)
        dfu2_dp2 = dfn.assemble(self.solid.forms['form.bi.df1_dp1_adj'])

        # map dfu2_dp2 to have p on the fluid domain
        solid_dofs, fluid_dofs = self.get_fsi_scalar_dofs()
        dfu2_dp2 = dfn.as_backend_type(dfu2_dp2).mat()
        dfu2_dp2 = linalg.reorder_mat_rows(dfu2_dp2, solid_dofs, fluid_dofs, self.fluid.state1['p'].size)

        dq_du, dp_du = self.solve_dqp1_du1_solid(adjoint=True)
        dfq2_du2 = 0 - dq_du
        dfp2_du2 = 0 - dp_du

        ## Do the linear algebra that solves for the adjoint states
        adj_uva = self.solid.get_state_vec()
        adj_qp = self.fluid.get_state_vec()

        adj_u_rhs, adj_v_rhs, adj_a_rhs, adj_q_rhs, adj_p_rhs = b

        # adjoint states for v, a, and q are explicit so we can solve for them
        self.solid.bc_base.apply(adj_v_rhs)
        adj_uva['v'][:] = adj_v_rhs

        self.solid.bc_base.apply(adj_a_rhs)
        adj_uva['a'][:] = adj_a_rhs

        # TODO: how to apply fluid boundary conditions in a generic way?
        adj_qp['q'][:] = adj_q_rhs

        adj_u_rhs -= dfv2_du2*adj_uva['v'] + dfa2_du2*adj_uva['a'] + dfq2_du2*adj_qp['q']

        bc_dofs = np.array(list(self.solid.bc_base.get_boundary_values().keys()), dtype=np.int32)
        self.solid.bc_base.apply(dfu2_du2, adj_u_rhs)
        dfp2_du2.zeroRows(bc_dofs, diag=0.0)
        # self.solid.bc_base.zero_columns(dfu2_du2, adj_u_rhs.copy(), diagonal_value=1.0)

        # solve the coupled system for pressure and displacement residuals
        dfu2_du2_mat = dfn.as_backend_type(dfu2_du2).mat()
        blocks = [[dfu2_du2_mat, dfp2_du2], [dfu2_dp2, 1.0]]

        dfup2_dup2 = linalg.form_block_matrix(blocks)
        adj_up, rhs = dfup2_dup2.getVecs()

        # calculate rhs vectors
        rhs[:adj_u_rhs.size()] = adj_u_rhs
        rhs[adj_u_rhs.size():] = adj_p_rhs

        # Solve the block linear system with LU factorization
        ksp = PETSc.KSP().create()
        ksp.setType(ksp.Type.PREONLY)

        pc = ksp.getPC()
        pc.setType(pc.Type.LU)

        ksp.setOperators(dfup2_dup2)
        ksp.solve(rhs, adj_up)

        adj_uva['u'][:] = adj_up[:adj_u_rhs.size()]
        adj_qp['p'][:] = adj_up[adj_u_rhs.size():]

        return linalg.concatenate(adj_uva, adj_qp)

    def apply_dres_dstate0_adj(self, x):
        adj_u2, adj_v2, adj_a2, adj_q2, adj_p2 = x

        ## Assemble sensitivity matrices
        # dt2 = it_params2['dt']
        solid = self.solid
        dt = self.dt
        # model.set_iter_params(**it_params2)

        dfu2_du1 = solid.cached_form_assemblers['bilin.df1_du0_adj'].assemble()
        dfu2_dv1 = solid.cached_form_assemblers['bilin.df1_dv0_adj'].assemble()
        dfu2_da1 = solid.cached_form_assemblers['bilin.df1_da0_adj'].assemble()

        dfv2_du1 = 0 - newmark.newmark_v_du0(dt)
        dfv2_dv1 = 0 - newmark.newmark_v_dv0(dt)
        dfv2_da1 = 0 - newmark.newmark_v_da0(dt)
        dfa2_du1 = 0 - newmark.newmark_a_du0(dt)
        dfa2_dv1 = 0 - newmark.newmark_a_dv0(dt)
        dfa2_da1 = 0 - newmark.newmark_a_da0(dt)

        ## Do the matrix vector multiplication that gets the RHS for the adjoint equations
        # Allocate a vector the for fluid side mat-vec multiplication
        b = x.copy()
        b['u'][:] = (dfu2_du1*adj_u2 + dfv2_du1*adj_v2 + dfa2_du1*adj_a2)
        b['v'][:] = (dfu2_dv1*adj_u2 + dfv2_dv1*adj_v2 + dfa2_dv1*adj_a2)
        b['a'][:] = (dfu2_da1*adj_u2 + dfv2_da1*adj_v2 + dfa2_da1*adj_a2)
        b['q'][:] = 0
        b['p'][:] = 0

        return b

    def apply_dres_dp_adj(self, x):
        # bsolid = self.solid.get_properties_vec()
        # bfluid = self.fluid.get_properties_vec()
        bsolid = self.solid.apply_dres_dp_adj(x[:3])
        bfluid = self.fluid.apply_dres_dp_adj(x[3:])
        return linalg.concatenate(bsolid, bfluid)

    def apply_dres_dcontrol_adj(self, x):
        # b = self.get_properties_vec()
        pass

class FSAIModel(FSIModel):
    """
    Represents a fluid-structure-acoustic interaction system
    """
    def __init__(self, solid, fluid, acoustic, fsi_verts):
        self.solid = solid
        self.fluid = fluid
        self.acoustic = acoustic
        
        state = linalg.concatenate(solid.get_state_vec(), fluid.get_state_vec(), acoustic.get_state_vec())
        self.state0 = state
        self.state1 = state.copy()

        control = linalg.BlockVec((np.array([1.0]),), ('psub',))
        self.control = control.copy()

        self.properties = linalg.concatenate(
            solid.get_properties_vec(), fluid.get_properties_vec(), acoustic.get_properties_vec())

        self._dt = 1.0

        self.fsi_verts = fsi_verts

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        #  the acoustic domain time step is ignored for WRA
        for domain in (self.solid, self.fluid):
            domain.dt = value

    ## Parameter setting methods
    def set_ini_state(self, state):
        sl_nblock = len(self.solid.state0.size)
        fl_nblock = len(self.fluid.state0.size)
        ac_nblock = len(self.acoustic.state0.size)

        sl_state = state[:sl_nblock]
        fl_state = state[sl_nblock:sl_nblock+fl_nblock]
        ac_state = state[sl_nblock+fl_nblock:sl_nblock+fl_nblock+ac_nblock]

        self.set_ini_solid_state(sl_state)
        self.set_ini_fluid_state(fl_state)
        self.set_ini_acoustic_state(ac_state)

    def set_fin_state(self, state):
        sl_nblock = len(self.solid.state0.size)
        fl_nblock = len(self.fluid.state0.size)
        ac_nblock = len(self.acoustic.state0.size)

        sl_state = state[:sl_nblock]
        fl_state = state[sl_nblock:sl_nblock+fl_nblock]
        ac_state = state[sl_nblock+fl_nblock:sl_nblock+fl_nblock+ac_nblock]

        self.set_fin_solid_state(sl_state)
        self.set_fin_fluid_state(fl_state)
        self.set_fin_acoustic_state(ac_state)

    def set_control(self, control):
        fl_control = self.fluid.control.copy()
        fl_control['psub'][:] = control['psub']
        self.fluid.set_control(fl_control)

    def set_properties(self, props):
        sl_nblock = len(self.solid.properties.size)
        fl_nblock = len(self.fluid.properties.size)
        ac_nblock = len(self.acoustic.properties.size)

        sl_props = props[:sl_nblock]
        fl_props = props[sl_nblock:sl_nblock+fl_nblock]
        ac_props = props[sl_nblock+fl_nblock:sl_nblock+fl_nblock+ac_nblock]

        self.solid.set_properties(sl_props)
        self.fluid.set_properties(fl_props)
        self.acoustic.set_properties(ac_props)

    ## Coupling methods
    def set_ini_solid_state(self, sl_state0):
        self.solid.set_ini_state(sl_state0)
    
    def set_ini_fluid_state(self, fl_state0):
        self.fluid.set_ini_state(fl_state0)

        # for explicit coupling
        sl_control = self.solid.control.copy()
        fsi_sdofs = self.solid.vert_to_sdof[self.fsi_verts].copy()
        sl_control = fl_state_to_sl_control(fl_state0, sl_control, fsi_sdofs)
        self.solid.set_control(sl_control)
    
    def set_ini_acoustic_state(self, ac_state0):
        self.acoustic.set_ini_state(ac_state0)

    def set_fin_solid_state(self, sl_state1):
        self.solid.set_fin_state(sl_state1)

        fl_control = self.fluid.control.copy()
        
        fsi_ref_config = self.solid.mesh.coordinates()[self.fsi_verts].reshape(-1)
        fsi_vdofs = self.solid.vert_to_vdof.reshape(-1, 2)[self.fsi_verts].reshape(-1).copy()
        fl_control = sl_state_to_fl_control(sl_state1, fl_control, fsi_ref_config, fsi_vdofs)
        
        self.fluid.set_control(fl_control)

    def set_fin_fluid_state(self, fl_state1):
        self.fluid.set_fin_state(fl_state1)

        ac_control = fl_state_to_ac_control(fl_state1, self.acoustic.control.copy())
        self.acoustic.set_control(ac_control)
    
    def set_fin_acoustic_state(self, ac_state1):
        self.acoustic.set_fin_state(ac_state1)

        control = ac_state_to_fl_control(ac_state1, self.fluid.control.copy())
        self.fluid.set_control(control)

    ## Empty parameter vectors
    def get_control_vec(self):
        ret = self.control.copy()
        ret.set(0.0)
        return ret

    def get_state_vec(self):
        ret = self.state0.copy()
        ret.set(0.0)
        return ret

    def get_properties_vec(self, set_default=True):
        ret = self.properties.copy()
        if not set_default:
            ret.set(0.0)
        return ret

    ## Solver methods
    def res(self):
        res_sl = self.solid.res()
        res_fl = self.fluid.res()
        res_ac = self.acoustic.res()
        return linalg.concatenate(res_sl, res_fl, res_ac)

    def solve_state1(self, ini_state, newton_solver_prm=None):
        if newton_solver_prm is None:
            newton_solver_prm = DEFAULT_NEWTON_SOLVER_PRM

        # Update form coefficients and initial guess
        self.set_fin_state(ini_state)

        sl_state1, _ = self.solid.solve_state1(newton_solver_prm)

        self.set_fin_solid_state(sl_state1)

        fluid_info, num_it = None, 0 
        fl_state1, ac_state1 = None, None
        abserr_max, relerr_max = newton_solver_prm['absolute_tolerance'], newton_solver_prm['relative_tolerance']
        abserr_prev, abserr, relerr = 0.0, 1.0, 1.0

        ## Solve the coupled fluid/acoustic system with a fixed point iteration:
        # solve fluids -> solve acoustics with updated fluids -> compute error in fluid and repeat if necessary
        fl_state1, fluid_info = self.fluid.solve_qp1()
        while abserr > abserr_max and relerr > relerr_max and num_it < 15:
            self.set_fin_fluid_state(fl_state1)

            ac_state1, _ = self.acoustic.solve_state1()
            self.set_fin_acoustic_state(ac_state1)
            print(linalg.dot(ac_state1, ac_state1))

            # compute the error/residual. The acoustic equations were solved previously so the residual is 
            # due only to the fluid
            fl_state1_new = self.fluid.solve_qp1()[0]
            fl_res = fl_state1 - fl_state1_new
            fl_state1 = fl_state1_new

            abserr = linalg.dot(fl_res, fl_res)
            relerr = abs(abserr - abserr_prev) / abserr
            abserr_prev = abserr
            print(abserr, relerr)
            # print(fl_res['q'], fl_res['p'])
            num_it += 1
        print("Iteration finished. Stats:")
        # print(abserr, abserr_max)
        # print(relerr, relerr_max)
        # print(num_it, '\n')

        step_info = {'fluid_info': fluid_info, 'num_it': num_it, 'abserr': abserr, 'relerr': relerr}

        return linalg.concatenate(sl_state1, fl_state1, ac_state1), step_info

    def _form_dflac_dflac(self):
        b = self.state0

        ## Solve the coupled fluid/acoustic system 
        # First compute some sensitivities that are needed
        *_, dq_dpsup, dp_dpsup = self.fluid.flow_sensitivity(*self.fluid.control.vecs, self.fluid.properties)
        
        # solve the coupled system for pressure and acoustic residuals
        dfq_dq = 1.0
        dfp_dp = 1.0
        dfpinc_dpinc = PETSc.Mat().createAIJ((b['pinc'].size, b['pinc'].size), nnz=b['pinc'].size)
        diag = PETSc.Vec().createSeq(b['pinc'].size)
        diag.set(1.0)
        dfpinc_dpinc.setDiagonal(diag)
        dfpref_dpref = 1.0

        # dfluid / dacoustic
        dfq_dpref = PETSc.Mat().createAIJ((b['q'].size, b['pref'].size), nnz=1)
        dfq_dpsup = -dq_dpsup
        dpsup_dpref = 1.0 # Supraglottal pressure is equal to very first reflected pressure
        dfq_dpref.setValue(0, 0, dfq_dpsup*dpsup_dpref)

        dfp_dpref = PETSc.Mat().createAIJ((b['p'].size, b['pref'].size), nnz=b['p'].size)
        dfp_dpsup = -dp_dpsup
        dfp_dpref.setValues(np.arange(b['p'].size, dtype=np.int32), 0, dfp_dpsup*dpsup_dpref)

        # dacoustic / dfluid
        dfpref_dq = PETSc.Mat().createAIJ((b['pref'].size, b['q'].size), nnz=2)
        dcontrol = self.acoustic.get_control_vec()
        dcontrol.set(0.0)
        dcontrol['qin'][:] = 1.0
        dfpref_dqin = self.acoustic.apply_dres_dcontrol(dcontrol)['pref'][:2]
        dqin_dq = 1.0
        dfpref_dq.setValues(np.array([0, 1], dtype=np.int32), 0, dfpref_dqin*dqin_dq)

        for mat in (dfq_dpref, dfp_dpref, dfpref_dq):
            mat.assemble()
        # breakpoint()

        blocks = [[   dfq_dq,    0.0,          0.0,    dfq_dpref], 
                  [      0.0, dfp_dp,          0.0,    dfp_dpref],
                  [      0.0,    0.0, dfpinc_dpinc,          0.0],
                  [dfpref_dq,    0.0,          0.0, dfpref_dpref]]

        A = linalg.form_block_matrix(blocks)
        # breakpoint()
        return A

    def solve_dres_dstate1(self, b):
        """
        Solve, dF/du x = f
        """
        x = self.get_state_vec()
        ## Assmeble any needed sensitivity matrices
        dq_du, dp_du = self.solve_dqp1_du1_solid(adjoint=False)
        dfq2_du2 = 0 - dq_du
        dfp2_du2 = 0 - dp_du

        ## Solve the solid system portion
        x[:3] = self.solid.solve_dres_dstate1(b[:3])

        ## Solve the coupled fluid/acoustic system 
        A = self._form_dflac_dflac()
        adj_z, rhs = A.getVecs()

        # Get only the q, p, pinc, pref (fluid/acoustic) portions of the residual
        b_flac = b[3:].copy()
        b_flac['q'] -= (dfq2_du2*x['u']).sum()
        b_flac['p'] -= dfp2_du2*x['u'].vec()
        rhs[:] = b_flac.to_ndarray()

        ksp = PETSc.KSP().create()
        ksp.setType(ksp.Type.PREONLY)

        pc = ksp.getPC()
        pc.setType(pc.Type.LU)

        ksp.setOperators(A)
        ksp.solve(rhs, adj_z)

        x[3:].set_vec(adj_z)
        # print(linalg.dot(x[3:], x[3:]))

        return x

    # Adjoint solver methods
    def solve_dres_dstate1_adj(self, b):
        """
        Solve, dF/du^T x = f
        """
        x = self.get_state_vec()

        ## Solve the coupled fluid/acoustic system first
        A = self._form_dflac_dflac()
        adj_z, rhs = A.getVecs()

        # Get only the q, p, pinc, pref (fluid/acoustic) portions of the residual
        rhs[:] = b[3:].to_ndarray()

        ksp = PETSc.KSP().create()
        ksp.setType(ksp.Type.PREONLY)

        pc = ksp.getPC()
        pc.setType(pc.Type.LU)

        ksp.setOperators(A)
        ksp.solveTranspose(rhs, adj_z)

        x[3:].set_vec(adj_z)

        ## Assemble sensitivity matrices
        dq_du, dp_du = self.solve_dqp1_du1_solid(adjoint=True)
        dfq2_du2 = 0 - dq_du
        dfp2_du2 = 0 - dp_du

        _adj_p = dfp2_du2.getVecRight()
        _adj_p[:] = x['p']

        b_uva = b[:3].copy()
        b_uva['u'] -= (dfq2_du2*x['q'] + dfn.PETScVector(dfp2_du2*_adj_p))

        x_uva = x[:3]
        x_uva[:] = self.solid.solve_dres_dstate1_adj(b_uva)

        return x

    def apply_dres_dstate0_adj(self, x):
        dfu2_dp1 = dfn.assemble(self.solid.forms['form.bi.df1_dp1_adj'])

        solid_dofs, fluid_dofs = self.get_fsi_scalar_dofs()
        dfu2_dp1 = dfn.as_backend_type(dfu2_dp1).mat()
        dfu2_dp1 = linalg.reorder_mat_rows(dfu2_dp1, solid_dofs, fluid_dofs, fluid_dofs.size)
        matvec_adj_p_rhs = dfu2_dp1*dfn.as_backend_type(x['u']).vec()

        b = self.get_state_vec()
        # Set uva blocks of b
        b[:3] = self.solid.apply_dres_dstate0_adj(x[:3])
        b['q'][:] = 0.0
        b['p'][:] = matvec_adj_p_rhs

        # Set acoustics state blocks of b
        b[-2:] = self.acoustic.apply_dres_dstate0_adj(x[-2:])
        return b
        
    def apply_dres_dp_adj(self, x):
        bsl = self.solid.apply_dres_dp_adj(x[:3])
        bfl = self.fluid.apply_dres_dp_adj(x[3:5])
        bac = self.acoustic.apply_dres_dp_adj(x[5:])
        return linalg.concatenate(bsl, bfl, bac)

    def apply_dres_dcontrol_adj(self, x):
        return self.get_control_vec()

class SolidModel:
    def __init__(self, solid):
        self.solid = solid

        self.state0 = self.solid.state0
        self.state1 = self.solid.state1

        self.control = self.solid.control

        self.properties = self.solid.properties

    @property
    def dt(self):
        return self.solid.dt

    @dt.setter
    def dt(self, value):
        self.solid.dt = value

    def set_ini_state(self, state):
        self.solid.set_ini_state(state)

    def set_fin_state(self, state):
        self.solid.set_fin_state(state)
    
    def set_control(self, control):
        self.solid.set_control(control)

    def set_properties(self, props):
        self.solid.set_properties(props)

    def solve_state1(self, ini_state, newton_solver_prm=None):
        """
        Solve for the final state given an initial guess
        """
        """
        Solve for the final state given an initial guess
        """
        solid = self.solid
        dt = solid.dt
        uva0 = solid.state0

        uva1 = solid.get_state_vec()

        # Update form coefficients and initial guess
        self.set_fin_state(ini_state)

        if newton_solver_prm is None:
            newton_solver_prm = DEFAULT_NEWTON_SOLVER_PRM

        dfn.solve(solid.f1 == 0, solid.u1, bcs=solid.bc_base, J=solid.df1_du1,
                  solver_parameters={"newton_solver": newton_solver_prm})

        res = dfn.assemble(self.solid.forms['form.un.f1'])
        self.solid.bc_base.apply(res)

        uva1['u'][:] = solid.u1.vector()
        uva1['v'][:] = newmark.newmark_v(uva1['u'], *uva0, dt)
        uva1['a'][:] = newmark.newmark_a(uva1['u'], *uva0, dt)

        self.set_fin_state(uva1)
        step_info = {}

        return uva1, step_info

    def get_state_vec(self):
        return self.solid.get_state_vec()

    def get_control_vec(self):
        return self.solid.get_control_vec()

    def get_properties_vec(self):
        return self.solid.get_properties_vec()


def sl_state_to_fl_control(sl_state, fl_control, fsi_ref_config, fsi_vdofs):
    u1_fluid = fsi_ref_config + sl_state['u'][fsi_vdofs]
    v1_fluid = sl_state['v'][fsi_vdofs]

    fl_control['usurf'][:] = u1_fluid
    fl_control['vsurf'][:] = v1_fluid
    return fl_control

def fl_state_to_sl_control(fl_state, sl_control, fsi_sdofs):
    p_ = np.zeros(sl_control['p'].size())
    p_[fsi_sdofs] = fl_state['p']
    
    sl_control['p'][:] = p_
    return sl_control

def ac_state_to_fl_control(ac_state, fl_control):
    fl_control['psup'] = ac_state['pref'][0]
    return fl_control

def fl_state_to_ac_control(fl_state, ac_control):
    ac_control['qin'][:] = fl_state['q']
    return ac_control