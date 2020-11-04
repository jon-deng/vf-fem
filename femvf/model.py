"""
Contains the Model class that couples fluid and solid behaviour
"""

from os import path

import numpy as np
import dolfin as dfn
from petsc4py import PETSc
# import ufl

from . import solids, fluids
from . import meshutils
from . import linalg

DEFAULT_NEWTON_SOLVER_PRM = {
    'linear_solver': 'petsc',
    'absolute_tolerance': 1e-7,
    'relative_tolerance': 1e-9}

def load_fsi_model(solid_mesh, fluid_mesh, Solid=solids.KelvinVoigt, Fluid=fluids.Bernoulli, 
                   fsi_facet_labels=('pressure',), fixed_facet_labels=('fixed',), coupling='explicit'):
    """
    Factory function that loads a model based on input solid/fluid meshes.

    The function will delegate to specific loading routines depending on how the mesh
    is input etc.

    Parameters
    ----------
    """
    # Load the solid
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

    # Load the fluid
    fluid = None
    if fluid_mesh is None and issubclass(Fluid, fluids.QuasiSteady1DFluid):
        xfluid, yfluid = meshutils.streamwise1dmesh_from_edges(mesh, facet_func, [facet_labels[label] for label in fsi_facet_labels])
        fluid = Fluid(xfluid, yfluid)
    elif fluid_mesh is None and not issubclass(Fluid, fluids.QuasiSteady1DFluid):
        raise ValueError(f"`fluid_mesh` cannot be `None` if the fluid is not 1D")
    else:
        raise ValueError(f"This function does not yet support input fluid meshes.")
    
    if coupling == 'explicit':
        model = ExplicitFSIModel(solid, fluid)
    elif coupling == 'implicit':
        model = ImplicitFSIModel(solid, fluid)

    return model

class FSIModel:
    """
    Represents a coupled system of a solid and a fluid model

    Parameters
    ----------
    mesh_path : str
        Path to a fenics mesh xml file. An xml file containing facet and cell functions are also
        loaded in the directory.
    facet_labels, cell_labels : dict of {str: int}
        A dictionary of named markers for facets and cells. `facet_labels` needs atleast two
        entries {'pressure': ...} and {'fixed': ...} denoting the facets with applied pressure
        and fixed conditions respectively.

    Attributes
    ----------
    solid : Solid
        A solid model object
    fluid : Fluid
        A fluid model object

    surface_vertices : array_like
        A list of vertex numbers on the pressure surface. They are ordered in increasing streamwise
        direction.
    """
    def __init__(self, solid, fluid):
        self.solid = solid
        self.fluid = fluid

        # Create a vertex marker from the boundary marker
        fsi_facet_ids = [solid.facet_labels[name] for name in solid.fsi_facet_labels]
        pressure_edges = np.array([nedge for nedge, fedge in enumerate(solid.facet_func.array()) 
                                   if fedge in set(fsi_facet_ids)])

        # TODO: Should replace with the commented code
        # fixed_facet_ids = [solid.facet_labels[name] for name in solid.fixed_facet_labels]
        # fixed_edges = np.array([nedge for nedge, fedge in enumerate(solid.mesh.edges().array()) 
        #                         if fedge in set(fixed_facet_ids)])
        fixed_edges = self.solid.facet_func.where_equal(self.solid.facet_labels['fixed'])

        pressure_vertices = meshutils.vertices_from_edges(pressure_edges, self.solid.mesh)
        fixed_vertices = meshutils.vertices_from_edges(fixed_edges, self.solid.mesh)

        surface_vertices = pressure_vertices
        surface_coordinates = self.solid.mesh.coordinates()[surface_vertices]

        # TODO: This will only work if you use a 1D fluid mesh where the mesh is aligned along the
        # surface
        # Sort the pressure surface vertices from inferior to superior
        idx_sort = meshutils.sort_vertices_by_nearest_neighbours(surface_coordinates)
        self.surface_vertices = surface_vertices[idx_sort]
        self.surface_coordinates = surface_coordinates[idx_sort]

        self.fixed_vertices = fixed_vertices
        self.fixed_corodinates = self.solid.mesh.coordinates()[fixed_vertices]

        self.state0 = linalg.concatenate(self.solid.state0, self.fluid.state0)
        self.state1 = linalg.concatenate(self.solid.state1, self.fluid.state1)
        self.control0 = linalg.BlockVec((), ())
        self.control1 = linalg.BlockVec((), ())
        self.properties = linalg.concatenate(self.solid.properties, self.fluid.properties)

    ## Methods for settings parameters of the model
    def set_ini_state(self, state):
        self.set_ini_solid_state(state[:3])
        self.set_ini_fluid_state(state[3:])

    def set_fin_state(self, state):
        self.set_fin_solid_state(state[:3])
        self.set_fin_fluid_state(state[3:])

    def set_ini_control(self, control):
        pass
    
    def set_fin_control(self, control):
        pass

    def set_properties(self, props):
        self.solid.set_properties(props[:len(self.solid.properties.size)])
        self.fluid.set_properties(props[len(self.solid.properties.size):])

    def set_time_step(self, dt):
        """
        Sets the time step.

        Parameters
        ----------
        dt : float
        """
        self.solid.set_time_step(dt)
        self.fluid.set_time_step(dt)


    def set_ini_params(self, state0, control0):
        """
        Sets all properties at the initial time.

        Parameters
        ----------
        uva0 : tuple of array_like
        qp0 : tuple of array_like
        dt : float
        fluid_props : dict
        solid_props : dict
        """
        self.set_ini_state(state0)
        self.set_ini_control(control0)

    def set_fin_params(self, state1, control1):
        """
        Sets all properties at the final time.

        Parameters
        ----------
        uva1 : tuple of array_like
        qp1 : tuple of array_like
        fluid_props : dict
        solid_props : dict
        """
        self.set_fin_state(state1)
        self.set_fin_control(control1)

    def set_iter_params(self, state0, control0, state1, control1, dt):
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
        self.set_time_step(dt)

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

        state0 = f.get_state(n)
        control0 = f.get_control(n)
        self.set_ini_params(state0, control0)

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
        control0 = f.get_control(n-1)

        state1 = f.get_state(n)
        control1 = f.get_control(n)

        dt = statefile.get_time(n) - statefile.get_time(n-1)

        # Assign the values to the model
        self.set_iter_params(state0, control0, state1, control1, dt)

    # These must be defined to properly exchange the forcing data between the solid and fluid
    def set_ini_solid_state(self, uva0):
        """
        Sets the state variables u, v, and a at the start of the step.

        Parameters
        ----------
        u0, v0, a0 : array_like
        """
        self.solid.set_ini_state(uva0)
        
        X_ref = self.solid.mesh.coordinates()[self.surface_vertices].reshape(-1)
        u0_fluid = X_ref + self.map_fsi_vector_from_solid_to_fluid(uva0[0])
        v0_fluid = self.map_fsi_vector_from_solid_to_fluid(uva0[1])
        self.fluid.set_ini_control((u0_fluid, v0_fluid))

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

        X_ref = self.solid.mesh.coordinates()[self.surface_vertices].reshape(-1)
        
        u1_fluid = X_ref + self.map_fsi_vector_from_solid_to_fluid(uva1[0])
        v1_fluid = self.map_fsi_vector_from_solid_to_fluid(uva1[1])
        self.fluid.set_fin_control((u1_fluid, v1_fluid))

    def set_ini_fluid_state(self, qp0):
        self.fluid.set_ini_state(qp0)

        p0_solid = self.map_fsi_scalar_from_fluid_to_solid(qp0[1])
        self.solid.set_ini_control(p0_solid)

    def set_fin_fluid_state(self, qp1):
        self.fluid.set_fin_state(qp1)

        p1_solid = self.map_fsi_scalar_from_fluid_to_solid(qp1[1])
        self.solid.set_fin_control(p1_solid)

    ## Solid / fluid interfacing functions
    def get_fsi_scalar_dofs(self):
        """
        Return dofs of the FSI interface on the solid and fluid

        This is needed to pass information between the two domains using conformal interfaces
        between them. Currently this is specifically made to work for the 1D fluid, so if you want
        to do something else, you'll have to think of how to generalized it.
        """
        sdof_solid = self.solid.vert_to_sdof[self.surface_vertices]
        sdof_fluid = np.arange(self.surface_vertices.size)

        return sdof_solid, sdof_fluid

    def get_fsi_vector_dofs(self):
        """
        Return dofs of the FSI interface on the solid and fluid

        This is needed to pass information between the two domains using conformal interfaces
        between them. Currently this is specifically made to work for the 1D fluid, so if you want
        to do something else, you'll have to think of how to generalized it.
        """
        vdof_solid = self.solid.vert_to_vdof.reshape(-1, 2)[self.surface_vertices].reshape(-1).copy()
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

        fluid_vector = self.fluid.get_surf_vector()
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

    def get_surf_state(self):
        """
        Returns the state (u, v, a) of surface vertices of the model.

        The displacement, u, returned is the actual position rather than the displacement relative
        to the reference configuration. Also, states are ordered in streamwise increasing order.

        Returns
        -------
        tuple of array_like
            A tuple of arrays of surface positions, velocities and accelerations.
        """
        vert_to_vdof = self.solid.vert_to_vdof
        surface_dofs = vert_to_vdof.reshape(-1, 2)[self.surface_vertices].flat

        u = self.solid.u0.vector()[surface_dofs].reshape(-1, 2)
        v = self.solid.v0.vector()[surface_dofs].reshape(-1, 2)
        a = self.solid.a0.vector()[surface_dofs].reshape(-1, 2)

        x_surface = (self.surface_coordinates + u, v, a)

        return x_surface

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

    def get_glottal_width(self):
        """
        Return glottal width
        """
        x_surface = self.get_surf_state()

        return self.fluid.get_glottal_width(x_surface)

    def get_exact_glottal_width(self):
        """
        Return glottal width
        """
        x_surface = self.get_surf_state()

        return self.fluid_props['y_midline'] - np.max(x_surface[0][..., 1])

    def get_collision_gap(self):
        """
        Return the smallest distance to the collision plane
        """
        u_surface = self.get_surf_state()[0]

        return self.solid.forms['coeff.prop.y_collision'].values()[0] - np.max(u_surface[..., 1])

    def get_ymax(self):
        """
        Return the maximum y-coordinate of the reference configuration
        """
        x_surface = self.get_surf_state()

        return np.max(x_surface[0][..., 1])

    def get_collision_verts(self):
        """
        Return vertex numbers of nodes in collision.
        """
        # import ipdb; ipdb.set_trace()
        u_surface = self.get_surf_state()[0]
        verts = self.surface_vertices[u_surface[..., 1] > self.y_collision.values()[0]]
        return verts

    ## Methods for getting vectors
    def get_state_vec(self):
        return linalg.concatenate(self.solid.get_state_vec(), self.fluid.get_state_vec())

    def get_control_vec(self):
        return self.control0.copy()

    def get_properties_vec(self):
        return linalg.concatenate(self.solid.get_properties_vec(), self.fluid.get_properties_vec())

    ## Solver functions
    ## Specific models have to implement these
    def res(self):
        """
        Return the residual vector, F
        """
        dt = self.solid.dt.vector()[0]
        u1, v1, a1 = self.solid.u1.vector(), self.solid.v1.vector(), self.solid.a1.vector()
        u0, v0, a0 = self.solid.u0.vector(), self.solid.v0.vector(), self.solid.a0.vector()
        res = self.get_state_vec()

        res['u'][:] = dfn.assemble(self.solid.forms['form.un.f1'])
        self.solid.bc_base.apply(res['u'])
        res['v'][:] = v1 - solids.newmark_v(u1, u0, v0, a0, dt)
        res['a'][:] = a1 - solids.newmark_a(u1, u0, v0, a0, dt)

        qp, *_ = self.fluid.solve_qp1()
        res['q'][:] = self.fluid.state1['q'] - qp['q']
        res['p'][:] = self.fluid.state1['p'] - qp['p']
        return res
    
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

    def res(self):
        """
        Return the residual vector, F
        """
        dt = self.solid.dt.vector()[0]
        u1, v1, a1 = self.solid.u1.vector(), self.solid.v1.vector(), self.solid.a1.vector()
        u0, v0, a0 = self.solid.u0.vector(), self.solid.v0.vector(), self.solid.a0.vector()
        res = self.get_state_vec()

        res['u'][:] = dfn.assemble(self.solid.forms['form.un.f1'])
        self.solid.bc_base.apply(res['u'])
        res['v'][:] = v1 - solids.newmark_v(u1, u0, v0, a0, dt)
        res['a'][:] = a1 - solids.newmark_a(u1, u0, v0, a0, dt)

        qp, *_ = self.fluid.solve_qp1()
        res['q'][:] = self.fluid.state1['q'] - qp['q']
        res['p'][:] = self.fluid.state1['p'] - qp['p']
        return res
    
    # Forward solver methods
    def solve_state1(self, ini_state, newton_solver_prm=None):
        """
        Solve for the final state given an initial guess
        """
        dt = self.solid.dt.vector()[0]
        solid = self.solid
        uva0 = self.solid.state0

        uva1 = self.solid.get_state_vec()

        # Update form coefficients and initial guess
        self.set_fin_state(ini_state)
        # self.set_iter_params(uva1=ini_state.vecs[:3], qp1=ini_state.vecs[3:])

        # TODO: You could implement this to use the non-linear solver only when collision is happening
        if newton_solver_prm is None:
            newton_solver_prm = DEFAULT_NEWTON_SOLVER_PRM

        dfn.solve(solid.f1 == 0, solid.u1, bcs=solid.bc_base, J=solid.df1_du1,
                  solver_parameters={"newton_solver": newton_solver_prm})

        res = dfn.assemble(self.solid.forms['form.un.f1'])
        self.solid.bc_base.apply(res)

        uva1['u'][:] = solid.u1.vector()
        uva1['v'][:] = solids.newmark_v(uva1['u'], *uva0, dt)
        uva1['a'][:] = solids.newmark_a(uva1['u'], *uva0, dt)

        self.set_fin_solid_state(uva1)
        qp1, fluid_info = self.fluid.solve_qp1()

        step_info = {'fluid_info': fluid_info}

        return linalg.concatenate(uva1, qp1), step_info

    def solve_dres_dstate1(self, b):
        """
        Solve, dF/du x = f
        """
        dt = self.solid.dt.vector()[0]
        x = self.get_state_vec()

        solid = self.solid

        dfu1_du1 = dfn.assemble(solid.df1_du1)
        dfv2_du2 = 0 - solids.newmark_v_du1(dt)
        dfa2_du2 = 0 - solids.newmark_a_du1(dt)

        dq_du, dp_du = self.solve_dqp1_du1_solid(adjoint=False)
        dfq2_du2 = 0 - dq_du
        dfp2_du2 = 0 - dp_du

        self.solid.bc_base.apply(dfu1_du1)
        dfn.solve(dfu1_du1, x['u'], b['u'], 'petsc')
        x['v'][:] = b['v'] - dfv2_du2*x['u']
        x['a'][:] = b['a'] - dfa2_du2*x['u']

        # qp1, fluid_info = self.fluid.solve_qp1()
        
        # breakpoint()
        x['q'][:] = b['q'] - dfq2_du2.inner(x['u'])
        x['p'][:] = b['p'] - dfn.PETScVector(dfp2_du2*x['u'].vec())
        return x

    # Adjoint solver methods
    def solve_dres_dstate1_adj(self, b):
        """
        Solve, dF/du^T x = f
        """
        ## Assemble sensitivity matrices
        # model.set_iter_params(**it_params)
        dt = self.solid.dt.vector()[0]

        dfu2_du2 = self.solid.cached_form_assemblers['bilin.df1_du1_adj'].assemble()
        dfv2_du2 = 0 - solids.newmark_v_du1(dt)
        dfa2_du2 = 0 - solids.newmark_a_du1(dt)

        dq_du, dp_du = self.solve_dqp1_du1_solid(adjoint=True)
        dfq2_du2 = 0 - dq_du
        dfp2_du2 = 0 - dp_du

        ## Do the linear algebra that solves for the adjoint states
        x = self.get_state_vec()
        adj_uva = x[:3]
        adj_qp = x[3:]

        adj_u_rhs, adj_v_rhs, adj_a_rhs, adj_q_rhs, adj_p_rhs = b

        # self.solid.bc_base.apply(adj_a_rhs)
        # self.solid.bc_base.apply(adj_v_rhs)
        adj_uva['a'][:] = adj_a_rhs
        adj_uva['v'][:] = adj_v_rhs

        # TODO: Think of how to apply fluid boundary conditions in a generic way.
        # There are no boundary conditions for the Bernoulli case because of the way it's coded but
        # this will be needed for different models
        adj_qp['q'][:] = adj_q_rhs
        adj_qp['p'][:] = adj_p_rhs

        _adj_p = dfp2_du2.getVecRight()
        _adj_p[:] = adj_qp['p']

        adj_u_rhs = adj_u_rhs - (
            dfv2_du2*adj_uva['v'] + dfa2_du2*adj_uva['a'] + dfq2_du2*adj_qp['q'] 
            + dfn.PETScVector(dfp2_du2*_adj_p))
        # print(adj_u_rhs.norm('l2'))
        # breakpoint()
        self.solid.bc_base.apply(dfu2_du2, adj_u_rhs)
        dfn.solve(dfu2_du2, adj_uva['u'], adj_u_rhs, 'petsc')

        return x

    def apply_dres_dstate0_adj(self, x):
        dt2 = self.solid.dt.vector()[0]
        dfu2_du1 = self.solid.cached_form_assemblers['bilin.df1_du0_adj'].assemble()
        dfu2_dv1 = self.solid.cached_form_assemblers['bilin.df1_dv0_adj'].assemble()
        dfu2_da1 = self.solid.cached_form_assemblers['bilin.df1_da0_adj'].assemble()
        dfu2_dp1 = dfn.assemble(self.solid.forms['form.bi.df1_dp1_adj'])

        dfv2_du1 = 0 - solids.newmark_v_du0(dt2)
        dfv2_dv1 = 0 - solids.newmark_v_dv0(dt2)
        dfv2_da1 = 0 - solids.newmark_v_da0(dt2)

        dfa2_du1 = 0 - solids.newmark_a_du0(dt2)
        dfa2_dv1 = 0 - solids.newmark_a_dv0(dt2)
        dfa2_da1 = 0 - solids.newmark_a_da0(dt2)

        solid_dofs, fluid_dofs = self.get_fsi_scalar_dofs()
        dfu2_dp1 = dfn.as_backend_type(dfu2_dp1).mat()
        dfu2_dp1 = linalg.reorder_mat_rows(dfu2_dp1, solid_dofs, fluid_dofs, fluid_dofs.size)
        matvec_adj_p_rhs = dfu2_dp1*dfn.as_backend_type(x['u']).vec()

        b = self.get_state_vec()
        b['u'][:] = dfu2_du1*x['u'] + dfv2_du1*x['v'] + dfa2_du1*x['a']
        b['v'][:] = dfu2_dv1*x['u'] + dfv2_dv1*x['v'] + dfa2_dv1*x['a']
        b['a'][:] = dfu2_da1*x['u'] + dfv2_da1*x['v'] + dfa2_da1*x['a']
        b['q'][:] = 0.0
        b['p'][:] = matvec_adj_p_rhs
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

class ImplicitFSIModel(FSIModel):
    
    def res(self):
        """
        Return the residual vector, F
        """
        dt = self.solid.dt.vector()[0]
        u1, v1, a1 = self.solid.u1.vector(), self.solid.v1.vector(), self.solid.a1.vector()
        u0, v0, a0 = self.solid.u0.vector(), self.solid.v0.vector(), self.solid.a0.vector()
        res = self.get_state_vec()

        res['u'][:] = dfn.assemble(self.solid.forms['form.un.f1'])
        self.solid.bc_base.apply(res['u'])
        res['v'][:] = v1 - solids.newmark_v(u1, u0, v0, a0, dt)
        res['a'][:] = a1 - solids.newmark_a(u1, u0, v0, a0, dt)

        qp, *_ = self.fluid.solve_qp1()
        res['q'][:] = self.fluid.state1['q'] - qp['q']
        res['p'][:] = self.fluid.state1['p'] - qp['p']
        return res
    
    # Forward solver methods
    def solve_state1(self, ini_state, newton_solver_prm=None):
        """
        Solve for the final state given an initial guess
        """
        dt = self.solid.dt.vector()[0]
        solid = self.solid

        # Set initial guesses for the states at the next time
        # uva1 = solid.get_state_vec()
        uva1 = ini_state[:3].copy()
        qp1 = ini_state[3:].copy()

        # Solve the coupled problem using fixed point iterations between the fluid and solid
        if newton_solver_prm is None:
            newton_solver_prm = DEFAULT_NEWTON_SOLVER_PRM

        # Calculate the initial residual
        self.set_fin_solid_state(uva1)
        self.set_fin_fluid_state(qp1)
        res0 = dfn.assemble(self.solid.f1)
        self.solid.bc_base.apply(res0)

        # Set tolerances for the fixed point iterations
        nit = 0
        abs_tol, rel_tol = newton_solver_prm['absolute_tolerance'], newton_solver_prm['relative_tolerance']
        max_nit = 10
        abs_err_prev, abs_err, rel_err = 1.0, np.inf, np.inf
        # *_, fluid_info = self.fluid.solve_qp0() 
        fluid_info = None
        while abs_err > abs_tol and rel_err > rel_tol and nit < max_nit:
            dfn.solve(solid.f1 == 0, solid.u1, bcs=solid.bc_base, J=solid.df1_du1,
                      solver_parameters={"newton_solver": newton_solver_prm})

            uva1['u'][:] = solid.u1.vector()
            uva1['v'][:] = solids.newmark_v(uva1['u'], *self.solid.state0, dt)
            uva1['a'][:] = solids.newmark_a(uva1['u'], *self.solid.state0, dt)
            # print(uva0['u'].norm('l2'))

            self.set_fin_solid_state(uva1)
            qp1, fluid_info = self.fluid.solve_qp1()

            self.set_fin_fluid_state(qp1)

            # Calculate the error in the solid residual with the updated pressures
            # self.set_iter_params(uva1=uva1, qp1=qp1)
            res = dfn.assemble(solid.f1)
            solid.bc_base.apply(res)
            # breakpoint()

            abs_err = res.norm('l2')
            rel_err = abs_err/abs_err_prev

            # breakpoint()
            nit += 1
        res = dfn.assemble(self.solid.forms['form.un.f1'])
        self.solid.bc_base.apply(res)
        # print(nit, res.norm('l2'))

        step_info = {'fluid_info': fluid_info,
                     'nit': nit, 'abs_err': abs_err, 'rel_err': rel_err}

        return linalg.concatenate(uva1, qp1), step_info

    def solve_dres_dstate1(self, b):
        """
        Solve, dF/du x = f
        """
        dt = self.solid.dt.vector()[0]
        x = self.get_state_vec()

        solid = self.solid

        dfu1_du1 = dfn.assemble(solid.df1_du1)
        dfv2_du2 = 0 - solids.newmark_v_du1(dt)
        dfa2_du2 = 0 - solids.newmark_a_du1(dt)

        dq_du, dp_du = self.solve_dqp1_du1_solid(adjoint=False)
        dfq2_du2 = 0 - dq_du
        dfp2_du2 = 0 - dp_du

        self.solid.bc_base.apply(dfu1_du1)
        dfn.solve(dfu1_du1, x['u'], b['u'], 'petsc')
        x['v'][:] = b['v'] - dfv2_du2*x['u']
        x['a'][:] = b['a'] - dfa2_du2*x['u']

        # qp1, fluid_info = self.fluid.solve_qp1()
        
        # breakpoint()
        x['q'][:] = b['q'] - dfq2_du2.inner(x['u'])
        x['p'][:] = b['p'] - dfn.PETScVector(dfp2_du2*x['u'].vec())
        return x

    # Adjoint solver methods
    def solve_dres_dstate1_adj(self, b):
        """
        Solve, dF/du^T x = f
        """
        ## Assemble sensitivity matrices
        # self.set_iter_params(**it_params)
        dt = self.solid.dt.vector()[0]

        dfu2_du2 = self.solid.cached_form_assemblers['bilin.df1_du1_adj'].assemble()
        dfv2_du2 = 0 - solids.newmark_v_du1(dt)
        dfa2_du2 = 0 - solids.newmark_a_du1(dt)
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
        dt2 = self.solid.dt.vector()[0]
        # model.set_iter_params(**it_params2)

        dfu2_du1 = solid.cached_form_assemblers['bilin.df1_du0_adj'].assemble()
        dfu2_dv1 = solid.cached_form_assemblers['bilin.df1_dv0_adj'].assemble()
        dfu2_da1 = solid.cached_form_assemblers['bilin.df1_da0_adj'].assemble()

        dfv2_du1 = 0 - solids.newmark_v_du0(dt2)
        dfv2_dv1 = 0 - solids.newmark_v_dv0(dt2)
        dfv2_da1 = 0 - solids.newmark_v_da0(dt2)
        dfa2_du1 = 0 - solids.newmark_a_du0(dt2)
        dfa2_dv1 = 0 - solids.newmark_a_dv0(dt2)
        dfa2_da1 = 0 - solids.newmark_a_da0(dt2)

        ## Do the matrix vector multiplication that gets the RHS for the adjoint equations
        # Allocate a vector the for fluid side mat-vec multiplication
        b = x.copy()
        b['u'][:] = (dfu2_du1*adj_u2 + dfv2_du1*adj_v2 + dfa2_du1*adj_a2)
        b['v'][:] = (dfu2_dv1*adj_u2 + dfv2_dv1*adj_v2 + dfa2_dv1*adj_a2)
        b['a'][:] = (dfu2_da1*adj_u2 + dfv2_da1*adj_v2 + dfa2_da1*adj_a2)
        b['q'][()] = 0
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
