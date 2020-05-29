"""
Contains the Model class that couples fluid and solid behaviour
"""

from os import path

import numpy as np
import dolfin as dfn
import ufl
from petsc4py import PETSc as pc

from . import solids, fluids
from . import constants as const
from . import meshutils
from .parameters.properties import FluidProperties

def load_fsi_model(solid_mesh, fluid_mesh, Solid=solids.KelvinVoigt, Fluid=fluids.Bernoulli):
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
        if ext.lower() == '.xml':
            # The solid mesh is an xml file
            mesh, facet_func, cell_func, facet_labels, cell_labels = meshutils.load_fenics_xmlmesh(solid_mesh)
        else:
            raise ValueError(f"Can't process mesh {solid_mesh} with extension {ext}")
    solid = Solid(mesh, facet_func, facet_labels, cell_func, cell_labels)

    # Load the fluid
    fluid = None
    if fluid_mesh is None and issubclass(Fluid, fluids.QuasiSteady1DFluid):
        xfluid, yfluid = meshutils.streamwise1dmesh_from_edges(mesh, facet_func, facet_labels['pressure'])
        fluid = Fluid(xfluid, yfluid)
    elif fluid_mesh is None and not issubclass(Fluid, fluids.Fluid1D):
        raise ValueError(f"`fluid_mesh` cannot be `None` if the fluid is not 1D")
    else:
        raise ValueError(f"This function does not yet support input fluid meshes.")

    model = ForwardModel(solid, fluid)

    return model

class ForwardModel:
    """
    Stores all the things related to the vocal fold forward model solved thru fenics.

    TODO: Instantiation is kind of messy and ugly. Prettify/clean it up.
        Class contains alot of extra, non-essential stuff. Think about what are the essential
        things that are included, how to compartmentalize the things etc.

    TODO: Improve assembly speed
        There are a number of ways to speed up assembly. Firstly the bilinear forms have
        components that remain constant under certain conditions. These could be cached to improved
        performance. There are some issues here however:
            adding the constant components incurs overhead due to different sparsity patterns
            multiplying a matrix by a scalar seems to use 3 threads for some reason

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

    form coefficients and states

    main forms for solving
    """
    def __init__(self, solid, fluid):
        self.solid = solid
        self.fluid = fluid

        # Create a vertex marker from the boundary marker
        pressure_edges = self.solid.facet_function.where_equal(self.solid.facet_labels['pressure'])
        fixed_edges = self.solid.facet_function.where_equal(self.solid.facet_labels['fixed'])

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

        self.cached_form_assemblers = {
            'bilin.df1_du1_adj': CachedBiFormAssembler(self.forms['form.bi.df1_du1_adj']),
            'bilin.df1_du0_adj': CachedBiFormAssembler(self.forms['form.bi.df1_du0_adj']),
            'bilin.df1_dv0_adj': CachedBiFormAssembler(self.forms['form.bi.df1_dv0_adj']),
            'bilin.df1_da0_adj': CachedBiFormAssembler(self.forms['form.bi.df1_da0_adj'])
        }

    @property
    def forms(self):
        return self.solid.forms

    # Solid / fluid interfacing functions
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

    def map_fsi_scalar_from_solid_to_fluid(self, solid_scalar):
        sdof_fluid, sdof_solid = self.get_fsi_scalar_dofs()

        fluid_scalar = self.fluid.get_surf_scalar()
        fluid_scalar[sdof_fluid] = solid_scalar[sdof_solid]
        return fluid_scalar

    def map_fsi_vector_from_solid_to_fluid(self, solid_vector):
        vdof_fluid, vdof_solid = self.get_fsi_vector_dofs()

        fluid_vector = self.fluid.get_surf_vector()
        fluid_vector[vdof_fluid] = solid_vector[vdof_solid]
        return fluid_vector

    def map_fsi_scalar_from_fluid_to_solid(self, fluid_scalar):
        sdof_fluid, sdof_solid = self.get_fsi_scalar_dofs()

        solid_scalar = dfn.Function(self.solid.scalar_fspace).vector()
        solid_scalar[sdof_solid] = fluid_scalar[sdof_fluid]
        return solid_scalar

    def map_fsi_vector_from_fluid_to_solid(self, fluid_vector):
        vdof_fluid, vdof_solid = self.get_fsi_vector_dofs()

        solid_vector = dfn.Function(self.solid.vector_fspace).vector()
        solid_vector[vdof_solid] = fluid_vector[vdof_fluid]
        return solid_vector

    # Core solver functions
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

    def get_pressure(self):
        """
        Calculate surface pressures using a bernoulli flow model.

        Parameters
        ----------
        fluid_props : dict
            A dictionary of fluid properties for the 1D bernoulli fluids model
        """
        # Update the pressure loading based on the deformed surface
        x_surface = self.get_surf_state()

        # Check that the surface doesn't cross over the midline
        if np.max(x_surface[0][..., 1]) > self.fluid.properties['y_midline']:
            raise RuntimeError('Model crossed symmetry line')

        q, pressure, fluid_info = self.fluid.fluid_pressure(x_surface)

        return q, pressure, fluid_info

    def get_flow_sensitivity(self):
        """
        Updates pressure sensitivity using a bernoulli flow model.

        Parameters
        ----------
        fluid_props : dict
            A dictionary of fluid properties

        Returns
        -------
        dp_du0 : np.ndarray
            Sensitivity of surface pressures w.r.t. the initial displacement.
        dq_du0 : np.ndarray
            Sensitivity of the flow rate w.r.t. the initial displacement.
        """
        # Calculate sensitivities of fluid quantities based on the deformed surface
        x_surface = self.get_surf_state()

        dq_du, dp_du = self.fluid.get_flow_sensitivity(x_surface)

        return dq_du, dp_du

    def get_flow_sensitivity_(self):
        return self.fluid.solve_fin_sensitivity()

    def get_flow_sensitivity_solid_ord(self, adjoint=False):
        """
        Updates pressure sensitivity using a bernoulli flow model.

        Parameters
        ----------
        fluid_props : dict
            A dictionary of fluid properties

        Returns
        -------
        dp_du0 : np.ndarray
            Sensitivity of surface pressures w.r.t. the initial displacement.
        dq_du0 : np.ndarray
            Sensitivity of the flow rate w.r.t. the initial displacement.
        """
        # Calculate sensitivities of fluid quantities based on the deformed surface
        x_surface = self.get_surf_state()

        dq_du, dp_du = self.fluid.get_flow_sensitivity_solid(self, x_surface, adjoint=adjoint)

        return dq_du, dp_du

    def assem_f1(self, u1=None):
        """
        Return the residual vector

        Parameters
        ----------
        u1 : dfn.cpp.la.Vector
        """
        return dfn.assemble(self.forms['forms.un.f1'])
        # M = self.assem_cache['M']
        # K = self.assem_cache['K']

        # dt = self.dt.values()[0]
        # gamma, beta = self.gamma.values()[0], self.beta.values()[0]

        # rm = self.rayleigh_m.values()[0]
        # rk = self.rayleigh_k.values()[0]

        # u0 = self.u0.vector()
        # v0 = self.v0.vector()
        # a0 = self.a0.vector()

        # if u1 is None:
        #     u1 = self.u1.vector()

        # v1 = newmark_v(u1, u0, v0, a0, dt, gamma, beta)
        # a1 = newmark_a(u1, u0, v0, a0, dt, gamma, beta)

        # return M*(a1 + rm*v1) + K*(u1 + rk*v1) + dfn.assemble(self.f1_nonlin)

    def assem_df1_du1(self):
        """
        Return the residual vector jacobian

        Parameters
        ----------
        u1 : dfn.cpp.la.Vector
        """
        return dfn.assemble(self.forms['form.bi.df1_du1'])

    def assem_df1_du1_adj(self):
        return self.cached_form_assemblers['bilin.df1_du1_adj'].assemble()

    def assem_df1_du0_adj(self):
        return self.cached_form_assemblers['bilin.df1_du0_adj'].assemble()

    def assem_df1_dv0_adj(self):
        return self.cached_form_assemblers['bilin.df1_dv0_adj'].assemble()

    def assem_df1_da0_adj(self):
        return self.cached_form_assemblers['bilin.df1_da0_adj'].assemble()

    # Convenience functions
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

    # Methods for setting model parameters
    def set_ini_solid_state(self, u0, v0, a0):
        """
        Sets the state variables u, v, and a at the start of the step.

        Parameters
        ----------
        u0, v0, a0 : array_like
        """
        self.solid.set_ini_state(u0, v0, a0)

        uv0_fluid = [self.map_fsi_vector_from_solid_to_fluid(y) for y in (u0, v0)]
        self.fluid.set_ini_surf_state(*uv0_fluid)

    def set_fin_solid_state(self, u1, v1, a1):
        """
        Sets the displacement at the end of the time step.

        This could be an initial guess in the case of non-linear governing equations, or a solved
        state so that the non-linear form can be linearized for the given state.

        Parameters
        ----------
        uva1 : tuple of array_like
        """
        self.solid.set_fin_state(u1, v1, a1)

        uv1_fluid = [self.map_fsi_vector_from_solid_to_fluid(y) for y in (u1, v1)]
        self.fluid.set_ini_surf_state(*uv1_fluid)

    def set_ini_fluid_state(self, q0, p0):
        self.fluid.set_ini_state(q0, p0)

        p0_solid = self.map_fsi_scalar_from_fluid_to_solid(p0)
        self.solid.set_ini_surf_pressure(p0_solid)

    def set_fin_fluid_state(self, q1, p1):
        self.fluid.set_fin_state(q1, p1)

        p1_solid = self.map_fsi_scalar_from_fluid_to_solid(p1)
        self.solid.set_ini_surf_pressure(p1_solid)

    def set_time_step(self, dt):
        """
        Sets the time step.

        Parameters
        ----------
        dt : float
        """
        self.solid.set_time_step(dt)
        self.fluid.set_time_step(dt)

    def set_solid_props(self, solid_props):
        """
        Sets solid properties given a dictionary of solid properties.

        Parameters
        ----------
        solid_props : properties.SolidProperties
        """
        self.solid.set_properties(solid_props)

    def set_fluid_props(self, fluid_props):
        """
        Sets fluid properties given a dictionary of fluid properties.

        This just sets the pressure vector given the fluid boundary conditions.

        Parameters
        ----------
        fluid_props : properties.FluidProperties
        """
        self.fluid.set_properties(fluid_props)

    # @profile
    def set_ini_params(self, uva0=None, qp0=None, solid_props=None, fluid_props=None):
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
        if uva0 is not None:
            self.set_ini_solid_state(*uva0)

        if qp0 is not None:
            self.set_ini_fluid_state(*qp0)

        if fluid_props is not None:
            self.set_fluid_props(fluid_props)

        if solid_props is not None:
            self.set_solid_props(solid_props)

    def set_fin_params(self, uva1=None, qp0=None, solid_props=None, fluid_props=None):
        """
        Sets all properties at the final time.

        Parameters
        ----------
        uva1 : tuple of array_like
        qp1 : tuple of array_like
        fluid_props : dict
        solid_props : dict
        """
        if uva1 is not None:
            self.set_fin_solid_state(*uva1)

        if qp1 is not None:
            self.set_ini_fluid_state(*qp1)

        if fluid_props is not None:
            self.set_fluid_props(fluid_props)

        if solid_props is not None:
            self.set_solid_props(solid_props)

    # @profile
    def set_iter_params(self, uva0=None, qp0=None, dt=None, uva1=None, qp1=None,
                        solid_props=None, fluid_props=None):
        """
        Sets all parameter values needed to integrate the model over a time step.

        The parameter specified at the final time (index 1) are initial guesses for the solution.
        One can then use a Newton method to iteratively solve for the actual final states.

        Unspecified parameters will have unchanged values.

        Parameters
        ----------
        uva0, uva1 : tuple of array_like
            Initial and final solid states
        qp0, qp1 : tuple of array_like
            Initial and final fluid states
        dt : float
        fluid_props : FluidProperties
        solid_props : SolidProperties
        """
        self.set_ini_params(uva0=uva0, qp0=qp0, solid_props=solid_props, fluid_props=fluid_props)
        self.set_fin_params(uva1=uva1, qp1=qp1)

        if dt is not None:
            self.set_time_step(dt)

    def set_params_fromfile(self, statefile, n, update_props=True):
        """
        Set all parameters needed to integrate the model from a recorded value.

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
        solid_props, fluid_props = None, None
        if update_props:
            fluid_props = statefile.get_fluid_props(n)
            solid_props = statefile.get_solid_props()

        uva0 = statefile.get_state(n)
        qp0 = statefile.get_fluid_state(n)

        # Assign the values to the model
        self.set_ini_params(uva0=uva0, qp0=qp0, solid_props=solid_props, fluid_props=fluid_props)
        return {'uva0': uva0, 'qp0': qp0, 'solid_props': solid_props, 'fluid_props': fluid_props}

    def set_iter_params_fromfile(self, statefile, n, set_final_state=True, update_props=True):
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
        solid_props, fluid_props = None, None
        if update_props:
            fluid_props = statefile.get_fluid_props(0)
            solid_props = statefile.get_solid_props()

        uva0 = statefile.get_state(n-1)
        qp0 = statefile.get_fluid_state(n-1)

        uva1 = None
        qp1 = None
        if set_final_state:
            uva1 = statefile.get_state(n)
            qp1 = statefile.get_fluid_state(n)

        dt = statefile.get_time(n) - statefile.get_time(n-1)

        # Assign the values to the model
        self.set_iter_params(uva0=uva0, qp0=qp0, uva1=uva1, qp1=qp1, dt=dt,
                             solid_props=solid_props, fluid_props=fluid_props)
        return {'uva0': uva0, 'uva1': uva1, 'qp0': qp0, 'qp1': qp1, 'dt': dt,
                'solid_props': solid_props, 'fluid_props': fluid_props, }

class CachedBiFormAssembler:
    """
    Assembles a bilinear form using a cached sparsity pattern

    Parameters
    ----------
    form : ufl.Form
    keep_diagonal : bool, optional
        Whether to preserve diagonals in the form
    """

    def __init__(self, form, keep_diagonal=True):
        self._form = form

        self._tensor = dfn.assemble(form, keep_diagonal=keep_diagonal)
        self._tensor.zero()

    @property
    def tensor(self):
        return self._tensor

    @property
    def form(self):
        return self._form

    def assemble(self):
        out = self.tensor.copy()
        return dfn.assemble(self.form, tensor=out)

# def load_xdmf(mesh_path, facet_labels, cell_labels):
#     """
#     Return mesh and facet/cell info

#     Parameters
#     ----------
#     mesh_path : str
#         Path to the mesh .xml file
#     facet_labels, cell_labels : dict(str: int)
#         Dictionaries providing integer identifiers for given labels

#     Returns
#     -------
#     mesh : dfn.Mesh
#         The mesh object
#     facet_function, cell_function : dfn.MeshFunction
#         A mesh function marking facets with integers. Marked elements correspond to the label->int
#         mapping given in facet_labels/cell_labels.
#     """
#     base_path, ext = path.splitext(mesh_path)

#     mesh = dfn.Mesh()
#     facet_function = dfn.MeshFunction('size_t', mesh, 1)
#     cell_function = dfn.MeshFunction('size_t', mesh, 2)

#     with dfn.XDMFFile(mesh_path) as f:
#         f.read(mesh)
#         for label in facet_labels():
#         f.read(facet_mvc)
#         f.read()
#     facet_function_path = base_path +  '_facet_region.xml'
#     cell_function_path = base_path + '_physical_region.xml'


#     return mesh, facet_function, cell_function
