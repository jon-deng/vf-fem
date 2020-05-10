"""
Contains the Model class that couples fluid and solid behaviour
"""

from os import path

import numpy as np
import dolfin as dfn
import ufl
from petsc4py import PETSc as pc

from . import solids
from . import fluids
from . import constants as const
from . import meshutils
from .parameters.properties import FluidProperties

def load_fsi_model(solid_mesh, fluid_mesh, Solid=solid.KelvinVoigt, Fluid=fluids.Bernoulli):
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
    if fluid_mesh is None:
        assert isinstance(Fluid, fluids.Fluid1D)
        xfluid, yfluid = meshutils.streamwise1dmesh_from_edges(mesh, facet_func, facet_labels['pressure'])
        fluid = Fluid(xfluid, yfluid)
    else:
        raise ValueError(f"This function only supports loading 1D fluid models currently.")

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

        # Sort the pressure surface vertices from inferior to superior
        idx_sort = meshutils.sort_vertices_by_nearest_neighbours(surface_coordinates)
        self.surface_vertices = surface_vertices[idx_sort]
        self.surface_coordinates = surface_coordinates[idx_sort]

        self.cached_form_assemblers = {
            'bilin.df1_du1_adj': CachedBiFormAssembler(self.forms['form.bi.df1_du1_adj']),
            'bilin.df1_du0_adj': CachedBiFormAssembler(self.forms['form.bi.df1_du0_adj']),
            'bilin.df1_dv0_adj': CachedBiFormAssembler(self.forms['form.bi.df1_dv0_adj']),
            'bilin.df1_da0_adj': CachedBiFormAssembler(self.forms['form.bi.df1_da0_adj'])
        }

    @property
    def forms(self):
        return self.solid.forms

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

    def get_surface_state(self):
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
        # import matplotlib.pyplot as plt
        # breakpoint()

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
        x_surface = self.get_surface_state()

        # Check that the surface doesn't cross over the midline
        if np.max(x_surface[0][..., 1]) > self.fluid.properties['y_midline']:
            raise RuntimeError('Model crossed symmetry line')

        q, pressure, fluid_info = self.fluid.fluid_pressure(x_surface)

        return q, pressure, fluid_info

    def set_pressure(self, pressure):
        """
        Set surface pressures

        Parameters
        ----------
        pressure : array_like
            An array of pressure values in surface vertex/fluid order
        """
        pressure_solidord = self.pressure_fluidord_to_solidord(pressure)
        self.forms['coeff.fsi.pressure'].vector()[:] = pressure_solidord

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
        x_surface = self.get_surface_state()

        dp_du0, dq_du0 = self.fluid.get_flow_sensitivity(self, x_surface)

        return dp_du0, dq_du0

    def pressure_fluidord_to_solidord(self, pressure):
        """
        Converts a pressure vector in surface vert. order to solid DOF order

        Parameters
        ----------
        pressure : array_like
        model : ufl.Coefficient
            The coefficient representing the pressure

        Returns
        -------
        xy_min, xy_sep :
            Locations of the minimum and separation areas, as well as surface pressures.
        """
        pressure_solid = dfn.Function(self.solid.scalar_fspace).vector()

        surface_verts = self.surface_vertices
        pressure_solid[self.solid.vert_to_sdof[surface_verts]] = pressure

        return pressure_solid

    # @profile
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
    def get_triangulation(self):
        from matplotlib.tri import Triangulation
        coords = self.get_cur_config()
        cells = self.solid.mesh.cells()
        return Triangulation(coords[:, 0], coords[:, 1], triangles=cells)

    def get_glottal_width(self):
        """
        Return glottal width
        """
        x_surface = self.get_surface_state()

        return self.fluid.get_glottal_width(x_surface)

    def get_exact_glottal_width(self):
        """
        Return glottal width
        """
        x_surface = self.get_surface_state()

        return self.fluid_props['y_midline'] - np.max(x_surface[0][..., 1])

    def get_collision_gap(self):
        """
        Return the smallest distance to the collision plane
        """
        u_surface = self.get_surface_state()[0]

        return self.solid.forms['coeff.prop.y_collision'].values()[0] - np.max(u_surface[..., 1])

    def get_ymax(self):
        """
        Return the maximum y-coordinate of the reference configuration
        """
        x_surface = self.get_surface_state()

        return np.max(x_surface[0][..., 1])

    def get_collision_verts(self):
        """
        Return vertex numbers of nodes in collision.
        """
        # import ipdb; ipdb.set_trace()
        u_surface = self.get_surface_state()[0]
        verts = self.surface_vertices[u_surface[..., 1] > self.y_collision.values()[0]]
        return verts

    # Methods for setting model parameters
    def set_ini_state(self, u0, v0, a0):
        """
        Sets the state variables u, v, and a at the start of the step.

        Parameters
        ----------
        u0, v0, a0 : array_like
        """
        self.solid.set_ini_state(u0, v0, a0)

    def set_fin_state(self, u1):
        """
        Sets the displacement at the end of the time step.

        Named `set_final_state` so that it pairs up with `set_initial_state`,  even though there is
        only an input `u`, instead of `u`, `v` and `a`.

        This could be an initial guess in the case of non-linear governing equations, or a solved
        state so that the non-linear form can be linearized for the given state.

        Parameters
        ----------
        u1 : array_like
        """
        self.solid.set_fin_state(u1)

    def set_ini_fluid_state(self, q0, p0):
        self.set_pressure(p0)

    def set_fin_fluid_state(self, q1, p1):
        # Not needed for steady Bernoulli model, but keep it here for consistency and maybe future
        # use
        raise NotImplementedError(
            "This method would not be needed unless you have an unsteady fluid model")

    def set_time_step(self, dt):
        """
        Sets the time step.

        Parameters
        ----------
        dt : float
        """
        self.solid.set_time_step(dt)

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
    def set_params(self, uva0=None, qp0=None, solid_props=None, fluid_props=None):
        """
        Set all parameters needed to integrate the model.

        Parameters
        ----------
        uva0 : array_like
        dt : float
        fluid_props : dict
        solid_props : dict
        """
        if uva0 is not None:
            self.set_ini_state(*uva0)

        if qp0 is not None:
            self.set_ini_fluid_state(*qp0)

        if fluid_props is not None:
            self.set_fluid_props(fluid_props)

        if solid_props is not None:
            self.set_solid_props(solid_props)

    # @profile
    def set_iter_params(self, uva0=None, qp0=None, dt=None, u1=None, solid_props=None, fluid_props=None):
        """
        Set parameter values needed to integrate the model over a time step.

        All parameter values are optional as some may remain constant. In this case, the current
        state of the parameter is simply unchanged.

        Parameters
        ----------
        uva0 : tuple of dfn.GenericVector
        dt : float
        u1 : dfn.GenericVector
        fluid_props : FluidProperties
        solid_props : SolidProperties
        u1 : dfn.GenericVector
        """
        self.set_params(uva0=uva0, qp0=qp0, solid_props=solid_props, fluid_props=fluid_props)

        if dt is not None:
            self.set_time_step(dt)

        if u1 is not None:
            self.set_fin_state(u1)

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
        self.set_params(uva0=uva0, qp0=qp0, solid_props=solid_props, fluid_props=fluid_props)
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
        u1 = None
        if set_final_state:
            u1 = statefile.get_u(n)

        dt = statefile.get_time(n) - statefile.get_time(n-1)

        # Assign the values to the model
        self.set_iter_params(uva0=uva0, qp0=qp0, dt=dt,
                             solid_props=solid_props, fluid_props=fluid_props, u1=u1)
        return {'uva0': uva0, 'qp0': qp0, 'dt': dt, 'solid_props': solid_props, 'fluid_props': fluid_props, 'u1': u1}

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
