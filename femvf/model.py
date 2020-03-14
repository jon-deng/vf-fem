"""
Contains the Model class that couples fluid and solid behaviour
"""
from . import forms
from . import fluids
from . import constants as const
from .properties import FluidProperties, LinearElasticRayleigh

from os import path

import numpy as np
import dolfin as dfn
import ufl
from petsc4py import PETSc as pc

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
    mesh : dfn.Mesh
    facet_function : dfn.MeshFunction
    cell_function : dfn.MeshFunction

    fluid_props : .properties.FluidProperties
    solid_props : .properties.SolidProperties

    surface_vertices : array_like
        A list of vertex numbers on the pressure surface. They are ordered in increasing streamwise
        direction.

    vector_function_space : dfn.VectorFunctionSpace
    scalar_function_space : dfn.FunctionSpace
    vertex_to_vdof : array_like
    vertex_to_sdof : array_like

    form coefficients and states

    main forms for solving
    """
    def __init__(self, mesh_path, facet_labels, cell_labels, forms=forms.linear_elastic_rayleigh):
        self.mesh, self.facet_function, self.cell_function = load_mesh(mesh_path, facet_labels, cell_labels)

        # Create a vertex marker from the boundary marker
        pressure_edges = self.facet_function.where_equal(facet_labels['pressure'])
        fixed_edges = self.facet_function.where_equal(facet_labels['fixed'])

        pressure_vertices = vertices_from_edges(pressure_edges, self.mesh)
        fixed_vertices = vertices_from_edges(fixed_edges, self.mesh)

        surface_vertices = pressure_vertices
        surface_coordinates = self.mesh.coordinates()[surface_vertices]

        # Sort the pressure surface vertices from inferior to superior
        idx_sort = sort_vertices_by_nearest_neighbours(surface_coordinates)
        self.surface_vertices = surface_vertices[idx_sort]
        self.surface_coordinates = surface_coordinates[idx_sort]

        ## Variational Forms
        self._forms = forms(self.mesh,
            self.facet_function, facet_labels, self.cell_function, cell_labels)

        # Add some commonly used things as parameters
        self.vector_function_space = self.forms['fspace.vector']
        self.scalar_function_space = self.forms['fspace.scalar']
        self.vert_to_vdof = dfn.vertex_to_dof_map(self.forms['fspace.vector']).reshape(-1, 2)
        self.vert_to_sdof = dfn.vertex_to_dof_map(self.forms['fspace.scalar'])

        self.dt = self.forms['coeff.time.dt']
        self.gamma = self.forms['coeff.time.gamma']
        self.beta = self.forms['coeff.time.beta']

        self.emod = self.forms['coeff.prop.emod']
        self.y_collision = self.forms['coeff.prop.y_collision']

        self.u0 = self.forms['coeff.state.u0']
        self.v0 = self.forms['coeff.state.v0']
        self.a0 = self.forms['coeff.state.a0']
        self.u1 = self.forms['coeff.arg.u1']

        self.f1 = self.forms['form.un.f1']
        self.df1_du1 = self.forms['form.bi.df1_du1']

        self.bc_base_adj = self.forms['bcs.base']
        self.bc_base = self.forms['bcs.base']

        self.scalar_trial = self.forms['trial.scalar']
        self.vector_trial = self.forms['trial.vector']

        self.df1_du1_mat = dfn.assemble(self.df1_du1)
        self.cached_form_assemblers = {
            'bilin.df1_du1_adj': CachedBiFormAssembler(self.forms['form.bi.df1_du1_adj']),
            'bilin.df1_du0_adj': CachedBiFormAssembler(self.forms['form.bi.df1_du0_adj']),
            'bilin.df1_dv0_adj': CachedBiFormAssembler(self.forms['form.bi.df1_dv0_adj']),
            'bilin.df1_da0_adj': CachedBiFormAssembler(self.forms['form.bi.df1_da0_adj'])
        }

        ## Set properties
        # Default property values are used
        # TODO: This is a bit weird because fluid_properties only needs the function
        # space of self to work, as long as that's defined before this then it should be okay
        self.fluid_props = FluidProperties(self)
        self.solid_props = LinearElasticRayleigh(self)

    @property
    def forms(self):
        return self._forms

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
        return self.mesh.coordinates()

    def get_cur_config(self):
        """
        Returns the current configuration of the body.

        Coordinates are in vertex order.

        Returns
        -------
        array_like
            An array of mesh coordinate point ordered with increasing vertices.
        """
        displacement = self.u0.vector()[self.vert_to_vdof]
        return self.mesh.coordinates() + displacement

    def get_surface_state(self):
        """
        Returns the state (u, v, a) of surface vertices of the model.

        The displacement, u, returned is the actual position rather than the displacement relative
        to the reference configuration. Also, states are ordered according to
        `self.surface_vertices`.

        Returns
        -------
        tuple of array_like
            A tuple of arrays of surface positions, velocities and accelerations.
        """
        surface_dofs = self.vert_to_vdof[self.surface_vertices].reshape(-1)

        u = self.u0.vector()[surface_dofs].reshape(-1, 2)
        v = self.v0.vector()[surface_dofs].reshape(-1, 2)
        a = self.a0.vector()[surface_dofs].reshape(-1, 2)

        x_surface = (self.surface_coordinates + u, v, a)

        return x_surface

    def get_pressure(self):
        """
        Updates pressure coefficient using a bernoulli flow model.

        Parameters
        ----------
        fluid_props : dict
            A dictionary of fluid properties for the 1D bernoulli fluids model
        """
        # Update the pressure loading based on the deformed surface
        x_surface = self.get_surface_state()

        # Check that the surface doesn't cross over the midline
        assert np.max(x_surface[0][..., 1]) < self.fluid_props['y_midline']

        pressure, fluid_info = fluids.get_pressure_form(self, x_surface, self.fluid_props)

        self.forms['coeff.fsi.pressure'].assign(pressure)

        return fluid_info

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
        dp_du0, dq_du0 = fluids.get_flow_sensitivity(self, x_surface, self.fluid_props)

        return dp_du0, dq_du0

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
    def get_glottal_width(self):
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

        return self.y_collision.values()[0] - np.max(u_surface[..., 1])

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
        self.forms['coeff.state.u0'].vector()[:] = u0
        self.forms['coeff.state.v0'].vector()[:] = v0
        self.forms['coeff.state.a0'].vector()[:] = a0

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
        self.forms['coeff.arg.u1'].vector()[:] = u1

    def set_time_step(self, dt):
        """
        Sets the time step.
        """
        self.forms['coeff.time.dt'].assign(dt)

    def set_solid_props(self, solid_props):
        """
        Sets solid properties given a dictionary of solid properties.

        Parameters
        ----------
        solid_props : properties.SolidProperties
        """
        forms = self.forms

        for label, value in solid_props.items():
            # assert prop_label in solid_props

            coefficient = forms['coeff.prop.'+label]

            # If the property is a field variable, values have to be assigned to every spot in
            # the vector
            if isinstance(coefficient, dfn.function.constant.Constant):
                coefficient.assign(value)
            else:
                coefficient.vector()[:] = value

    def set_fluid_props(self, fluid_props):
        """
        Sets fluid properties given a dictionary of fluid properties.

        This just sets the pressure vector given the fluid boundary conditions.

        Parameters
        ----------
        fluid_props : properties.FluidProperties
        """
        self.fluid_props = fluid_props

    def set_params(self, x0=None, solid_props=None, fluid_props=None):
        """
        Set all parameters needed to integrate the model.

        Parameters
        ----------
        x0 : array_like
        dt : float
        fluid_props : dict
        solid_props : dict
        """
        if x0 is not None:
            self.set_ini_state(*x0)

        if fluid_props is not None:
            self.set_fluid_props(fluid_props)

        if solid_props is not None:
            self.set_solid_props(solid_props)

        fluid_info = self.get_pressure()

        return fluid_info

    def set_iter_params(self, x0=None, dt=None, u1=None, solid_props=None, fluid_props=None):
        """
        Set parameter values needed to integrate the model over a time step.

        All parameter values are optional as some may remain constant. In this case, the current
        state of the parameter is simply unchanged.

        Parameters
        ----------
        x0 : tuple of dfn.GenericVector
        dt : float
        u1 : dfn.GenericVector
        fluid_props : FluidProperties
        solid_props : SolidProperties
        u1 : dfn.GenericVector
        """
        self.set_params(x0=x0, solid_props=solid_props, fluid_props=fluid_props)

        if dt is not None:
            self.set_time_step(dt)

        if u1 is not None:
            self.set_fin_state(u1)

        fluid_info = self.get_pressure()

        return fluid_info

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

        x0 = statefile.get_state(n)

        # Assign the values to the model
        fluid_info = self.set_params(x0=x0, solid_props=solid_props, fluid_props=fluid_props)

        return fluid_info

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

        x0 = statefile.get_state(n-1)
        u1 = None
        if set_final_state:
            u1 = statefile.get_u(n)

        dt = statefile.get_time(n) - statefile.get_time(n-1)

        # Assign the values to the model
        fluid_info = self.set_iter_params(x0=x0, dt=dt,
                                          solid_props=solid_props, fluid_props=fluid_props, u1=u1)

        return fluid_info, fluid_props

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

def load_mesh(mesh_path, facet_labels, cell_labels):
    """
    Return mesh and facet/cell info

    Parameters
    ----------
    mesh_path : str
        Path to the mesh .xml file
    facet_labels, cell_labels : dict(str: int)
        Dictionaries providing integer identifiers for given labels

    Returns
    -------
    mesh : dfn.Mesh
        The mesh object
    facet_function, cell_function : dfn.MeshFunction
        A mesh function marking facets with integers. Marked elements correspond to the label->int
        mapping given in facet_labels/cell_labels.
    """
    base_path, ext = path.splitext(mesh_path)
    facet_function_path = base_path +  '_facet_region.xml'
    cell_function_path = base_path + '_physical_region.xml'

    if ext == '':
        mesh_path = mesh_path + '.xml'

    mesh = dfn.Mesh(mesh_path)
    facet_function = dfn.MeshFunction('size_t', mesh, facet_function_path)
    cell_function = dfn.MeshFunction('size_t', mesh, cell_function_path)

    return mesh, facet_function, cell_function

def vertices_from_edges(edge_indices, mesh):
    """
    Return vertices associates with a set of edges
    """
    edge_to_vertex = np.array([[vertex.index() for vertex in dfn.vertices(edge)]
                                for edge in dfn.edges(mesh)])

    vertices = np.unique(edge_to_vertex[edge_indices].reshape(-1))
    return vertices

def sort_vertices_by_nearest_neighbours(vertex_coordinates):
    """
    Return an index list sorting the vertices based on its nearest neighbours

    For the case of a collection of vertices along the surface of a mesh, this should sort them
    along an increasing or decreasing surface coordinate.

    This is mainly used to sort the inferior-superior direction is oriented along the positive x axis.

    Parameters
    ----------
    vertex_coordinates : (..., 2) array_like
        An array of surface coordinates, with x and y locations stored in the last dimension.
    """
    # Determine the very first coordinate
    idx_sort = [np.argmin(vertex_coordinates[..., 0])]

    while len(idx_sort) < vertex_coordinates.shape[0]:
        # Calculate array of distances to every other coordinate
        vector_distances = vertex_coordinates - vertex_coordinates[idx_sort[-1]]
        distances = np.sum(vector_distances**2, axis=-1)**0.5
        distances[idx_sort] = np.nan

        idx_sort.append(np.nanargmin(distances))

    return np.array(idx_sort)
