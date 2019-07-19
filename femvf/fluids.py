"""
Functionality related to fluids
"""

import autograd
from autograd import numpy as np

from petsc4py import PETSc

FLUID_PROP_LABELS = ('p_sub', 'p_sup', 'a_sub', 'a_sup', 'rho', 'y_midline')
SEPARATION_FACTOR = 1.1

def set_pressure_form(form, coordinates, vertices, vertex_to_sdof, fluid_props):
    """
    Sets the nodal values of the pressure ufl.Coefficient.

    Parameters
    ----------
    form : ufl.Coefficient
        The coefficient representing the pressure
    coordinates : (NUM_VERTICES, GEOMETRIC_DIM) np.ndarray
        An array containing vertex numbers of the surface vertices
    vertices : (NUM_VERTICES,) np.ndarray
        An array containing vertex numbers of the surface vertices
    vertex_to_sdof : (NUM_VERTICES,) np.ndarray
        An array containing the scalar degree of freedom corresponding to a vertex
    fluid_props : dict
        A dictionary of fluid property keyword arguments.

    Returns
    -------
    xy_min, xy_sep :
        Locations of the minimum and separation areas
    """
    pressure, info = fluid_pressure(coordinates, fluid_props)

    form.vector()[vertex_to_sdof[vertices]] = pressure

    info['pressure'] = pressure
    return info

def fluid_pressure(coordinates, fluid_props):
    """
    Computes the pressure loading at a series of surface nodes according to Pelorson (1994)

    Parameters
    ----------
    coordinates : (NUM_VERTICES, GEOMETRIC_DIM) np.array
        Coordinates of surface vertices ordered along the flow direction (increasing x coordinate).
    fluid_props : dict
        A dictionary of fluid properties.

    Returns
    -------
    p : np.ndarray
        An array of pressure vectors for each each vertex
    xy_min, xy_sep: (2,) np.ndarray
        The coordinates of the vertices at minimum and separation areas
    """
    y_midline = fluid_props['y_midline']
    p_sup, p_sub = fluid_props['p_sup'], fluid_props['p_sub']
    rho = fluid_props['rho']

    area = 2 * (y_midline - coordinates[:, 1])
    # a_sub = area[0]
    a_sub = fluid_props['a_sub']

    idx_min = np.argmin(area)
    a_min = area[idx_min]

    # The separation pressure is computed at the node before 'total' separation
    a_sep = SEPARATION_FACTOR * a_min
    idx_sep = np.argmax(np.logical_and(area >= a_sep, np.arange(area.size) > idx_min)) - 1

    # 1D Bernoulli approximation of the flow
    p_sep = p_sup
    flow_rate_sqr = (p_sep - p_sub)/(1/2*rho*(1/a_sub**2-1/a_sep**2))

    p = p_sub + 1/2*rho*flow_rate_sqr*(1/a_sub**2 - 1/area**2)

    # Calculate the pressure along the separation edge
    # Separation happens between vertex i and i+1, so adjust the bernoulli pressure at vertex i
    # based on where separation occurs
    num = (a_sep - area[idx_sep])
    den = (area[idx_sep+1] - area[idx_sep])
    factor = num/den

    separation = np.zeros(coordinates.shape[0], dtype=np.bool)
    separation[idx_sep] = 1

    attached = np.ones(coordinates.shape[0], dtype=np.bool)
    attached[idx_sep:] = 0

    p = attached*p + separation*factor*p[idx_sep]

    flow_rate = flow_rate_sqr**0.5

    xy_min = coordinates[idx_min]
    xy_sep = coordinates[idx_sep]
    info = {'flow_rate': flow_rate,
            'xy_min': xy_min,
            'xy_sep': xy_sep}
    return p, info

def set_flow_sensitivity(coordinates, vertices, vertex_to_sdof, vertex_to_vdof, fluid_props):
    """
    Returns sparse matrices/vectors for the sensitivity of pressure and flow rate to displacement.

    Parameters
    ----------
    coordinates : (NUM_VERTICES, GEOMETRIC_DIM) np.ndarray
        An array containing vertex numbers of the surface vertices
    vertices : (NUM_VERTICES,) np.ndarray
        An array containing vertex numbers of the surface vertices
    vertex_to_sdof : (NUM_VERTICES,) np.ndarray
        An array containing the degree of freedom corresponding to each vertex
    vertex_to_vdof : (NUM_VERTICES, u.geometric_dimension) np.ndarray
        An array containing the degree of freedom of each u corresponding to each vertex
    fluid_props : dict
        A dictionary of fluid properties.

    Returns
    -------
    dp_du : PETSc.Mat
        Sensitivity of pressure with respect to displacement
    dq_du : PETSc.Vec
        Sensitivity of flow rate with respect to displacement
    """
    _dp_du, _dq_du = flow_sensitivity(coordinates, fluid_props)

    dp_du = PETSc.Mat().create(PETSc.COMM_SELF)
    dp_du.setType('aij')
    dp_du.setSizes([vertex_to_sdof.size, vertex_to_vdof.size])

    nnz = np.zeros(vertex_to_sdof.size, dtype=np.int32)
    nnz[vertex_to_sdof[vertices]] = vertices.size*2
    dp_du.setPreallocationNNZ(list(nnz))

    dp_du.setValues(vertex_to_sdof[vertices], vertex_to_vdof[vertices].reshape(-1), _dp_du)
    dp_du.assemblyBegin()
    dp_du.assemblyEnd()

    dq_du = PETSc.Vec().create(PETSc.COMM_SELF).createSeq(vertex_to_vdof.size)

    dq_du.setValues(vertex_to_vdof[vertices].reshape(-1), _dq_du)
    dq_du.assemblyBegin()
    dq_du.assemblyEnd()

    return dp_du, dq_du

def flow_sensitivity(coordinates, fluid_props):
    """
    Returns the derivative of flow properties with respect to the displacement.

    Parameters
    ----------
    coordinates : (NUM_VERTICES, 2) np.array
        Coordinates of surface vertices
    fluid_props : dict
        A dictionary of fluid property keyword arguments.
    """
    y_midline = fluid_props['y_midline']
    p_sup, p_sub = fluid_props['p_sup'], fluid_props['p_sub']
    rho = fluid_props['rho']

    area = 2 * (y_midline - coordinates[:, 1])
    darea_dy = -2 # darea_dx = 0

    # a_sub = area[0]
    a_sub = fluid_props['a_sub']

    idx_min = np.argmin(area)
    a_min = area[idx_min]

    a_sep = SEPARATION_FACTOR * a_min
    da_sep_da_min = SEPARATION_FACTOR
    idx_sep = np.argmax(np.logical_and(area >= a_sep, np.arange(area.size) > idx_min)) - 1

    # 1D Bernoulli approximation of the flow
    coeff = 2*(p_sup - p_sub)/rho
    flow_rate_sqr = coeff/(1/a_sub**2-1/a_sep**2)
    dflow_rate_sqr_da_sub = -coeff / (1/a_sub**2-1/a_sep**2)**2 * (-2/a_sub**3)
    dflow_rate_sqr_da_sep = -coeff / (1/a_sub**2-1/a_sep**2)**2 * (2/a_sep**3)

    assert coordinates.size%2 == 0
    j_sep = 2*idx_sep + 1
    j_min = 2*idx_min + 1
    j_sub = 1

    ## Calculate the pressure sensitivity
    dp_du = np.zeros((coordinates.size//2, coordinates.size))
    for i in range(idx_sep+1):
        j = 2*i + 1

        # p[i] = p_sub + 1/2*rho*flow_rate_sqr*(1/a_sub**2 + 1/area[i]**2)
        dp_darea = 1/2*rho*flow_rate_sqr*(2/area[i]**3)
        dp_darea_sep = 1/2*rho*dflow_rate_sqr_da_sep*(1/a_sub**2 - 1/area[i]**2)
        dp_darea_sub = 1/2*rho*dflow_rate_sqr_da_sub*(1/a_sub**2 - 1/area[i]**2) \
                       + 1/2*rho*flow_rate_sqr*(-2/a_sub**3)

        dp_du[i, j] += dp_darea * darea_dy
        dp_du[i, j_min] += dp_darea_sep * da_sep_da_min * darea_dy
        dp_du[i, j_sub] += dp_darea_sub * darea_dy

    # Account for factor on separation pressure
    p_sep = p_sub + 1/2*rho*flow_rate_sqr*(1/a_sub**2 - 1/area[idx_sep]**2)
    p = p_sub + 1/2*rho*flow_rate_sqr*(1/a_sub**2 - 1/area**2)
    p_sep = p[idx_sep]
    dp_sep_du = dp_du[idx_sep, :]

    num = (a_sep - area[idx_sep])
    dnum_dy_min = da_sep_da_min*darea_dy
    dnum_dy_sep = -1*darea_dy

    den = (area[idx_sep+1] - area[idx_sep])
    dden_dy_sep1 = 1*darea_dy
    dden_dy_sep = -darea_dy

    factor = num/den

    dfactor_du = np.zeros(coordinates.size)
    dfactor_du[j_min] += dnum_dy_min/den
    dfactor_du[j_sep] += dnum_dy_sep/den - num/den**2*dden_dy_sep
    dfactor_du[j_sep+2] = -num/den**2*dden_dy_sep1

    dp_du[idx_sep, :] = factor*dp_sep_du + dfactor_du*p_sep

    ## Calculate the flow rate sensitivity
    dflow_rate_du = np.zeros(coordinates.size)
    dflow_rate_du[j_min] += dflow_rate_sqr_da_sep / (2*flow_rate_sqr**(1/2)) * da_sep_da_min * darea_dy
    dflow_rate_du[j_sub] += dflow_rate_sqr_da_sub / (2*flow_rate_sqr**(1/2)) * darea_dy

    #dp_du = pressure_sensitivity_ad(coordinates, fluid_props)

    return dp_du, dflow_rate_du

def pressure_sensitivity_ad(coordinates, fluid_props):
    """
    Returns the derivative of fluid pressure with respect to displacement u.

    This is done via the autograd, autodifferentiation package.

    Parameters
    ----------
    coordinates : (NUM_VERTICES, 2) np.array
        Coordinates of surface vertices
    fluid_props : dict
        A dictionary of fluid property keyword arguments.
    """
    dp_du = autograd.jacobian(
        lambda x: fluid_pressure(np.reshape(x, (-1, 2)), fluid_props)[0])
    dp_du_ad = dp_du(coordinates.reshape(-1))

    return dp_du_ad
