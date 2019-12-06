"""
Functionality related to fluids
"""

import numpy as np

import dolfin as dfn
from petsc4py import PETSc

SEPARATION_FACTOR = 1.1

def fluid_pressure(x, fluid_props):
    """
    Computes the pressure loading at a series of surface nodes according to Pelorson (1994)

    Parameters
    ----------
    x : tuple of (u, v, a) each of (NUM_VERTICES, GEOMETRIC_DIM) np.ndarray
        States of the surface vertices, ordered following the flow (increasing x coordinate).
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

    # Calculate transverse plane areas using the y component of 'u' (aka x[0]) of the surface
    area = 2 * (y_midline - x[0][:, 1])
    dt_area = -2 * (x[1][:, 1])

    a_sub = fluid_props['a_sub']

    idx_min = np.argmin(area)
    a_min = area[idx_min]
    dt_a_min = dt_area[idx_min]

    # The separation pressure is computed at the node before 'total' separation
    a_sep = SEPARATION_FACTOR * a_min
    dt_a_sep = SEPARATION_FACTOR * dt_a_min
    idx_sep = np.argmax(np.logical_and(area >= a_sep, np.arange(area.size) > idx_min)) - 1

    # 1D Bernoulli approximation of the flow
    p_sep = p_sup
    flow_rate_sqr = 2/rho*(p_sep - p_sub)/(a_sub**-2 - a_sep**-2)
    dt_flow_rate_sqr = 2/rho*(p_sep - p_sub)*-1*(a_sub**-2 - a_sep**-2)**-2 * (2*a_sep**-3 * dt_a_sep)

    p = p_sub + 1/2*rho*flow_rate_sqr*(1/a_sub**2 - 1/area**2)

    # Calculate the pressure along the separation edge
    # Separation happens between vertex i and i+1, so adjust the bernoulli pressure at vertex i
    # based on where separation occurs
    num = (a_sep - area[idx_sep])
    den = (area[idx_sep+1] - area[idx_sep])
    factor = num/den

    separation = np.zeros(x[0].shape[0], dtype=np.bool)
    separation[idx_sep] = 1

    attached = np.ones(x[0].shape[0], dtype=np.bool)
    attached[idx_sep:] = 0

    p = attached*p + separation*factor*p[idx_sep]

    flow_rate = flow_rate_sqr**0.5
    dt_flow_rate = 0.5*flow_rate_sqr**(-0.5) * dt_flow_rate_sqr

    xy_min = x[0][idx_min]
    xy_sep = x[0][idx_sep]
    info = {'flow_rate': flow_rate,
            'dt_flow_rate': dt_flow_rate,
            'idx_min': idx_min,
            'idx_sep': idx_sep,
            'xy_min': xy_min,
            'xy_sep': xy_sep,
            'a_min': a_min,
            'a_sep': a_sep}
    return p, info

def get_pressure_form(model, x, fluid_props):
    """
    Returns the ufl.Coefficient pressure.

    Parameters
    ----------
    model : ufl.Coefficient
        The coefficient representing the pressure
    x : tuple of (u, v, a) each of (NUM_VERTICES, GEOMETRIC_DIM) np.ndarray
        States of the surface vertices, ordered following the flow (increasing x coordinate).
    fluid_props : dict
        A dictionary of fluid property keyword arguments.

    Returns
    -------
    xy_min, xy_sep :
        Locations of the minimum and separation areas, as well as surface pressures.
    """
    pressure = dfn.Function(model.scalar_function_space)

    pressure_vector, info = fluid_pressure(x, fluid_props)
    surface_verts = model.surface_vertices
    pressure.vector()[model.vert_to_sdof[surface_verts]] = pressure_vector

    info['pressure'] = pressure_vector
    return pressure, info

def flow_sensitivity(x, fluid_props):
    """
    Returns the sensitivities of flow properties at a surface state.

    Parameters
    ----------
    x : tuple of (u, v, a) each of (NUM_VERTICES, GEOMETRIC_DIM) np.ndarray
        States of the surface vertices, ordered following the flow (increasing x coordinate).
    fluid_props : dict
        A dictionary of fluid property keyword arguments.
    """
    y_midline = fluid_props['y_midline']
    p_sup, p_sub = fluid_props['p_sup'], fluid_props['p_sub']
    rho = fluid_props['rho']

    area = 2 * (y_midline - x[0][:, 1])
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

    assert x[0].size%2 == 0
    j_sep = 2*idx_sep + 1
    j_min = 2*idx_min + 1
    j_sub = 1

    ## Calculate the pressure sensitivity
    dp_du = np.zeros((x[0].size//2, x[0].size))
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

    dfactor_du = np.zeros(x[0].size)
    dfactor_du[j_min] += dnum_dy_min/den
    dfactor_du[j_sep] += dnum_dy_sep/den - num/den**2*dden_dy_sep
    dfactor_du[j_sep+2] = -num/den**2*dden_dy_sep1

    dp_du[idx_sep, :] = factor*dp_sep_du + dfactor_du*p_sep

    ## Calculate the flow rate sensitivity
    dflow_rate_du = np.zeros(x[0].size)
    dflow_rate_du[j_min] += dflow_rate_sqr_da_sep / (2*flow_rate_sqr**(1/2)) * da_sep_da_min * darea_dy
    dflow_rate_du[j_sub] += dflow_rate_sqr_da_sub / (2*flow_rate_sqr**(1/2)) * darea_dy

    #dp_du = pressure_sensitivity_ad(coordinates, fluid_props)

    return dp_du, dflow_rate_du

def get_flow_sensitivity(model, x, fluid_props):
    """
    Returns sparse matrices/vectors for the sensitivity of pressure and flow rate to displacement.

    Parameters
    ----------
    model
    x : tuple of (u, v, a) each of (NUM_VERTICES, GEOMETRIC_DIM) np.ndarray
        States of the surface vertices, ordered following the flow (increasing x coordinate).
    fluid_props : dict
        A dictionary of fluid properties.

    Returns
    -------
    dp_du : PETSc.Mat
        Sensitivity of pressure with respect to displacement
    dq_du : PETSc.Vec
        Sensitivity of flow rate with respect to displacement
    """
    _dp_du, _dq_du = flow_sensitivity(x, fluid_props)

    dp_du = PETSc.Mat().create(PETSc.COMM_SELF)
    dp_du.setType('aij')
    dp_du.setSizes([model.vert_to_sdof.size, model.vert_to_vdof.size])

    pressure_vertices = model.surface_vertices
    nnz = np.zeros(model.vert_to_sdof.size, dtype=np.int32)
    nnz[model.vert_to_sdof[pressure_vertices]] = pressure_vertices.size*2
    dp_du.setPreallocationNNZ(list(nnz))

    dp_du.setValues(model.vert_to_sdof[pressure_vertices],
                    model.vert_to_vdof[pressure_vertices].reshape(-1), _dp_du)
    dp_du.assemblyBegin()
    dp_du.assemblyEnd()

    # You should be able to create your own vector from scratch too but there are a couple of things
    # you have to set like local to global mapping that need to be there in order to interface with
    # a particular fenics setup. I just don't know what it needs to use.
    # TODO: Figure this out, since it also applies to matrices

    # dq_du = PETSc.Vec().create(PETSc.COMM_SELF).createSeq(vert_to_vdof.size)
    # dq_du.setValues(vert_to_vdof[surface_verts].reshape(-1), _dq_du)
    # dq_du.assemblyBegin()
    # dq_du.assemblyEnd()

    dq_du = dfn.Function(model.vector_function_space).vector()
    dq_du[model.vert_to_vdof[pressure_vertices].reshape(-1)] = _dq_du

    return dp_du, dq_du
