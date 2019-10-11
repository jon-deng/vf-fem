from matplotlib import tri

def triangulation(mesh, x, vert_to_vdof):
    """
    Returns a triangulation for a mesh.

    Parameters
    ----------
    mesh : dolfin.mesh
    x : dolfin.vector
    scalar : dolfin.vector, optional

    Returns
    -------
    matplotlib.triangulation
    """
    delta_xy = x[0].vector()[vert_to_vdof.reshape(-1)].reshape(-1, 2)
    xy_current = mesh.coordinates() + delta_xy

    out = tri.Triangulation(xy_current[:, 0], xy_current[:, 1], triangles=mesh.cells())

    return out
