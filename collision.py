"""
Routines for implementing collision
"""
import numpy as np
import dolfin as dfn
import forms as frm

def detect_collision(mesh, vert_marker, omega_contact, u0, v0, a0, vert_to_vdof,
                     domainid_contact=frm.domainid_contact):
    """
    Detects if collision is happening and at what vertices.

    Parameters
    ----------
    mesh : dolfin.mesh
        The mesh
    u0 : dolfin.Function
        The displacement

    Returns
    -------
    np.ndarray
        An array of vertex numbers in collision
    """
    # Grabbing a copy of the mesh coordinates, so we can reset them later
    xy_reference = mesh.coordinates().copy()

    # Modify mesh coordinates to the current configuration
    mesh.coordinates()[...] = xy_reference + u0.vector()[vert_to_vdof.reshape(-1)].reshape(-1, 2)

    # Mark coordinates that have passed the contact plane
    vert_marker.set_all(0)
    omega_contact.mark(vert_marker, domainid_contact)
    _vert_coll = np.array(vert_marker.where_equal(domainid_contact))

    # Eliminate coordinates with velocity directed away from the contact plane
    # atleast_1d is used because the boolean condition leads to a scalar bool value that
    # leads to singleton axes for 1d, 1 element arrays
    vert_coll = None
    vert_coll = _vert_coll[np.atleast_1d(v0.vector()[_vert_coll] >= 0)]

    # Reset the mesh coordinates to their original configuration
    mesh.coordinates()[...] = xy_reference

    return vert_coll

def set_collision(lhs_tensor, rhs_tensor, u0, v0, a0, verts, vert_to_vdof):
    """
    Sets the left hand and right hand side tensors for collision.

    Collision is implemented by:
    - Detecting what nodes are on the contact plane. This is done outside of the function.
    - These nodes are given an initial y-velocity of 0.0.
        - I think this would be like instantaneously stopping the surface, and then removing the
          surface
    - Now compute the nodal forces at the contact plane. If the nodal force is pointing towards the
    surface, then we also set a dirichlet boundary condition on it and if the nodal force is
    pointing away from the contact plane then we don't enforce a dirichlet boundary condition on it.

    Parameters
    ----------
    verts : np.ndarray
        A list of vertex numbers which are in collision with the glottal midplane.
    """
    ## Specify DirichletBC at collision nodes
    # Set the y-velocity and acceleration at collision nodes
    # Collision happens if
    idx_coll = vert_to_vdof[verts, 1]
    v0.vector()[idx_coll] = 0
    a0.vector()[idx_coll] = 0

    # Compute nodal forces. If the nodal force is less <= 0 then the node is no longer in collision
    nodal_forces = dfn.assemble(frm.force_form)
    idx_coll = idx_coll[nodal_forces[idx_coll] >= 0]

    # Apply BCs
    lhs_tensor.ident(idx_coll)
    lhs_tensor.apply('insert')

    rhs_tensor[idx_coll] = u0.vector()[idx_coll]
    rhs_tensor.apply('insert')
