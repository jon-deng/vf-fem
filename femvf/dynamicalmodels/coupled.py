"""
Contains class definitions for coupled dynamical systems models
"""

import dolfin as dfn
from petsc4py import PETSc as ptc

import blocklinalg.linalg as bla

from .base import DynamicalSystem


class FSIMap:
    """
    Represents a mapping between two domains (fluid and solid)

    This mapping involves a 1-to-1 correspondence between DOFs of vectors on the two domains
    """
    def __init__(self, ndof_fluid, ndof_solid, fluid_dofs, solid_dofs, comm=None):
        """
        Parameters
        ----------
        ndof_fluid, ndof_solid : int
            number of DOFS on the fluid and solid domains
        fluid_dofs, solid_dofs : array
            arrays of corresponding dofs on the fluid and solid side domains
        comm : None or PETSc.Comm
            MPI communicator. Not really used here since I never run stuff in parallel.
        """
        self.N_FLUID = ndof_fluid
        self.N_SOLID = ndof_solid

        self.dofs_fluid = fluid_dofs
        self.dofs_solid = solid_dofs

        self.fluid_to_solid_idx = {idxf: idxs for idxf, idxs in zip(fluid_dofs, solid_dofs)}
        self.solid_to_fluid_idx = {idxf: idxs for idxf, idxs in zip(fluid_dofs, solid_dofs)}

        self.jac_fluid_to_solid = self.assem_jac_fluid_to_solid(comm)
        self.jac_solid_to_fluid = self.assem_jac_solid_to_fluid(comm)
    
    def map_fluid_to_solid(self, fluid_vec, solid_vec):
        solid_vec[self.dofs_solid] = fluid_vec[self.dofs_fluid]

    def map_solid_to_fluid(self, solid_vec, fluid_vec):
        fluid_vec[self.dofs_solid] = solid_vec[self.dofs_fluid]

    def assem_jac_fluid_to_solid(self, comm):
        A = ptc.Mat().create(comm)
        A.setSizes([self.N_SOLID, self.N_FLUID])
        A.setUp()
        for jj, ii in self.fluid_to_solid_idx.items():
            A.setValue(ii, jj, 1)
        A.assemble()
        return A

    def assem_jac_solid_to_fluid(self, comm):
        A = ptc.Mat().create(comm)
        A.setSizes([self.N_FLUID, self.N_SOLID])
        A.setUp()
        for jj, ii in self.solid_to_fluid_idx.items():
            A.setValue(ii, jj, 1)
        A.assemble()
        return A

class FSIDynamicalSystem(DynamicalSystem):
    """
    Class representing a fluid-solid coupled dynamical system
    """

    def __init__(self, solid_model, fluid_model):
        self.solid = solid_model
        self.fluid = fluid_model

        self.models = (self.solid, self.fluid)
        self.state = bla.concatenate([model.state for model in self.models])
        self.statet = bla.concatenate([model.statet for model in self.models])

        self.dstate = bla.concatenate([model.dstate for model in self.models])
        self.dstatet = bla.concatenate([model.dstatet for model in self.models])

        # Set extra stuff needed for the FSI
        self.ymid = self.solid.properties['y_coll']
        self.solid_area = dfn.Function(self.solid.forms['function_space.scalar']).vector()

        self.solid_xref = self.solid.XREF

    def set_state(self, state):
        self.state[:] = state

        ## The below are needed to communicate FSI interactions
        # Set solid_area
        self.solid_area[:] = self.ymid - (self.solid_xref + self.solid.state['u'])[1::2]

        # map solid_area to fluid area

        # map fluid pressure to solid pressure

    def set_statet(self, statet):
        self.statet[:] = statet


    # have to override the default set_properties method because the so
    # the solid property can't be set using solid.properties[:] = ....
    # properties manually using setter methods
    def set_properties(self, props):
        
        nsolid = self.solid.properties.bsize
        self.solid.set_properties(props[:nsolid])
        self.fluid.set_properties(props[nsolid:])

    # Extra methods needed for FSI

def assign_vec_into_subvecs(vec, subvecs):
    """
    Assigns a BlockVector to a sequence of sub BlockVectors

    Parameters
    ----------
    vec : BlockVec
    subvecs : List of BlockVec
    """
    # Check that vector sizes are compatible
    # subvecs_total_size should concatenate the sizes of all subvecs
    # subvecs_total_size == vec.size??

    # Store the current part of `vec` that has not been assigned to a subvec
    _vec = vec
    for subvec in subvecs:
        subvec_block_size = subvec.bsize
        subvec[:] = _vec[:subvec_block_size]
        _vec = _vec[subvec_block_size:]

