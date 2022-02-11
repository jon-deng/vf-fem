"""
Contains class definitions for coupled dynamical systems models
"""

from petsc4py import PETSc as ptc

class FSIMap:
    """
    Represents a mapping between two domains (fluid and solid)

    This mapping involves a 1-to-1 correspondence between DOFs on the two domains
    """
    def __init__(self, ndof_fluid, ndof_solid, fluid_dofs, solid_dofs):
        self.N_FLUID = ndof_fluid
        self.N_SOLID = ndof_solid

        self.dofs_fluid = fluid_dofs
        self.dofs_solid = solid_dofs

        self.fluid_to_solid_idx = {idxf: idxs for idxf, idxs in zip(fluid_dofs, solid_dofs)}
        self.solid_to_fluid_idx = {idxf: idxs for idxf, idxs in zip(fluid_dofs, solid_dofs)}
    
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