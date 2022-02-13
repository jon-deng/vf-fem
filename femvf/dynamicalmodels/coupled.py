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

    def assem_jac_fluid_to_solid(self, comm=None):
        A = ptc.Mat().create(comm)
        A.setSizes([self.N_SOLID, self.N_FLUID])
        A.setUp()
        for jj, ii in self.fluid_to_solid_idx.items():
            A.setValue(ii, jj, 1)
        A.assemble()
        return A

    def assem_jac_solid_to_fluid(self, comm=None):
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

    def __init__(self, solid_model, fluid_model, solid_fsi_dofs, fluid_fsi_dofs):
        self.solid = solid_model
        self.fluid = fluid_model
        self.models = (self.solid, self.fluid)

        self.state = bla.concatenate_vec([model.state for model in self.models])
        self.statet = bla.concatenate_vec([model.statet for model in self.models])

        self.dstate = bla.concatenate_vec([model.dstate for model in self.models])
        self.dstatet = bla.concatenate_vec([model.dstatet for model in self.models])

        # Set extra stuff needed for FSI
        self.ymid = self.solid.properties['y_coll']
        self.solid_area = dfn.Function(self.solid.forms['function_space.scalar']).vector()
        self.dsolid_area = dfn.Function(self.solid.forms['function_space.scalar']).vector()
        # have to compute dslarea_du here as sensitivity of solid area wrt displacement function

        self.solid_xref = self.solid.XREF

        # solid and fluid fsi dofs should be created when the two models are created
        self.fsimap = FSIMap(
            self.fluid.state['p'].size, self.solid_area.size(), fluid_fsi_dofs, solid_fsi_dofs)

        # These area jacobians of the mapping of scalars at the FSI interface from one domain to the
        # other
        self._dsolid_dfluid = self.fsimap.assem_jac_fluid_to_solid()
        self._dfluid_dsolid = self.fsimap.assem_jac_solid_to_fluid()

        # mats = [[0, self._dsolid_dfluid]]
        # self.dslicontrol_dflstate = bla.BlockMat(mats)

        # dslarea_dslstate # will have to compute the sensitivity of solid area to solid state; also a blockmatrix
        # mats = [[0, self._dfluid_dsolid*dslarea_dslstate]]
        # self.dflicontrol_dslstate = bla.BlockMat(mats)

        # mats = [[]]
        # self.dslicontrol_dflstatet = bla.BlockMat(mats)

        # mats = [[]]
        # self.dflicontrol_dslstatet = bla.BlockMat(mats)

    def set_state(self, state):
        self.state[:] = state

        ## The below are needed to communicate FSI interactions
        # Set solid_area
        self.solid_area[:] = self.ymid - (self.solid_xref + self.solid.state['u'])[1::2]

        # map solid_area to fluid area
        fluid_control = self.fluid.icontrol.copy()
        self.fsimap.map_solid_to_fluid(self.solid_area, fluid_control['area'])
        self.fluid.set_icontrol(fluid_control)

        # map fluid pressure to solid pressure
        solid_control = self.solid.icontrol.copy()
        self.fsimap.map_fluid_to_solid(self.fluid.state['p'], solid_control['p'])
        self.solid.set_icontrol(solid_control)

    def set_dstate(self, dstate):
        self.dstate[:] = dstate

        ## The below are needed to communicate FSI interactions
        # map linearized state to linearized solid area
        self.dsolid_area[:] = - (self.dstate['u'])[1::2]

        # map linearized solid area to fluid area
        dfluid_control = self.fluid.dicontrol.copy()
        dfluid_control['area'] = self._dfluid_dsolid * self.dsolid_area
        self.fluid.set_dicontrol(dfluid_control)

        # map linearized fluid pressure to solid pressure
        dsolid_control = self.solid.icontrol.copy()
        dsolid_control['p'] = self._dsolid_dfluid * self.fluid.dstate['p']
        self.solid.set_dicontrol(dsolid_control)

    # Since the fluid has no time dependence there should be no need to set FSI interactions here
    # for the specialized 1D Bernoulli model so I've left it empty for now
    def set_statet(self, statet):
        self.statet[:] = statet

        # ## The below are needed to communicate FSI interactions
        # # Set solid_area
        # self.solid_area[:] = self.ymid - (self.solid_xref + self.solid.statet['u'])[1::2]

        # # map solid_area to fluid area
        # fluid_control = self.fluid.icontrol.copy()
        # self.fsimap.map_solid_to_fluid(self.solid_area, fluid_control['area'])
        # self.fluid.set_icontrol(fluid_control)

        # # map fluid pressure to solid pressure
        # solid_control = self.solid.icontrol.copy()
        # self.fsimap.map_fluid_to_solid(self.fluid.state['p'], solid_control['p'])
        # self.solid.set_icontrol(solid_control)
        # pass

    def set_dstatet(self, dstatet):
        self.dstatet[:] = dstatet

        # ## The below are needed to communicate FSI interactions
        # # map linearized state to linearized solid area
        # self.dsolid_area[:] = - (self.dstate['u'])[1::2]

        # # map linearized solid area to fluid area
        # dfluid_control = self.fluid.dicontrol.copy()
        # dfluid_control['area'] = self.fsimap.assem_jac_solid_to_fluid() * self.dsolid_area
        # self.fluid.set_dicontrol(dfluid_control)

        # # map linearized fluid pressure to solid pressure
        # dsolid_control = self.solid.icontrol.copy()
        # dsolid_control['p'] = self.fsimap.assem_jac_fluid_to_solid() * self.fluid.dstate['p']
        # self.solid.set_dicontrol(dsolid_control)
        # pass


    # have to override the default set_properties method because the so
    # the solid property can't be set using solid.properties[:] = ....
    # properties manually using setter methods
    def set_properties(self, props):
        
        nsolid = self.solid.properties.bsize
        self.solid.set_properties(props[:nsolid])
        self.fluid.set_properties(props[nsolid:])

    def assem_res(self):
        return bla.concatenate_vec([model.assem_res() for model in self.models])

    def assem_dres_dstate(self):
        dfsolid_dxsolid = self.models[0].assem_dres_dstate()
        dfsolid_dxfluid = self.models[0].assem_dres_dicontrol() * self.dslicontrol_dflstate

        dffluid_dxfluid = self.models[1].assem_dres_dstate()
        dffluid_dxsolid = self.models[1].assem_dres_dicontrol() * self.dflicontrol_dflstate
        bmats = [
            [dfsolid_dxsolid, dfsolid_dxfluid],
            [dffluid_dxsolid, dffluid_dxfluid]]
        return bla.concatenate_mat(bmats)

    def assem_dres_dstatet(self):
        dfsolid_dxsolid = self.models[0].assem_dres_dstatet()
        dfsolid_dxfluid = self.models[0].assem_dres_dicontrol() * self.dslicontrol_dflstate

        dffluid_dxfluid = self.models[1].assem_dres_dstatet()
        dffluid_dxsolid = self.models[1].assem_dres_dicontrol() * self.dflicontrol_dflstate
        bmats = [
            [dfsolid_dxsolid, dfsolid_dxfluid],
            [dffluid_dxsolid, dffluid_dxfluid]]
        return bla.concatenate_mat(bmats)

    def assem_dres_dprops(self):
        dfsolid_dxsolid = self.models[0].assem_dres_dprops()
        dfsolid_dxfluid = self.models[0].assem_dres_dicontrol() * self.dslicontrol_dflstatet

        dffluid_dxfluid = self.models[1].assem_dres_dprops()
        dffluid_dxsolid = self.models[1].assem_dres_dicontrol() * self.dslicontrol_dflstatet
        bmats = [
            [dfsolid_dxsolid, dfsolid_dxfluid],
            [dffluid_dxsolid, dffluid_dxfluid]]
        return bla.concatenate_mat(bmats)

    

def assign_vec_into_subvecs(vec, subvecs):
    """
    Assigns a BlockVector to a sequence of sub BlockVectors

    Parameters
    ----------
    vec : BlockVec
    subvecs : List of BlockVec
    """
    # Check that vector sizes are compatible
    # subvecs_total_size should concatenate_vec the sizes of all subvecs
    # subvecs_total_size == vec.size??

    # Store the current part of `vec` that has not been assigned to a subvec
    _vec = vec
    for subvec in subvecs:
        subvec_block_size = subvec.bsize
        subvec[:] = _vec[:subvec_block_size]
        _vec = _vec[subvec_block_size:]

