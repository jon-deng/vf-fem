"""
Generic nonlinear dynamical system class

The class `BaseDynamicalModel` mainly specifies the attributes/methods that
specific dynamical system classes must contain but does not contain any
implementation details.
"""

from typing import TypeVar, Union

from blockarray import subops
from blockarray import blockvec as bv, blockmat as bm

T = TypeVar('T')
Vector = Union[subops.DfnVector, subops.GenericSubarray, subops.PETScVector]
Matrix = Union[subops.DfnMatrix, subops.GenericSubarray, subops.PETScMatrix]

BlockVec = bv.BlockVector[Vector]
BlockMat = bm.BlockMatrix[Matrix]

class BaseDynamicalModel:

    # def __init__()

    def set_state(self, state: BlockVec):
        raise NotImplementedError()

    def set_statet(self, statet: BlockVec):
        raise NotImplementedError()

    def set_control(self, control: BlockVec):
        raise NotImplementedError()

    def set_prop(self, prop: BlockVec):
        raise NotImplementedError()


    def set_dstate(self, dstate: BlockVec):
        raise NotImplementedError()

    def set_dstatet(self, dstatet: BlockVec):
        raise NotImplementedError()

    def set_dcontrol(self, dcontrol: BlockVec):
        raise NotImplementedError()


    def assem_res(self) -> BlockVec:
        raise NotImplementedError()

    def assem_dres_dstate(self) -> BlockVec:
        raise NotImplementedError()

    def assem_dres_dstatet(self) -> BlockMat:
        raise NotImplementedError()

    def assem_dres_dcontrol(self) -> BlockMat:
        raise NotImplementedError()

    def assem_dres_dprop(self) -> BlockMat:
        raise NotImplementedError()
