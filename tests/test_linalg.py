"""
Test linalg functions
"""

import unittest

import numpy as np
from petsc4py import PETSc

from femvf import linalg

class Test_form_block_matrix(unittest.TestCase):
    def runTest(self):
        # A = [[1, 2, 3],
        #      [0, 0, 4],
        #      [0, 0, 5]]
        # B = [[6, 7],
        #      [8, 9],
        #      [0, 0]]
        # C = [[10, 11, 12],
        #      [ 0,  0, 13]]
        # D = [[1, 0],
        #      [0, 1]]
        Ai, Aj, Av = [0, 3, 4, 5], [0, 1, 2, 2, 2], [1, 2, 3, 4, 5]
        Bi, Bj, Bv = [0, 2, 4, 4], [0, 1, 0, 1], [6, 7, 8, 9]
        Ci, Cj, Cv = [0, 3, 4], [0, 1, 2, 2], [10, 11, 12, 13]
        Di, Dj, Dv = [0, 1, 2], [0, 1], [1, 1]

        def set_csr(mat, i, j, v):
            i, j = np.array(i, dtype=np.int32), np.array(j, dtype=np.int32),
            v = np.array(v, dtype=np.float)
            mat.setUp()
            mat.setValuesCSR(i, j, v)
            mat.assemble()
            return mat

        A = PETSc.Mat().create()
        A.setSizes([3, 3])
        A = set_csr(A, Ai, Aj, Av)

        B = PETSc.Mat().create()
        B.setSizes([3, 2])
        B = set_csr(B, Bi, Bj, Bv)

        C = PETSc.Mat().create()
        C.setSizes([2, 3])
        C = set_csr(C, Ci, Cj, Cv)

        D = PETSc.Mat().create()
        D.setSizes([2, 2])
        D = set_csr(D, Di, Dj, Dv)

        blocks = [[A, B], [C, 1]]
        block_mat = linalg.form_block_matrix(blocks)

        ABCD = [[1, 2, 3, 6, 7],
                [0, 0, 4, 8, 9],
                [0, 0, 5, 0, 0],
                [10, 11, 12, 1, 0],
                [0, 0, 13, 0, 1]]

        # print(block_mat[:, :])

        self.assertTrue(np.all(block_mat[:, :] == np.array(ABCD)))

class Test_reorder_mat_rows(unittest.TestCase):
    def runTest(self):
        A = [[0, 0, 0, 0],
             [1, 2, 3, 4],
             [0, 5, 6, 7],
             [0, 0, 8, 9],
             [0, 0, 0, 10],
             [0, 0, 0, 0]]
        Ai = np.array([0, 0, 4, 7, 9, 10, 10], dtype=np.int32)
        Aj = np.array([0, 1, 2, 3, 1, 2, 3, 2, 3, 3], dtype=np.int32)
        Av = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float)

        A = PETSc.Mat().create()
        A.setSizes([6, 4])
        A.setUp()
        A.setPreallocationNNZ(Ai[1:]-Ai[:-1])

        A.setValuesCSR(Ai, Aj, Av)
        A.assemble()

        # print(A[:, :])

        rows_in = np.array([1, 2, 3, 4])
        rows_out = np.array([3, 2, 1, 0])
        A_reorder = linalg.reorder_mat_rows(A, rows_in, rows_out, 4)

        B = [[0, 0, 0, 10],
             [0, 0, 8, 9],
             [0, 5, 6, 7],
             [1, 2, 3, 4]]
        self.assertTrue(np.all(A_reorder[:, :] == np.array(B)))
        # print(A_reorder[:, :])

if __name__ == '__main__':
    test = Test_form_block_matrix()
    test.runTest()

    test = Test_reorder_mat_rows()
    test.runTest()
