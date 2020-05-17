"""
This module contains various utilities for sparse linear algebra
"""

import numpy as np
from petsc4py import PETSc

def form_block_matrix(blocks, finalize=True):
    """
    Form a monolithic block matrix by combining matrices in `block_mat`

    Parameters
    ----------
    block_mat : [[]]
        A list of lists containing the matrices forming the blocks of the desired block matrix.
        These are organized as:
            [[mat00, ..., mat0n],
             [mat10, ..., mat1n],
             [  ..., ...,   ...],
             [matm0, ..., matmn]]
    """
    blocks_shape = get_blocks_shape(blocks)
    blocks_sizes = get_blocks_sizes(blocks, blocks_shape)
    block_row_sizes, block_col_sizes = blocks_sizes

    blocks_csr = get_blocks_csr(blocks, blocks_shape)
    i_mono, j_mono, v_mono = get_block_matrix_csr(blocks_csr, blocks_shape, blocks_sizes)

    ## Create a monolithic matrix to contain the block matrix
    block_mat = PETSc.Mat()
    block_mat.create(PETSc.COMM_SELF)
    block_mat.setSizes([np.sum(block_row_sizes), np.sum(block_col_sizes)])

    block_mat.setUp()

    ## Insert the values into the matrix
    nnz = i_mono[1:] - i_mono[:-1]
    block_mat.setPreallocationNNZ(nnz)
    block_mat.setValuesCSR(i_mono, j_mono, v_mono)

    if finalize:
        block_mat.assemble()

    return block_mat

def get_blocks_shape(blocks):
    """
    Return the shape of the block matrix, and the sizes of each block
    """
    ## Get the block sizes
    # also check that the same number of columns are supplied in each row
    M_BLOCK = len(blocks)
    N_BLOCK = len(blocks[0])
    for row in range(1, M_BLOCK):
        assert N_BLOCK == len(blocks[row])

    return M_BLOCK, N_BLOCK

def get_blocks_sizes(blocks, blocks_shape):
    """
    Return the sizes of each block in the block matrix

    Returns
    -------
    block_row_sizes, block_col_sizes: np.ndarray
        An array containing the number of rows/columns along each row/column block. For example, if
        the block matrix contains blocks of shape
        [[(2, 5), (2, 6)],
         [(7, 5), (7, 6)]]
        `block_row_sizes` and `block_col_sizes` will be [2, 7] and [5, 6], respectively.
    """
    M_BLOCK, N_BLOCK = blocks_shape

    ## Calculate an array containing the n_rows in each row block
    # and n_columns in each column block
    block_row_sizes = -1*np.ones(M_BLOCK, dtype=np.intp)
    block_col_sizes = -1*np.ones(N_BLOCK, dtype=np.intp)

    # check that row/col sizes are consistent with the other row/col block sizes
    for row in range(M_BLOCK):
        for col in range(N_BLOCK):
            block = blocks[row][col]
            shape = None
            if isinstance(block, PETSc.Mat):
                shape = block.getSize()
            elif isinstance(block, (int, float)):
                # Use -1 to indicate a variable size 'diagonal' matrix, that will adopt
                # the shape of neighbouring blocks to form a proper block matrix
                shape = (-1, -1)
            else:
                raise ValueError("Blocks can only be matrices or floats")

            for block_sizes in (block_row_sizes, block_col_sizes):
                if block_sizes[row] == -1:
                    block_sizes[row] = shape[0]
                else:
                    assert (block_sizes[row] == shape[0]
                            or shape[0] == -1)

    # convert any purely variable size blocks to size 1 blocks
    block_row_sizes = np.where(block_row_sizes == -1, 1, block_row_sizes)
    block_col_sizes = np.where(block_col_sizes == -1, 1, block_col_sizes)
    return block_row_sizes, block_col_sizes

def get_blocks_csr(blocks, blocks_shape):
    """
    Return the CSR format data for each block in a block matrix form
    """
    M_BLOCK, N_BLOCK = blocks_shape

    # Grab all the CSR format values and put them into a block list form
    i_block = []
    j_block = []
    v_block = []

    for row in range(M_BLOCK):
        i_block_row = []
        j_block_row = []
        v_block_row = []
        for col in range(N_BLOCK):
            block = blocks[row][col]
            if isinstance(block, PETSc.Mat):
                i, j, v = block.getValuesCSR()
                i_block_row.append(i)
                j_block_row.append(j)
                v_block_row.append(v)
            else:
                # In this case the block should just be a constant value, like 1.0
                # to indicate an identity matrix
                i_block_row.append(None)
                j_block_row.append(None)
                v_block_row.append(block)
        i_block.append(i_block_row)
        j_block.append(j_block_row)
        v_block.append(v_block_row)

    return i_block, j_block, v_block

def get_block_matrix_csr(blocks_csr, blocks_shape, blocks_sizes):
    """
    Return csr data associated with monolithic block matrix
    """
    i_block, j_block, v_block = blocks_csr
    M_BLOCK, N_BLOCK = blocks_shape
    block_row_sizes, block_col_sizes = blocks_sizes

    # block_row_offsets = np.concatenate(([0] + np.cumsum(block_row_sizes)[:-1]))
    block_col_offsets = np.concatenate(([0] + np.cumsum(block_col_sizes)[:-1]))

    i_mono = [0]
    j_mono = []
    v_mono = []

    for row in range(M_BLOCK):
        for local_row in range(block_row_sizes[row]):
            j_mono_row = []
            v_mono_row = []
            for col in range(N_BLOCK):
                # get the CSR data associated with the specific block
                i, j, v = i_block[row][col], j_block[row][col], v_block[row][col]

                # if the block is not a matrix, handle this case as if it's a diagonal block
                if i is None:
                    if v != 0 or row == col:
                        # only set the 'diagonal' if the matrix dimension is appropriate
                        # i.e. if the block is a tall rectangular one, don't keep
                        # writing diagonals when the row index > # cols since this is
                        # undefined
                        if local_row < block_col_sizes[col]:
                            j_mono_row += [local_row + block_col_offsets[col]]
                            v_mono_row += [v]
                else:
                    istart = i[local_row]
                    iend = i[local_row+1]

                    j_mono_row += (j[istart:iend] + block_col_offsets[col]).tolist()
                    v_mono_row += v[istart:iend].tolist()
            i_mono += [i_mono[-1] + len(v_mono_row)]
            j_mono += j_mono_row
            v_mono += v_mono_row

    i_mono = np.array(i_mono, dtype=np.int32)
    j_mono = np.array(j_mono, dtype=np.int32)
    v_mono = np.array(v_mono, dtype=np.float)
    return i_mono, j_mono, v_mono
