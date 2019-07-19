"""
Contains transformations of unit shapes to deformed shapes
"""

import numpy as np

def bilinear(X, x1, x2, x3, x4):
    """
    Transforms unit square to new square.

    Square node numbers increase counter clockwise starting with 1 in the lower left corner.

    Parameters
    ----------
    x1, x2, x3, x4 : New square corner coordinates
    """
    N1 = (1/2 - X[..., 0]) * (1/2 - X[..., 1])
    N2 = (1/2 + X[..., 0]) * (1/2 - X[..., 1])
    N3 = (1/2 + X[..., 0]) * (1/2 + X[..., 1])
    N4 = (1/2 - X[..., 0]) * (1/2 + X[..., 1])

    x = N1[..., None]*x1 + N2[..., None]*x2 \
        + N3[..., None]*x3 + N4[..., None]*x4
    return x
