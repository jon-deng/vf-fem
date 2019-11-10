"""
Contains special linear operator definitions.
"""

from dolfin.cpp.la import LinearOperator

class LinCombOfMats(LinearOperator):
    def __init__(self, *args):
        super(LinCombOfMats, self).__init__()
        self.ops = args

    def mult(self, x, y):
        for op in ops:
            scale, mat = op
            y[:] += scale * (mat * x)

# class ProdOfMats(LinearOperator):
#     def __init__(*args):


#     def mult():