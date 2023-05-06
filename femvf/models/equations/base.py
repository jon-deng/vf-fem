"""
This module defines the basic non-linear residual
"""

from typing import Callable, Mapping, Any

class SolidResidual:
    """
    Represents a symbolic (`UFL`) residual
    """

    def __init__(self, res, param):
        self._res = res
        self._param = param

    _param: Mapping[str, Any]
    _res: Any

    @property
    def param(self):
        """
        Return parameters of the residual
        """
        return self._param

    def __getitem__(self, key):
        return self.param[key]

    def __neg__(self):
        return ResidualMultiple(self, -1)

    def __mul__(self, scalar):
        return ResidualMultiple(self, scalar)

    def __rmul__(self, other: 'BaseResidual'):
        return ResidualSum(self, other)

    def __add__(self, other: 'BaseResidual'):
        return ResidualSum(self, other)

    def __radd__(self, other: 'BaseResidual'):
        return ResidualSum(self, other)


class ResidualSum(BaseResidual):

    def __init__(self, res1, res2):
        self._res = res1._res + res2._res
        self._param = res1.param + res2.param

class ResidualMultiple(BaseResidual):

    def __init__(self, res, scalar):
        self._res = scalar*res._res
        self._param = res1.param + res2.param
