"""
This module defines the basic non-linear residual
"""

from typing import Mapping, Any, Callable

import ufl

class BaseResidual:

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

class UFLResidual(BaseResidual):
    """
    Represents a symbolic residual using UFL

    Parameters
    ----------
    res:
        A form in the UFL language
    param:
        A mapping from names to form parameters
    """

    def __init__(self, res, param):
        self._res = res
        self._param = param

class JAXResidual(BaseResidual):
    """
    Represents a symbolic residual using JAX

    Parameters
    ----------
    res : Callable
        A residual function implemented using JAX
    param:
        A mapping from names to form parameters
    """

    def __init__(self, res, param):
        self._res = res
        self._param = param
