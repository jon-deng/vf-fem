"""
Transient fluid model definition

TODO: Change smoothing parameters to all be expressed in units of length
(all the smoothing parameters have a smoothing effect that occurs over a small length.
The smaller the length, the sharper the smoothing. )
"""

from numpy.typing import ArrayLike
import jax

from blockarray import blockvec as bla
from . import base

from ..jaxutils import blockvec_to_dict, flatten_nested_dict

from ..equations import fluid

## 1D Bernoulli approximation codes


class Model(base.BaseTransientModel):

    def __init__(self, residual: fluid.JaxResidual):
        res, (state, control, prop) = residual.res, residual.res_args

        self._res = jax.jit(res)
        self._dres = lambda state, control, prop, tangents: jax.jvp(
            res, (state, control, prop), tangents
        )[1]

        self.state0 = bla.BlockVector(list(state.values()), labels=[list(state.keys())])
        self.state1 = self.state0.copy()

        self.control = bla.BlockVector(
            list(control.values()), labels=[list(control.keys())]
        )

        self.prop = bla.BlockVector(list(prop.values()), labels=[list(prop.keys())])

        self.primals = (
            blockvec_to_dict(self.state1),
            blockvec_to_dict(self.control),
            blockvec_to_dict(self.prop),
        )

    @property
    def fluid(self):
        return self

    ## Parameter setting functions
    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        self._dt = value

    def set_ini_state(self, state):
        """
        Set the initial fluid state
        """
        self.state0[:] = state

    def set_fin_state(self, state):
        """
        Set the final fluid state
        """
        self.state1[:] = state

    def set_control(self, control):
        """
        Set the final surface displacement and velocity
        """
        self.control[:] = control

    def set_prop(self, prop):
        """
        Set the fluid properties
        """
        self.prop[:] = prop

    ## Residual functions
    # TODO: Make remaining residual/solving functions
    def assem_res(self):
        labels = self.state1.labels
        subvecs = self._res(*self.primals)
        subvecs, shape = flatten_nested_dict(subvecs, labels)
        return bla.BlockVector(subvecs, shape, labels)

    ## Solver functions
    def solve_state1(self, state1, options=None):
        """
        Return the final flow state
        """
        info = {}
        return self.state1 - self.assem_res(), info


class PredefinedModel(Model):
    def __init__(self, mesh: ArrayLike, *args, **kwargs):
        residual = self._make_residual(mesh, *args, **kwargs)
        super().__init__(residual)

    def _make_residual(self, mesh: ArrayLike, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")


class BernoulliSmoothMinSep(PredefinedModel):
    """
    Bernoulli fluid model with separation at the minimum
    """

    def _make_residual(self, mesh):
        return fluid.BernoulliSmoothMinSep(mesh)


class BernoulliFixedSep(PredefinedModel):
    """
    Bernoulli fluid model with separation at the minimum
    """

    def _make_residual(self, mesh, idx_sep=0):
        return fluid.BernoulliFixedSep(mesh, idx_sep=idx_sep)


class BernoulliAreaRatioSep(PredefinedModel):
    """
    Bernoulli fluid model with separation at the minimum
    """

    def _make_residual(self, mesh):
        return fluid.BernoulliAreaRatioSep(mesh)
