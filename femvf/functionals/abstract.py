"""
Contains definition of the abstract functional
"""

import dolfin as dfn

def new_statefile(self, f):
    statefile_updated = self._f is None or self._f != f

    if not statefile_updated:
        assert self._f == f

    return statefile_updated

def update_cache(func):
    def wrapped_func(self, f, *args, **kwargs):
        """
        Reruns the `eval` function if the passed in file instance is different from what was last
        run and the functional needs caching, as specified by `CACHE`
        """
        if self.CACHE and new_statefile(self, f):
            # Run the functional to calculate the objective function and update any cached values
            self(f)

        return func(self, f, *args, **kwargs)

    return wrapped_func


class AbstractFunctional:
    """
    This class represents a functional to be computed given a solved forward model

    After running a functional over forward model states, the value is cached in _value.

    To make a new functional ....

    Parameters
    ----------
    model : model.ForwardModel
        The forward model instance
    funcs : tuple
        A tuple of other functionals, that are computed as part of the given functional

    Attributes
    ----------
    model : model.ForwardModel
    funcs : tuple of Functional
        A collection of functional instances used in computing the given functional
    constants : dict
        A dictionary of additional options of how to compute the functional
    value : float
        The value of the functional. This should be a constant any time it's computed because
        the instance of the functional is always tied to the same file.
    cache : dict
        A dictionary of cached values. These are specific to the functional and values are likely
        to be cached if they are needed to compute sensitivities and are expensive to compute.
    """
    CACHE = True

    ## Subclasses should also add a `default_constants` class attribute
    default_constants = {}

    def __init__(self, model, *funcs):
        self.model = model
        self._forms = self.form_definitions(model)
        self.funcs = tuple(funcs)

        import copy
        self.constants = copy.deepcopy(type(self).default_constants)

        # A cache of other quantities
        self.cache = dict()

        # A cache for the current value of the functional, and the file it was evaluated on
        self._value = None
        self._f = None

    def __call__(self, f):
        if new_statefile(self, f):
            self._value = self.eval(f)

        self._f = f
        return self._value

    @property
    def forms(self):
        return self._forms

    @update_cache
    def duva(self, f, n, iter_params0, iter_params1):
        """
        Return the sensitivity of the functional to :math:`u^n`

        Parameters
        ----------
        n : int
            Index of state
        f : statefile.StateFile
            The history of states to compute it over
        iter_params0, iter_params1 : dict
            Dictionary of parameters that specify iteration n. These are parameters fed into
            `model.ForwardModel.set_iter_params` with signature:
            `(uva0=None, dt=None, u1=None, solid_props=None, fluid_props=None)`

            `iter_params0` specifies the parameters needed to map the states at `n-1` to the states
            at `n+0`.
        """
        in_duva = self.eval_duva(f, n, iter_params0, iter_params1)

        ret_duva = []
        for dz in in_duva:
            if isinstance(dz, float):
                _dz = dfn.Function(self.model.solid.vector_fspace).vector()
                _dz[:] = dz
                ret_duva.append(_dz)
            else:
                ret_duva.append(dz)
        return tuple(ret_duva)

    @update_cache
    def dqp(self, f, n, iter_params0, iter_params1):
        return self.eval_dqp(f, n, iter_params0, iter_params1)

    @update_cache
    def dp(self, f):
        return self.eval_dp(f)

    ## Subclasses have to implement these methods
    @staticmethod
    def form_definitions(model):
        """
        Return a dictionary of form definitions needed for computing the functional

        Only need to implement if the function relies of some forms defined in UFL
        """
        return {}

    def eval(self, f):
        """
        Return the value of the objective function for the state history in `f`
        """
        raise NotImplementedError("`eval` must be implemented")

    def eval_duva(self, f, n, iter_params0, iter_params1):
        """
        Return the sensitivity of the functional to :math:`u^n`

        Parameters
        ----------
        n : int
            Index of state
        f : statefile.StateFile
            The history of states to compute it over
        iter_params0, iter_params1 : dict
            Dictionary of parameters that specify iteration n. These are parameters fed into
            `model.ForwardModel.set_iter_params` with signature:
            `(uva0=None, dt=None, u1=None, solid_props=None, fluid_props=None)`

            `iter_params0` specifies the parameters needed to map the states at `n-1` to the states
            at `n+0`.
        """
        raise NotImplementedError("`eval_duva` must be implemented")

    def eval_dqp(self, f, n, iter_params0, iter_params1):
        raise NotImplementedError("You have to implement this")

    def eval_dp(self, f):
        """
        Return the sensitivity of the functional with respect to the parameters.
        """
        raise NotImplementedError("`eval_dp` must be implemented")

    def eval_dt(self, f, n):
        """
        Return the sensitivity of the functional with respect to the time step.
        """
        raise NotImplementedError("`eval_dt` must be implemented")

    # def __add__(self, other):

    # def __sub__(self, other):

    # def __mul__(self, other):

    # def __truediv__(self, other):

    # def __pow__(self, other):

    # def __radd__(self, other):
    #     return NotImplementedError("")

    # def __rsub__(self, other):
    #     return NotImplementedError("")

    # def __rmul__(self, other):
    #     return NotImplementedError("")

    # def __rtruediv__(self, other):
    #     return NotImplementedError("")

    # def __neg__(self):

    # def __pos__(self):


# class ScalarMultiple(AbstractFunctional):

# class Sum(AbstractFunctional):
#     def eval(self, f):
#         return self.funcs[0](f) + self.funcs[1](f)

#     def eval_duva(self, n, p0, p1):
#         return self.funcs[0].duva(f, n, p0, p1) + self.funcs[1].duva(f, n, p0, p1)

#     def eval_dp(n, p0, p1):
#         return self.funcs[0].duva(f, n, p0, p1) + self.funcs[1].duva(f, n, p0, p1)

# class ProductOfFuncs(AbstractFunctional):
#     def eval(self, f):
#         return
#     def eval_duva(self, n, p0, p1):

# class QuotientOfFuncs(AbstractFunctional):
#     def eval(self, f):
#         return
#     def eval_duva(self, n, p0, p1):
#     def eval_dp(n, p0, p1):

# class PowerOfFuncs(AbstractFunctional):
#     def eval(self, f):
#         return
#     def eval_duva(self, n, p0, p1):
#     def eval_dp(n, p0, p1):

# class ScalarMultiple(AbstractFunctional):
#     def eval(self, f):
#         return
#     def eval_duva(self, n, p0, p1):
#     def eval_dp(n, p0, p1):
