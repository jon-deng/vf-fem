"""
Contains definition of the abstract functional
"""

import dolfin as dfn

def new_statefile(self, f):
    """
    Return if `f` has been updated
    """
    statefile_updated = self._f is None or self._f != f

    if not statefile_updated:
        assert self._f == f

    return statefile_updated

def update_cache(func):
    """
    Return a decorated instance method that updates a cache attribute on new input files
    """
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

    This class should not be used directly to create functional except in special instances.

    After running a functional over forward model states, the value is cached in _value.

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
    # TODO : The code for functionals is quite hard to follow.
    # Rethink what are the basic things that are required. Not all the things in __init__
    # are likely needed so you can get rid of irrelevant stuff
    CACHE = True

    ## Subclasses should also add a `default_constants` class attribute
    default_constants = {}

    def __init__(self, model, *funcs):
        self.model = model

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

    @update_cache
    def dstate(self, f, n):
        return self.eval_dstate(f, n)

    @update_cache
    def dprops(self, f):
        return self.eval_dprops(f)

    @update_cache
    def ddt(self, f, n):
        """
        Return dg/ddt
        """
        return self.eval_ddt(f, n)

    @update_cache
    def dt0(self, f, n):
        """
        Return dg/ddt
        """
        return self.eval_dt0(f, n)

    ## Subclasses have to implement these methods
    def eval(self, f):
        """
        Return the value of the objective function for the state history in `f`
        """
        raise NotImplementedError("`eval` must be implemented")

    def eval_dstate(self, f, n):
        """
        Return dg/d(u, v, a)

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
        raise NotImplementedError("`eval_dstate` must be implemented")

    def eval_dprops(self, f):
        """
        Return the dg/dsolidparameters
        """
        raise NotImplementedError("`eval_dprops` must be implemented")

    def eval_dt0(self, f, n):
        """
        Return the dg/dt0
        """
        raise NotImplementedError("`eval_dt0` must be implemented")

    def eval_ddt(self, f, n):
        """
        Return the dg/ddt
        """
        raise NotImplementedError("`eval_ddt` must be implemented")
