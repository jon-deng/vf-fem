"""
Implements some basic math operations tha can be used to combine functionals into new functionals.
"""
import numpy as np
from .abstract import AbstractFunctional

def add(funca, funcb):
    """
    Add functionals
    """
    if funca.model != funcb.model:
        raise ValueError("Functionals must use the same model")

    return Sum(funca.model, funca, funcb)

def product(funca, funcb):
    if funca.model != funcb.model:
        raise ValueError("Functionals must use the same model")

    return Product(funca.model, funca, funcb)

class Sum(AbstractFunctional):
    """
    A functional formed from a sum of two other functionals
    """
    def eval(self, f):
        funca, funcb = self.funcs
        return funca(f) + funcb(f)

    def eval_duva(self, f, n, iter_params0, iter_params1):
        funca, funcb = self.funcs
        fa_duva = funca.duva(f, n, iter_params0, iter_params1)
        fb_duva = funcb.duva(f, n, iter_params0, iter_params1)
        return fa_duva + fb_duva

    def eval_dqp(self, f, n, iter_params0, iter_params1):
        funca, funcb = self.funcs
        fa_dqp = funca.dqp(f, n, iter_params0, iter_params1)
        fb_dqp = funcb.dqp(f, n, iter_params0, iter_params1)
        return fa_dqp + fb_dqp

    def eval_dp(self, f):
        funca, funcb = self.funcs

        fa_dp = funca.dp(f)
        fb_dp = funcb.dp(f)
        return fa_dp + fb_dp

class Product(AbstractFunctional):
    """
    A functional formed from a product of two other functionals
    """
    def eval(self, f):
        funca, funcb = self.funcs
        return funca(f) * funcb(f)

    def eval_duva(self, f, n, iter_params0, iter_params1):
        funca, funcb = self.funcs
        fa_duva = funca.duva(f, n, iter_params0, iter_params1)
        fb_duva = funcb.duva(f, n, iter_params0, iter_params1)
        return fa_duva*funcb(f) + funca(f)*fb_duva

    def eval_dqp(self, f, n, iter_params0, iter_params1):
        funca, funcb = self.funcs
        fa_dqp = funca.dqp(f, n, iter_params0, iter_params1)
        fb_dqp = funcb.dqp(f, n, iter_params0, iter_params1)
        return fa_dqp*funcb(f) + funca(f)*fb_dqp

    def eval_dp(self, f):
        funca, funcb = self.funcs

        fa_dp = funca.dp(f)
        fb_dp = funcb.dp(f)
        return fa_dp*funcb(f) + funca(f)*fb_dp

class Power(AbstractFunctional):
    """
    A functional formed by raising a functional to the power of another
    """
    def eval(self, f):
        funca, funcb = self.funcs
        return funca(f) ** funcb(f)

    def eval_duva(self, f, n, iter_params0, iter_params1):
        funca, funcb = self.funcs
        fa, fb = funca(f), funcb(f)
        fa_duva = funca.duva(f, n, iter_params0, iter_params1)
        fb_duva = funcb.duva(f, n, iter_params0, iter_params1)
        return fb*fa**(fb-1) * fa_duva + np.log(fa)*fa**fb*fb_duva

    def eval_dqp(self, f, n, iter_params0, iter_params1):
        funca, funcb = self.funcs
        fa_dqp = funca.dqp(f, n, iter_params0, iter_params1)
        fb_dqp = funcb.dqp(f, n, iter_params0, iter_params1)
        return fa_dqp*funcb(f) + funca(f)*fb_dqp

    def eval_dp(self, f):
        funca, funcb = self.funcs

        fa_dp = funca.dp(f)
        fb_dp = funcb.dp(f)
        return fa_dp*funcb(f) + funca(f)*fb_dp

class Constant(AbstractFunctional):
    """
    Functional that always evaluates to a constant
    """
    func_types = ()
    default_constants = {
        'value': 0.0
    }

    def eval(self, f):
        return self.constants['value']

    def eval_duva(self, f, n, iter_params0, iter_params1):
        return (0.0, 0.0, 0.0)

    def eval_dp(self, f):
        return None
