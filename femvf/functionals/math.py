"""
Implements some basic math operations tha can be used to combine functionals into new functionals.
"""

from .abstract import AbstractFunctional

def add(funca, funcb):
    """
    Add functionals
    """
    if funca.model != funcb.model:
        raise ValueError("Functionals must use the same model")

    class Sum(AbstractFunctional):
        def eval(self, f):
            funca, funcb = self.funcs
            return funca(f) + funcb(f)

        def eval_duva(self, f, n, iter_params0, iter_params1):
            funca, funcb = self.funcs
            funca_duva = funca.duva(f, n, iter_params0, iter_params1)
            funcb_duva = funcb.duva(f, n, iter_params0, iter_params1)
            return funca_duva + funcb_duva

        def eval_dqp(self, f, n, iter_params0, iter_params1):
            funca, funcb = self.funcs
            funca_dqp = funca.dqp(f, n, iter_params0, iter_params1)
            funcb_dqp = funcb.dqp(f, n, iter_params0, iter_params1)
            return funca_dqp + funcb_dqp

        def eval_dp(self, f):
            funca, funcb = self.funcs

            funca_dp = funca.dp(f)
            funcb_dp = funcb.dp(f)
            return funca_dp + funcb_dp

    return Sum(funca.model, funca, funcb)

def sub(funca, funcb):
    if funca.model != funcb.model:
        raise ValueError("Functionals must use the same model")

    class Difference(AbstractFunctional):
        def eval(self, f):
            funca, funcb = self.funcs
            return funca(f) - funcb(f)

        def eval_duva(self, f, n, iter_params0, iter_params1):
            funca, funcb = self.funcs
            funca_duva = funca.duva(f, n, iter_params0, iter_params1)
            funcb_duva = funcb.duva(f, n, iter_params0, iter_params1)
            return funca_duva - funcb_duva

        def eval_dqp(self, f, n, iter_params0, iter_params1):
            funca, funcb = self.funcs
            funca_dqp = funca.dqp(f, n, iter_params0, iter_params1)
            funcb_dqp = funcb.dqp(f, n, iter_params0, iter_params1)
            return funca_dqp - funcb_dqp

        def eval_dp(self, f):
            funca, funcb = self.funcs

            funca_dp = funca.dp(f)
            funcb_dp = funcb.dp(f)
            return funca_dp - funcb_dp

    return Difference(funca.model, funca, funcb)

