from .basic import Functional

class UnaryFunctional(Functional):
    def __init__(self, model, g0, **kwargs):
        super(UnaryFunctional, self).__init__(model, **kwargs)
        self.func_args = (g0)

    @property
    def g0(self):
        return self.func_args[0]

class BinaryFunctional(Functional):
    def __init__(self, model, g0, g1, **kwargs):
        super(BinaryFunctional, self).__init__(model, **kwargs)
        self.func_args = (g0, g1)

    @property
    def g0(self):
        return self.func_args[0]

    @property
    def g1(self):
        return self.func_args[1]

class Penalty(BinaryFunctional):
    def __init__(self, model, g0, g1, **kwargs):
        super(Penalty, self).__init__(model, g0, g1, **kwargs)

        self.kwargs.setdefault('lambda_penalty', 1.0)

    def eval(self, f):
        return self.func_args[0](f) + 0.5*self.kwargs['lambda_penalty']*self.func_args[1](f)**2

    def du(self, f, n, iter_params0, iter_params1):
        return self.func_args[0].du(f, n, iter_params0, iter_params1) \
               + self.kwargs['lambda_penalty']*self.func_args[1].du(f, n, iter_params0, iter_params1)

    def dv(self, f, n, iter_params0, iter_params1):
        return self.func_args[0].dv(f, n, iter_params0, iter_params1) \
               + self.kwargs['lambda_penalty']*self.func_args[1].dv(f, n, iter_params0, iter_params1)

    def da(self, f, n, iter_params0, iter_params1):
        return self.func_args[0].da(f, n, iter_params0, iter_params1) \
               + self.kwargs['lambda_penalty']*self.func_args[1].da(f, n, iter_params0, iter_params1)

    def dp(self, f):
        return self.func_args[0].dp(f) \
               + self.kwargs['lambda_penalty']*self.func_args[1].dp(f)
