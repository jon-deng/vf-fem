"""
Implements some basic math operations tha can be used to combine functionals into new functionals.
"""
import numpy as np
from .abstract import AbstractFunctional
# from .basic import Scalar


def add(funa, funb):
    """
    Add functionals
    """
    funa, funb = _convert_float(funa, funb)
    _validate(funa, funb)

    return Sum(funa.model, funa, funb)

def sub(funa, funb):
    """
    Add functionals
    """
    funa, funb = _convert_float(funa, funb)
    _validate(funa, funb)

    return Sum(funa.model, funa, mul(-1.0, funb))

def mul(funa, funb):
    funa, funb = _convert_float(funa, funb)
    _validate(funa, funb)

    return Product(funa.model, funa, funb)

def power(funa, funb):
    funa, funb = _convert_float(funa, funb)
    _validate(funa, funb)

    if isinstance(funb, Scalar):
        return ScalarPower(funa.model, funa, funb)
    else:
        return Power(funa.model, funa, funb)

def _validate(*funs):
    _model = funs[0].model

    for fun in funs[1:]:
        if fun.model != _model:
            raise ValueError("Functionals must use the same model")

def _convert_float(funa, funb):
    """
    Converts floats to scalar functionals
    """
    funa_isfunc = isinstance(funa, AbstractFunctional)
    funb_isfunc = isinstance(funb, AbstractFunctional)

    if funa_isfunc and funb_isfunc:
        pass
    elif funa_isfunc and isinstance(funb, float):
        funb = Scalar(funa.model, funb)
    elif funb_isfunc and isinstance(funa, float):
        funa = Scalar(funb.model, funa)
    else:
        raise TypeError("something went wrong")

    return funa, funb


class Sum(AbstractFunctional):
    """
    A functional representing a + b, where a and b are functionals
    """
    def __init__(self, model, funa, funb):
        super().__init__(model, funa, funb)

    def eval(self, f):
        funa, funb = self.funcs
        return funa(f) + funb(f)

    @staticmethod
    def _sum_drule(funa, funb, dname, *args):
        dfuna = getattr(funa, dname)
        dfunb = getattr(funb, dname)

        # a, b = funa(*args), funb(*args)
        da, db = dfuna(*args), dfunb(*args)
        return da + db

    def eval_dstate(self, f, n):
        return self._sum_drule(*self.funcs, 'dstate', f, n)

    def eval_dprops(self, f):
        return self._sum_drule(*self.funcs, 'dprops', f)

    def eval_ddt(self, f, n):
        return self._sum_drule(*self.funcs, 'ddt', f, n)

    def eval_dt0(self, f, n):
        return self._sum_drule(*self.funcs, 'dt0', f, n)

class Product(AbstractFunctional):
    """
    A functional representing a * b, where a and b are functionals
    """
    def __init__(self, model, funa, funb):
        super().__init__(model, funa, funb)

    def eval(self, f):
        funa, funb = self.funcs
        return funa(f) * funb(f)

    @staticmethod
    def _product_drule(funa, funb, dname, *args):
        dfuna = getattr(funa, dname)
        dfunb = getattr(funb, dname)

        a, b = funa(args[0]), funb(args[0])
        da, db = dfuna(*args), dfunb(*args)

        return da*b + a*db

    def eval_dstate(self, f, n):
        return self._product_drule(*self.funcs, 'dstate', f, n)

    def eval_dprops(self, f):
        return self._product_drule(*self.funcs, 'dprops', f)

    def eval_ddt(self, f, n):
        return self._product_drule(*self.funcs, 'ddt', f, n)

    def eval_dt0(self, f, n):
        return self._product_drule(*self.funcs, 'dt0', f, n)

class Power(AbstractFunctional):
    """
    A functional representing a ** b, where a and b are functionals
    """
    def eval(self, f):
        funa, funb = self.funcs
        return funa(f) ** funb(f)

    @staticmethod
    def _power_drule(funa, funb, dname, *args):
        dfuna = getattr(funa, dname)
        dfunb = getattr(funb, dname)

        a, b = funa(args[0]), funb(args[0])
        da, db = dfuna(*args), dfunb(*args)
        
        assert a > 0 # This is not defined if a < 0
        return b*a**(b-1)*da + np.log(a)*a**b*db

    def eval_dstate(self, f, n):
        return self._power_drule(*self.funcs, 'dstate', f, n)

    def eval_dprops(self, f):
        return self._power_drule(*self.funcs, 'dprops', f)

    def eval_ddt(self, f, n):
        return self._power_drule(*self.funcs, 'ddt', f, n)

    def eval_dt0(self, f, n):
        return self._power_drule(*self.funcs, 'dt0', f, n)

class ScalarPower(AbstractFunctional):
    """
    A functional representing a ** b, where a is a functional and b a scalar
    """
    def eval(self, f):
        funa, funb = self.funcs
        a, b = funa(f), funb(f)
        return a**b

    @staticmethod
    def _scalarpower_drule(funa, funb, dname, *args):
        dfuna = getattr(funa, dname)
        dfunb = getattr(funb, dname)

        a, b = funa(args[0]), funb(args[0])
        da = dfuna(*args)
        # db = dfunb(*args)

        return b*a**(b-1) * da

    def eval_dstate(self, f, n):
        return self._scalarpower_drule(*self.funcs, 'dstate', f, n)

    def eval_dprops(self, f):
        return self._scalarpower_drule(*self.funcs, 'dprops', f)

    def eval_ddt(self, f, n):
        return self._scalarpower_drule(*self.funcs, 'ddt', f, n)

    def eval_dt0(self, f, n):
        return self._scalarpower_drule(*self.funcs, 'dt0', f, n)

class Scalar(AbstractFunctional):
    """
    Functional that always evaluates to a constant scalar
    """
    func_types = ()
    default_constants = {
        'value': 0.0
    }

    def __init__(self, model, val):
        self._val = val
        super().__init__(model)

    def eval(self, f):
        return self._val

    def eval_dstate(self, f, n):
        return self.model.get_state_vec()

    def eval_dprops(self, f):
        dprops = self.model.get_properties_vec()
        dprops.set(0.0)
        return dprops

    def eval_ddt(self, f, n):
        return 0.0

    def eval_dt0(self, f, n):
        return 0.0
