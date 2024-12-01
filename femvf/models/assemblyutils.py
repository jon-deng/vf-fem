"""
FEniCS form assembly utilities
"""

import ufl
import dolfin as dfn

from blockarray.subops import zero_mat

from femvf.equations.form import Form


class CachedUFLFormAssembler:
    """
    Assemble a UFL/Dolfin form and cache the sparsity pattern

    Parameters
    ----------
    form : ufl.Form
    kwargs :
        Keyword arguments of `dfn.assemble`
    """

    def __init__(self, form: ufl.Form, **kwargs):
        self._form = form

        tensor = kwargs.pop('tensor', None)
        if tensor is None:
            if len(form.arguments()) == 0:
                tensor = dfn.Scalar()
            elif len(form.arguments()) == 1:
                tensor = dfn.PETScVector()
            elif len(form.arguments()) == 2:
                tensor = dfn.PETScMatrix()
            else:
                raise ValueError("Form arity must be between 0 and 2")

        self._tensor = tensor
        self._kwargs = kwargs

    @property
    def tensor(self):
        return self._tensor

    @property
    def form(self):
        return self._form

    def assemble(self):
        return dfn.assemble(self.form, tensor=self.tensor, **self._kwargs)


Tensor = dfn.PETScMatrix | dfn.PETScVector

class FormAssembler:
    """
    Assemble associated UFL forms from a `Form` instance

    Parameters
    ----------
    form : Form
    """

    def __init__(self, form: Form):
        self._form = form
        self._cached_assemblers = {}

    @property
    def form(self) -> Form:
        return self._form

    @property
    def assemblers(self) -> dict[str, CachedUFLFormAssembler | Tensor]:
        return self._cached_assemblers

    def assemble_from_cache(self, cache_key: str):
        if cache_key in self.assemblers:
            assembler = self.assemblers[cache_key]
            if isinstance(assembler, CachedUFLFormAssembler):
                return assembler.assemble()
            else:
                return assembler
        else:
            raise KeyError(f"{cache_key}")

    def assemble(self, form_key: str):
        """
        Assemble a residual from the form
        """
        cache_key = form_key

        if cache_key not in self.assemblers:
            ufl_form = self.form.ufl_forms[form_key]
            self.assemblers[cache_key] = CachedUFLFormAssembler(ufl_form)

        return self.assemble_from_cache(cache_key)

    def assemble_derivative(
            self,
            form_key: str,
            coefficient_key: str,
            adjoint: bool=False
        ):
        """
        Assemble a residual from the form
        """
        def gen_cache_key(form_key: str, coefficient_key: str, adjoint: bool):
            if adjoint:
                return f"d{form_key}_d{coefficient_key}_adj"
            else:
                return f"d{form_key}_d{coefficient_key}"

        cache_key = gen_cache_key(form_key, coefficient_key, adjoint)
        if cache_key not in self.assemblers:
            coeff = self.form[coefficient_key]
            ufl_form = self.form.ufl_forms[form_key]

            if coeff in ufl_form.coefficients():
                if adjoint:
                    form = dfn.adjoint(dfn.derivative(ufl_form, coeff))
                else:
                    form = dfn.derivative(ufl_form, coeff)
                self.assemblers[cache_key] = CachedUFLFormAssembler(form)
            else:
                m_form = ufl_form.arguments()[0].function_space().dim()
                n_coeff = coeff.function_space().dim()
                if adjoint:
                    const_mat = dfn.PETScMatrix(zero_mat(n_coeff, m_form))
                else:
                    const_mat = dfn.PETScMatrix(zero_mat(m_form, n_coeff))
                self.assemblers[cache_key] = const_mat

        return self.assemble_from_cache(cache_key)
