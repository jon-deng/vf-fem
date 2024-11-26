"""
FEniCS form assembly utilities
"""

import ufl
import dolfin as dfn

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
    def assemblers(self) -> dict[str, CachedUFLFormAssembler]:
        return self._cached_assemblers


    def assemble(self, residual_key: str):
        """
        Assemble a residual from the form
        """
        # TODO: Update this hard-coded value for multiple residuals keys
        # in `Form`
        form_key = f"{residual_key}"

        if form_key not in self.assemblers:
            form = self.form.ufl_forms[residual_key]
            self.assemblers[form_key] = CachedUFLFormAssembler(form)

        return self.assemblers[form_key].assemble()

    def assemble_derivative(
            self,
            residual_key: str,
            coefficient_key: str,
            adjoint: bool=False
        ):
        """
        Assemble a residual from the form
        """
        if adjoint:
            form_key = f"d{residual_key}_d{coefficient_key}_adj"
        else:
            form_key = f"d{residual_key}_d{coefficient_key}"

        if form_key not in self.assemblers:
            coeff = self.form[coefficient_key]
            form = dfn.derivative(self.form.ufl_forms[residual_key], coeff)
            if adjoint:
                form = dfn.adjoint(form)
            self.assemblers[form_key] = CachedUFLFormAssembler(form)

        return self.assemblers[form_key].assemble()
