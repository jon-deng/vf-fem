import ufl
import dolfin as dfn


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

