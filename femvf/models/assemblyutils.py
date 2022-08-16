import ufl
import dolfin as dfn

class CachedFormAssembler:
    """
    Assembles a bilinear form using a cached sparsity pattern

    Parameters
    ----------
    form : ufl.Form
    kwargs :
        Keyword arguments of `dfn.assemble` except for `tensor`
    """

    def __init__(self, form: ufl.Form, **kwargs):
        self._form = form

        if 'tensor' not in kwargs:
            if len(form.arguments()) == 0:
                tensor = dfn.Scalar()
            elif len(form.arguments()) == 1:
                tensor = dfn.PETScVector()
            elif len(form.arguments()) == 2:
                tensor = dfn.PETScMatrix()
            else:
                raise ValueError("Form arity must be between 0 and 2")
        else:
            tensor = kwargs['tensor']
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