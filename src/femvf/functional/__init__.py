"""
This package contains definitions of functionals

A functional is mapping from the time history of all states/controls/parameters to a real number
.. ::math {(u, v, a, q, p; t, params)_n for n in {0, 1, ..., N-1}}
i.e. a functional should take in the entire time history of states from a forward model run and return a
real number.

For computing the sensitivity of the functional through the discrete adjoint method, you also need
the sensitivity of the functional with respect to the n'th state. This function has the signature
```dfunctional_du(model, n, f, ....) -> float, dict```, 
where the parameters have the same uses as defined previously. Parameter `n` is the state to
compute the sensitivity with respect to.
"""
