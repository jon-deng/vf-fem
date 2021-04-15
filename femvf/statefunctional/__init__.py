"""
This package contains definitions of state functionals

A state functional should map a single (state/control/parameter set) combination to a real number,
unlike the functionals defined in `functionals` which depend on all states over a time period.

For computing the sensitivity of the functional through the discrete adjoint method, you also need
the sensitivity of the functional with respect to the n'th state. This function has the signature
```dfunctional_du(state, control, parameters) -> float```
"""
