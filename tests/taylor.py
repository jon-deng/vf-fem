"""
Taylor convergence test
"""

from typing import TypeVar, Callable, Optional

import numpy as np

# import blockarray.linalg as bla

T = TypeVar('T')
U = TypeVar('U')


def taylor_convergence(
    x0: T,
    dx: T,
    f: Callable[[T], U],
    jac: Callable[[T], U],
    norm: Optional[Callable[[T], float]] = None,
    rel_err_tol: float = 1e-8,
    abs_err_tol: float = 1e-8,
    conv_rate_tol: float = 1e-2,
):
    """
    Apply the Taylor series convergence test to a linearization

    Parameters
    ----------
    x0, dx: T
        The linearization point and linearization direction
    f, jac: Callable[[T], U]
        The non-linear function and its linearization

        The non-linear function generally takes a numeric type input,
        (`BlockVector`, `NDArray`, `float`, etc.) and returns another
        numeric type output (`BlockVector`, `NDArray`, `float`, etc.).
        The linearization should return a linearized output.

        Note that types `T` and `U` can be the same or different; for example,
        if `f` maps scalars to scalars and if `f` maps vectors to scalars,
        respectively.
    norm: Callable[[U], float]
        A norm function to measure the size of function outputs.
    """
    if norm is None:
        norm = np.linalg.norm

    # Start with the largest step and move to original
    alphas = 2 ** np.arange(4)[::-1]
    res_ns = [f(x0 + alpha * dx) for alpha in alphas]
    res_0 = f(x0)

    dres_exacts = [res_n - res_0 for res_n in res_ns]
    dres_linear = jac(x0, dx)

    abs_errs = np.array(
        [
            norm(dres_exact - alpha * dres_linear)
            for dres_exact, alpha in zip(dres_exacts, alphas)
        ]
    )
    err_magnitudes = np.array(
        [
            1 / 2 * norm(dres_exact + alpha * dres_linear)
            for dres_exact, alpha in zip(dres_exacts, alphas)
        ]
    )
    with np.errstate(invalid='ignore'):
        conv_rates = np.log(abs_errs[:-1] / abs_errs[1:]) / np.log(
            alphas[:-1] / alphas[1:]
        )
        rel_errs = abs_errs / err_magnitudes

    print(
        "||dres_linear||, ||dres_exact||"
        f" = {norm(dres_linear)}, {norm(dres_exacts[-1])}"
    )
    print("Relative errors: ", rel_errs)
    print("Convergence rates: ", np.array(conv_rates))

    pass_rel_err = rel_errs[-1] < rel_err_tol
    pass_abs_err = abs_errs[-1] < abs_err_tol
    pass_conv_rate = np.isclose(conv_rates, 2.0, atol=conv_rate_tol)
    assert pass_rel_err or pass_abs_err or pass_conv_rate
    return abs_errs, err_magnitudes, conv_rates
