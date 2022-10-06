"""
Taylor convergence test
"""

import numpy as np

# import blockarray.linalg as bla

def taylor_convergence(x0, dx, res, jac, norm=None):
    """
    Test the Taylor convergence of a linearization
    """
    if norm is None:
        norm = np.linalg.norm

    # Start with the largest step and move to original
    alphas = 2**np.arange(4)[::-1]
    res_ns = [res(x0+alpha*dx) for alpha in alphas]
    res_0 = res(x0)

    dres_exacts = [res_n-res_0 for res_n in res_ns]
    dres_linear = jac(x0, dx)

    errs = np.array([
        norm(dres_exact-alpha*dres_linear)
        for dres_exact, alpha in zip(dres_exacts, alphas)
    ])
    err_magnitudes = np.array([
        1/2*norm(dres_exact+alpha*dres_linear)
        for dres_exact, alpha in zip(dres_exacts, alphas)
    ])
    with np.errstate(invalid='ignore'):
        conv_rates = (
            np.log(errs[:-1]/errs[1:])
            / np.log(alphas[:-1]/alphas[1:])
        )
        rel_errs = errs/err_magnitudes

    print(
        "||dres_linear||, ||dres_exact||"
        f" = {norm(dres_linear)}, {norm(dres_exacts[-1])}"
    )
    print("Relative errors: ", rel_errs)
    print("Convergence rates: ", np.array(conv_rates))
    return errs, err_magnitudes, conv_rates