"""
Definitions of newmark, time discretization scheme

Units are in CGS
"""


def newmark_v(u, u0, v0, a0, dt, gamma=1 / 2, beta=1 / 4):
    """
    Returns the Newmark method velocity update.

    Parameters
    ----------
    u : ufl.Argument or ufl.Coefficient
        A trial function, to solve for a displacement, or a coefficient function
    u0, v0, a0 : ufl.Coefficient
        The initial states
    gamma, beta : float
        Newmark time integration parameters

    Returns
    -------
    v1
    """
    return (
        gamma / beta / dt * (u - u0)
        - (gamma / beta - 1.0) * v0
        - dt * (gamma / 2.0 / beta - 1.0) * a0
    )


def newmark_v_du1(dt, gamma=1 / 2, beta=1 / 4):
    """See `newmark_v`"""
    return gamma / beta / dt


def newmark_v_du0(dt, gamma=1 / 2, beta=1 / 4):
    """See `newmark_v`"""
    return -gamma / beta / dt


def newmark_v_dv0(dt, gamma=1 / 2, beta=1 / 4):
    """See `newmark_v`"""
    return -(gamma / beta - 1.0)


def newmark_v_da0(dt, gamma=1 / 2, beta=1 / 4):
    """See `newmark_v`"""
    return -dt * (gamma / 2.0 / beta - 1.0)


def newmark_v_dt(u, u0, v0, a0, dt, gamma=1 / 2, beta=1 / 4):
    """See `newmark_v`"""
    return -gamma / beta / dt**2 * (u - u0) - (gamma / 2.0 / beta - 1.0) * a0


def newmark_a(u, u0, v0, a0, dt, gamma=1 / 2, beta=1 / 4):
    """
    Returns the Newmark method acceleration update.

    Parameters
    ----------
    u : ufl.Argument
    u0, v0, a0 : ufl.Argument
        Initial states
    gamma, beta : float
        Newmark time integration parameters

    Returns
    -------
    a1
    """
    return 1 / beta / dt**2 * (u - u0 - dt * v0) - (1 / 2 / beta - 1) * a0


def newmark_a_du1(dt, gamma=1 / 2, beta=1 / 4):
    """See `newmark_a`"""
    return 1.0 / beta / dt**2


def newmark_a_du0(dt, gamma=1 / 2, beta=1 / 4):
    """See `newmark_a`"""
    return -1.0 / beta / dt**2


def newmark_a_dv0(dt, gamma=1 / 2, beta=1 / 4):
    """See `newmark_a`"""
    return -1.0 / beta / dt


def newmark_a_da0(dt, gamma=1 / 2, beta=1 / 4):
    """See `newmark_a`"""
    return -(1 / 2 / beta - 1)


def newmark_a_dt(u, u0, v0, a0, dt, gamma=1 / 2, beta=1 / 4):
    """See `newmark_a`"""
    return -2 / beta / dt**3 * (u - u0 - dt * v0) + 1 / beta / dt**2 * (-v0)


def newmark_error_estimate(a1, a0, dt, beta=1 / 4):
    """
    Return an estimate of the truncation error in `u` over the step.

    Error is esimated using eq (18) in [1]. Note that their paper defines $\beta2$ as twice $\beta$
    in the original newmark notation (used here). Therefore the beta term is multiplied by 2.

    [1] A simple error estimator and adaptive time stepping procedure for dynamic analysis.
    O. C. Zienkiewicz and Y. M. Xie. Earthquake Engineering and Structural Dynamics, 20:871-887
    (1991).

    Parameters
    ----------
    a1 : dfn.Vector()
        The newmark prediction of acceleration at :math:`n+1`
    a0 : dfn.Vector()
        The newmark prediction of acceleration at :math:`n`
    dt : float
        The time step integrated over
    beta : float
        The newmark-beta method :math:`beta` parameter

    Returns
    -------
    dfn.Vector()
        An estimate of the error in :math:`u_{n+1}`
    """
    return 0.5 * dt**2 * (2 * beta - 1 / 3) * (a1 - a0)
