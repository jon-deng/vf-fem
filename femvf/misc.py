"""
Don't know what to do with this yet.
"""

import numpy as np

def get_dynamic_fluid_props(fluid_props, time):
    """
    Returns fluid properties at specified time given a time varying set of properties.

    Parameters
    ----------
    fluid_props : dict
        Time varying set of fluid properties. If a property called 'key' is time varying, there
        should be a key called 'key_time' that stores time points at which the property changes.
        fluid_props[]
    """
    out = {}

    dynamic_properties = ['p_sub', 'p_sup']
    for key in dynamic_properties:
        if f'{key}_time' in fluid_props:
            out[key] = np.interp(time, fluid_props[f'{key}_time'], fluid_props[key])
        else:
            out[key] = fluid_props[key]

    for key, value in fluid_props.items():
        if key not in dynamic_properties:
            out[key] = value

    return out

def get_fundamental_freq_phase_amp(f, sample_freq):
    """
    Return an estimate of fundamental frequency and phase using DFT

    Parameters
    ----------
    f : array_like of float
        Discrete (real) signals sampled at certain frequency.
    sample_freq : float
        Sampling frequency in [Hz]

    Returns
    -------
    freq0, phi0, a0 : float
        Fundamental frequency [Hz], phase shift [time], and amplitude [signal] estimates
    """
    N = f.size

    dft_f = np.fft.rfft(f)
    freq = np.arange(N//2+1) / (N/sample_freq)
    phase = np.angle(dft_f)
    mag = np.abs(dft_f)

    idx_f0 = np.argmax(mag)

    f0 = freq[idx_f0]
    tphase0 = phase[idx_f0]/(2*np.pi) * 1/f0
    amp0 = 2*mag[idx_f0]/N

    return f0, tphase0, amp0

## Below are a collection of smoothened discrete operators. These are used in the bernoulli code
# def smooth_minimum(x, alpha=-1000):
#     """
#     Return the smooth approximation to the minimum element of x.

#     Parameters
#     ----------
#     x : array_like
#         Array of values to compute the minimum of
#     alpha : float
#         Factor that control the sharpness of the minimum. The function approaches the true minimum
#         function as `alpha` approachs negative infinity.
#     """
#     weights = np.exp(alpha*x)
#     return np.sum(x*weights) / np.sum(weights)

# def dsmooth_minimum_dx(x, alpha=-1000):
#     """
#     Return the derivative of the smooth minimum with respect to x.

#     Parameters
#     ----------
#     x : array_like
#         Array of values to compute the minimum of
#     alpha : float
#         Factor that control the sharpness of the minimum. The function approaches the true minimum
#         function as `alpha` approachs negative infinity.
#     """
#     weights = np.exp(alpha*x)
#     return weights/np.sum(weights)*(1+alpha*(x - smooth_minimum(x, alpha)))

# def sigmoid(x):
#     return 1/(1+np.exp(-x))

# def dsigmoid_dx(x):
#     sig = sigmoid(x)
#     return sig * (1-sig)

# def mirr_logistic(x, x0, k=100):
#     """
#     Return the mirrored logistic function evaluated at x-x0
#     """
#     arg = k*(x-x0)
#     return sigmoid(arg)

# def dmirr_logistic_dx(x, x0, k=10):
#     """
#     Return the logistic function evaluated at x-xref
#     """
#     arg = k*(x-x0)
#     darg_dx = k
#     return dsigmoid_dx(arg) * darg_dx

# def dmirr_logistic_dx0(x, x0, k=10):
#     """
#     Return the logistic function evaluated at x-xref
#     """
#     arg = k*(x-x0)
#     darg_dx0 = -k
#     return dsigmoid_dx(arg) * darg_dx0

# def gaussian_selection(x, y, y0, sigma=1.0):
#     """
#     Return the `x` value from an `(x, y)` pair where `y` equals `y0`.

#     Weights are computed according to a gaussian distribution.

#     Parameters
#     ----------
#     x, y : array_like
#         Paired array of values
#     sigma : float
#         Standard deviation of the selection criteria
#     """
#     assert x.size == y.size
#     weights = (sigma*(2*np.pi)**0.5)**-1 * np.exp(-0.5*((y-yref)/sigma)**2)

#     return np.sum(x*weights) / np.sum(weights)

# def dgaussian_selection_dx(x, y, y0, sigma=0.1):
#     """
#     Return the `x` value from an `(x, y)` pair where `y` equals `y0`.

#     Weights are computed according to a gaussian distribution.

#     Parameters
#     ----------
#     x, y : array_like
#         Paired array of values
#     sigma : float
#         Standard deviation of the selection criteria
#     """
#     assert x.size == y.size
#     weights = (sigma*(2*np.pi)**0.5)**-1 * np.exp(-0.5*((y-yref)/sigma)**2)

#     return np.sum(x*weights) / np.sum(weights)

# def dgaussian_selection_dy(x, y, y0, sigma=0.1):
#     """
#     Return the `x` value from an `(x, y)` pair where `y` equals `y0`.

#     Weights are computed according to a gaussian distribution.

#     Parameters
#     ----------
#     x, y : array_like
#         Paired array of values
#     sigma : float
#         Standard deviation of the selection criteria
#     """
#     assert x.size == y.size
#     weights = (sigma*(2*np.pi)**0.5)**-1 * np.exp(-0.5*((y-yref)/sigma)**2)

#     return np.sum(x*weights) / np.sum(weights)

# def dgaussian_selection_dy0(x, y, y0, sigma=0.1):
#     """
#     Return the `x` value from an `(x, y)` pair where `y` equals `y0`.

#     Weights are computed according to a gaussian distribution.

#     Parameters
#     ----------
#     x, y : array_like
#         Paired array of values
#     sigma : float
#         Standard deviation of the selection criteria
#     """
#     assert x.size == y.size
#     weights = (sigma*(2*np.pi)**0.5)**-1 * np.exp(-0.5*((y-yref)/sigma)**2)

#     return np.sum(x*weights) / np.sum(weights)