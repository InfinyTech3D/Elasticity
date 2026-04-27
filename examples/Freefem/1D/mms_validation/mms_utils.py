"""Shared utilities for 1D MMS validation."""

import numpy as np


# ---------------------------------------------------------------------------
# Quadrature rules
# Approximation: integral_{x_i}^{x_{i+1}} g(x) dx
#                ~= (h/2) * sum_k( w_k * g(x_k) )
# where x_k = (x_i + x_{i+1})/2 + (h/2)*xi_k
# ---------------------------------------------------------------------------

def midpoint_quadrature(g, x1, x2):
    """1-point Gauss rule: n_g=1, xi=[0], w=[2]."""
    h   = x2 - x1
    x_k = 0.5 * (x1 + x2)
    return (h / 2.0) * 2.0 * g(x_k)


def gauss2_quadrature(g, x1, x2):
    """2-point Gauss rule: n_g=2, xi=[-1/sqrt(3), +1/sqrt(3)], w=[1, 1]."""
    h     = x2 - x1
    x_mid = 0.5 * (x1 + x2)
    xi    = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    x_k   = x_mid + (h / 2.0) * xi
    return (h / 2.0) * (g(x_k[0]) + g(x_k[1]))
