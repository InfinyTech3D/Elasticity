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


# ---------------------------------------------------------------------------
# FEM assembly
# ---------------------------------------------------------------------------

def assemble_nodal_forces(f_body, nodes, quadrature):
    """
    Assemble the consistent nodal force vector F_i = integral f_body(x) phi_i(x) dx.

    f_body    : callable x -> float
    nodes     : 1-D array of node coordinates
    quadrature: midpoint_quadrature or gauss2_quadrature
    """
    forces = np.zeros(len(nodes))
    for i in range(len(nodes) - 1):
        x1, x2 = nodes[i], nodes[i + 1]
        h = x2 - x1
        forces[i]     += quadrature(lambda x, x1=x1, x2=x2, h=h: f_body(x) * (x2 - x) / h, x1, x2)
        forces[i + 1] += quadrature(lambda x, x1=x1, x2=x2, h=h: f_body(x) * (x - x1) / h, x1, x2)
    return forces
