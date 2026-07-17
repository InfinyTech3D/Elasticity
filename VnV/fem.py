"""1D FEM numerics: quadrature, body-force assembly, error norms."""

import numpy as np

# 1D Gauss-Legendre points/weights on [-1, 1].
_GAUSS_LEGENDRE_1D = {
    1: (np.array([0.0]),
        np.array([2.0])),
    2: (np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]),
        np.array([1.0, 1.0])),
}


def line_quadrature(n_pts):
    """Gauss-Legendre rule on a line segment; returns a callable for int_{x1}^{x2} g dx."""
    if n_pts not in _GAUSS_LEGENDRE_1D:
        raise ValueError(f"line_quadrature: {n_pts}-point rule not supported")
    # xi : points, w : weights
    xi, w = _GAUSS_LEGENDRE_1D[n_pts]

    def rule(g, x1, x2):
        h   = x2 - x1
        x_k = 0.5 * (x1 + x2) + 0.5 * h * xi
        return 0.5 * h * sum(w_i * g(x_i) for w_i, x_i in zip(w, x_k))
    return rule


# H1 needs >=2 pts: the P1 gradient superconverges at the midpoint, so 1-pt fakes O(h^2).
L2_QUADRATURE_1D = line_quadrature(2)
H1_QUADRATURE_1D = line_quadrature(2)


def source_integration(f_body, nodes, edges, quadrature):
    """Assembles nodal loads F_i = int f_body(x) phi_i(x) dx from a source field."""
    forces = np.zeros(len(nodes))
    for a, b in edges:
        x1, x2 = nodes[a], nodes[b]
        h = x2 - x1
        # P1-only: 2-node edge and linear hat shape functions; TODO generalize per element type.
        forces[a] += quadrature(lambda x, x1=x1, x2=x2, h=h: f_body(x) * (x2 - x) / h, x1, x2)
        forces[b] += quadrature(lambda x, x1=x1, x2=x2, h=h: f_body(x) * (x - x1) / h, x1, x2)
    return forces


def _error_norm(nodes, edges, u_h, diff, quadrature):
    """Error norm sqrt(sum_e int diff^2 dx); diff(ua, ub, x1, x2, x) is (discrete - exact)."""
    total = 0.0
    for a, b in edges:
        x1, x2 = nodes[a], nodes[b]
        ua, ub = u_h[a], u_h[b]
        def integrand(x):
            d = diff(ua, ub, x1, x2, x)
            return d * d
        total += quadrature(integrand, x1, x2)
    return np.sqrt(total)


def l2_error(nodes, edges, u_h, u_ex, quadrature):
    """L2 error norm; the discrete field is the interpolant of the nodal values."""
    def diff(ua, ub, x1, x2, x):
        # P1-only: linear interpolant from 2 nodal values; TODO generalize per element type.
        u_interp = ua + (ub - ua) * (x - x1) / (x2 - x1)
        return u_interp - u_ex(x)
    return _error_norm(nodes, edges, u_h, diff, quadrature)


def h1_seminorm_error(nodes, edges, u_h, du_ex, quadrature):
    """H1 semi-norm error; the discrete field is the gradient of that interpolant."""
    def diff(ua, ub, x1, x2, x):
        # P1-only: constant element gradient from 2 nodal values; TODO generalize per element type.
        du_h = (ub - ua) / (x2 - x1)
        return du_h - du_ex(x)
    return _error_norm(nodes, edges, u_h, diff, quadrature)
