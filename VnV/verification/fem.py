"""Dim-generic FEM numerics: facet rules, load assembly, and error norms (SOFA quadrature)."""

import numpy as np

import Sofa.SofaFEM

# 1D Gauss-Legendre points/weights on [-1, 1].
_GAUSS_LEGENDRE_1D = {
    1: (np.array([0.0]),
        np.array([2.0])),
    2: (np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]),
        np.array([1.0, 1.0])),
}


# Reference-triangle rule over {xi,eta>=0, xi+eta<=1}; physical weight is w_ref * 2*area.
_TRI_QUADRATURE = {
    1: (np.array([[1/3, 1/3]]),
        np.array([1/2])),
    3: (np.array([[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]]),
        np.array([1/6, 1/6, 1/6])),
}


# --- Facet rules: boundary integration, yielding (coords, w, N) ---

def edge_line_rule(n_pts=2):
    """Facet rule for a 2D boundary edge."""
    if n_pts not in _GAUSS_LEGENDRE_1D:
        raise ValueError(f"edge_line_rule: {n_pts}-point rule not supported")
    xi_pts, w_pts = _GAUSS_LEGENDRE_1D[n_pts]

    def rule(xe):
        xe_arr = np.asarray(xe, float)            # (2, 2)
        Le = np.linalg.norm(xe_arr[1] - xe_arr[0])
        for xi, wi in zip(xi_pts, w_pts):
            t = 0.5 * (xi + 1.0)
            N = np.array([1.0 - t, t])
            coords = N @ xe_arr                   # (2,)
            yield coords, wi * Le / 2.0, N
    return rule


def quad_face_rule(n_pts=2):
    """Facet rule for a 3D boundary quad face."""
    if n_pts not in _GAUSS_LEGENDRE_1D:
        raise ValueError(f"quad_face_rule: {n_pts}-point rule not supported")
    xi_pts, w_pts = _GAUSS_LEGENDRE_1D[n_pts]

    def rule(xe):
        xe_arr = np.asarray(xe, float)            # (4, 3)
        for xi, wi in zip(xi_pts, w_pts):
            for eta, wj in zip(xi_pts, w_pts):
                N = 0.25 * np.array([
                    (1 - xi) * (1 - eta),
                    (1 + xi) * (1 - eta),
                    (1 + xi) * (1 + eta),
                    (1 - xi) * (1 + eta),
                ])
                dN_dxi  = 0.25 * np.array([-(1 - eta),  (1 - eta), (1 + eta), -(1 + eta)])
                dN_deta = 0.25 * np.array([-(1 - xi),  -(1 + xi),  (1 + xi),  (1 - xi) ])
                t_xi   = dN_dxi  @ xe_arr         # (3,)
                t_eta  = dN_deta @ xe_arr
                dA     = np.linalg.norm(np.cross(t_xi, t_eta))
                coords = N @ xe_arr               # (3,)
                yield coords, wi * wj * dA, N
    return rule


def tri_face_rule(n_pts=3):
    """Facet rule for a 3D boundary triangle."""
    if n_pts not in _TRI_QUADRATURE:
        raise ValueError(f"tri_face_rule: {n_pts}-point rule not supported")
    pts, wts = _TRI_QUADRATURE[n_pts]

    def rule(xe):
        xe_arr = np.asarray(xe, float)            # (3, 3)
        x0, x1, x2 = xe_arr
        area = 0.5 * np.linalg.norm(np.cross(x1 - x0, x2 - x0))
        for (xi, eta), w_ref in zip(pts, wts):
            N      = np.array([1.0 - xi - eta, xi, eta])
            coords = N @ xe_arr                    # (3,)
            yield coords, w_ref * 2.0 * area, N
    return rule


def point_facet_rule():
    """Facet rule for a 1D boundary vertex; the surface integral is a point evaluation."""
    def rule(xe):
        xe_arr = np.asarray(xe, float)            # (1, 1)
        yield xe_arr[0], 1.0, np.array([1.0])
    return rule


# --- Assembly: consistent nodal loads over elements and facets ---

def source_integration(f_body, nodes, conn, element_rule):
    """Assemble consistent nodal body forces F_a = sum_e int N_a * f_body dx."""
    nodes = np.asarray(nodes)
    dim   = nodes.shape[1]
    F     = np.zeros((len(nodes), dim))
    for elem in conn:
        idx = np.asarray(elem)
        xe  = nodes[idx]                          # (n_local, dim)
        for coords, w, N, _ in element_rule(xe):
            f_components = f_body(*coords)
            for a, node in enumerate(idx):
                for d in range(dim):
                    F[node, d] += N[a] * f_components[d] * w
    return F


def source_integration_boundary(traction, nodes, facets, facet_rule):
    """Assemble consistent nodal forces F_a += int N_a * traction dS over boundary facets."""
    nodes = np.asarray(nodes)
    dim   = nodes.shape[1]
    F     = np.zeros((len(nodes), dim))
    for facet in facets:
        idx = np.asarray(facet)
        xe  = nodes[idx]                          # (n_local, dim)
        for coords, w, N in facet_rule(xe):
            T = traction(*coords)
            for a, node in enumerate(idx):
                for d in range(dim):
                    F[node, d] += N[a] * T[d] * w
    return F


# --- Integration over the mesh, reusing SOFA's FiniteElement kernel for the element math ---

def integrate_over_mesh(nodes, node_indices, element, degree, integrand):
    """Integrate a quantity over the mesh: sum over elements and their quadrature points.

    `integrand(element_nodes, point, shape_values, physical_gradients)` returns the scalar to
    integrate at one quadrature point:
        element_nodes      the mesh-node indices of the current element
        point              physical coordinates of the quadrature point   (dim,)
        shape_values       shape-function values N_a                       (nodes_per_element,)
        physical_gradients dN_a/dx_d                                       (nodes_per_element, dim)
    """
    nodes = np.asarray(nodes)
    dim = nodes.shape[1]

    # Reference-space data is identical for every element of this type, so fetch it once.
    weights, shape_values, reference_grads = Sofa.SofaFEM.quadrature_data(element, dim, degree)

    total = 0.0
    # node_indices[e] = the mesh-node indices that form element e.
    for element_nodes in node_indices:
        node_coordinates = nodes[element_nodes]                                 # (nodes_per_element, dim)
        physical_gradients, measures = Sofa.SofaFEM.element_mapping(element, node_coordinates, reference_grads)
        for q, weight in enumerate(weights):
            point = shape_values[q] @ node_coordinates
            total += integrand(element_nodes, point, shape_values[q], physical_gradients[q]) * weight * measures[q]
    return total


def l2_error(nodes, node_indices, element, degree, u_h, u_exact):
    """L2 error norm: sqrt( integral over the mesh of ||u_h - u_exact||^2 )."""
    u_h = np.asarray(u_h)

    def integrand(element_nodes, point, shape_values, physical_gradients):
        u_h_value  = shape_values @ u_h[element_nodes]     # interpolated displacement at the point
        difference = u_h_value - u_exact(*point)
        return difference @ difference

    return float(np.sqrt(integrate_over_mesh(nodes, node_indices, element, degree, integrand)))


def h1_semi_error(nodes, node_indices, element, degree, u_h, grad_u_exact):
    """H1 semi-norm error: sqrt( integral over the mesh of ||grad u_h - grad u_exact||_F^2 )."""
    u_h = np.asarray(u_h)

    def integrand(element_nodes, point, shape_values, physical_gradients):
        grad_u_h   = u_h[element_nodes].T @ physical_gradients   # (component_i, direction_d) = d(u_i)/dx_d
        difference = grad_u_h - grad_u_exact(*point)             # grad_u_exact(point)[i, j] = d(u_i)/dx_j
        return np.sum(difference * difference)

    return float(np.sqrt(integrate_over_mesh(nodes, node_indices, element, degree, integrand)))


# --- Dispatch: element kind -> its boundary-facet rule ---

FACET_RULES = {
    "edge": point_facet_rule,
    "tri":  edge_line_rule,
    "quad": edge_line_rule,
    "tet":  tri_face_rule,
    "hexa": quad_face_rule,
}
