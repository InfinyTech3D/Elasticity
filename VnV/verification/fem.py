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


# --- Error norms over the mesh (evaluated with SOFA's FiniteElement quadrature) ---

def mesh_quadrature(topology, nodes, degree):
    """Whole-mesh quadrature (points, weights, N, physical gradients, conn) from SOFA's rules."""
    return Sofa.SofaFEM.element_quadrature(topology, np.asarray(nodes, float), int(degree))


def l2_error(q, u_h, u_ex):
    """Vector L2 error norm sqrt( int ||u_h - u_ex||^2 dx ) over the mesh."""
    u_loc  = np.asarray(u_h)[q.conn]                       # (Nelem, n_local, dim)
    u_h_g  = np.einsum("eqa,ead->eqd", q.shape, u_loc)     # u_h at the Gauss points
    pts    = q.points.reshape(-1, u_loc.shape[-1])
    u_ex_g = np.array([u_ex(*p) for p in pts]).reshape(u_h_g.shape)
    diff   = u_h_g - u_ex_g
    return float(np.sqrt(np.sum(np.sum(diff * diff, axis=-1) * q.weights)))


def h1_semi_error(q, u_h, grad_u_ex):
    """Vector H1 semi-norm error sqrt( int ||grad u_h - grad u_ex||_F^2 dx ) over the mesh."""
    u_loc   = np.asarray(u_h)[q.conn]                      # (Nelem, n_local, dim)
    grad_uh = np.einsum("eqda,eai->eqid", q.grad, u_loc)   # [e,q,i,d] = du_i/dx_d
    pts     = q.points.reshape(-1, u_loc.shape[-1])
    grad_ue = np.array([grad_u_ex(*p) for p in pts]).reshape(grad_uh.shape)
    diff    = grad_uh - grad_ue
    return float(np.sqrt(np.sum(np.sum(diff * diff, axis=(-1, -2)) * q.weights)))


# --- Dispatch: element kind -> its boundary-facet rule ---

FACET_RULES = {
    "edge": point_facet_rule,
    "tri":  edge_line_rule,
    "quad": edge_line_rule,
    "tet":  tri_face_rule,
    "hexa": quad_face_rule,
}
