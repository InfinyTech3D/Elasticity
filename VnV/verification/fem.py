"""Dim-generic FEM numerics: element/facet rules, load assembly, error norms."""

import numpy as np

# 1D Gauss-Legendre points/weights on [-1, 1].
_GAUSS_LEGENDRE_1D = {
    1: (np.array([0.0]),
        np.array([2.0])),
    2: (np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]),
        np.array([1.0, 1.0])),
}


# --- Reference-element shape functions and quadrature tables ---

def _shape_q1(xi, eta):
    """2D Q1 shape on [-1, 1]^2. Node order: (-,-), (+,-), (+,+), (-,+)."""
    N = 0.25 * np.array([
        (1 - xi) * (1 - eta),
        (1 + xi) * (1 - eta),
        (1 + xi) * (1 + eta),
        (1 - xi) * (1 + eta),
    ])
    dN_dxi  = 0.25 * np.array([-(1 - eta),  (1 - eta), (1 + eta), -(1 + eta)])
    dN_deta = 0.25 * np.array([-(1 - xi),  -(1 + xi),  (1 + xi),  (1 - xi) ])
    return N, dN_dxi, dN_deta


def _shape_hex_q1(xi, eta, zeta):
    """3D Q1 hex shape on [-1, 1]^3. Node order: bottom face CCW, then top face CCW."""
    N = 0.125 * np.array([
        (1 - xi) * (1 - eta) * (1 - zeta),
        (1 + xi) * (1 - eta) * (1 - zeta),
        (1 + xi) * (1 + eta) * (1 - zeta),
        (1 - xi) * (1 + eta) * (1 - zeta),
        (1 - xi) * (1 - eta) * (1 + zeta),
        (1 + xi) * (1 - eta) * (1 + zeta),
        (1 + xi) * (1 + eta) * (1 + zeta),
        (1 - xi) * (1 + eta) * (1 + zeta),
    ])
    dN_dxi = 0.125 * np.array([
        -(1 - eta) * (1 - zeta),  (1 - eta) * (1 - zeta),
         (1 + eta) * (1 - zeta), -(1 + eta) * (1 - zeta),
        -(1 - eta) * (1 + zeta),  (1 - eta) * (1 + zeta),
         (1 + eta) * (1 + zeta), -(1 + eta) * (1 + zeta),
    ])
    dN_deta = 0.125 * np.array([
        -(1 - xi) * (1 - zeta), -(1 + xi) * (1 - zeta),
         (1 + xi) * (1 - zeta),  (1 - xi) * (1 - zeta),
        -(1 - xi) * (1 + zeta), -(1 + xi) * (1 + zeta),
         (1 + xi) * (1 + zeta),  (1 - xi) * (1 + zeta),
    ])
    dN_dzeta = 0.125 * np.array([
        -(1 - xi) * (1 - eta), -(1 + xi) * (1 - eta),
        -(1 + xi) * (1 + eta), -(1 - xi) * (1 + eta),
         (1 - xi) * (1 - eta),  (1 + xi) * (1 - eta),
         (1 + xi) * (1 + eta),  (1 - xi) * (1 + eta),
    ])
    return N, dN_dxi, dN_deta, dN_dzeta


# Reference-triangle rule over {xi,eta>=0, xi+eta<=1}; physical weight is w_ref * 2*area.
_TRI_QUADRATURE = {
    1: (np.array([[1/3, 1/3]]),
        np.array([1/2])),
    3: (np.array([[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]]),
        np.array([1/6, 1/6, 1/6])),
}


# Reference-tetrahedron rule over {>=0, xi+eta+zeta<=1}; physical weight is w_ref * 6*vol.
_TET_A4 = (5.0 - np.sqrt(5.0)) / 20.0
_TET_B4 = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0
_TET_QUADRATURE = {
    1: (np.array([[1/4, 1/4, 1/4]]),
        np.array([1/6])),
    4: (np.array([[_TET_A4, _TET_A4, _TET_A4],
                  [_TET_B4, _TET_A4, _TET_A4],
                  [_TET_A4, _TET_B4, _TET_A4],
                  [_TET_A4, _TET_A4, _TET_B4]]),
        np.array([1/24, 1/24, 1/24, 1/24])),
}


# --- Element rules: map a reference rule onto a physical element ---
# rule(xe) yields (coords, w, N, dN_phys) per Gauss point:
#   xe (n_local, dim) physical nodes; coords (dim,); w scalar (× detJ or area);
#   N (n_local,) shape values; dN_phys (dim, n_local) physical-coord gradients.

def quad_q1_rule(n_pts=2):
    """Element rule for Q1 quads: tensor-product Gauss-Legendre on [-1,1]^2."""
    if n_pts not in _GAUSS_LEGENDRE_1D:
        raise ValueError(f"quad_q1_rule: {n_pts}-point rule not supported")
    xi_pts, w_pts = _GAUSS_LEGENDRE_1D[n_pts]

    def rule(xe):
        xe_arr = np.asarray(xe, float)            # (4, 2)
        x_col, y_col = xe_arr[:, 0], xe_arr[:, 1]
        for xi, wi in zip(xi_pts, w_pts):
            for eta, wj in zip(xi_pts, w_pts):
                N, dN_dxi, dN_deta = _shape_q1(xi, eta)
                J = np.array([[dN_dxi  @ x_col, dN_dxi  @ y_col],
                              [dN_deta @ x_col, dN_deta @ y_col]])
                detJ   = np.linalg.det(J)
                Jinv   = np.linalg.inv(J)
                coords = np.array([N @ x_col, N @ y_col])
                dN_dx  = Jinv[0, 0] * dN_dxi + Jinv[1, 0] * dN_deta
                dN_dy  = Jinv[0, 1] * dN_dxi + Jinv[1, 1] * dN_deta
                dN_phys = np.stack([dN_dx, dN_dy])      # (2, 4) [d, a]
                yield coords, wi * wj * detJ, N, dN_phys
    return rule


def tri_p1_rule(n_pts=3):
    """Element rule for P1 triangles."""
    if n_pts not in _TRI_QUADRATURE:
        raise ValueError(f"tri_p1_rule: {n_pts}-point rule not supported")
    pts, wts = _TRI_QUADRATURE[n_pts]

    def rule(xe):
        xe_arr = np.asarray(xe, float)            # (3, 2)
        (x0, y0), (x1, y1), (x2, y2) = xe_arr
        A2    = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        area  = abs(A2) / 2.0
        dN_dx = np.array([(y1 - y2) / A2, (y2 - y0) / A2, (y0 - y1) / A2])
        dN_dy = np.array([(x2 - x1) / A2, (x0 - x2) / A2, (x1 - x0) / A2])
        dN_phys = np.stack([dN_dx, dN_dy])        # (2, 3) [d, a]
        for (xi, eta), w_ref in zip(pts, wts):
            N      = np.array([1.0 - xi - eta, xi, eta])
            coords = np.array([N @ xe_arr[:, 0], N @ xe_arr[:, 1]])
            yield coords, w_ref * 2.0 * area, N, dN_phys
    return rule


def hex_q1_rule(n_pts=2):
    """Element rule for Q1 hexes: tensor-product Gauss-Legendre on [-1,1]^3."""
    if n_pts not in _GAUSS_LEGENDRE_1D:
        raise ValueError(f"hex_q1_rule: {n_pts}-point rule not supported")
    xi_pts, w_pts = _GAUSS_LEGENDRE_1D[n_pts]

    def rule(xe):
        xe_arr = np.asarray(xe, float)            # (8, 3)
        x_col, y_col, z_col = xe_arr[:, 0], xe_arr[:, 1], xe_arr[:, 2]
        for xi, wi in zip(xi_pts, w_pts):
            for eta, wj in zip(xi_pts, w_pts):
                for zeta, wk in zip(xi_pts, w_pts):
                    N, dN_dxi, dN_deta, dN_dzeta = _shape_hex_q1(xi, eta, zeta)
                    J = np.array([
                        [dN_dxi   @ x_col, dN_dxi   @ y_col, dN_dxi   @ z_col],
                        [dN_deta  @ x_col, dN_deta  @ y_col, dN_deta  @ z_col],
                        [dN_dzeta @ x_col, dN_dzeta @ y_col, dN_dzeta @ z_col],
                    ])
                    detJ   = np.linalg.det(J)
                    Jinv   = np.linalg.inv(J)
                    coords = np.array([N @ x_col, N @ y_col, N @ z_col])
                    dN_dx  = Jinv[0, 0] * dN_dxi + Jinv[1, 0] * dN_deta + Jinv[2, 0] * dN_dzeta
                    dN_dy  = Jinv[0, 1] * dN_dxi + Jinv[1, 1] * dN_deta + Jinv[2, 1] * dN_dzeta
                    dN_dz  = Jinv[0, 2] * dN_dxi + Jinv[1, 2] * dN_deta + Jinv[2, 2] * dN_dzeta
                    dN_phys = np.stack([dN_dx, dN_dy, dN_dz])   # (3, 8) [d, a]
                    yield coords, wi * wj * wk * detJ, N, dN_phys
    return rule


def tet_p1_rule(n_pts=4):
    """Element rule for P1 tetrahedra."""
    if n_pts not in _TET_QUADRATURE:
        raise ValueError(f"tet_p1_rule: {n_pts}-point rule not supported")
    pts, wts = _TET_QUADRATURE[n_pts]

    # Reference-coord gradients of (N0..N3) wrt (xi, eta, zeta), constant.
    dN_ref = np.array([[-1.0, -1.0, -1.0],
                       [ 1.0,  0.0,  0.0],
                       [ 0.0,  1.0,  0.0],
                       [ 0.0,  0.0,  1.0]])       # (4, 3) [a, ref-axis]

    def rule(xe):
        xe_arr = np.asarray(xe, float)            # (4, 3)
        x0 = xe_arr[0]
        # Jacobian columns are the edge vectors from node 0: x = x0 + J @ (xi,eta,zeta).
        J     = np.column_stack([xe_arr[1] - x0, xe_arr[2] - x0, xe_arr[3] - x0])
        detJ  = np.linalg.det(J)
        vol   = abs(detJ) / 6.0
        Jinv  = np.linalg.inv(J)
        dN_phys = (dN_ref @ Jinv).T               # (3, 4) [d, a]
        for (xi, eta, zeta), w_ref in zip(pts, wts):
            N      = np.array([1.0 - xi - eta - zeta, xi, eta, zeta])
            coords = N @ xe_arr                    # (3,)
            yield coords, w_ref * 6.0 * vol, N, dN_phys
    return rule


def edge_p1_rule(n_pts=2):
    """Element rule for P1 edges (1D domain)."""
    if n_pts not in _GAUSS_LEGENDRE_1D:
        raise ValueError(f"edge_p1_rule: {n_pts}-point rule not supported")
    xi_pts, w_pts = _GAUSS_LEGENDRE_1D[n_pts]

    def rule(xe):
        xe_arr = np.asarray(xe, float)            # (2, 1)
        x_col  = xe_arr[:, 0]
        L      = x_col[1] - x_col[0]
        dN_phys = np.array([[-1.0 / L, 1.0 / L]]) # (1, 2) [d, a]
        for xi, wi in zip(xi_pts, w_pts):
            t      = 0.5 * (xi + 1.0)
            N      = np.array([1.0 - t, t])
            coords = np.array([N @ x_col])        # (1,)
            yield coords, wi * abs(L) / 2.0, N, dN_phys
    return rule


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


# --- Error norms over the mesh ---

def l2_error(nodes, conn, u_h, u_ex, element_rule):
    """Vector L2 error norm sqrt( int ||u_h - u_ex||^2 dx ) over the mesh."""
    nodes = np.asarray(nodes)
    u     = np.asarray(u_h)
    total = 0.0
    for elem in conn:
        idx   = np.asarray(elem)
        xe    = nodes[idx]                        # (n_local, dim)
        u_loc = u[idx]                            # (n_local, dim)
        for coords, w, N, _ in element_rule(xe):
            u_h_g  = N @ u_loc                    # (dim,)
            u_ex_g = np.asarray(u_ex(*coords))    # (dim,)
            diff   = u_h_g - u_ex_g
            total += float(np.dot(diff, diff)) * w
    return np.sqrt(total)


def h1_semi_error(nodes, conn, u_h, grad_u_ex, element_rule):
    """Vector H1 semi-norm error sqrt( int ||grad u_h - grad u_ex||_F^2 dx ) over the mesh."""
    nodes = np.asarray(nodes)
    u     = np.asarray(u_h)
    total = 0.0
    for elem in conn:
        idx   = np.asarray(elem)
        xe    = nodes[idx]                        # (n_local, dim)
        u_loc = u[idx]                            # (n_local, dim)
        for coords, w, _, dN_phys in element_rule(xe):
            grad_uh = (dN_phys @ u_loc).T         # (dim, dim) [i, d]
            grad_ue = np.asarray(grad_u_ex(*coords))  # [i, j] = du_i/dx_j
            diff    = grad_uh - grad_ue
            total  += float(np.sum(diff * diff)) * w
    return np.sqrt(total)


# --- Dispatch: element kind -> its element rule and boundary-facet rule ---

ELEMENT_RULES = {
    "edge": edge_p1_rule,
    "tri":  tri_p1_rule,
    "quad": quad_q1_rule,
    "tet":  tet_p1_rule,
    "hexa": hex_q1_rule,
}

FACET_RULES = {
    "edge": point_facet_rule,
    "tri":  edge_line_rule,
    "quad": edge_line_rule,
    "tet":  tri_face_rule,
    "hexa": quad_face_rule,
}
