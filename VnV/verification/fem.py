"""Dim-generic FEM numerics: error norms integrated with SOFA's quadrature."""

import numpy as np

import Sofa.SofaFEM


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
