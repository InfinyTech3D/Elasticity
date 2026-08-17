"""Dim-generic FEM numerics: SOFA's quadrature, the fields sampled on it, and the norms over them."""

import numpy as np

import Sofa.SofaFEM


class MeshQuadrature:
    """The reference->physical mapping of a whole mesh, evaluated once and shared by every norm.

    Each norm used to walk the mesh itself, calling Sofa.SofaFEM.element_mapping once per element,
    so a level paid for the same mapping eight times over -- once per quantity -- and paid a pybind
    crossing per element on each pass. Here SOFA maps every element in one call and the norms become
    numpy expressions over the result, with no Python loop over elements or quadrature points.

    Array layout throughout: `e` element, `q` quadrature point, `a` node within an element,
    `c` field component, `d` spatial direction.
    """

    def __init__(self, nodes, node_indices, element, degree):
        self.nodes = np.asarray(nodes)
        # node_indices[e] = the mesh-node indices that form element e.
        self.node_indices = np.asarray(node_indices)
        dim = self.nodes.shape[1]

        # Reference-space data is identical for every element of this type, so fetch it once.
        self.weights, self.shape_values, reference_gradients = \
            Sofa.SofaFEM.quadrature_data(element, dim, degree)
        self.physical_gradients, measures = Sofa.SofaFEM.element_mapping_batch(
            element, self.nodes, self.node_indices, reference_gradients)

        self.points = np.einsum('qa,ead->eqd', self.shape_values, self.nodes[self.node_indices])
        self.scale = self.weights * measures        # weight * measure at every (element, point)

    def integrate(self, values):
        """Integrate a scalar given at every (element, quadrature point)."""
        return float(np.sum(values * self.scale))

    def values(self, field):
        """A nodal field at every quadrature point: (e, q, c)."""
        return np.einsum('qa,eac->eqc', self.shape_values, np.asarray(field)[self.node_indices])

    def grads(self, field):
        """Gradient of a nodal field: (e, q, c, d) = d(u_c)/dx_d."""
        return np.einsum('eac,eqad->eqcd',
                         np.asarray(field)[self.node_indices], self.physical_gradients)

    def sample(self, field):
        """A manufactured field at every quadrature point: the one seam where it meets this mesh."""
        return field(self.points)


def l2(quadrature, values):
    """L2 norm of a vector field given at every quadrature point: (e, q, c) -> scalar."""
    return float(np.sqrt(quadrature.integrate(np.sum(values * values, axis=-1))))


def frobenius(quadrature, tensors):
    """L2 norm of a tensor field, Frobenius over its last two axes: (e, q, c, d) -> scalar."""
    return float(np.sqrt(quadrature.integrate(np.sum(tensors * tensors, axis=(-2, -1)))))


def energy(quadrature, gradients, energy_density):
    """Elastic energy of a displacement gradient field: integral over the mesh of psi."""
    return quadrature.integrate(energy_density(gradients))


def orthogonality_defect(quadrature, grad_h, grad_error, grad_exact, tangent):
    """a(e, u_h) with e = u_h - u: the Galerkin orthogonality defect.

    Zero when u_h solves the continuous variational problem against its own space, which needs the
    load functional integrated exactly. It is what separates |U - U_h| from 0.5 ||e||_E^2.

    The tangent is taken at the exact solution, which is where a nonlinear material would have to
    linearize; for a linear one it is constant and the argument is ignored. Neither tensor needs
    symmetrizing first: the tangent already carries the symmetric part of what it is applied to, and
    its result is symmetric, so contracting against the full gradient drops nothing.
    """
    return quadrature.integrate(np.sum(tangent(grad_exact, grad_error) * grad_h, axis=(-2, -1)))
