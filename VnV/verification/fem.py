"""Dim-generic FEM numerics: error norms integrated with SOFA's quadrature."""

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

    def interpolate(self, field):
        """A nodal field at every quadrature point: (e, q, c)."""
        return np.einsum('qa,eac->eqc', self.shape_values, np.asarray(field)[self.node_indices])

    def gradient(self, field):
        """Gradient of a nodal field: (e, q, c, d) = d(u_c)/dx_d."""
        return np.einsum('eac,eqad->eqcd',
                         np.asarray(field)[self.node_indices], self.physical_gradients)


def l2_error(quadrature, u_h, u_exact):
    """L2 error norm: sqrt( integral over the mesh of ||u_h - u_exact||^2 )."""
    difference = quadrature.interpolate(u_h) - u_exact(quadrature.points)
    return float(np.sqrt(quadrature.integrate(np.sum(difference * difference, axis=-1))))


def h1_semi_error(quadrature, u_h, grad_u_exact):
    """H1 semi-norm error: sqrt( integral over the mesh of ||grad u_h - grad u_exact||_F^2 )."""
    difference = quadrature.gradient(u_h) - grad_u_exact(quadrature.points)
    return float(np.sqrt(quadrature.integrate(np.sum(difference * difference, axis=(-2, -1)))))


def exact_l2_norm(quadrature, u_exact):
    """L2 norm of the exact field: the magnitude l2_error is small or large *relative to*."""
    value = u_exact(quadrature.points)
    return float(np.sqrt(quadrature.integrate(np.sum(value * value, axis=-1))))


def exact_h1_semi_norm(quadrature, grad_u_exact):
    """H1 semi-norm of the exact field: the magnitude h1_semi_error is measured against."""
    gradient = grad_u_exact(quadrature.points)
    return float(np.sqrt(quadrature.integrate(np.sum(gradient * gradient, axis=(-2, -1)))))


def energy(quadrature, u_h, energy_density):
    """Elastic energy of a discrete displacement field: integral over the mesh of psi(grad u_h)."""
    return quadrature.integrate(energy_density(quadrature.gradient(u_h)))


def exact_energy(quadrature, grad_u_exact, energy_density):
    """Elastic energy of the exact field, on the same mesh and quadrature as the discrete one."""
    return quadrature.integrate(energy_density(grad_u_exact(quadrature.points)))


def orthogonality_defect(quadrature, u_h, grad_u_exact, constitutive):
    """a(e, u_h) with e = u_h - u: the Galerkin orthogonality defect.

    Zero when u_h solves the continuous variational problem against its own space, which needs the
    load functional integrated exactly. It is what separates |U - U_h| from 0.5 ||e||_E^2.
    """
    grad_u_h = quadrature.gradient(u_h)
    grad_e = grad_u_h - grad_u_exact(quadrature.points)
    strain_e = 0.5 * (grad_e + np.swapaxes(grad_e, -2, -1))
    strain_uh = 0.5 * (grad_u_h + np.swapaxes(grad_u_h, -2, -1))
    return quadrature.integrate(np.sum(constitutive(strain_e) * strain_uh, axis=(-2, -1)))


def energy_norm_error(quadrature, u_h, grad_u_exact, energy_density):
    """Energy norm of the error: sqrt( 2 * integral of psi(grad u_h - grad u_exact) ).

    The norm the Galerkin solution actually minimizes in -- a material-weighted H1 semi-norm.
    """
    difference = quadrature.gradient(u_h) - grad_u_exact(quadrature.points)
    return float(np.sqrt(2.0 * quadrature.integrate(energy_density(difference))))
