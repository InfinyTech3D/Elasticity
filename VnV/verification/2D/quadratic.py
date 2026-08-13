"""2D quadratic manufactured solution: a load the P1/Q1 space represents exactly."""

import numpy as np

from ..manufactured import ManufacturedSolution, lame

AMPLITUDE = 0.1

# In-plane direction masks; the embedding space extends them (see prescribe_displacement_on).
IN_PLANE_MASKS = {"left": [1, 0], "right": [1, 0], "bottom": [0, 1], "top": [0, 1]}


class Quadratic2D(ManufacturedSolution):
    """u = A[x^2, y^2]: body force constant and traction linear, so both are nodally exact."""

    traction_on = ("left", "right", "bottom", "top")

    def __init__(self, deck, spatial_dimensions):
        super().__init__(deck, spatial_dimensions)
        self.mu, self.lam = lame(self.material, spatial_dimensions)

    @property
    def prescribe_displacement_on(self):
        # Each face prescribes only the component that is constant on it -- u_x on the x-faces is
        # A x^2 at fixed x -- so the nodal values carry the Dirichlet data without error either.
        return {region: self._pad_mask(mask) for region, mask in IN_PLANE_MASKS.items()}

    def u(self, point):
        x, y = point[0], point[1]
        return self._embed(AMPLITUDE * np.array([x**2, y**2]))

    def grad_u(self, point):
        x, y = point[0], point[1]
        return self._embed(AMPLITUDE * np.array([[2.0 * x, 0.0],
                                                 [0.0, 2.0 * y]]))

    def source(self, point):
        # f = -div(sigma) = -(lam + mu) grad(div u) - mu laplacian(u), and both are constant here:
        # grad(div u) = laplacian(u) = 2A[1, 1], so f = -2A(lam + 2mu)[1, 1] everywhere.
        magnitude = -2.0 * AMPLITUDE * (self.lam + 2.0 * self.mu)
        return self._embed(magnitude * np.ones(2))

    def stress(self, point):
        # sigma = lambda tr(eps) I + 2 mu eps, linear in the coordinates
        return self.constitutive(self.strain(self.grad_u(point)))
