"""2D trigonometric manufactured solution (linear elasticity)."""

import numpy as np

from ..manufactured import ManufacturedSolution, lame

AMPLITUDE = 0.1

# In-plane direction masks; the embedding space extends them (see prescribe_displacement_on).
IN_PLANE_MASKS = {"left": [1, 0], "right": [1, 0], "bottom": [0, 1], "top": [0, 1]}


class Trigonometric2D(ManufacturedSolution):
    """u = A[sin(kx x)cos(ky y), cos(kx x)sin(ky y)], kx=2pi/L, ky=2pi/W: ux fixed on x-faces, uy on y-faces, traction elsewhere."""

    traction_on = ("left", "right", "bottom", "top")

    def __init__(self, deck, spatial_dimensions):
        super().__init__(deck, spatial_dimensions)
        geom = deck["geometry"]
        self.kx = 2.0 * np.pi / geom["length"]
        self.ky = 2.0 * np.pi / geom["width"]
        self.mu, self.lam = lame(self.material, spatial_dimensions)

    @property
    def prescribe_displacement_on(self):
        # Fixing the out-of-plane components is what keeps the stiffness matrix regular: their
        # block is decoupled from the in-plane one and carries no source, so it has a rigid
        # translation for a null mode -- and u = 0 there, so the constraint costs no physics.
        return {region: self._pad_mask(mask) for region, mask in IN_PLANE_MASKS.items()}

    def _embed(self, in_plane):
        """Place an in-plane vector or tensor in the embedding space, padded with zeros."""
        embedded = np.zeros((self.spatial_dimensions,) * in_plane.ndim)
        embedded[(slice(0, 2),) * in_plane.ndim] = in_plane
        return embedded

    def u(self, point):
        x, y = point[0], point[1]
        kx, ky = self.kx, self.ky
        return self._embed(AMPLITUDE * np.array([np.sin(kx * x) * np.cos(ky * y),
                                                 np.cos(kx * x) * np.sin(ky * y)]))

    def grad_u(self, point):
        x, y = point[0], point[1]
        kx, ky = self.kx, self.ky
        ss = np.sin(kx * x) * np.sin(ky * y)
        cc = np.cos(kx * x) * np.cos(ky * y)
        return self._embed(AMPLITUDE * np.array([[ kx * cc, -ky * ss],
                                                 [-kx * ss,  ky * cc]]))

    def source(self, point):
        # f = -div(sigma); the field is constant out of plane, so f has no out-of-plane component
        lam, mu = self.lam, self.mu
        x, y = point[0], point[1]
        kx, ky = self.kx, self.ky
        ux = np.sin(kx * x) * np.cos(ky * y)
        uy = np.cos(kx * x) * np.sin(ky * y)
        d2ux_dxx = -kx**2 * ux
        d2ux_dyy = -ky**2 * ux
        d2ux_dxy = -kx * ky * np.cos(kx * x) * np.sin(ky * y)
        d2uy_dxx = -kx**2 * uy
        d2uy_dyy = -ky**2 * uy
        d2uy_dxy = -kx * ky * np.sin(kx * x) * np.cos(ky * y)
        fx = -((lam + 2 * mu) * d2ux_dxx + lam * d2uy_dxy + mu * (d2ux_dyy + d2uy_dxy))
        fy = -(mu * (d2ux_dxy + d2uy_dxx) + lam * d2ux_dxy + (lam + 2 * mu) * d2uy_dyy)
        return self._embed(AMPLITUDE * np.array([fx, fy]))

    def stress(self, point):
        # sigma = lambda tr(eps) I + 2 mu eps, in the embedding space: at spatial_dimensions = 3
        # this is plane strain, eps_zz = 0 but sigma_zz = lambda tr(eps)
        return self.constitutive(self.strain(self.grad_u(point)))
