"""3D trigonometric manufactured solution (linear elasticity, full Hooke)."""

import numpy as np

from ..manufactured import ManufacturedSolution, lame

AMPLITUDE = 0.1


class Trigonometric3D(ManufacturedSolution):
    """u_i oscillates along all axes; each component fixed on its own faces, traction elsewhere."""

    prescribe_displacement_on = {"left": [1, 0, 0], "right": [1, 0, 0],
                                 "bottom": [0, 1, 0], "top": [0, 1, 0],
                                 "front": [0, 0, 1], "back": [0, 0, 1]}
    traction_on = ("left", "right", "bottom", "top", "front", "back")

    def __init__(self, deck, spatial_dimensions):
        super().__init__(deck, spatial_dimensions)
        geom = deck["geometry"]
        self.kx = 2.0 * np.pi / geom["length"]
        self.ky = 2.0 * np.pi / geom["width"]
        self.kz = 2.0 * np.pi / geom["height"]
        self.mu, self.lam = lame(self.material, spatial_dimensions)

    def u(self, point):
        x, y, z = point[0], point[1], point[2]
        kx, ky, kz = self.kx, self.ky, self.kz
        return AMPLITUDE * np.array([np.sin(kx * x) * np.cos(ky * y) * np.cos(kz * z),
                                     np.cos(kx * x) * np.sin(ky * y) * np.cos(kz * z),
                                     np.cos(kx * x) * np.cos(ky * y) * np.sin(kz * z)])

    def grad_u(self, point):
        x, y, z = point[0], point[1], point[2]
        kx, ky, kz = self.kx, self.ky, self.kz
        sx, cx = np.sin(kx * x), np.cos(kx * x)
        sy, cy = np.sin(ky * y), np.cos(ky * y)
        sz, cz = np.sin(kz * z), np.cos(kz * z)
        return AMPLITUDE * np.array([
            [ kx * cx * cy * cz, -ky * sx * sy * cz, -kz * sx * cy * sz],
            [-kx * sx * sy * cz,  ky * cx * cy * cz, -kz * cx * sy * sz],
            [-kx * sx * cy * sz, -ky * cx * sy * sz,  kz * cx * cy * cz],
        ])

    def source(self, point):
        # f = -div(sigma); for this field it reduces to mu*|k|^2 u + (lam+mu) (k.T) (component-wise)
        lam, mu = self.lam, self.mu
        kx, ky, kz = self.kx, self.ky, self.kz
        k2 = kx**2 + ky**2 + kz**2
        ksum = kx + ky + kz
        ux, uy, uz = self.u(point)
        return np.array([ux * (mu * k2 + (lam + mu) * kx * ksum),
                         uy * (mu * k2 + (lam + mu) * ky * ksum),
                         uz * (mu * k2 + (lam + mu) * kz * ksum)])

    def stress(self, point):
        # sigma = lambda tr(eps) I + 2 mu eps
        lam, mu = self.lam, self.mu
        G = self.grad_u(point)
        eps = 0.5 * (G + G.T)
        return lam * np.trace(eps) * np.eye(3) + 2 * mu * eps
