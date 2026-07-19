"""2D trigonometric manufactured solution (linear elasticity, plane stress)."""

import numpy as np

from ..manufactured import ManufacturedSolution

AMPLITUDE = 0.1


def lame(material):
    """Plane-stress Lame parameters (lambda, mu) from the material dict."""
    E, nu = material["youngModulus"], material["poissonRatio"]
    lam = E * nu / (1.0 - nu**2)
    mu = E / (2.0 * (1.0 + nu))
    return lam, mu


class Trigonometric2D(ManufacturedSolution):
    """u = A[sin(kx x)cos(ky y), cos(kx x)sin(ky y)], kx=2pi/L, ky=2pi/W: ux fixed on x-faces, uy on y-faces, traction elsewhere."""

    prescribe_displacement_on = {"left": [1, 0], "right": [1, 0],
                                 "bottom": [0, 1], "top": [0, 1]}
    traction_on = ("left", "right", "bottom", "top")

    def __init__(self, deck):
        super().__init__(deck)
        geom = deck["geometry"]
        self.kx = 2.0 * np.pi / geom["length"]
        self.ky = 2.0 * np.pi / geom["width"]

    def u(self, point):
        x, y = point[0], point[1]
        kx, ky = self.kx, self.ky
        return AMPLITUDE * np.array([np.sin(kx * x) * np.cos(ky * y),
                                     np.cos(kx * x) * np.sin(ky * y)])

    def grad_u(self, point):
        x, y = point[0], point[1]
        kx, ky = self.kx, self.ky
        ss = np.sin(kx * x) * np.sin(ky * y)
        cc = np.cos(kx * x) * np.cos(ky * y)
        return AMPLITUDE * np.array([[ kx * cc, -ky * ss],
                                     [-kx * ss,  ky * cc]])

    def source(self, point, material):
        # f = -div(sigma), linear elasticity with plane-stress Lame parameters
        lam, mu = lame(material)
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
        return AMPLITUDE * np.array([fx, fy])

    def stress(self, point, material):
        # sigma = lambda tr(eps) I + 2 mu eps
        lam, mu = lame(material)
        G = self.grad_u(point)
        exx, eyy = G[0, 0], G[1, 1]
        exy = 0.5 * (G[0, 1] + G[1, 0])
        tr = exx + eyy
        sxx = lam * tr + 2 * mu * exx
        syy = lam * tr + 2 * mu * eyy
        sxy = 2 * mu * exy
        return np.array([[sxx, sxy], [sxy, syy]])
