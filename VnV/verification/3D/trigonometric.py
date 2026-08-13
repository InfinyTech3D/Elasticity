"""3D trigonometric manufactured solution (linear elasticity, full Hooke)."""

import numpy as np
import sympy as sp

from ..manufactured import ManufacturedSolution

AMPLITUDE = 0.1


class Trigonometric3D(ManufacturedSolution):
    """u_i oscillates along all axes; each component fixed on its own faces, traction elsewhere."""

    prescribe_displacement_on = {"left": [1, 0, 0], "right": [1, 0, 0],
                                 "bottom": [0, 1, 0], "top": [0, 1, 0],
                                 "front": [0, 0, 1], "back": [0, 0, 1]}
    traction_on = ("left", "right", "bottom", "top", "front", "back")

    def displacement(self, coordinates):
        geom = self.deck["geometry"]
        kx = 2.0 * np.pi / geom["length"]
        ky = 2.0 * np.pi / geom["width"]
        kz = 2.0 * np.pi / geom["height"]
        x, y, z = coordinates
        return [AMPLITUDE * sp.sin(kx * x) * sp.cos(ky * y) * sp.cos(kz * z),
                AMPLITUDE * sp.cos(kx * x) * sp.sin(ky * y) * sp.cos(kz * z),
                AMPLITUDE * sp.cos(kx * x) * sp.cos(ky * y) * sp.sin(kz * z)]
