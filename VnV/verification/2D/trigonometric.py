"""2D trigonometric manufactured solution (linear elasticity)."""

import numpy as np
import sympy as sp

from ..manufactured import ManufacturedSolution

AMPLITUDE = 0.1

# In-plane direction masks; the embedding space extends them (see prescribe_displacement_on).
IN_PLANE_MASKS = {"left": [1, 0], "right": [1, 0], "bottom": [0, 1], "top": [0, 1]}


class Trigonometric2D(ManufacturedSolution):
    """u = A[sin(kx x)cos(ky y), cos(kx x)sin(ky y)], kx=2pi/L, ky=2pi/W: ux fixed on x-faces, uy on y-faces, traction elsewhere."""

    traction_on = ("left", "right", "bottom", "top")

    @property
    def prescribe_displacement_on(self):
        # Fixing the out-of-plane components is what keeps the stiffness matrix regular: their
        # block is decoupled from the in-plane one and carries no source, so it has a rigid
        # translation for a null mode -- and u = 0 there, so the constraint costs no physics.
        return {region: self._pad_mask(mask) for region, mask in IN_PLANE_MASKS.items()}

    def displacement(self, coordinates):
        geom = self.deck["geometry"]
        kx = 2.0 * np.pi / geom["length"]
        ky = 2.0 * np.pi / geom["width"]
        x, y = coordinates[0], coordinates[1]
        return self._embed([AMPLITUDE * sp.sin(kx * x) * sp.cos(ky * y),
                            AMPLITUDE * sp.cos(kx * x) * sp.sin(ky * y)])
