"""1D sinusoidal manufactured solution."""

import numpy as np
import sympy as sp

from ..manufactured import ManufacturedSolution


class Sinusoidal1D(ManufacturedSolution):
    """u(x) = [sin(k x)], k = 2 pi / L: prescribed (clamped) at 'left', traction at 'right'."""

    prescribe_displacement_on = {"left": [1]}
    traction_on = ("right",)

    def displacement(self, coordinates):
        k = 2.0 * np.pi / self.deck["geometry"]["length"]
        return [sp.sin(k * coordinates[0])]
