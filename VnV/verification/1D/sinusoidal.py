"""1D sinusoidal manufactured solution."""

import numpy as np

from ..manufactured import ManufacturedSolution


class Sinusoidal1D(ManufacturedSolution):
    """u(x) = [sin(k x)], k = 2 pi / L: prescribed (clamped) at 'left', traction at 'right'."""

    prescribe_displacement_on = {"left": [1]}
    traction_on = ("right",)

    def __init__(self, deck):
        super().__init__(deck)
        self.k = 2.0 * np.pi / deck["geometry"]["length"]

    def u(self, point):
        return np.array([np.sin(self.k * point[0])])

    def grad_u(self, point):
        return np.array([[self.k * np.cos(self.k * point[0])]])

    def source(self, point, material):
        # f = -div(sigma) = -E u''
        E = material["youngModulus"]
        return np.array([E * self.k**2 * np.sin(self.k * point[0])])

    def stress(self, point, material):
        # sigma_xx = E u'
        E = material["youngModulus"]
        return np.array([[E * self.k * np.cos(self.k * point[0])]])
