"""1D quadratic manufactured solution: a load the P1 space represents exactly."""

import numpy as np

from ..manufactured import ManufacturedSolution

AMPLITUDE = 0.1


class Quadratic1D(ManufacturedSolution):
    """u(x) = [A x^2]: body force constant, so the nodal source is the source; clamped at 'left'."""

    prescribe_displacement_on = {"left": [1]}
    traction_on = ("right",)

    def constitutive(self, strain):
        # no lame() branch exists at d = 1, where lambda + 2 mu = E collapses Hooke to sigma = E eps
        return self.material["youngModulus"] * strain

    def u(self, point):
        return np.array([AMPLITUDE * point[0] ** 2])

    def grad_u(self, point):
        return np.array([[2.0 * AMPLITUDE * point[0]]])

    def source(self, point):
        # f = -d(sigma)/dx = -2 A E, constant, so its nodal interpolant is exact
        return np.array([-2.0 * AMPLITUDE * self.material["youngModulus"]])

    def stress(self, point):
        # sigma_xx = E u' = 2 A E x
        return self.constitutive(self.strain(self.grad_u(point)))
