"""1D quadratic manufactured solution: a load the P1 space represents exactly."""

from ..manufactured import ManufacturedSolution

AMPLITUDE = 0.1


class Quadratic1D(ManufacturedSolution):
    """u(x) = [A x^2]: body force constant, so the nodal source is the source; clamped at 'left'."""

    prescribe_displacement_on = {"left": [1]}
    traction_on = ("right",)

    def displacement(self, coordinates):
        return [AMPLITUDE * coordinates[0] ** 2]
