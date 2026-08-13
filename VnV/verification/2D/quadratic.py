"""2D quadratic manufactured solution: a load the P1/Q1 space represents exactly."""

from ..manufactured import ManufacturedSolution

AMPLITUDE = 0.1

# In-plane direction masks; the embedding space extends them (see prescribe_displacement_on).
IN_PLANE_MASKS = {"left": [1, 0], "right": [1, 0], "bottom": [0, 1], "top": [0, 1]}


class Quadratic2D(ManufacturedSolution):
    """u = A[x^2, y^2]: body force constant and traction linear, so both are nodally exact."""

    traction_on = ("left", "right", "bottom", "top")

    @property
    def prescribe_displacement_on(self):
        # Each face prescribes only the component that is constant on it -- u_x on the x-faces is
        # A x^2 at fixed x -- so the nodal values carry the Dirichlet data without error either.
        return {region: self._pad_mask(mask) for region, mask in IN_PLANE_MASKS.items()}

    def displacement(self, coordinates):
        x, y = coordinates[0], coordinates[1]
        return self._embed([AMPLITUDE * x**2, AMPLITUDE * y**2])
