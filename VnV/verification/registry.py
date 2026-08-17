"""Name -> class registries for deck-driven verification tests."""

from ..geometry.geometry import Bar1D, Beam2D, Beam3D
from .manufactured import (Quadratic1D, Quadratic2D, Trigonometric1D, Trigonometric2D,
                           Trigonometric3D)
from .materials import LinearElastic

# deck "geometry.type" -> geometry class
GEOMETRIES = {"Bar1D": Bar1D, "Beam2D": Beam2D, "Beam3D": Beam3D}

# deck "material.type" -> material class. Field and material are independent axes: the deck names
# one of each and the pair is a study, so a new material costs a line here rather than a class per
# combination.
MATERIALS = {"LinearElastic": LinearElastic}

# (topological dim of the geometry, deck "function") -> manufactured-field class. The key is
# the dimension of the PDE, not of the space it is solved in: that is spatial_dimensions, which
# the field takes as a constructor argument.
FIELDS = {
    (1, "quadratic"): Quadratic1D,
    (1, "trigonometric"): Trigonometric1D,
    (2, "quadratic"): Quadratic2D,
    (2, "trigonometric"): Trigonometric2D,
    (3, "trigonometric"): Trigonometric3D,
}
