"""Name -> class registries for deck-driven verification tests."""

import importlib

from ..geometry.geometry import Bar1D, Beam2D, Beam3D

# deck "geometry.type" -> geometry class
GEOMETRIES = {"Bar1D": Bar1D, "Beam2D": Beam2D, "Beam3D": Beam3D}

# (topological dim of the geometry, deck "function") -> manufactured-solution class. The key is
# the dimension of the PDE, not of the space it is solved in: that is spatial_dimensions, which
# the solution takes as a constructor argument.
# Solution modules live under verification/<dim>/; the leading-digit dirs need importlib.
_sinusoidal_1d = importlib.import_module("VnV.verification.1D.sinusoidal")
_trigonometric_2d = importlib.import_module("VnV.verification.2D.trigonometric")
_trigonometric_3d = importlib.import_module("VnV.verification.3D.trigonometric")
SOLUTIONS = {
    (1, "sinusoidal"): _sinusoidal_1d.Sinusoidal1D,
    (2, "trigonometric"): _trigonometric_2d.Trigonometric2D,
    (3, "trigonometric"): _trigonometric_3d.Trigonometric3D,
}
