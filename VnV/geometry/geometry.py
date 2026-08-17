"""Tool-agnostic geometry descriptions for the VnV suite."""

from abc import ABC, abstractmethod

import numpy as np


class Geometry(ABC):
    """A shape parameterization plus its named boundary regions."""

    dim: int          # topological dimension of the shape (and of the elements meshing it)
    extents: list

    # Extent names in axis order; a shape carries the first `dim` of them as attributes.
    EXTENT_NAMES = ("length", "width", "height")

    @property
    def named_extents(self) -> dict:
        """Extent by name, in axis order: how a manufactured field asks for one of them."""
        return {name: getattr(self, name) for name in self.EXTENT_NAMES[:self.dim]}

    def _init_space(self, spatial_dimensions):
        """Set the dimension of the space the shape is embedded in; defaults to its own."""
        self.spatial_dimensions = self.dim if spatial_dimensions is None else spatial_dimensions
        if not self.dim <= self.spatial_dimensions <= 3:
            raise ValueError(f"{type(self).__name__}: spatial_dimensions must be in "
                             f"[{self.dim}, 3], got {self.spatial_dimensions}")

    def _normal(self, axis, sign):
        """Outward unit normal along `axis`, expressed in the embedding space."""
        normal = [0.0] * self.spatial_dimensions
        normal[axis] = sign
        return normal

    @property
    @abstractmethod
    def regions(self) -> dict:
        """Named boundary regions as ``{name: predicate(point) -> bool}``."""

    @property
    @abstractmethod
    def normals(self) -> dict:
        """Outward unit normal per region as ``{name: [n_x, ...]}`` (length spatial_dimensions)."""


class Bar1D(Geometry):
    """A 1D bar occupying ``[0, length]`` along x."""

    dim = 1

    def __init__(self, length, spatialDimensions=None):
        self._init_space(spatialDimensions)
        self.length = length

    @property
    def extents(self) -> list:
        """Box max corner [Lx, Ly, Lz]; inactive axes are zero."""
        return [self.length, 0.0, 0.0]

    @property
    def regions(self) -> dict:
        L = self.length
        return {
            "left":  lambda p: np.isclose(p[0], 0.0),
            "right": lambda p: np.isclose(p[0], L),
        }

    @property
    def normals(self) -> dict:
        return {"left": self._normal(0, -1.0), "right": self._normal(0, 1.0)}


class Beam2D(Geometry):
    """A 2D rectangle occupying ``[0, length] x [0, width]`` in the xy-plane."""

    dim = 2

    def __init__(self, length, width, spatialDimensions=None):
        self._init_space(spatialDimensions)
        self.length = length
        self.width = width

    @property
    def extents(self) -> list:
        """Box max corner [Lx, Ly, Lz]; inactive axes are zero."""
        return [self.length, self.width, 0.0]

    @property
    def regions(self) -> dict:
        L, W = self.length, self.width
        return {
            "left":   lambda p: np.isclose(p[0], 0.0),
            "right":  lambda p: np.isclose(p[0], L),
            "bottom": lambda p: np.isclose(p[1], 0.0),
            "top":    lambda p: np.isclose(p[1], W),
        }

    @property
    def normals(self) -> dict:
        return {"left": self._normal(0, -1.0), "right": self._normal(0, 1.0),
                "bottom": self._normal(1, -1.0), "top": self._normal(1, 1.0)}


class Beam3D(Geometry):
    """A 3D box occupying ``[0, length] x [0, width] x [0, height]``."""

    dim = 3

    def __init__(self, length, width, height, spatialDimensions=None):
        self._init_space(spatialDimensions)
        self.length = length
        self.width = width
        self.height = height

    @property
    def extents(self) -> list:
        """Box max corner [Lx, Ly, Lz]."""
        return [self.length, self.width, self.height]

    @property
    def regions(self) -> dict:
        L, W, H = self.length, self.width, self.height
        return {
            "left":   lambda p: np.isclose(p[0], 0.0),
            "right":  lambda p: np.isclose(p[0], L),
            "bottom": lambda p: np.isclose(p[1], 0.0),
            "top":    lambda p: np.isclose(p[1], W),
            "front":  lambda p: np.isclose(p[2], 0.0),
            "back":   lambda p: np.isclose(p[2], H),
        }

    @property
    def normals(self) -> dict:
        return {"left": self._normal(0, -1.0), "right": self._normal(0, 1.0),
                "bottom": self._normal(1, -1.0), "top": self._normal(1, 1.0),
                "front": self._normal(2, -1.0), "back": self._normal(2, 1.0)}
