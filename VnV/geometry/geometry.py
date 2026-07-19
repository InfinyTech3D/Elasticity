"""Tool-agnostic geometry descriptions for the VnV suite."""

from abc import ABC, abstractmethod

import numpy as np


class Geometry(ABC):
    """A shape parameterization plus its named boundary regions."""

    dim: int
    extents: list

    @property
    @abstractmethod
    def regions(self) -> dict:
        """Named boundary regions as ``{name: predicate(point) -> bool}``."""

    @property
    @abstractmethod
    def normals(self) -> dict:
        """Outward unit normal per region as ``{name: [n_x, ...]}`` (length dim)."""


class Bar1D(Geometry):
    """A 1D bar occupying ``[0, length]`` along x."""

    dim = 1

    def __init__(self, length):
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
        return {"left": [-1.0], "right": [1.0]}


class Beam2D(Geometry):
    """A 2D rectangle occupying ``[0, length] x [0, width]`` in the xy-plane."""

    dim = 2

    def __init__(self, length, width):
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
        return {"left": [-1.0, 0.0], "right": [1.0, 0.0],
                "bottom": [0.0, -1.0], "top": [0.0, 1.0]}


class Beam3D(Geometry):
    """A 3D box occupying ``[0, length] x [0, width] x [0, height]``."""

    dim = 3

    def __init__(self, length, width, height):
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
        return {"left": [-1.0, 0.0, 0.0], "right": [1.0, 0.0, 0.0],
                "bottom": [0.0, -1.0, 0.0], "top": [0.0, 1.0, 0.0],
                "front": [0.0, 0.0, -1.0], "back": [0.0, 0.0, 1.0]}
