"""Tool-agnostic geometry descriptions for the VnV suite."""

from abc import ABC, abstractmethod

import numpy as np


class Geometry(ABC):
    """A shape parameterization plus its named boundary regions."""

    dim: int
    element: str

    @property
    @abstractmethod
    def regions(self) -> dict:
        """Named boundary regions as ``{name: predicate(point) -> bool}``."""


class Bar1D(Geometry):
    """A 1D bar occupying ``[0, length]`` along x."""

    dim = 1
    element = "edge"

    def __init__(self, length):
        self.length = length

    @property
    def regions(self) -> dict:
        L = self.length
        return {
            "left":  lambda p: np.isclose(p[0], 0.0),
            "right": lambda p: np.isclose(p[0], L),
        }
