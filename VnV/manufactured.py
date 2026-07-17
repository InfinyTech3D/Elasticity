"""Manufactured solutions: exact fields plus the instructions to apply BCs."""

from abc import ABC, abstractmethod

import numpy as np


class ManufacturedSolution(ABC):
    """An exact displacement field that drives a verification test and knows how to apply BCs."""

    prescribe_displacement_on = ()    # regions where u is prescribed (a clamp when u = 0)
    traction_on = ()                  # regions where the derived traction is applied

    @abstractmethod
    def u(self, x):
        """Exact displacement u(x)."""

    @abstractmethod
    def du(self, x):
        """Exact derivative u'(x); used for the H1 norm and the boundary traction."""

    @abstractmethod
    def source(self, x, E):
        """Body force f(x) = -E u''(x) that produces u under Young's modulus E."""


class Sinusoidal1D(ManufacturedSolution):
    """u(x) = sin(2 pi x): prescribed (clamped) at 'left', tip traction at 'right'."""

    prescribe_displacement_on = ("left",)
    traction_on = ("right",)

    def u(self, x):
        return np.sin(2.0 * np.pi * x)

    def du(self, x):
        return 2.0 * np.pi * np.cos(2.0 * np.pi * x)

    def source(self, x, E):
        # f = -E u'' with u = sin(2 pi x)
        return 4.0 * np.pi**2 * E * np.sin(2.0 * np.pi * x)
