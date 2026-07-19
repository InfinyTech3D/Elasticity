"""Manufactured solutions: exact fields plus the instructions to apply BCs."""

from abc import ABC, abstractmethod


class ManufacturedSolution(ABC):
    """An exact displacement field that drives a verification test and knows how to apply BCs."""

    prescribe_displacement_on = {}    # {region: direction mask} where u is prescribed (fixed comps)
    traction_on = ()                  # regions where the derived traction is applied

    def __init__(self, deck):
        self.deck = deck

    @abstractmethod
    def u(self, point):
        """Displacement vector u at a point, shape (dim,)."""

    @abstractmethod
    def grad_u(self, point):
        """Displacement gradient du_i/dx_j at a point, shape (dim, dim); for the H1 seminorm."""

    @abstractmethod
    def source(self, point, material):
        """Body force vector f = -div(sigma) at a point, shape (dim,)."""

    @abstractmethod
    def stress(self, point, material):
        """Cauchy stress sigma_ij at a point, shape (dim, dim); a face traction is sigma . n."""
