"""Manufactured solutions: exact fields plus the instructions to apply BCs."""

from abc import ABC, abstractmethod

import Sofa.SofaDeformable

# Spatial dimension -> SOFA's own Young/Poisson -> (mu, lambda) converter. Going through the
# bindings keeps the manufactured source term on exactly the constitutive branch the solver takes
# (plane stress at 2, plane strain at 3). No 1D variant is bound: at d=1, lambda + 2 mu = E, and
# the 1D solutions write E directly.
_LAME = {2: Sofa.SofaDeformable.toLameParameters2D, 3: Sofa.SofaDeformable.toLameParameters3D}


def lame(material, spatial_dimensions):
    """Lame parameters (mu, lambda) of the material, for the given embedding space."""
    return _LAME[spatial_dimensions](material["youngModulus"], material["poissonRatio"])


class ManufacturedSolution(ABC):
    """An exact displacement field that drives a verification test and knows how to apply BCs."""

    prescribe_displacement_on = {}    # {region: direction mask} where u is prescribed (fixed comps)
    traction_on = ()                  # regions where the derived traction is applied

    def __init__(self, deck, spatial_dimensions):
        self.deck = deck
        self.material = deck["material"]
        self.spatial_dimensions = spatial_dimensions

    def _pad_mask(self, mask):
        """Extend a mask to the embedding space, fixing the out-of-plane components."""
        return list(mask) + [1] * (self.spatial_dimensions - len(mask))

    @abstractmethod
    def u(self, point):
        """Displacement vector u at a point, shape (spatial_dimensions,)."""

    @abstractmethod
    def grad_u(self, point):
        """Displacement gradient du_i/dx_j, shape (spatial_dimensions,) squared; for the H1 seminorm."""

    @abstractmethod
    def source(self, point):
        """Body force vector f = -div(sigma) at a point, shape (spatial_dimensions,)."""

    @abstractmethod
    def stress(self, point):
        """Cauchy stress sigma_ij at a point, square; a face traction is sigma . n."""
