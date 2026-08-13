"""Manufactured solutions: exact fields plus the instructions to apply BCs."""

from abc import ABC, abstractmethod

import numpy as np
import sympy as sp

import Sofa.SofaDeformable


def toLameParameters1D(youngModulus, poissonRatio):
    """(mu, lambda) at d = 1, where lambda + 2 mu = E collapses Hooke to sigma = E eps."""
    return 0.5 * youngModulus, 0.0


# Spatial dimension -> SOFA's own Young/Poisson -> (mu, lambda) converter. Going through the
# bindings keeps the manufactured source term on exactly the constitutive branch the solver takes
# (plane stress at 2, plane strain at 3).
_LAME = {1: toLameParameters1D,
         2: Sofa.SofaDeformable.toLameParameters2D,
         3: Sofa.SofaDeformable.toLameParameters3D}

_COORDINATES = sp.symbols("x y z")


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
        dimensions = spatial_dimensions
        self.mu, self.lam = lame(self.material, dimensions)

        self.coordinates = sp.Matrix(_COORDINATES[:dimensions])
        displacement = sp.Matrix(self.displacement(self.coordinates))
        gradient = displacement.jacobian(self.coordinates)
        stress = self.constitutive_law(self.strain(gradient))
        source = -sp.Matrix([sum(sp.diff(stress[i, j], self.coordinates[j])
                                 for j in range(dimensions))
                             for i in range(dimensions)])

        self.u = self._at_point(displacement, (dimensions,))
        self.grad_u = self._at_point(gradient, (dimensions, dimensions))
        self.stress = self._at_point(stress, (dimensions, dimensions))
        self.source = self._at_point(source, (dimensions,))
        self.constitutive, self.energy_density = self._compile_material_laws()

    @abstractmethod
    def displacement(self, coordinates):
        """Exact displacement as sympy expressions, one per embedding-space component."""

    @staticmethod
    def strain(gradient):
        """Small strain tensor eps = 1/2 (grad u + grad u^T) of a displacement gradient."""
        return (gradient + gradient.T) / 2

    def constitutive_law(self, strain):
        """Hooke's law sigma = lambda tr(eps) I + 2 mu eps, symbolic in the strain."""
        return self.lam * strain.trace() * sp.eye(self.spatial_dimensions) + 2 * self.mu * strain

    def energy_density_law(self, gradient):
        """Strain energy density psi = 1/2 sigma : eps, the integrand of the elastic energy."""
        strain = self.strain(gradient)
        stress = self.constitutive_law(strain)
        return sum(stress[i] * strain[i] for i in range(len(strain))) / 2

    def _at_point(self, expression, shape):
        """Compile an expression of the coordinates into f(point) -> ndarray of `shape`."""
        evaluate = sp.lambdify(list(self.coordinates), expression, "numpy")
        return lambda point: np.asarray(evaluate(*point), dtype=float).reshape(shape)

    def _compile_material_laws(self):
        """Compile the material laws into callables over a whole second-order tensor."""
        dimensions = self.spatial_dimensions
        tensor = sp.MatrixSymbol("T", dimensions, dimensions)
        constitutive = sp.lambdify(tensor, self.constitutive_law(sp.Matrix(tensor)), "numpy")
        energy_density = sp.lambdify(tensor, self.energy_density_law(sp.Matrix(tensor)), "numpy")
        return constitutive, energy_density

    def _pad_mask(self, mask):
        """Extend a mask to the embedding space, fixing the out-of-plane components."""
        return list(mask) + [1] * (self.spatial_dimensions - len(mask))

    def _embed(self, in_plane):
        """Place an in-plane displacement in the embedding space, padded with zeros."""
        return list(in_plane) + [0] * (self.spatial_dimensions - len(in_plane))
