"""Constitutive laws: a material declares its strain energy, and the rest is differentiated from it.

Symbolic and numpy only -- no SOFA. The parameters arrive as numbers, converted by the deck, which is
where SOFA is the authority on which constitutive branch they belong to.
"""

from abc import ABC, abstractmethod

import numpy as np
import sympy as sp


def _tensor(dimensions, prefix):
    """A d x d matrix of scalar symbols, so lambdified code indexes plain broadcastable arrays."""
    return sp.Matrix(dimensions, dimensions, lambda i, j: sp.Symbol(f"{prefix}_{i}_{j}"))


class Material(ABC):
    """A material as one strain energy density; its stress and tangent are derived from that.

    Declaring psi and differentiating it is the argument the manufactured fields already make for
    declaring only their displacement, applied one level down: a hand-written stress and a
    hand-written energy can disagree with each other, and psi = 1/2 sigma : eps is an identity only
    while psi stays quadratic.

    `stress` is dpsi/d(grad u), which is the Cauchy stress for a small-strain material and the first
    Piola stress for a hyperelastic one -- the same expression either way, which is why the source
    term derived from it needs no branch. `tangent` is the second derivative acting on a tensor: for
    a linear material it is the constant elasticity tensor, and it is what the Galerkin orthogonality
    defect contracts against.
    """

    @abstractmethod
    def energy_density(self, gradient):
        """psi as a function of the displacement gradient, symbolic in that matrix's entries."""

    def stress(self, gradient):
        """dpsi/d(grad u) at `gradient`: differentiated against placeholders, then substituted."""
        placeholder = _tensor(gradient.rows, "G")
        return self._stress(placeholder).subs(dict(zip(placeholder, gradient)))

    def tangent(self, gradient, direction):
        """d2psi/d(grad u)^2 at `gradient`, contracted with `direction`."""
        placeholder = _tensor(gradient.rows, "G")
        stress = self._stress(placeholder)
        rows, columns = placeholder.rows, placeholder.cols
        action = sp.Matrix(rows, columns, lambda i, j: sum(
            sp.diff(stress[i, j], placeholder[k, l]) * direction[k, l]
            for k in range(rows) for l in range(columns)))
        return action.subs(dict(zip(placeholder, gradient)))

    def _stress(self, placeholder):
        """The stress in terms of the placeholder symbols, differentiated once from psi."""
        psi = self.energy_density(placeholder)
        return sp.Matrix(placeholder.rows, placeholder.cols,
                         lambda i, j: sp.diff(psi, placeholder[i, j]))


class LinearElastic(Material):
    """Hooke: psi = 1/2 lambda tr(eps)^2 + mu eps:eps on the symmetric gradient eps."""

    def __init__(self, mu, lam):
        self.mu = mu
        self.lam = lam

    def energy_density(self, gradient):
        strain = (gradient + gradient.T) / 2
        return self.lam * strain.trace() ** 2 / 2 + self.mu * sum(e ** 2 for e in strain)


def compile_laws(material, dimensions):
    """psi and the tangent action as callables over batches of (..., d, d) tensors.

    One call covers every quadrature point of a mesh. Each tangent component is lambdified on its own
    for the same reason the fields are: a component that does not depend on its arguments compiles to
    a *scalar*, and stacking scalars with array-valued siblings is where the obvious version breaks --
    which is the common case here, since a linear material's tangent is constant.
    """
    gradient, direction = _tensor(dimensions, "G"), _tensor(dimensions, "D")
    arguments = list(gradient) + list(direction)
    energy_law = sp.lambdify(list(gradient), material.energy_density(gradient), "numpy")
    tangent_law = [sp.lambdify(arguments, entry, "numpy")
                   for entry in material.tangent(gradient, direction)]

    def columns_of(values):
        values = np.asarray(values, dtype=float)
        return values, [values[..., i, j]
                        for i in range(dimensions) for j in range(dimensions)]

    def energy_density(gradients):
        values, columns = columns_of(gradients)
        return np.broadcast_to(np.asarray(energy_law(*columns), dtype=float), values.shape[:-2])

    def tangent(at, applied_to):
        base, base_columns = columns_of(at)
        applied, applied_columns = columns_of(applied_to)
        batch = np.broadcast_shapes(base.shape[:-2], applied.shape[:-2])
        stress = np.empty(batch + (dimensions, dimensions), dtype=float)
        for (i, j), component in zip(np.ndindex(dimensions, dimensions), tangent_law):
            stress[..., i, j] = np.broadcast_to(
                np.asarray(component(*base_columns, *applied_columns), dtype=float), batch)
        return stress

    return energy_density, tangent
