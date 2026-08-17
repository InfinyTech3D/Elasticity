"""The quantities a level measures: one class per metric, so a metric is defined in one place."""

from abc import ABC, abstractmethod

import numpy as np

from VnV.verification.fem import energy, frobenius, l2, orthogonality_defect


class Measurement:
    """One solved level, sampled once: the fields at every quadrature point, and the energies.

    The exact energy is integrated in its own right rather than derived from the energy norm.
    Writing e = u_h - u for the error field, G = grad u, G_h = grad u_h, and C for the
    elasticity tensor, psi is a quadratic form, so the energy density of the difference is not
    the difference of the energy densities:

        psi(G_h) - psi(G) = psi(G_h - G) + (G_h - G) : C : G

    The trailing cross term is linear in the error and vanishes nowhere pointwise. It cancels
    only after integrating over the domain, and only if Galerkin orthogonality a(e, u_h) = 0
    holds, which is what reduces the energy difference to the energy norm of the error:

        U_h - U = -0.5 ||e||_E^2 + a(e, u_h)

    Orthogonality needs u_h to satisfy the continuous variational problem against its own
    space, which needs the load functional integrated exactly. FEMSourceTerm integrates it
    with a fixed rule, so a(e, u_h) = L_h(u_h) - L(u_h) is nonzero and of the same order as
    the energy error, leaving dU a mixture of the two. Deriving dU from Enorm would assume
    the defect away; integrating U separately is what lets aeuh measure it.

    The two energies live here rather than inside a metric because three metrics read them: Enorm's
    scale, dU's value and scale, and aeuh's scale.
    """

    def __init__(self, quadrature, displacement, solution, cpp_energy):
        self.quadrature = quadrature
        self.solution = solution
        self.cpp_energy = cpp_energy

        # Sampled here rather than inside each norm: the exact field is a lambdified sympy expression
        # evaluated one component at a time over every quadrature point, which is the most expensive
        # thing a level does outside the solve, and five norms used to ask for it separately.
        self.u_h = quadrature.values(displacement)
        self.u_exact = quadrature.sample(solution.u)
        self.grad_h = quadrature.grads(displacement)
        self.grad_exact = quadrature.sample(solution.grad_u)

        # The error fields the norms are taken over. A metric measures on these and takes its noise
        # floor on the exact ones, so the two go through the same reducer and stay commensurate.
        self.u_error = self.u_h - self.u_exact
        self.grad_error = self.grad_h - self.grad_exact

        self.energy_h = energy(quadrature, self.grad_h, solution.energy_density)
        self.energy_exact = energy(quadrature, self.grad_exact, solution.energy_density)

    @property
    def cpp_ratio(self):
        """SOFA's own potential energy over the Python quadrature's energy; should read 1."""
        return self.cpp_energy / self.energy_h


class Metric(ABC):
    """A quantity measured at every level: how to measure it, and what it is small relative to."""

    name = ""

    # The norms are what the order-of-accuracy test is about; dU and aeuh exist to keep each other
    # honest -- dU only reads as the energy error while aeuh converges faster, and the pair has to
    # satisfy dU = 0.5 Enorm^2 + aeuh. That makes them a cross-check on the energy identity rather
    # than a claim of their own, so they are computed on every level but shown only with the
    # diagnostics on.
    reported = True

    @abstractmethod
    def measure(self, m):
        """The metric's value on one level."""

    @abstractmethod
    def scale(self, m):
        """Reference magnitude the relative noise floor is taken against.

        Each metric needs a magnitude of its own dimension for the relative floor to mean anything:
        the norms scale with the field, the energy metrics with the energy. The exact norms are
        integrated on the level's own mesh and rule, so the floor and the error it gates are
        commensurate by construction.
        """


class L2(Metric):
    """Root of the integrated squared error, e = u_h - u."""

    name = "L2"

    def measure(self, m):
        return l2(m.quadrature, m.u_error)

    def scale(self, m):
        return l2(m.quadrature, m.u_exact)


class H1(Metric):
    """H1 semi-norm of the error: the same, on grad u_h - grad u, Frobenius."""

    name = "H1"

    def measure(self, m):
        return frobenius(m.quadrature, m.grad_error)

    def scale(self, m):
        return frobenius(m.quadrature, m.grad_exact)


class EnergyNorm(Metric):
    """The material-weighted H1 semi-norm the Galerkin solution minimizes in."""

    name = "Enorm"

    def measure(self, m):
        return float(np.sqrt(2.0 * energy(m.quadrature, m.grad_error,
                                          m.solution.energy_density)))

    def scale(self, m):
        return np.sqrt(2.0 * m.energy_exact)


class EnergyGap(Metric):
    """|U - U_h|: half of the energy-identity cross-check, not a convergence claim of its own."""

    name = "dU"
    reported = False

    def measure(self, m):
        return abs(m.energy_exact - m.energy_h)

    def scale(self, m):
        return m.energy_exact


class OrthogonalityDefect(Metric):
    """|a(e, u_h)|: the other half, nonzero because FEMSourceTerm uses a fixed quadrature rule."""

    name = "aeuh"
    reported = False

    def measure(self, m):
        return abs(orthogonality_defect(m.quadrature, m.grad_h, m.grad_error,
                                        m.solution.constitutive))

    def scale(self, m):
        return m.energy_exact


# Order matters: it is the column order of every table and the key order of every record.
#
# The orders these are expected to converge at are the deck's `expect`, not a constant here: they
# follow from the element, not from the metric, so a P2 deck states different numbers for the same
# five metrics. For P1 they read L2 -> 2, H1 -> 1, Enorm -> 1 (it is the material-weighted H1
# semi-norm); dU is quadratic in the error so it would be 2 under Galerkin orthogonality, but it also
# carries the load consistency error a(e, u_h) -- which FEMSourceTerm's fixed quadrature rule makes
# O(h^2) too, so dU measures a mixture of the two. aeuh is that defect, reported so the mixture is
# visible rather than silent: it is only safe to read dU as the energy error while aeuh converges
# faster.
METRICS = [L2(), H1(), EnergyNorm(), EnergyGap(), OrthogonalityDefect()]

REPORTED = [metric for metric in METRICS if metric.reported]
