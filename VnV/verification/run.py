"""Deck-driven verification runner: dispatch a deck to an MMS convergence study."""

import json
import os
import sys

import numpy as np

# Make the VnV package importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import Sofa
import Sofa.Core
import Sofa.Simulation

from VnV.verification.registry import GEOMETRIES, SOLUTIONS
from VnV.verification.scene import MMSScene
from VnV.verification.fem import (energy, energy_norm_error, exact_energy, h1_semi_error,
                                  l2_error, orthogonality_defect)
from VnV.sofa.conventions import CONTAINER, ELEMENT_CPP

# Rates to expect for P1: L2 -> 2, H1 -> 1, Enorm -> 1 (it is the material-weighted H1 semi-norm).
# dU is quadratic in the error so it would be 2 under Galerkin orthogonality, but it also carries
# the load consistency error a(e, u_h) -- which FEMSourceTerm's fixed quadrature rule makes O(h^2)
# too, so dU measures a mixture of the two. aeuh is that defect, reported so the mixture is visible
# rather than silent: it is only safe to read dU as the energy error while aeuh converges faster.
METRICS = ("L2", "H1", "Enorm", "dU", "aeuh")


def run(deck_path):
    with open(deck_path) as f:
        deck = json.load(f)

    geo_spec = dict(deck["geometry"])
    geometry = GEOMETRIES[geo_spec.pop("type")](**geo_spec)
    solution = SOLUTIONS[(geometry.dim, deck["function"])](deck, geometry.spatial_dimensions)
    element = deck["element"]
    degree = deck["quadratureDegree"]
    element_name = ELEMENT_CPP[element]     # SOFA geometry name expected by Sofa.SofaFEM

    header = f"{'mesh':>12} {'h':>10}"
    for name in METRICS:
        header += f" {name:>12} {'rate':>6}"
    # cpp/py: SOFA's own potential energy against the Python quadrature; should read 1.
    print(header + f" {'cpp/py':>8}")

    prev = None
    for res in deck["mesh"]["sizes"]:
        root = Sofa.Core.Node("root")
        MMSScene(geometry=geometry, material=deck["material"], force_field=deck["forceField"],
                 element=element, resolution=res, solvers=deck["solvers"], mms=solution,
                 source_quadrature_degree=deck["sourceQuadratureDegree"]).build(root)
        Sofa.Simulation.init(root)
        Sofa.Simulation.animate(root, root.dt.value)
        beam = root.beam.Beam
        nodes = beam.dofs.rest_position.array()
        uh = beam.dofs.position.array() - nodes
        # node_indices[e] = the mesh-node indices forming element e (from the topology).
        node_indices = getattr(beam.topology, CONTAINER[element][1]).array()

        def u_ex(*coords):
            return solution.u(np.asarray(coords))

        def grad_u_ex(*coords):
            return solution.grad_u(np.asarray(coords))

        psi = solution.energy_density
        u_energy = energy(nodes, node_indices, element_name, degree, uh, psi)

        # The exact energy is integrated in its own right rather than derived from the energy norm.
        # Writing e = u_h - u for the error field, G = grad u, G_h = grad u_h, and C for the
        # elasticity tensor, psi is a quadratic form, so the energy density of the difference is not
        # the difference of the energy densities:
        #
        #     psi(G_h) - psi(G) = psi(G_h - G) + (G_h - G) : C : G
        #
        # The trailing cross term is linear in the error and vanishes nowhere pointwise. It cancels
        # only after integrating over the domain, and only if Galerkin orthogonality a(e, u_h) = 0
        # holds, which is what reduces the energy difference to the energy norm of the error:
        #
        #     U_h - U = -0.5 ||e||_E^2 + a(e, u_h)
        #
        # Orthogonality needs u_h to satisfy the continuous variational problem against its own
        # space, which needs the load functional integrated exactly. FEMSourceTerm integrates it
        # with a fixed rule, so a(e, u_h) = L_h(u_h) - L(u_h) is nonzero and of the same order as
        # the energy error, leaving dU a mixture of the two. Deriving dU from Enorm would assume
        # the defect away; integrating U separately is what lets aeuh measure it.
        u_energy_exact = exact_energy(nodes, node_indices, element_name, degree, grad_u_ex, psi)

        current = {
            "L2":    l2_error(nodes, node_indices, element_name, degree, uh, u_ex),
            "H1":    h1_semi_error(nodes, node_indices, element_name, degree, uh, grad_u_ex),
            "Enorm": energy_norm_error(nodes, node_indices, element_name, degree, uh, grad_u_ex, psi),
            "dU":    abs(u_energy_exact - u_energy),
            "aeuh":  abs(orthogonality_defect(nodes, node_indices, element_name, degree, uh,
                                              grad_u_ex, solution.constitutive)),
        }

        # Read from the force field itself rather than Node.computeEnergy(), whose subtree sum also
        # picks up the point-load ConstantForceField that stands in for a traction in 1D.
        cpp_energy = beam.FEM.getPotentialEnergy()

        h = 1.0 / (res[0] - 1)
        row = f"{'x'.join(str(r) for r in res):>12} {h:>10.5f}"
        for name in METRICS:
            rate = "" if prev is None else f"{np.log(current[name] / prev[1][name]) / np.log(h / prev[0]):.2f}"
            row += f" {current[name]:>12.3e} {rate:>6}"
        print(row + f" {cpp_energy / u_energy:>8.5f}")

        prev = (h, current)
        Sofa.Simulation.unload(root)


if __name__ == "__main__":
    run(sys.argv[1])
