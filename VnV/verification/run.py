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
from VnV.verification.fem import l2_error, h1_semi_error
from VnV.sofa.conventions import CONTAINER, ELEMENT_CPP


def run(deck_path):
    with open(deck_path) as f:
        deck = json.load(f)

    geo_spec = dict(deck["geometry"])
    geometry = GEOMETRIES[geo_spec.pop("type")](**geo_spec)
    solution = SOLUTIONS[(geometry.dim, deck["function"])](deck)
    element = deck["element"]
    degree = deck["quadratureDegree"]
    element_name = ELEMENT_CPP[element]     # SOFA geometry name expected by Sofa.SofaFEM

    print(f"{'mesh':>12} {'h':>10} {'L2':>12} {'rateL2':>7} {'H1':>12} {'rateH1':>7}")
    prev = None
    for res in deck["mesh"]["sizes"]:
        root = Sofa.Core.Node("root")
        MMSScene(geometry=geometry, material=deck["material"], force_field=deck["forceField"],
                 element=element, resolution=res, solvers=deck["solvers"], mms=solution).build(root)
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

        l2 = l2_error(nodes, node_indices, element_name, degree, uh, u_ex)
        h1 = h1_semi_error(nodes, node_indices, element_name, degree, uh, grad_u_ex)
        h = 1.0 / (res[0] - 1)
        if prev:
            rl2 = f"{np.log(l2 / prev[1]) / np.log(h / prev[0]):.2f}"
            rh1 = f"{np.log(h1 / prev[2]) / np.log(h / prev[0]):.2f}"
        else:
            rl2 = rh1 = ""
        mesh = "x".join(str(r) for r in res)
        print(f"{mesh:>12} {h:>10.5f} {l2:>12.3e} {rl2:>7} {h1:>12.3e} {rh1:>7}")
        prev = (h, l2, h1)
        Sofa.Simulation.unload(root)


if __name__ == "__main__":
    run(sys.argv[1])
