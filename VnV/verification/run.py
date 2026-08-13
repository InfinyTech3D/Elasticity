"""Deck-driven verification runner: dispatch a deck to an MMS convergence study."""

import json
import os
import pathlib
import sys

import numpy as np

# Make the VnV package importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import Sofa
import Sofa.Core
import Sofa.Simulation

from VnV.verification.registry import GEOMETRIES, SOLUTIONS
from VnV.verification.scene import MMSScene
from VnV.verification.fem import (energy, energy_norm_error, exact_energy, exact_h1_semi_norm,
                                  exact_l2_norm, h1_semi_error, l2_error, orthogonality_defect)
from VnV.sofa.conventions import CONTAINER, ELEMENT_CPP

# Rates to expect for P1: L2 -> 2, H1 -> 1, Enorm -> 1 (it is the material-weighted H1 semi-norm).
# dU is quadratic in the error so it would be 2 under Galerkin orthogonality, but it also carries
# the load consistency error a(e, u_h) -- which FEMSourceTerm's fixed quadrature rule makes O(h^2)
# too, so dU measures a mixture of the two. aeuh is that defect, reported so the mixture is visible
# rather than silent: it is only safe to read dU as the energy error while aeuh converges faster.
METRICS = ("L2", "H1", "Enorm", "dU", "aeuh")

# The order-of-accuracy test compares the observed order against the order above; the observed one
# is only an estimate of it inside the asymptotic range.
FORMAL_ORDER = {"L2": 2.0, "H1": 1.0, "Enorm": 1.0, "dU": 2.0, "aeuh": 2.0}

# A metric stops measuring the discretization and starts measuring its contaminants -- round-off,
# the norm quadrature, the linear solve -- once it drops to their level. The floor is this fraction
# of a reference magnitude of the metric's own dimension (see `scales` below), so one number serves
# all five. It catches round-off; it does not catch a loose solver tolerance, which sits far above
# it and needs a tolerance-tightening run to expose.
NOISE_FLOOR_RELATIVE = 1e-9


def refinement_sweep(mesh, extents):
    """Element counts and mesh spacing per level, coarsest first.

    A deck states the coarsest element count and how many times to refine it, so refinement at a
    fixed ratio on every axis is structural: there is no per-level list for a hand-written value to
    drift in, and the constant ratio the settling test assumes is the ratio the meshes have.
    """
    ratio = mesh["refinementRatio"]
    sweep = []
    for level in range(mesh["levels"]):
        elements = [count * ratio ** level for count in mesh["elements"]]
        # The classical mesh parameter is the largest element diameter, which the coarsest direction
        # sets; extents carries zeros for inactive axes, so zip against `elements` drops them.
        spacing = max(extent / count for extent, count in zip(extents, elements))
        sweep.append((elements, spacing))
    return sweep


def observed_order(history, metric):
    """Order from the two finest levels in `history`, or None plus the reason there is none.

    Reasons are reported rather than blanked so a discarded pair says why it was discarded:
      floor  one of the two errors is down among its contaminants, so the ratio is noise
      osc    the error stopped decreasing monotonically (convergence ratio R <= 0)
      div    the error is growing, or growing in increments (R >= 1)
    """
    if len(history) < 2:
        return None, ""

    current, previous = history[-1], history[-2]
    error, before = current["value"][metric], previous["value"][metric]
    if error <= current["floor"][metric] or before <= previous["floor"][metric]:
        return None, "floor"

    if len(history) >= 3:
        # Stern et al.'s convergence ratio on the triplet, R = (e_i - e_i-1) / (e_i-1 - e_i-2):
        # 0 < R < 1 is monotone convergence, R <= 0 oscillatory, R >= 1 divergent. It reads the
        # *increments*, so unlike e_i < e_i-1 it catches a series that oscillates while still
        # happening to decrease on this one pair. It does not catch a series that stalls -- a
        # stall gives a small positive R -- which is what the noise floor above is for.
        # Their R is formed from signed solution values, since in solution verification the error is
        # unknown; a manufactured solution hands us the error itself, so R is formed from the error
        # norms instead. The ratio is unchanged (both differences flip sign), but a norm cannot go
        # negative, so a solution oscillating about the exact field can still show a monotone norm.
        increment = error - before
        previous_increment = before - history[-3]["value"][metric]
        ratio = increment / previous_increment if previous_increment else np.inf
        if ratio <= 0.0:
            return None, "osc"
        if ratio >= 1.0:
            return None, "div"
    elif error >= before:
        return None, "div"

    return np.log(error / before) / np.log(current["h"] / previous["h"]), ""


def asymptotic_range(history, metric, tolerance):
    """Trailing levels over which the observed order has settled, per Roache's requirement that it
    stop changing under refinement.

    The settling test compares consecutive observed orders against each other, never against the
    formal order: selecting the range by closeness to the answer being verified would make the
    order-of-accuracy test circular. Returns the retained orders, finest last.
    """
    orders = [(index, level["order"][metric][0]) for index, level in enumerate(history)
              if level["order"][metric][0] is not None]

    # Contiguity matters: a readable order with a discarded pair beneath it does not extend a run.
    retained = []
    for index, order in reversed(orders):
        if retained and (index + 1 != retained[0][0] or abs(order - retained[0][1]) > tolerance):
            break
        retained.insert(0, (index, order))

    # One order cannot demonstrate that anything settled -- it takes two to compare.
    return retained if len(retained) >= 2 else []


def summarize(history, tolerance, labels):
    """Per metric: the settled range and the order in it, or None where nothing settled."""
    summary = {}
    for name in METRICS:
        retained = asymptotic_range(history, name, tolerance)
        orders = [order for _, order in retained]
        # The coarsest level whose pair survived; the pair spans it and the level below, so that
        # lower level is the first one inside the asymptotic range.
        summary[name] = None if not retained else {
            "from": labels[retained[0][0] - 1],
            "order": orders[-1],
            "spread": max(orders) - min(orders),
        }
    return summary


def print_summary(summary):
    """Per metric: where the observed order settled, what it is there, and what was expected."""
    rows = {"settled from": [], "order p": [], "spread": [], "expected": []}
    for name in METRICS:
        settled = summary[name]
        rows["settled from"].append(settled["from"] if settled else "not reached")
        rows["order p"].append(f"{settled['order']:.2f}" if settled else "--")
        rows["spread"].append(f"{settled['spread']:.2f}" if settled else "--")
        rows["expected"].append(f"{FORMAL_ORDER[name]:.1f}")

    print()
    for label, cells in rows.items():
        row = f"{label:>12} {'':>10}"
        for cell in cells:
            row += f" {cell:>12} {'':>6}"
        print(row)


def print_overview(results):
    """One line per deck: the settled order per metric, `--` where none was reached.

    The per-deck tables above distinguish *why* a metric has no order -- floor, oscillation,
    divergence, never settled -- which this collapses to `--` for the sake of one line per deck.
    A deck that raised reads `error` instead, so a crash is never mistaken for a metric that simply
    did not settle.
    """
    print()
    header = f"{'deck':<44}"
    for name in METRICS:
        header += f" {name:>9}"
    print(header)

    for name, summary in results:
        row = f"{name:<44}"
        for metric in METRICS:
            if summary is None:
                cell = "error"
            else:
                settled = summary[metric]
                cell = f"{settled['order']:.2f}" if settled else "--"
            row += f" {cell:>9}"
        print(row)

    row = f"{'expected':<44}"
    for metric in METRICS:
        row += f" {FORMAL_ORDER[metric]:>9.1f}"
    print(row)


def run_all(directory):
    """Every deck in the tree, each with its own table, then one line per deck."""
    results = []
    for path in sorted(directory.glob("*D/*.json")):
        name = f"{path.parent.name}/{path.stem}"
        print(f"\n--- {name} ---")
        try:
            results.append((name, run(path)))
        except Exception as error:      # one deck that blows up must not hide the other eight
            print(f"  failed: {type(error).__name__}: {error}")
            results.append((name, None))
    print_overview(results)


def run(deck_path):
    with open(deck_path) as f:
        deck = json.load(f)

    geo_spec = dict(deck["geometry"])
    geometry = GEOMETRIES[geo_spec.pop("type")](**geo_spec)
    solution = SOLUTIONS[(geometry.dim, deck["function"])](deck, geometry.spatial_dimensions)
    element = deck["element"]
    degree = deck["quadratureDegree"]
    element_name = ELEMENT_CPP[element]     # SOFA geometry name expected by Sofa.SofaFEM

    header = f"{'elements':>12} {'h':>10}"
    for name in METRICS:
        header += f" {name:>12} {'rate':>6}"
    # cpp/py: SOFA's own potential energy against the Python quadrature; should read 1.
    print(header + f" {'cpp/py':>8}")

    history = []
    labels = []
    for elements, h in refinement_sweep(deck["mesh"], geometry.extents):
        # RegularGridTopology's `n` counts grid points, not cells: nodes = elements + 1 per axis,
        # and its spacing is extent/(n-1). Converting here keeps that the only place the two
        # conventions meet.
        res = [count + 1 for count in elements]
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

        # Each metric needs a reference magnitude of its own dimension for the relative floor to
        # mean anything: the norms scale with the field, the energy metrics with the energy. The
        # exact norms are integrated on this mesh and rule, so the floor and the error it gates are
        # commensurate by construction.
        scales = {
            "L2":    exact_l2_norm(nodes, node_indices, element_name, degree, u_ex),
            "H1":    exact_h1_semi_norm(nodes, node_indices, element_name, degree, grad_u_ex),
            "Enorm": np.sqrt(2.0 * u_energy_exact),
            "dU":    u_energy_exact,
            "aeuh":  u_energy_exact,
        }

        # Read from the force field itself rather than Node.computeEnergy(), whose subtree sum also
        # picks up the point-load ConstantForceField that stands in for a traction in 1D.
        cpp_energy = beam.FEM.getPotentialEnergy()

        history.append({"h": h, "value": current,
                        "floor": {name: NOISE_FLOOR_RELATIVE * scales[name] for name in METRICS}})
        history[-1]["order"] = {name: observed_order(history, name) for name in METRICS}

        label = 'x'.join(str(count) for count in elements)
        labels.append(label)
        row = f"{label:>12} {h:>10.5f}"
        for name in METRICS:
            order, reason = history[-1]["order"][name]
            row += f" {current[name]:>12.3e} {f'{order:.2f}' if order is not None else reason:>6}"
        print(row + f" {cpp_energy / u_energy:>8.5f}")

        Sofa.Simulation.unload(root)

    summary = summarize(history, deck["asymptoticTolerance"], labels)
    print_summary(summary)
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: run.py <deck>.json | --all")
    if sys.argv[1] == "--all":
        run_all(pathlib.Path(__file__).parent)
    else:
        run(sys.argv[1])
