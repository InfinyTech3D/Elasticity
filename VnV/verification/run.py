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
from VnV.verification.fem import MeshQuadrature
from VnV.verification.metrics import METRICS, REPORTED, Measurement
from VnV.verification.report import (clear_progress, print_overview, print_progress, print_summary,
                                     print_table)
from VnV.verification.study import ConvergenceStudy, LevelResult
from VnV.sofa.conventions import CONTAINER, ELEMENT_CPP
from VnV.sofa.scene import load_plugins

FIGURE_DIR = pathlib.Path(__file__).parent / "figures"

# The records are the run's evidence rather than a rendering of it, so they keep their own directory.
RESULTS_DIR = pathlib.Path(__file__).parent / "results"


def newton_diagnostics(newton):
    """Iterations, residual reduction and stopping status of the Newton solve at one level.

    The standards require the algebraic error to be shown negligible before a refinement study means
    anything; these are what SOFA exposes to show it. `residualGraph` is a map Data that binds to
    Python as its string form, `"residual v0 v1 ..."`, holding *squared* norms (NewtonRaphsonSolver
    pushes squaredResidualNorm) with entry 0 taken before the first iteration -- hence the sqrt for a
    norm and len-1 for the iteration count.
    """
    if newton is None:
        return None

    values = []
    for token in newton.residualGraph.value.split():
        try:
            values.append(float(token))
        except ValueError:      # the map key, not one of its values
            continue

    status = newton.status.value
    # ConvergedEquilibrium means the iterations never started because the residual was already zero.
    # On a loaded MMS scene that is not a success, it means the load never reached the system.
    healthy = status.startswith("Converged") and status != "ConvergedEquilibrium"
    # r/r0, not r: a residual is a force while the error norms are not, so only the dimensionless
    # reduction can be held against anything else in the table. Undefined when it started at zero.
    reduction = float(np.sqrt(values[-1] / values[0])) if values and values[0] else float("nan")
    # The problem is linear, so a healthy solve converges in one iteration; anything more is the
    # stopping criterion fighting round-off, not physics.
    return {"iterations": max(len(values) - 1, 0),
            "reduction": reduction,
            "status": "ok" if healthy else status}


def pcg_diagnostics(linear):
    """The linear solver's residual history at one level, per Newton iteration, and its tolerance.

    `PCGLinearSolver` fills a `graph` Data with one curve per call to solve(), so one per Newton
    iteration, keyed `Error <n>`. It clears the map on AnimateBeginEvent, so the read has to sit
    after the animate and before the unload. The values are r.M^-1.r over b.b -- squared, and
    weighted by the preconditioner when one is attached -- which is exactly the quantity the solve
    tests against `tolerance`, so they are passed on unconverted.

    A map Data binds to Python as one line per key, `<key> <v0> <v1> ...`. This key carries a space,
    so the first two tokens of a line belong to the key; parsing the line as floats the way the
    Newton graph is parsed would read the iteration number as a residual. Returns None for a direct
    solver, which has no iterations to report.
    """
    graph = getattr(linear, 'graph', None) if linear is not None else None
    if graph is None:
        return None

    curves = {}
    for line in graph.value.splitlines():
        tokens = line.split()
        if len(tokens) < 3 or tokens[0] != "Error":
            continue
        curves[int(tokens[1])] = [float(token) for token in tokens[2:]]

    tolerance = getattr(linear, 'tolerance', None)
    return {"curves": curves, "tolerance": float(tolerance.value) if tolerance else None}


def refinement_sweep(mesh, extents):
    """Grid cells per axis and mesh spacing per level, coarsest first.

    A deck states the coarsest cell count and how many times to refine it, so refinement at a fixed
    ratio on every axis is structural: there is no per-level list for a hand-written value to drift
    in, and the constant ratio the settling test assumes is the ratio the meshes have.

    Cells, not elements: what the deck controls is the grid `RegularGridTopology` lays down, and the
    topology mappings then split each cell -- 6 tetrahedra per hexahedron, 2 triangles per quad. Those
    elements are larger than the cell (the tetrahedra span its body diagonal), but by a factor fixed
    across levels, so it cancels in the ratio the pairwise order divides by and `h` stays the honest
    mesh parameter for a rate.
    """
    ratio = mesh["refinementRatio"]
    sweep = []
    for level in range(mesh["levels"]):
        cells = [count * ratio ** level for count in mesh["cells"]]
        # The classical mesh parameter is the largest cell diameter, which the coarsest direction
        # sets; extents carries zeros for inactive axes, so zip against `cells` drops them.
        spacing = max(extent / count for extent, count in zip(extents, cells))
        sweep.append((cells, spacing))
    return sweep


def plots_module():
    """matplotlib stays optional: a run that only prints tables must not need it installed."""
    from VnV.verification import plots
    return plots


def write_results(record):
    """The record to disk, one file per deck, named as its figures are."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{record['deck'].replace('/', '_')}.json"
    with open(path, "w") as f:
        # default=float: the orders come out of numpy, and json refuses a float64 it is not told about.
        json.dump(record, f, indent=2, default=float)
    print(f"  wrote {path.relative_to(RESULTS_DIR.parent)}")


def finish_deck(study, output):
    """Close out a deck: the record first, then the figures drawn from it.

    One record serves the file and both figures, so a figure redrawn from disk and the run's own
    cannot come from different numbers.
    """
    record = study.to_json()
    if output["results"]:
        write_results(record)
    if not (output["residuals"] or output["convergence"]):
        return

    FIGURE_DIR.mkdir(exist_ok=True)
    for path in plots_module().render(record, FIGURE_DIR, output["residuals"],
                                     output["convergence"]):
        print(f"  wrote {path.relative_to(FIGURE_DIR.parent)}")
    if output["residuals"] and not any(level["pcg"] for level in record["levels"]):
        print("  no linear-solver residuals to plot: this deck solves directly")


def run_all(directory, output):
    """Every deck in the tree, each with its own table, then one line per deck.

    Returns [(name, study)] with a None study where the deck raised, so the caller can take an exit
    code off the sweep as a whole.
    """
    results = []
    for path in sorted(directory.glob("*D/*.json")):
        name = f"{path.parent.name}/{path.stem}"
        print(f"\n--- {name} ---")
        try:
            results.append((name, run(path, output)))
        except Exception as error:      # one deck that blows up must not hide the other eight
            clear_progress()            # it raised mid-sweep, so the bar still owns the line
            print(f"  failed: {type(error).__name__}: {error}")
            results.append((name, None))
    # The metric list is the caller's: a deck that raised has no study to ask for its own.
    print_overview(results, METRICS if output["diagnostics"] else REPORTED)
    return results


def run(deck_path, output):
    with open(deck_path) as f:
        deck = json.load(f)

    deck_file = pathlib.Path(deck_path)
    deck_name = f"{deck_file.parent.name}/{deck_file.stem}"
    geo_spec = dict(deck["geometry"])
    geometry = GEOMETRIES[geo_spec.pop("type")](**geo_spec)
    solution = SOLUTIONS[(geometry.dim, deck["function"])](deck, geometry.spatial_dimensions)
    element = deck["element"]
    degree = deck["quadratureDegree"]
    element_name = ELEMENT_CPP[element]     # SOFA geometry name expected by Sofa.SofaFEM

    noise_floor = deck["noiseFloorRelative"]

    sweep = refinement_sweep(deck["mesh"], geometry.extents)
    study = ConvergenceStudy(deck_name, element, solution.equation, METRICS,
                             deck["asymptoticTolerance"], deck["expect"],
                             deck["expectTolerance"])
    opened = False                      # the deck's window, raised on the first level that has curves
    for level, (cells, h) in enumerate(sweep, start=1):
        label = 'x'.join(str(count) for count in cells)
        print_progress(level, len(sweep), label, "sofa")

        # RegularGridTopology's `n` counts grid points, not cells: nodes = cells + 1 per axis, and
        # its spacing is extent/(n-1). Converting here keeps that the only place the two conventions
        # meet.
        res = [count + 1 for count in cells]
        root = Sofa.Core.Node("root")
        MMSScene(geometry=geometry, material=deck["material"], force_field=deck["forceField"],
                 element=element, resolution=res, solvers=deck["solvers"], mms=solution,
                 source_quadrature_degree=deck["sourceQuadratureDegree"]).build(root)
        Sofa.Simulation.init(root)
        Sofa.Simulation.animate(root, root.dt.value)
        beam = root.beam.Beam

        # The solve has returned, so its residual history exists; the norms below are the long part of
        # a level, which is exactly when a live window should already be showing this level. The window
        # is another process's, so this only posts the curve -- drawing it costs this run nothing.
        pcg = pcg_diagnostics(getattr(beam, 'linearSolver', None))
        live_windows = output["live"]
        if live_windows is not None and pcg and pcg["curves"]:
            if not opened:
                live_windows.figure(deck_name, solution.equation, element, len(sweep),
                                    pcg["tolerance"])
                opened = True
            live_windows.add(deck_name, label, pcg["curves"])

        print_progress(level, len(sweep), label, "norms")
        nodes = beam.dofs.rest_position.array()
        uh = beam.dofs.position.array() - nodes
        # node_indices[e] = the mesh-node indices forming element e (from the topology).
        node_indices = getattr(beam.topology, CONTAINER[element][1]).array()

        # The mapping is shared by every norm below rather than rebuilt inside each of them.
        quadrature = MeshQuadrature(nodes, node_indices, element_name, degree)
        # The energy comes off the force field itself rather than Node.computeEnergy(), whose subtree
        # sum also picks up the point-load ConstantForceField that stands in for a traction in 1D.
        measurement = Measurement(quadrature, uh, solution, beam.FEM.getPotentialEnergy())

        # The element count the mapping actually produced, since the deck states cells: one row of
        # node_indices per element, so 6x the cells for tetrahedra and 2x for triangles.
        study.add(LevelResult(
            label=label, h=h, elements=len(node_indices),
            values={metric.name: metric.measure(measurement) for metric in METRICS},
            floors={metric.name: noise_floor * metric.scale(measurement)
                    for metric in METRICS},
            cpp_ratio=measurement.cpp_ratio,
            newton=newton_diagnostics(getattr(beam, 'newton', None)),
            pcg=pcg))

        Sofa.Simulation.unload(root)

    clear_progress()
    # Every table and figure reads the study's own retained levels, so the green run and the reported
    # order cannot tell different stories about the same sweep.
    print_table(study, output["diagnostics"])
    print_summary(study, output["diagnostics"])
    finish_deck(study, output)
    return study


if __name__ == "__main__":
    arguments = sys.argv[1:]
    # The residual figure exists while the sweep runs, so it has a live form; the convergence figure is
    # the outcome of the whole sweep and only exists at the end, so it is written or not at all.
    flags = {flag: flag in arguments for flag in ("--diagnostics-on", "--residuals-live",
                                                  "--residuals-write", "--convergence-write",
                                                  "--results-write")}
    arguments = [argument for argument in arguments if argument not in flags]
    if len(arguments) != 1:
        sys.exit("usage: run.py <deck>.json | --all [--diagnostics-on] [--residuals-live] "
                 "[--residuals-write] [--convergence-write] [--results-write]")

    output = {"diagnostics": flags["--diagnostics-on"],
              "residuals": flags["--residuals-write"],
              "convergence": flags["--convergence-write"],
              "results": flags["--results-write"],
              "live": None}

    # Whether a window can open at all is decided here, on this machine, rather than left for the plot
    # process to discover: a batch or ssh session with no display still gets its figure, on disk.
    live_residuals = flags["--residuals-live"]
    if live_residuals and not plots_module().interactive():
        print("no display for --residuals-live: writing the residual figure instead")
        output["residuals"], live_residuals = True, False

    if live_residuals:
        from VnV.verification.live import LiveWindows
        FIGURE_DIR.mkdir(exist_ok=True)
        output["live"] = LiveWindows(FIGURE_DIR / "live.log")

    # Before the first table: the plugin load messages are the scene's, not a level's, so they belong
    # above the tables rather than interleaved with their rows.
    load_plugins()
    try:
        if arguments[0] == "--all":
            failed = [name for name, study in run_all(pathlib.Path(__file__).parent, output)
                      if study is None or not study.ok()]
        else:
            study = run(arguments[0], output)
            failed = [] if study.ok() else [study.deck]
    finally:
        # Even if the sweep raised: the windows drawn so far are worth keeping, and the pipe has to be
        # closed for the child to know nothing more is coming. It is never waited on -- that is what
        # frees the terminal while the figures stay up.
        if output["live"] is not None:
            output["live"].detach()

    # Non-zero so a sweep can gate something. The tables above already say which metric fell short and
    # why, so this only names the decks.
    if failed:
        sys.exit(f"\nfailed: {', '.join(failed)}")
