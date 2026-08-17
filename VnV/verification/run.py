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
from VnV.verification.metrics import FORMAL_ORDER, METRICS, NOISE_FLOOR_RELATIVE, Measurement
from VnV.verification.study import ConvergenceStudy, LevelResult
from VnV.sofa.conventions import CONTAINER, ELEMENT_CPP
from VnV.sofa.scene import load_plugins

# The reported subset, for the overview table -- which prints a column per metric even for a deck that
# raised and therefore has no study to ask.
REPORTED = [metric for metric in METRICS if metric.reported]

GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"

PROGRESS_WIDTH = 20

FIGURE_DIR = pathlib.Path(__file__).parent / "figures"

# The records are the run's evidence rather than a rendering of it, so they keep their own directory.
RESULTS_DIR = pathlib.Path(__file__).parent / "results"

# The rates-only table's per-metric column, wide enough for the `not reached` the summary can print
# underneath it. With the diagnostics on, a metric spans its value and rate columns instead.
RATE_COLUMN = 11


def paint(cell, accepted):
    """Green if the settling test kept this rate, red if it discarded it.

    The cell arrives already padded: the escape sequences are characters as far as a format width is
    concerned, so colouring before padding would widen the column by the length of the codes. Colour
    is a reading aid rather than data, hence the tty guard -- a redirected run stays plain text.
    """
    if not cell.strip() or not sys.stdout.isatty():
        return cell
    return f"{GREEN if accepted else RED}{cell}{RESET}"


def print_progress(level, levels, label, phase):
    """The level being run and which half of it, rewritten in place -- the table waits for the sweep.

    The finest levels dominate the wall clock, so the bar tracks levels rather than time: it is there
    to say which mesh is running, not to predict when it finishes. `phase` separates the SOFA side of
    a level -- scene build, init, solve -- from the Python norms, the two places a level can sit for a
    while. The label is padded so the phase does not shift as the element counts grow.
    """
    if not sys.stdout.isatty():
        return
    filled = round(PROGRESS_WIDTH * level / levels)
    bar = "=" * filled + "-" * (PROGRESS_WIDTH - filled)
    # \r to reuse the line and \033[K to drop whatever the previous, possibly longer, line left.
    sys.stdout.write(f"\r  {level}/{levels} [{bar}] {label:<14} {phase}\033[K")
    sys.stdout.flush()


def clear_progress():
    """Leave the line the bar was using empty, so the table starts on a clean row."""
    if sys.stdout.isatty():
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()


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


def print_table(study, show_diagnostics):
    """One row per level, printed once the sweep is over.

    A rate can only be coloured after the finest level exists: the settled range is the *trailing*
    run of orders, so whether a rate belongs to it is not known when its level is solved. Hence the
    whole table waits for the study's retained level indices, and the trailing run of green rates is
    what reports the asymptotic range.

    The order-of-accuracy test's evidence is the sequence of orders in the reported norms, so that is
    the default table and the metric name sits over its own rate column. The error magnitudes, the dU
    and aeuh cross-check, the cpp/py energy check and the Newton residuals are what those orders were
    computed from or against rather than the evidence itself, so they wait for `show_diagnostics`. The
    solve's `status` does not: a rate off an unconverged level is not evidence of anything, so the
    guard stays in both tables.
    """
    metrics = study.metrics if show_diagnostics else study.reported
    accepted = {metric.name: study.accepted(metric.name) for metric in metrics}
    header = f"{'cells':>12} {'h':>10}"
    # What the deck asked for is cells; what the mapping made of them is an element count worth
    # seeing, since it is 6x or 2x the cells for the split element types.
    if show_diagnostics:
        header += f" {'elements':>9}"
    for metric in metrics:
        header += (f" {metric.name:>12} {'rate':>6}" if show_diagnostics
                   else f" {metric.name:>{RATE_COLUMN}}")
    if show_diagnostics:
        # cpp/py: SOFA's own potential energy against the Python quadrature; should read 1.
        # nIt / r/r0: the Newton solve, so algebraic error polluting the fine end is visible.
        header += f" {'cpp/py':>8} {'nIt':>4} {'r/r0':>8}"
    print(header + " status")

    for index, level in enumerate(study.levels):
        row = f"{level.label:>12} {level.h:>10.5f}"
        if show_diagnostics:
            row += f" {level.elements:>9}"
        for metric in metrics:
            rate = level.rates[metric.name]
            # A discarded pair reads red whether it was discarded for a stated reason or simply fell
            # outside the settled run.
            cell = f"{rate.value:.2f}" if rate.value is not None else rate.reason
            accepted_here = index in accepted[metric.name]
            if show_diagnostics:
                row += f" {level.values[metric.name]:>12.3e} {paint(f'{cell:>6}', accepted_here)}"
            else:
                row += f" {paint(f'{cell:>{RATE_COLUMN}}', accepted_here)}"

        newton = level.newton
        if show_diagnostics:
            row += f" {level.cpp_ratio:>8.5f}"
            if newton is None:
                row += f" {'':>4} {'':>8}"
            else:
                row += f" {newton['iterations']:>4} {newton['reduction']:>8.1e}"
        # The status is the last column, so it needs no padding; `no newton` takes no colour -- the
        # deck has no Newton solve to pass or fail, which is neither acceptance nor rejection.
        if newton is None:
            row += " no newton"
        else:
            row += f" {paint(newton['status'], newton['status'] == 'ok')}"
        print(row)


def print_summary(study, show_diagnostics):
    """Per metric: the order in the settled range, its spread there, and what was expected.

    Which levels the range covers is not restated here -- the green rates in the table above are it.
    `show_diagnostics` picks which metrics and which column widths, so these rows stay under the
    metric they belong to in either table.
    """
    metrics = study.metrics if show_diagnostics else study.reported
    settled = [study.verdict(metric.name) for metric in metrics]
    rows = [
        # The order row is the run's verdict, so it carries the table's colours on the same rule the
        # rates do: green where the settling test produced an order, red where it never reached one.
        # Not green for agreeing with `expected` -- that comparison is the reader's to make, and
        # colouring it would need a tolerance on the answer the test is supposed to be measuring.
        ("order p", [f"{s.order:.2f}" if s else "not reached" for s in settled],
         [s is not None for s in settled]),
        ("spread", [f"{s.spread:.2f}" if s else "--" for s in settled], None),
        ("expected", [f"{study.expected[metric.name]:.1f}" for metric in metrics], None),
    ]

    print()
    for label, cells, coloured in rows:
        # Past the cells and h columns, plus the element count the diagnostics table inserts there.
        row = f"{label:>12} {'':>10}" + (f" {'':>9}" if show_diagnostics else "")
        for index, cell in enumerate(cells):
            padded = f"{cell:>12} {'':>6}" if show_diagnostics else f"{cell:>{RATE_COLUMN}}"
            row += f" {padded if coloured is None else paint(padded, coloured[index])}"
        print(row)

    unconverged = study.unconverged()
    if unconverged:
        print(f"\n  Newton did not converge at: {', '.join(unconverged)}"
              f" -- the orders above are contaminated by algebraic error at those levels.")


def print_overview(results, show_diagnostics):
    """One line per deck: the settled order per metric, `--` where none was reached.

    The per-deck tables above distinguish *why* a metric has no order -- floor, oscillation,
    divergence, never settled -- which this collapses to `--` for the sake of one line per deck.
    A deck that raised reads `error` instead, so a crash is never mistaken for a metric that simply
    did not settle. The orders carry the same colours they do per deck.
    """
    metrics = METRICS if show_diagnostics else REPORTED
    print()
    header = f"{'deck':<44}"
    for metric in metrics:
        header += f" {metric.name:>9}"
    print(header + f" {'solver':>9}")

    for name, study in results:
        row = f"{name:<44}"
        for metric in metrics:
            settled = None if study is None else study.verdict(metric.name)
            cell = "error" if study is None else f"{settled.order:.2f}" if settled else "--"
            row += f" {paint(f'{cell:>9}', settled is not None)}"
        # A settled order means nothing at a level where the solve did not converge, so the count of
        # such levels rides along on the same line rather than living only in the per-deck table.
        if study is None:
            solver_cell = "error"
        else:
            unconverged = len(study.unconverged())
            solver_cell = "ok" if not unconverged else f"{unconverged} bad"
        print(row + f" {paint(f'{solver_cell:>9}', solver_cell == 'ok')}")

    row = f"{'expected':<44}"
    for metric in metrics:
        row += f" {FORMAL_ORDER[metric.name]:>9.1f}"
    print(row + f" {'ok':>9}")


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
    """Every deck in the tree, each with its own table, then one line per deck."""
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
    print_overview(results, output["diagnostics"])


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

    sweep = refinement_sweep(deck["mesh"], geometry.extents)
    study = ConvergenceStudy(deck_name, element, solution.equation, METRICS,
                             deck["asymptoticTolerance"], FORMAL_ORDER)
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
            floors={metric.name: NOISE_FLOOR_RELATIVE * metric.scale(measurement)
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
            run_all(pathlib.Path(__file__).parent, output)
        else:
            run(arguments[0], output)
    finally:
        # Even if the sweep raised: the windows drawn so far are worth keeping, and the pipe has to be
        # closed for the child to know nothing more is coming. It is never waited on -- that is what
        # frees the terminal while the figures stay up.
        if output["live"] is not None:
            output["live"].detach()
