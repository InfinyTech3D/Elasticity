"""Deck-driven verification runner: dispatch a deck to an MMS convergence study."""

import argparse
import json
import os
import pathlib
import sys
import traceback

import numpy as np

# Make the VnV package importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import Sofa
import Sofa.Core
import Sofa.Simulation

from VnV.verification.deck import Deck
from VnV.verification.scene import MMSScene
from VnV.verification.fem import MeshQuadrature
from VnV.verification.metrics import METRICS, REPORTED, Measurement
from VnV.verification.report import (clear_progress, print_overview, print_progress, print_summary,
                                     print_table)
from VnV.verification.study import ConvergenceStudy, LevelResult
from VnV.sofa.conventions import CONTAINER, ELEMENT_CPP
from VnV.sofa.scene import load_plugins

# Decks live in the <dim>D/ subdirectories of this file's own directory, and only decks do.
DECK_ROOT = pathlib.Path(__file__).parent

# Where `results/` and `figures/` land unless --output-dir says otherwise. Two directories rather than
# one because a record is the run's evidence while a figure is a rendering of it. Pointing a run
# somewhere else is what lets a baseline survive the next run, which is how two formulations get
# compared on the same deck.
DEFAULT_OUTPUT_DIR = DECK_ROOT

# Drawing is an inspection concern, so add_visual_style requires this itself rather than putting it
# in PLUGINS for every headless sweep to pay for. It is what provides VisualStyle.
VISUAL_PLUGIN = "Sofa.Component.Visual"


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

    The iteration number orders the solves and is then dropped, because ordering is all it is ever
    used for -- the figure lays them end to end and never names one. Keeping it as a dict key would
    mean carrying it through JSON, which stringifies keys, and every reader converting back before
    sorting so that "10" does not come before "2".
    """
    graph = getattr(linear, 'graph', None) if linear is not None else None
    if graph is None:
        return None

    solves = []
    for line in graph.value.splitlines():
        tokens = line.split()
        if len(tokens) < 3 or tokens[0] != "Error":
            continue
        solves.append((int(tokens[1]), [float(token) for token in tokens[2:]]))

    tolerance = getattr(linear, 'tolerance', None)
    return {"curves": [residuals for _, residuals in sorted(solves)],
            "tolerance": float(tolerance.value) if tolerance else None}


def plots_module():
    """matplotlib stays optional: a run that only prints tables must not need it installed."""
    from VnV.verification import plots
    return plots


def resolve_deck(path):
    """A deck path, taken relative to the deck root when it does not resolve where it was typed.

    The two entry points below run from different working directories -- runSofa chdirs to the
    scene's own directory while it loads it -- so `2D/trigonometric_tri.json` has to mean the same
    deck either way.
    """
    path = pathlib.Path(path)
    return path if path.is_absolute() or path.exists() else DECK_ROOT / path


def level_label(cells):
    """A mesh named by its cells per axis, as every table and figure names it."""
    return 'x'.join(str(count) for count in cells)


def build_level(deck, cells, root):
    """The scene for one mesh, on `root`. Both the sweep and createScene build through here."""
    # RegularGridTopology's `n` counts grid points, not cells: nodes = cells + 1 per axis, and its
    # spacing is extent/(n-1). Converting here keeps that the only place the two conventions meet.
    res = [count + 1 for count in cells]
    MMSScene(deck, res).build(root)


def add_visual_style(root):
    """Make the scene visible. Inspection only -- a sweep never draws, so this is not in build_level.

    Nothing in an MMS scene draws by default: there is no visual model, and the components that can
    draw themselves are gated on display flags no one has set. `showForceFields` is what puts the
    mesh on screen, since the FEM force field renders its own elements. `showBehaviorModels` is what
    puts the boundary conditions there: BaseROI::draw returns early without it, so a BoxROI's
    selection -- the thing most worth looking at -- would be invisible. The GUI's display-flags panel
    toggles the rest from here.

    Its own RequiredPlugin so the scene stays self-contained under runSofa, as the rest of it is.
    """
    root.addObject('RequiredPlugin', name='visual', pluginName=[VISUAL_PLUGIN])
    root.addObject('VisualStyle', displayFlags='showForceFields showBehaviorModels')


def measure(deck, beam, label, h, pcg):
    """One solved mesh as a LevelResult: the norms, their floors, and how the solve went."""
    nodes = beam.dofs.rest_position.array()
    uh = beam.dofs.position.array() - nodes
    # node_indices[e] = the mesh-node indices forming element e (from the topology).
    node_indices = getattr(beam.topology, CONTAINER[deck.element][1]).array()

    # The mapping is shared by every norm rather than rebuilt inside each of them. ELEMENT_CPP is the
    # SOFA geometry name Sofa.SofaFEM expects.
    quadrature = MeshQuadrature(nodes, node_indices, ELEMENT_CPP[deck.element],
                                deck.quadrature_degree)
    # The energy comes off the force field itself rather than Node.computeEnergy(), whose subtree sum
    # also picks up the point-load ConstantForceField that stands in for a traction in 1D.
    measurement = Measurement(quadrature, uh, deck.solution, beam.FEM.getPotentialEnergy())

    # The element count the mapping actually produced, since the deck states cells: one row of
    # node_indices per element, so 6x the cells for tetrahedra and 2x for triangles.
    return LevelResult(
        label=label, h=h, elements=len(node_indices),
        values={metric.name: metric.measure(measurement) for metric in METRICS},
        floors={metric.name: deck.noise_floor_relative * metric.scale(measurement)
                for metric in METRICS},
        cpp_ratio=measurement.cpp_ratio,
        newton=newton_diagnostics(getattr(beam, 'newton', None)),
        pcg=pcg)


def write_results(record, output_dir):
    """The record to disk, one file per deck, named as its figures are."""
    directory = output_dir / "results"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record['deck'].replace('/', '_')}.json"
    with open(path, "w") as f:
        # default=float: the orders come out of numpy, and json refuses a float64 it is not told about.
        json.dump(record, f, indent=2, default=float)
    print(f"  wrote {path.relative_to(output_dir)}")


def finish_deck(study, args):
    """Close out a deck: the record first, then the figures drawn from it.

    One record serves the file and both figures, so a figure redrawn from disk and the run's own
    cannot come from different numbers.
    """
    record = study.to_json()
    if args.results_write:
        write_results(record, args.output_dir)
    if not (args.residuals_write or args.convergence_write):
        return

    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for path in plots_module().render(record, figure_dir, args.residuals_write,
                                      args.convergence_write):
        print(f"  wrote {path.relative_to(args.output_dir)}")
    if args.residuals_write and not any(level["pcg"] for level in record["levels"]):
        print("  no linear-solver residuals to plot: this deck solves directly")


def run_all(directory, args):
    """Every deck in the tree, each with its own table, then one line per deck.

    Returns [(name, study)] with a None study where the deck raised, so the caller can take an exit
    code off the sweep as a whole.
    """
    results = []
    for path in sorted(directory.glob("*D/*.json")):
        name = f"{path.parent.name}/{path.stem}"
        print(f"\n--- {name} ---")
        try:
            results.append((name, run(path, args)))
        except Exception as error:      # one deck that blows up must not hide the other eight
            clear_progress()            # it raised mid-sweep, so the bar still owns the line
            print(f"  failed: {type(error).__name__}: {error}")
            # A single deck run outside --all raises and prints its own stack; inside --all the sweep
            # has to keep going, so the stack is only printed when it is asked for. Without it, four
            # decks failing on a new formulation is four re-runs to find four line numbers.
            # On stdout, not print_exc's default stderr: the stack has to stay next to the deck it
            # belongs to, and a redirected run buffers the two streams independently.
            if args.traceback:
                traceback.print_exc(file=sys.stdout)
            results.append((name, None))
    # The metric list is the caller's: a deck that raised has no study to ask for its own.
    print_overview(results, METRICS if args.diagnostics_on else REPORTED)
    return results


def run(deck_path, args):
    # Parsed and checked before anything is built, so a mistyped key costs a message rather than a
    # sweep that dies once the first mesh has already been solved.
    deck = Deck.load(deck_path)

    sweep = deck.levels()
    study = ConvergenceStudy(deck, METRICS)
    opened = False                      # the deck's window, raised on the first level that has curves
    for level, (cells, h) in enumerate(sweep, start=1):
        label = level_label(cells)
        print_progress(level, len(sweep), label, "sofa")

        root = Sofa.Core.Node("root")
        build_level(deck, cells, root)
        Sofa.Simulation.init(root)
        Sofa.Simulation.animate(root, root.dt.value)
        beam = root.beam.Beam

        # The solve has returned, so its residual history exists; the norms below are the long part of
        # a level, which is exactly when a live window should already be showing this level. The window
        # is another process's, so this only posts the curve -- drawing it costs this run nothing.
        pcg = pcg_diagnostics(getattr(beam, 'linearSolver', None))
        live_windows = args.live
        if live_windows is not None and pcg and pcg["curves"]:
            if not opened:
                live_windows.figure(deck.name, deck.equation, deck.element, len(sweep),
                                    pcg["tolerance"])
                opened = True
            live_windows.add(deck.name, label, pcg["curves"])

        print_progress(level, len(sweep), label, "norms")
        study.add(measure(deck, beam, label, h, pcg))

        Sofa.Simulation.unload(root)

    clear_progress()
    # Every table and figure reads the study's own retained levels, so the green run and the reported
    # order cannot tell different stories about the same sweep.
    print_table(study, args.diagnostics_on)
    print_summary(study, args.diagnostics_on)
    finish_deck(study, args)
    return study


def createScene(root):
    """runSofa entry point: `runSofa -l SofaPython3 run.py --argv <deck> --argv <level>`.

    One file, two jobs: `python -m VnV.verification.run` measures, runSofa looks. runSofa imports this
    file as a module named after it, so `__name__` is never "__main__" here and the CLI below does not
    run -- and both go through build_level, so what is inspected cannot drift from what is swept.

    Building is all this does. runSofa owns the init, the stepping and the window, which is why the
    scene arrives unsolved: the point is the setup, and the tables already report the answer. The
    controllers have still run by the time you see it -- they fill their Data on the init-done event --
    so a BoxROI's selection and a clamp's indices are there to read in the undeformed scene.
    """
    if len(sys.argv) != 3:
        sys.exit("run.py under runSofa expects: --argv <deck> --argv <level>")
    deck = Deck.load(resolve_deck(sys.argv[1]))
    cells, _ = deck.levels()[int(sys.argv[2])]
    add_visual_style(root)
    build_level(deck, cells, root)


def parse_arguments():
    """The CLI. `args` is the whole run configuration -- there is no second dict built from it."""
    parser = argparse.ArgumentParser(description=__doc__)
    # A positional with nargs='?' is allowed in a mutually exclusive group, which is what states
    # "a deck or --all, exactly one" declaratively rather than by counting leftover tokens.
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("deck", nargs="?", help="path to a deck")
    target.add_argument("--all", action="store_true", help="every deck under <dim>D/")
    parser.add_argument("--diagnostics-on", action="store_true",
                        help="the wide table: error magnitudes, dU/aeuh, elements, cpp/py, Newton")
    parser.add_argument("--results-write", action="store_true",
                        help="write the record everything else is drawn from")
    parser.add_argument("--convergence-write", action="store_true",
                        help="write the error-against-h figure")
    # The residual figure exists while the sweep runs, so it has a live form; the convergence figure
    # is the outcome of the whole sweep and only exists at the end, so it is written or not at all.
    parser.add_argument("--residuals-write", action="store_true",
                        help="write the linear-solver residual figure")
    parser.add_argument("--residuals-live", action="store_true",
                        help="the same figure in a window that fills in as the sweep runs")
    parser.add_argument("--traceback", action="store_true",
                        help="full stack for a deck that raises under --all, which otherwise "
                             "reports one line so the remaining decks still run")
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR,
                        help="parent of results/ and figures/ (default: alongside the decks)")
    args = parser.parse_args()
    args.live = None            # filled in below if a window can be opened at all
    return args


if __name__ == "__main__":
    args = parse_arguments()

    # Whether a window can open at all is decided here, on this machine, rather than left for the plot
    # process to discover: a batch or ssh session with no display still gets its figure, on disk.
    if args.residuals_live and not plots_module().interactive():
        print("no display for --residuals-live: writing the residual figure instead")
        args.residuals_write, args.residuals_live = True, False

    if args.residuals_live:
        from VnV.verification.live import LiveWindows
        figure_dir = args.output_dir / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        args.live = LiveWindows(figure_dir / "live.log")

    # Before the first table: the plugin load messages are the scene's, not a level's, so they belong
    # above the tables rather than interleaved with their rows.
    load_plugins()
    try:
        if args.all:
            failed = [name for name, study in run_all(DECK_ROOT, args)
                      if study is None or not study.ok()]
        else:
            study = run(resolve_deck(args.deck), args)
            failed = [] if study.ok() else [study.name]
    finally:
        # Even if the sweep raised: the windows drawn so far are worth keeping, and the pipe has to be
        # closed for the child to know nothing more is coming. It is never waited on -- that is what
        # frees the terminal while the figures stay up.
        if args.live is not None:
            args.live.detach()

    # Non-zero so a sweep can gate something. The tables above already say which metric fell short and
    # why, so this only names the decks.
    if failed:
        sys.exit(f"\nfailed: {', '.join(failed)}")
