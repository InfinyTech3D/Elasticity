"""Figures from a verification sweep. No SOFA here -- the runner reads the Data, this draws it.

Everything is drawn from the runner's result record (`--results-write`), so `python -m
VnV.verification.plots <result>.json` redraws a sweep without solving it again.
"""

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")       # a batch run writes files; `interactive()` switches this for a live window
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullLocator

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e8e7e3"

# One ordered ramp per verdict: the hue says whether the solve reached the floor, the step within the
# hue says which refinement level it was, so a curve carries both at once. Sampled away from each
# ramp's pale end, where a step stops holding contrast against the surface.
RAMPS = {True: "Greens", False: "Reds", None: "Greys"}
RAMP_RANGE = (0.45, 0.95)


def interactive():
    """Switch to a backend that can open a window, and report whether one is really available.

    `use` succeeding is not proof: Tk accepts the name and only fails when the first figure looks for
    a display, so a throwaway figure is the probe. A headless run says so once and writes instead of
    dying on the first level.
    """
    for backend in ("TkAgg", "QtAgg", "MacOSX"):
        try:
            matplotlib.use(backend, force=True)
            plt.close(plt.figure())
            return True
        except Exception:
            continue
    matplotlib.use("Agg", force=True)
    return False


def ramp_colour(verdict, index, count):
    """Step `index` of `count` along the ramp belonging to `verdict`, coarsest light to finest dark."""
    start, end = RAMP_RANGE
    position = end if count < 2 else start + (end - start) * index / (count - 1)
    return to_hex(colormaps[RAMPS[verdict]](position))


# The same two hues, unstepped: on the convergence figure a level's place is its position along h, so
# the ramp has no second job and both verdicts take their ramp's darkest step.
ACCEPTED = ramp_colour(True, 0, 1)
DISCARDED = ramp_colour(False, 0, 1)

# The convergence figure's markers. Fill tells the three norms apart where their curves crowd, shape
# tells the element apart between figures -- a cell of the same kind carries the same outline whichever
# deck drew it. Both are redundant with a label on the plot, which is why neither reaches the key.
MARKER_FILL = {"L2": INK, "H1": to_hex(colormaps["Blues"](0.72)), "Enorm": SURFACE}
MARKER_SHAPE = {"edge": "o", "quad": "s", "hexa": "s", "tri": "^", "tet": "^"}


def _style(axis):
    """Recessive frame: the data is the only thing that should draw attention."""
    axis.set_facecolor(SURFACE)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=INK_SOFT, labelsize=9)
    # A log decade's minor ticks stipple the spine without saying anything the labels do not.
    axis.tick_params(which="minor", length=0)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))


def _title(axis, equation):
    """The manufactured field alone: which deck and which figure this is belongs in a caption.

    The pad is what the title's own lines grow into, so a wordy 3D field needs no special case.
    """
    axis.set_title(equation, fontsize=11, color=INK, pad=12)


def _element_note(axis, element, corner):
    """Which element ran, inside the frame, so a figure lifted into a larger one carries it along.

    `corner` because the free corner is not the same on both figures: errors climb to the right, so
    the top left is empty there, while a residual history starts at its highest and occupies it.
    """
    x, alignment = (0.02, "left") if corner == "left" else (0.98, "right")
    # Ink and a size above the tick labels: it is a statement about the figure, not an axis annotation.
    axis.text(x, 0.97, f"Element: {element.capitalize()}", transform=axis.transAxes,
              ha=alignment, va="top", fontsize=11, color=INK)


def _frame(figure, axis, equation, element, tolerance):
    """Everything that does not depend on the data, so a live figure can stand up before it has any."""
    figure.patch.set_facecolor(SURFACE)
    _style(axis)
    if tolerance:
        axis.axhline(tolerance, color=INK_SOFT, linewidth=1.2, linestyle=(0, (4, 3)))
    axis.set_yscale("log")
    _title(axis, equation)
    _element_note(axis, element, "right")
    axis.set_xlabel("cumulative PCG iterations", fontsize=9, color=INK_SOFT)
    axis.set_ylabel("residual  (r M^-1 r) / (b b)", fontsize=9, color=INK_SOFT)


def _draw_level(axis, label, curves, tolerance, index, count):
    """One level: its solves laid end to end, coloured by verdict and stepped by refinement."""
    residuals, seams = [], []
    for iteration in sorted(curves):
        if residuals:                   # not the first solve, so this point is a handover
            seams.append(len(residuals))
        residuals.extend(curves[iteration])

    # Each solve is tested against the floor in its own right, and the last value of each is where it
    # stopped, so one solve short of the floor condemns the level. Without a reported tolerance there
    # is nothing to test against and the level stays neutral.
    converged = all(curves[key][-1] <= tolerance for key in curves) if tolerance else None
    colour = ramp_colour(converged, index, count)

    axis.plot(range(len(residuals)), residuals, color=colour, linewidth=1.8,
              marker="o", markersize=3.5, label=label)
    axis.plot(seams, [residuals[seam] for seam in seams], linestyle="none", marker="o",
              markersize=7, markerfacecolor=SURFACE, markeredgecolor=colour, markeredgewidth=1.6)
    if converged is False:
        axis.plot(len(residuals) - 1, residuals[-1], linestyle="none", marker="x",
                  markersize=8, markeredgewidth=2.0, color=colour)


def _legend(figure, axis):
    """The level key, rebuilt from whatever is on the axis -- a live figure calls this per level."""
    handles, labels = axis.get_legend_handles_labels()
    while figure.legends:               # a live redraw replaces its key rather than stacking keys
        figure.legends[0].remove()
    # Wrapped at four, so a long sweep's labels stay inside the figure instead of running off it;
    # the reserved strip grows with the row count the wrap produces.
    columns = 4
    rows = -(-len(labels) // columns)
    figure.legend(handles, labels, title="cells per axis", loc="lower center", frameon=False,
                  ncol=columns, fontsize=9, title_fontsize=9)
    figure.tight_layout(rect=(0, 0.04 + 0.055 * rows, 1, 1))


def pcg_residuals(equation, element, levels, tolerance, path):
    """PCG residual history against cumulative iterations, one curve per refinement level.

    `levels` is [(label, {newton iteration: [residuals]})] coarsest first. Cost is the question, so
    the Newton solves of a level are laid end to end on one axis: where a curve ends is what that
    level spent, and how the ends spread out is how the cost scales under refinement. A hollow ring
    marks where one Newton solve hands over to the next.

    The residuals are SOFA's own quantity, r.M^-1.r over b.b, and `tolerance` is the floor it tests
    them against, both drawn unconverted.

    Hue is the verdict and the step within it is the level, so the legend still names every run while
    a level that stopped above the floor reads red among the greens. Its final point also carries an x,
    so the verdict does not rest on hue alone.
    """
    figure, axis = plt.subplots(figsize=(7.6, 4.6))
    _frame(figure, axis, equation, element, tolerance)
    for index, (label, curves) in enumerate(levels):
        _draw_level(axis, label, curves, tolerance, index, len(levels))
    _legend(figure, axis)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def _error_frame(figure, axis, equation, element):
    figure.patch.set_facecolor(SURFACE)
    _style(axis)
    # After _style: setting a scale installs that scale's own locators, dropping the integer one.
    axis.set_xscale("log")
    axis.set_yscale("log")
    _title(axis, equation)
    _element_note(axis, element, "left")
    axis.set_xlabel("mesh spacing h", fontsize=9, color=INK_SOFT)
    axis.set_ylabel("Norm", fontsize=9, color=INK_SOFT)


def _draw_metric(axis, metric, marker):
    """One metric's errors against h, its segments coloured by the verdict on the rate they carry."""
    spacings, errors, accepted = metric["h"], metric["error"], metric["accepted"]

    # A rate belongs to a *pair* of levels, so the segment between them is what the settling test
    # kept or discarded -- the same cell the table paints, from the same verdicts. Colouring the
    # points instead would have to invent a verdict for the coarsest level, which has no rate at all.
    for index in range(1, len(spacings)):
        axis.plot(spacings[index - 1:index + 1], errors[index - 1:index + 1],
                  color=ACCEPTED if accepted[index] else DISCARDED, linewidth=2.0,
                  solid_capstyle="round", zorder=3)
    # Marker fill separates the three norms at a glance where they crowd; the shape says which element
    # the deck ran. Neither is in the key -- the line's own label names it, these only help the eye.
    axis.plot(spacings, errors, linestyle="none", marker=marker, markersize=6,
              markerfacecolor=MARKER_FILL[metric["name"]], markeredgecolor=INK_SOFT,
              markeredgewidth=1.2, zorder=4)

    # Hue is the verdict here, so identity is a label on the line rather than a colour in a key.
    axis.annotate(metric["name"], (spacings[0], errors[0]), textcoords="offset points",
                  xytext=(9, 0), ha="left", va="center", fontsize=9, color=INK)

    # The reported order is the rate on the finest pair, so it is printed on the segment that produced
    # it: the number a reviewer wants is then on the figure rather than only in the table. Where that
    # pair was discarded the reason takes its place, in the colour that says it was discarded.
    rate, reason = metric["rate"][-1], metric["reason"][-1]
    text = f"{rate:.2f}" if rate is not None else reason
    if text:
        # Geometric means: on log-log axes that is the midpoint of the segment.
        anchor = ((spacings[-2] * spacings[-1]) ** 0.5, (errors[-2] * errors[-1]) ** 0.5)
        axis.annotate(text, anchor, textcoords="offset points", xytext=(6, -8), ha="left", va="top",
                      fontsize=9, color=ACCEPTED if accepted[-1] else DISCARDED)


def _discarded_note(axis):
    """The one thing colour needs to say, inside the frame and only when there is red to explain.

    Green is every rate the settling test was willing to read, so it needs no entry; a sweep that
    settled from its first pair has nothing to explain at all and keeps a bare frame.
    """
    axis.legend([Line2D([], [], color=DISCARDED, linewidth=2.0)], ["pre-asymptotic regime"],
                loc="lower right", frameon=False, fontsize=9, handlelength=1.6,
                labelcolor=INK_SOFT)


def error_convergence(equation, element, series, path):
    """Error against mesh spacing, log-log, one line per reported norm.

    `series` is [{name, h, error, rate, reason, accepted}] with the levels coarsest first, so
    refinement reads right to left and a line's slope is its observed order. What the figure has to say
    is which part of that line the order-of-accuracy test was allowed to read: the green run at the
    fine end is the asymptotic range, drawn from the same retained levels the table colours, and the
    rate printed on the last segment is the order the run reports.
    """
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    _error_frame(figure, axis, equation, element)
    for metric in series:
        _draw_metric(axis, metric, MARKER_SHAPE[element])

    spacings = series[0]["h"]
    # Room on the right for the labels, which hang off the coarsest point.
    axis.set_xlim(min(spacings) / 1.2, max(spacings) * 2.4)
    # A sweep is a handful of levels inside a decade or two, where a log axis labels minor ticks and
    # says nothing about the meshes. The meshes are the ticks instead.
    axis.set_xticks(spacings)
    axis.set_xticklabels([f"{spacing:.3g}" for spacing in spacings])
    axis.xaxis.set_minor_locator(NullLocator())

    # A segment exists per pair, so the coarsest level's own entry is not a discarded rate but no rate.
    if any(not accepted for metric in series for accepted in metric["accepted"][1:]):
        _discarded_note(axis)
    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def _series(result):
    """The convergence figure's input, pulled from a result record."""
    levels = result["levels"]
    return [{"name": name,
             "h": [level["h"] for level in levels],
             "error": [level["metrics"][name]["error"] for level in levels],
             "rate": [level["metrics"][name]["rate"] for level in levels],
             "reason": [level["metrics"][name]["reason"] for level in levels],
             "accepted": [level["metrics"][name]["accepted"] for level in levels]}
            for name in result["reported"]]


def _residual_levels(result):
    """The residual figure's input: (label, curves) per level that has any, and the shared tolerance.

    JSON keys are strings, so the Newton iteration numbers come back as text and have to be integers
    again before anything sorts them -- `"10"` sorts before `"2"`.
    """
    levels, tolerance = [], None
    for level in result["levels"]:
        pcg = level["pcg"]
        if not pcg or not pcg["curves"]:
            continue
        tolerance = pcg["tolerance"]
        levels.append((level["label"], {int(key): values for key, values in pcg["curves"].items()}))
    return levels, tolerance


def render(result, figure_dir, residuals, convergence):
    """Write the asked-for figures for one result record and return the paths written.

    The runner renders the record it just wrote, and this module's `__main__` renders one off disk,
    so a post-processed figure and the run's own cannot come from different numbers.
    """
    stem = result["deck"].replace("/", "_")
    written = []
    if convergence:
        path = figure_dir / f"{stem}_error.png"
        error_convergence(result["equation"], result["element"], _series(result), path)
        written.append(path)
    if residuals:
        levels, tolerance = _residual_levels(result)
        if levels:      # a direct solver has no iterations, so there is nothing to draw
            path = figure_dir / f"{stem}_pcg.png"
            pcg_residuals(result["equation"], result["element"], levels, tolerance, path)
            written.append(path)
    return written


class LivePlot:
    """The same figure, drawn while the sweep runs: one level joins it as soon as its solve returns.

    A level's residual history exists the moment `animate` returns, well before its error norms are
    integrated, so the window fills in during the part of the level that actually takes the time. The
    granularity is one curve per level and cannot be finer -- a solve is a single call into SOFA, and
    the `graph` Data only becomes readable once it comes back.

    `levels` is the sweep length, known up front, so the ramp step of an early level is the step it
    will still have when the sweep ends and the figure stops changing colour under the reader.
    """

    def __init__(self, equation, element, levels, tolerance):
        self.levels = levels
        self.tolerance = tolerance
        self.drawn = 0
        self.figure, self.axis = plt.subplots(figsize=(7.6, 4.6))
        _frame(self.figure, self.axis, equation, element, tolerance)
        plt.show(block=False)
        self._paint()

    def add(self, label, curves):
        _draw_level(self.axis, label, curves, self.tolerance, self.drawn, self.levels)
        self.drawn += 1
        _legend(self.figure, self.axis)
        self._paint()

    def _paint(self):
        """Render this figure now, rather than leaving it for an idle moment that will not come.

        `plt.pause` would only `draw_idle()` -- scheduling the render -- and then hand the GUI loop the
        few milliseconds it is given, which is not enough to finish a figure of this size, so the paint
        stays pending until the next call and the curves arrive in batches. It also works on whichever
        figure is globally active, which is the wrong one as soon as a second deck opens its own. A
        synchronous draw on our own canvas, then a flush so the window manager blits it, puts the level
        on screen before the sweep moves on.

        The window is still frozen *between* these calls: the solve is a blocking call into SOFA and
        the norms are numpy, so no events are served while a level runs. Only another thread or process
        would fix that, and it would buy responsiveness, not earlier curves.
        """
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()


if __name__ == "__main__":
    # Redraw a finished sweep from its record: a change to a figure costs a redraw, not a re-run, and
    # the figures land beside the record they came from rather than in the runner's output directory.
    if len(sys.argv) < 2:
        sys.exit("usage: python -m VnV.verification.plots <result>.json ...")
    for argument in sys.argv[1:]:
        record = pathlib.Path(argument)
        for written in render(json.loads(record.read_text()), record.parent, True, True):
            print(f"wrote {written}")
