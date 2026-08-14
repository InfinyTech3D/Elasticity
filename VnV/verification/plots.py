"""Figures from a verification sweep. No SOFA here -- the runner reads the Data, this draws it."""

import matplotlib
matplotlib.use("Agg")       # a batch run writes files; `interactive()` switches this for a live window
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import to_hex
from matplotlib.ticker import MaxNLocator

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


def _frame(figure, axis, deck, tolerance):
    """Everything that does not depend on the data, so a live figure can stand up before it has any."""
    figure.patch.set_facecolor(SURFACE)
    _style(axis)
    if tolerance:
        axis.axhline(tolerance, color=INK_SOFT, linewidth=1.2, linestyle=(0, (4, 3)))
    axis.set_yscale("log")
    axis.set_title(f"{deck} -- PCG cost per level", fontsize=11, color=INK)
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


def pcg_residuals(deck, levels, tolerance, path):
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
    _frame(figure, axis, deck, tolerance)
    for index, (label, curves) in enumerate(levels):
        _draw_level(axis, label, curves, tolerance, index, len(levels))
    _legend(figure, axis)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


class LivePlot:
    """The same figure, drawn while the sweep runs: one level joins it as soon as its solve returns.

    A level's residual history exists the moment `animate` returns, well before its error norms are
    integrated, so the window fills in during the part of the level that actually takes the time. The
    granularity is one curve per level and cannot be finer -- a solve is a single call into SOFA, and
    the `graph` Data only becomes readable once it comes back.

    `levels` is the sweep length, known up front, so the ramp step of an early level is the step it
    will still have when the sweep ends and the figure stops changing colour under the reader.
    """

    def __init__(self, deck, levels, tolerance):
        self.levels = levels
        self.tolerance = tolerance
        self.drawn = 0
        self.figure, self.axis = plt.subplots(figsize=(7.6, 4.6))
        _frame(self.figure, self.axis, deck, tolerance)
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
