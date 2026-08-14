"""Figures from a verification sweep. No SOFA here -- the runner reads the Data, this draws it."""

import matplotlib
matplotlib.use("Agg")       # a batch runner writes files and never opens a window
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
    figure.patch.set_facecolor(SURFACE)
    _style(axis)

    for index, (label, curves) in enumerate(levels):
        residuals, seams = [], []
        for iteration in sorted(curves):
            if residuals:                   # not the first solve, so this point is a handover
                seams.append(len(residuals))
            residuals.extend(curves[iteration])

        # Each solve is tested against the floor in its own right, and the last value of each is where
        # it stopped, so one solve short of the floor condemns the level. Without a reported tolerance
        # there is nothing to test against and the level stays neutral.
        converged = all(curves[key][-1] <= tolerance for key in curves) if tolerance else None
        colour = ramp_colour(converged, index, len(levels))

        axis.plot(range(len(residuals)), residuals, color=colour, linewidth=1.8,
                  marker="o", markersize=3.5, label=label)
        axis.plot(seams, [residuals[seam] for seam in seams], linestyle="none", marker="o",
                  markersize=7, markerfacecolor=SURFACE, markeredgecolor=colour, markeredgewidth=1.6)
        if converged is False:
            axis.plot(len(residuals) - 1, residuals[-1], linestyle="none", marker="x",
                      markersize=8, markeredgewidth=2.0, color=colour)

    if tolerance:
        axis.axhline(tolerance, color=INK_SOFT, linewidth=1.2, linestyle=(0, (4, 3)))

    axis.set_yscale("log")
    axis.set_title(f"{deck} -- PCG cost per level", fontsize=11, color=INK)
    axis.set_xlabel("cumulative PCG iterations", fontsize=9, color=INK_SOFT)
    axis.set_ylabel("residual  (r M^-1 r) / (b b)", fontsize=9, color=INK_SOFT)

    handles, labels = axis.get_legend_handles_labels()
    # Wrapped at four, so a long sweep's labels stay inside the figure instead of running off it;
    # the reserved strip grows with the row count the wrap produces.
    columns = 4
    rows = -(-len(labels) // columns)
    figure.legend(handles, labels, title="cells per axis", loc="lower center", frameon=False,
                  ncol=columns, fontsize=9, title_fontsize=9)
    figure.tight_layout(rect=(0, 0.04 + 0.055 * rows, 1, 1))
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)
