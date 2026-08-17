"""The run's terminal output: the progress bar and the three tables.

No numerics and no SOFA -- `sys` is the only import, and it has to stay that way. Everything here reads
a study it is handed, or a list of them, and prints. Keeping it separate is what lets a second kind of
test (a single-mesh comparison rather than a sweep) reuse the colours and the column widths instead of
copying them, and what lets a column be edited without a SOFA build to import the module.
"""

import sys

GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"

PROGRESS_WIDTH = 20

# The widths `print_table` and `print_summary` have to agree on, and the only ones named: the summary
# pads blank cells across the level columns so its rows sit under the metric they belong to, so a width
# written out twice is a table that silently stops lining up. Widths used inside one function only --
# the diagnostics table's trailing columns, the overview's -- stay literals where they are read.
LABEL_WIDTH, H_WIDTH, ELEMENTS_WIDTH = 12, 10, 9

# A metric's slot: one rate column by default, a value column plus a rate column with diagnostics on.
# The rates-only column is wide enough for the `not reached` the summary can print underneath it.
RATE_COLUMN = 11
VALUE_WIDTH, RATE_WIDTH = 12, 6


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
    header = f"{'cells':>{LABEL_WIDTH}} {'h':>{H_WIDTH}}"
    # What the deck asked for is cells; what the mapping made of them is an element count worth
    # seeing, since it is 6x or 2x the cells for the split element types.
    if show_diagnostics:
        header += f" {'elements':>{ELEMENTS_WIDTH}}"
    for metric in metrics:
        header += (f" {metric.name:>{VALUE_WIDTH}} {'rate':>{RATE_WIDTH}}" if show_diagnostics
                   else f" {metric.name:>{RATE_COLUMN}}")
    if show_diagnostics:
        # cpp/py: SOFA's own potential energy against the Python quadrature; should read 1.
        # nIt / r/r0: the Newton solve, so algebraic error polluting the fine end is visible.
        header += f" {'cpp/py':>8} {'nIt':>4} {'r/r0':>8}"
    print(header + " status")

    for index, level in enumerate(study.levels):
        row = f"{level.label:>{LABEL_WIDTH}} {level.h:>{H_WIDTH}.5f}"
        if show_diagnostics:
            row += f" {level.elements:>{ELEMENTS_WIDTH}}"
        for metric in metrics:
            rate = level.rates[metric.name]
            # A discarded pair reads red whether it was discarded for a stated reason or simply fell
            # outside the settled run.
            cell = f"{rate.value:.2f}" if rate.value is not None else rate.reason
            accepted_here = index in accepted[metric.name]
            if show_diagnostics:
                row += (f" {level.values[metric.name]:>{VALUE_WIDTH}.3e}"
                        f" {paint(f'{cell:>{RATE_WIDTH}}', accepted_here)}")
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
    judgements = [study.passed(metric.name) for metric in metrics]
    rows = [
        # The order row is the run's verdict, so it carries the table's colours: green where the
        # settled order is the one the deck expects, red where it is not or where none was reached.
        # The deck states the tolerance that comparison needs, which is what makes this a verdict
        # rather than the reader's own judgement -- and the settling test still never looks at
        # `expected`, so the range was chosen independently of the answer it is now held against.
        # dU and aeuh are not judged, so they keep the older rule: green wherever an order was reached.
        ("order p", [f"{s.order:.2f}" if s else "not reached" for s in settled],
         [reached is not None if judgement is None else judgement
          for judgement, reached in zip(judgements, settled)]),
        ("spread", [f"{s.spread:.2f}" if s else "--" for s in settled], None),
        ("expected", [f"{study.expected[metric.name]:.1f}" for metric in metrics], None),
    ]

    print()
    for label, cells, coloured in rows:
        # The label sits in the cells column; the rest is blank padding across the level columns,
        # plus the element count the diagnostics table inserts there.
        row = (f"{label:>{LABEL_WIDTH}} {'':>{H_WIDTH}}"
               + (f" {'':>{ELEMENTS_WIDTH}}" if show_diagnostics else ""))
        for index, cell in enumerate(cells):
            padded = (f"{cell:>{VALUE_WIDTH}} {'':>{RATE_WIDTH}}" if show_diagnostics
                      else f"{cell:>{RATE_COLUMN}}")
            row += f" {padded if coloured is None else paint(padded, coloured[index])}"
        print(row)

    unconverged = study.unconverged()
    if unconverged:
        print(f"\n  Newton did not converge at: {', '.join(unconverged)}"
              f" -- the orders above are contaminated by algebraic error at those levels.")


def print_overview(results, metrics):
    """One line per deck: the settled order per metric, `--` where none was reached.

    The per-deck tables above distinguish *why* a metric has no order -- floor, oscillation,
    divergence, never settled -- which this collapses to `--` for the sake of one line per deck.
    A deck that raised reads `error` instead, so a crash is never mistaken for a metric that simply
    did not settle. The orders carry the same colours they do per deck.

    What an order is held against is the deck's own `expect`, so there is no expectation row across
    decks: each deck's summary above prints the numbers that deck was judged by.

    `metrics` is passed in rather than read from a module here: a deck that raised has no study to ask
    for its metric list, and importing the list would pull SOFA into this module through `metrics.py`.
    """
    print()
    header = f"{'deck':<44}"
    for metric in metrics:
        header += f" {metric.name:>9}"
    print(header + f" {'solver':>9}")

    for name, study in results:
        row = f"{name:<44}"
        for metric in metrics:
            settled = None if study is None else study.verdict(metric.name)
            judgement = None if study is None else study.passed(metric.name)
            cell = "error" if study is None else f"{settled.order:.2f}" if settled else "--"
            # A judged metric takes the verdict's colour; dU and aeuh only say whether an order was
            # reached at all, and a deck that raised reads red on every column.
            row += f" {paint(f'{cell:>9}', settled is not None if judgement is None else judgement)}"
        # A settled order means nothing at a level where the solve did not converge, so the count of
        # such levels rides along on the same line rather than living only in the per-deck table.
        if study is None:
            solver_cell = "error"
        else:
            unconverged = len(study.unconverged())
            solver_cell = "ok" if not unconverged else f"{unconverged} bad"
        print(row + f" {paint(f'{solver_cell:>9}', solver_cell == 'ok')}")
