"""A convergence sweep as one object: the levels, the rates read off them, and what settled.

No SOFA here -- the runner measures, this holds the measurements and draws conclusions from them.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Rate:
    """An observed order, or the reason there is none.

    Reasons are carried rather than blanked so a discarded pair says why it was discarded:
      floor  one of the two errors is down among its contaminants, so the ratio is noise
      osc    the error stopped decreasing monotonically (convergence ratio R <= 0)
      div    the error is growing, or growing in increments (R >= 1)
    """

    value: float | None
    reason: str = ""


@dataclass
class LevelResult:
    """One level of a study: what was measured on it, and what the solve cost.

    `values` and `floors` are keyed by metric name. `rates` is filled by the study when the level is
    added, since a rate needs the levels beneath this one.
    """

    label: str
    h: float | None
    elements: int
    values: dict
    floors: dict
    cpp_ratio: float
    newton: dict | None
    pcg: dict | None
    rates: dict = field(default_factory=dict)


@dataclass
class Verdict:
    """The order in a metric's settled range, and which levels that range covers."""

    order: float
    spread: float
    # The levels whose order was retained. The table colours exactly these rates, so the asymptotic
    # range is shown by the run of green rather than named on a line of its own.
    levels: list


class ConvergenceStudy:
    """A refinement sweep: levels in, an order of accuracy per metric out.

    `metrics` is the metric objects the runner measured with -- only their `name` and `reported` are
    read here, so this module stays independent of how a metric is computed. Everything else comes off
    the deck, and is copied out of it here rather than reached for later: the constructor is the one
    place that touches the deck's shape, so the rest of the class -- and the record it writes -- reads
    plain attributes and would survive a study built from something other than a deck.

    `expected` is the order per metric name the deck states, and `expect_tolerance` how far the settled
    order may sit from it before the metric counts as failed.
    """

    def __init__(self, deck, metrics):
        self.name = deck.name
        self.element = deck.element
        self.equation = deck.equation
        self.metrics = list(metrics)
        self.reported = [metric for metric in self.metrics if metric.reported]
        self.tolerance = deck.asymptotic_tolerance
        self.expected = deck.expect
        self.expect_tolerance = deck.expect_tolerance
        self.levels = []

    def add(self, level):
        """Append a level, then read the rates its arrival makes available."""
        self.levels.append(level)
        level.rates = {metric.name: self._rate(metric.name) for metric in self.metrics}

    def _rate(self, metric):
        """Order from the two finest levels so far, or None plus the reason there is none."""
        if len(self.levels) < 2:
            return Rate(None)

        current, previous = self.levels[-1], self.levels[-2]
        error, before = current.values[metric], previous.values[metric]
        if error <= current.floors[metric] or before <= previous.floors[metric]:
            return Rate(None, "floor")

        if len(self.levels) >= 3:
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
            previous_increment = before - self.levels[-3].values[metric]
            ratio = increment / previous_increment if previous_increment else np.inf
            if ratio <= 0.0:
                return Rate(None, "osc")
            if ratio >= 1.0:
                return Rate(None, "div")
        elif error >= before:
            return Rate(None, "div")

        return Rate(np.log(error / before) / np.log(current.h / previous.h))

    def settled(self, metric):
        """Trailing levels over which the observed order has settled, per Roache's requirement that
        it stop changing under refinement.

        The settling test compares consecutive observed orders against each other, never against the
        formal order: selecting the range by closeness to the answer being verified would make the
        order-of-accuracy test circular. Returns (index, order) pairs, finest last.
        """
        orders = [(index, level.rates[metric].value) for index, level in enumerate(self.levels)
                  if level.rates[metric].value is not None]

        # Contiguity matters: a readable order with a discarded pair beneath it does not extend a run.
        retained = []
        for index, order in reversed(orders):
            if retained and (index + 1 != retained[0][0] or abs(order - retained[0][1]) > self.tolerance):
                break
            retained.insert(0, (index, order))

        # One order cannot demonstrate that anything settled -- it takes two to compare.
        return retained if len(retained) >= 2 else []

    def verdict(self, metric):
        """The order in this metric's settled range, or None where nothing settled."""
        retained = self.settled(metric)
        if not retained:
            return None
        orders = [order for _, order in retained]
        return Verdict(order=orders[-1], spread=max(orders) - min(orders),
                       levels=[index for index, _ in retained])

    def accepted(self, metric):
        """The level indices whose rate the settling test kept, as a set for the tables to test."""
        return {index for index, _ in self.settled(metric)}

    def passed(self, metric):
        """Whether the settled order is the one the deck expects, or None if this metric is not judged.

        This is where the run stops measuring and starts judging, and the two are kept apart on
        purpose: `settled` above chooses its range by comparing consecutive orders to each other and
        never looks at `expected`, so the range is picked independently of the answer being checked.
        Only afterwards is the order it produced held against the deck's number. Selecting the range by
        closeness to `expected` would be circular; judging an independently chosen range is not.

        Only the reported metrics are judged. dU and aeuh are the energy cross-check, and aeuh reaches
        no order at all where the load is exactly representable -- the quadratic fields, where
        orthogonality holds so the defect sits at round-off and grows under refinement. Holding it to
        an expected order would fail those decks for behaving exactly as they should.
        """
        if metric not in {reported.name for reported in self.reported}:
            return None
        verdict = self.verdict(metric)
        # bool(): the order comes out of numpy, and a numpy bool is not JSON-serializable -- it would
        # reach the record as 1.0 through `default=float` rather than as true.
        return bool(verdict is not None
                    and abs(verdict.order - self.expected[metric]) <= self.expect_tolerance)

    def ok(self):
        """Every judged metric settled where the deck said it would, on a sweep whose solves converged.

        The solve is part of the verdict rather than a footnote to it: an order that looks settled over
        levels whose Newton solve never converged is not evidence of anything.
        """
        return not self.unconverged() and all(self.passed(metric.name) for metric in self.reported)

    def unconverged(self):
        """Labels of the levels whose solve did not converge.

        An order that looks settled while the solve never converged is not evidence of anything, so
        this rides alongside the orders rather than being left to the per-level rows.
        """
        return [level.label for level in self.levels
                if (level.newton["status"] if level.newton else None) not in ("ok", None)]

    def summary(self):
        """Per metric: the settled range and the order in it, or None where nothing settled."""
        metrics = {}
        for metric in self.metrics:
            verdict = self.verdict(metric.name)
            metrics[metric.name] = None if verdict is None else {
                "order": verdict.order, "spread": verdict.spread, "levels": verdict.levels}
        return {"metrics": metrics, "unconverged": self.unconverged()}

    def to_json(self):
        """The sweep as a JSON-able record: what was measured, and what the run concluded from it.

        Everything a figure needs is in here, verdicts included, so redrawing is a post-processing step
        rather than another sweep and a redrawn figure cannot reach a different conclusion than the table
        did. It is also the run's own record of a verification exercise, which is the thing a report
        quotes -- hence the diagnostics too, whether or not they were printed.
        """
        accepted = {metric.name: self.accepted(metric.name) for metric in self.metrics}
        levels = []
        for index, level in enumerate(self.levels):
            metrics = {}
            for metric in self.metrics:
                rate = level.rates[metric.name]
                metrics[metric.name] = {"error": level.values[metric.name],
                                        "floor": level.floors[metric.name],
                                        "rate": rate.value,
                                        # Empty unless the pair was discarded for a stated reason.
                                        "reason": rate.reason,
                                        "accepted": index in accepted[metric.name]}
            levels.append({"label": level.label, "h": level.h, "elements": level.elements,
                           "cpp": level.cpp_ratio, "metrics": metrics,
                           "newton": level.newton, "pcg": level.pcg})

        return {"deck": self.name,
                # The element the deck ran, which the convergence figure turns into a marker shape.
                "element": self.element,
                # The manufactured field in math form, which titles the figures and can be quoted in a
                # write-up: it comes off the solution itself, so it names the problem that was solved.
                "equation": self.equation,
                # Which metrics the convergence figure draws, decided here rather than in the drawing
                # code: dU and aeuh are the energy cross-check, recorded but not a convergence claim.
                "reported": [metric.name for metric in self.reported],
                "expected": {metric.name: self.expected[metric.name] for metric in self.metrics},
                "expectTolerance": self.expect_tolerance,
                # The judgement, stored rather than left to be recomputed: null where a metric is not
                # judged. `ok` is the deck's verdict and what the runner's exit code is taken from.
                "passed": {metric.name: self.passed(metric.name) for metric in self.metrics},
                "ok": self.ok(),
                "summary": self.summary(),
                "levels": levels}
