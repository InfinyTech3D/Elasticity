"""Run the mesh-refinement convergence study for every 3D MMS case.

Loops cases × nu × nx and writes per-(case, nu) text tables and convergence
plots into the shared `results/` directory. Mirrors the 2D driver minus the
plane-stress / plane-strain `dim` axis (3D has a single constitutive branch).
"""

import numpy as np

from sinus_neumann import mms as sinus_neumann_mms

from solid import (
    RESULTS_DIR,
    load_params,
    solve_solid,
    element_hex,
)
from output import write_convergence_table, plot_convergence


def convergence_study(elem_specs, mms, L, E, nu, nx_values):
    """
    Run convergence study for each element type in elem_specs, write a
    per-(element) text table, and one shared plot with L²/H¹ for every
    element on the same axes.

    elem_specs : list of dicts with keys 'elem', 'label', 'l2_style', 'h1_style'
    """
    print(f"\n  PoissonRatio = {nu}", flush=True)
    hdr = (f"{'nx':>5} | {'h':>10} | {'L2':>14} | {'rate_L2':>7} "
           f"| {'H1':>14} | {'rate_H1':>7}")

    plot_series, hs_ref = [], None
    for spec in elem_specs:
        elem, label = spec["elem"], spec["label"]
        tag         = label.replace(" ", "_")
        stem        = f"convergence_{mms.name}_{tag}_nu{nu}"

        print(f"\n── {label}  {mms.name}  nu={nu} ──\n{hdr}", flush=True)

        rows, hs, l2s, h1s = [], [], [], []
        for k, nx in enumerate(nx_values):
            ny = nz = nx
            h  = L / (nx - 1)
            sol = solve_solid(elem, mms, L, E, nu, nx, ny, nz)
            l2  = elem.compute_l2(sol, mms, L)
            h1  = elem.compute_h1(sol, mms, L)

            rate_l2 = (f"{np.log(l2 / l2s[-1]) / np.log(h / hs[-1]):.2f}"
                       if k > 0 else "")
            rate_h1 = (f"{np.log(h1 / h1s[-1]) / np.log(h / hs[-1]):.2f}"
                       if k > 0 else "")
            print(f"{nx:5d} | {h:10.4f} | {l2:14.6e} | {rate_l2:>7} "
                  f"| {h1:14.6e} | {rate_h1:>7}", flush=True)
            rows.append({"nx": nx, "h": h,
                         "L2": l2, "rate_L2": rate_l2,
                         "H1": h1, "rate_H1": rate_h1})
            hs.append(h); l2s.append(l2); h1s.append(h1)

        write_convergence_table(stem, rows, RESULTS_DIR)
        plot_series.append({"label": f"{label} L²",
                            "errors": l2s, "style": spec["l2_style"]})
        plot_series.append({"label": f"{label} H¹",
                            "errors": h1s, "style": spec["h1_style"]})
        hs_ref = hs

    title = f"Convergence — {mms.name}  nu={nu}"
    plot_convergence(f"convergence_{mms.name}_nu{nu}",
                     hs_ref, plot_series, title=title, results_dir=RESULTS_DIR)


if __name__ == "__main__":
    cfg  = load_params()
    L    = cfg["length"]
    E    = cfg["youngModulus"]
    conv = cfg["convergence"]

    specs = [
        {"elem": element_hex, "label": "Q1 hex",
         "l2_style": "bo-", "h1_style": "rs--"},
    ]

    for mms in (sinus_neumann_mms,):
        nx_vals = conv["nx_values"][mms.name]
        print(f"\n══ {mms.name} ══")
        for nu in conv["nu_values"]:
            convergence_study(specs, mms, L, E, nu, nx_vals)
