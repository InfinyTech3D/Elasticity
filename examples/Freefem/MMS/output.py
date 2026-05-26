"""Shared output helpers across MMS 1D/2D/3D drivers (tables + plots)."""

import os
import numpy as np
import matplotlib.pyplot as plt

from plot_utils import annotate_convergence_rates


def write_convergence_table(stem, rows, results_dir):
    """
    Write convergence table to <results_dir>/<stem>.txt.

    rows : list of dicts with keys 'nx', 'h', and one key per error column.
           Rate columns are strings (empty for the first row).
    """
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{stem}.txt")
    err_keys = [k for k in rows[0] if k not in ("nx", "h")]
    header   = f"{'nx':>6} | {'h':>10}" + "".join(f" | {k:>16}" for k in err_keys)
    with open(path, "w") as f:
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for row in rows:
            line = f"{row['nx']:6d} | {row['h']:10.4f}"
            for k in err_keys:
                v = row[k]
                line += f" | {v:16.6e}" if isinstance(v, float) else f" | {v:>16}"
            f.write(line + "\n")


def plot_convergence(stem, hs, series, title, results_dir, ylabel="Error"):
    """
    Save log-log convergence plot to <results_dir>/<stem>.png.

    series : list of {"label", "errors", "style"?} dicts
    Per-segment convergence rates are annotated above each line segment.
    """
    os.makedirs(results_dir, exist_ok=True)
    h_arr   = np.array(hs)
    default = ["bo-", "rs--", "g^:", "m^-"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, s in enumerate(series):
        style = s.get("style", default[i % len(default)])
        e_arr = np.array(s["errors"])
        ax.loglog(h_arr, e_arr, style, label=s["label"], linewidth=2, markersize=7)
        annotate_convergence_rates(ax, h_arr, e_arr)
    ax.set_xlabel("h")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, f"{stem}.png"), dpi=150)
    plt.close(fig)
