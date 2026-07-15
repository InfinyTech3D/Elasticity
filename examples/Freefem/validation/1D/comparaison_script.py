"""
1D Bar Simulation - Distributed Load - Comparison File

Physical case: bar fixed at x=0, free at x=L, uniform distributed load q.
Analytical solution (E*u'' + q = 0, u(0)=0, u'(L)=0):

    u(x) = (q/E) * (L*x - x**2/2)
"""
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sofa_bar_distributed import sofaRun
from pyfreefem import FreeFemRunner


def _rms(a, b):
    return np.linalg.norm(a - b) / np.sqrt(a.size)


def _write_freefem_results(path, x, u):
    x_final = x + u
    with open(path, 'w') as f:
        f.write(f"{'x_initial':>12}  {'x_final':>12}  {'u_x':>12}\n")
        f.write("-" * 42 + "\n")
        for xi, xf, ui in zip(x, x_final, u):
            f.write(f"{xi:12.6f}  {xf:12.6f}  {ui:12.6f}\n")


def _default_params_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "params_distributed.json")


def plot_bar_geometry(x0, u_sofa, scale=None):
    
    x_deformed = x0 + u_sofa
    if scale is None:
        span = x0[-1] - x0[0]
        max_disp = np.max(np.abs(u_sofa)) or 1e-12
        scale = 0.15 * span / max_disp  

    fig, ax = plt.subplots(figsize=(9, 2.5))
    y_rest, y_def = 1.0, 0.0

    ax.plot(x0, [y_rest] * len(x0), 'o-', color="gray", label="Rest configuration")
    ax.plot(x0 + scale * u_sofa, [y_def] * len(x0), 'o-', color="crimson",
            label=f"Deformed (SOFA, ×{scale:.1f} for visibility)")
    ax.plot([0, 0], [y_def - 0.3, y_rest + 0.3], 'k-', linewidth=3)  # fixed end marker

    ax.set_yticks([y_def, y_rest])
    ax.set_yticklabels(["deformed", "rest"])
    ax.set_xlabel("x")
    ax.set_title("1D Bar — Distributed Load — Rest vs Deformed (schematic)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(y_def - 0.6, y_rest + 0.6)
    fig.tight_layout()
    return fig


if __name__ == "__main__":

    config_file = sys.argv[1] if len(sys.argv) > 1 else _default_params_path()
    with open(config_file) as f:
        cfg = json.load(f)

    length        = float(cfg["length"])
    nx            = int(cfg["nx"])
    q             = float(cfg["q"])
    young_modulus = float(cfg["youngModulus"])
    poisson_ratio = float(cfg["poissonRatio"])

    # === Run FF ===
    runner = FreeFemRunner("freefem_bar_distributed.edp")
    exports = runner.execute({
        'youngModulus': young_modulus,
        'q': q,
        'nx': nx,
        'length': length,
    })
    x_ff = exports['xcoords']
    u_ff = exports['u[]']

    os.makedirs("results", exist_ok=True)
    _write_freefem_results(os.path.join("results", "freefem_distributed_results.txt"), x_ff, u_ff)

    # === Run SOFA ===
    x_sofa, u_sofa = sofaRun(length=length
            , q=q
            , young_modulus=young_modulus
            , poisson_ratio=poisson_ratio
            , nx=nx)

    # ========== analytical sol ===============
    u_exact = (q / young_modulus) * (length * x_ff - x_ff**2 / 2.0)

    
    rms_sofa_vs_ff    = _rms(u_sofa, u_ff)
    rms_sofa_vs_exact = _rms(u_sofa, u_exact)


    # --- Compare Results ---
    with open("results/comparison_distributed_results.txt", 'w') as f:
        header = f"{'x':>10}  {'u_exact':>12}  {'u_ff':>12}  {'u_sofa':>12}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for x, ue, uff, us in zip(x_ff, u_exact, u_ff, u_sofa):
            f.write(f"{x:10.4f}  {ue:12.6f}  {uff:12.6f}  {us:12.6f}\n")

        f.write("\n")
        f.write("RMS error norms (||a - b||_2 / sqrt(n))\n")
        f.write("-" * 40 + "\n")
        f.write(f"  RMS(sofa, ff)     = {rms_sofa_vs_ff:.6e}\n")
        f.write(f"  RMS(sofa, exact)  = {rms_sofa_vs_exact:.6e}\n")
        

    print("RMS error norms (||a - b||_2 / sqrt(n))")
    print("-" * 40)
    print(f"  RMS(sofa, ff)     = {rms_sofa_vs_ff:.6e}")
    print(f"  RMS(sofa, exact)  = {rms_sofa_vs_exact:.6e}")
    

    # ==== plot 
    fig, ax = plt.subplots()
    ax.plot(x_ff,   u_exact, label="Analytical", linestyle="--", color="black")
    ax.plot(x_ff,   u_ff,    label="FreeFEM",    marker="o", markersize=4, linestyle="none")
    ax.plot(x_sofa, u_sofa,  label="SOFA",       marker="x", markersize=5, linestyle="none")
    ax.set_xlabel("x")
    ax.set_ylabel("Displacement u(x)")
    ax.set_title("1D Bar — Distributed Load — Displacement Comparison")
    ax.legend()
    fig.savefig("results/comparison_distributed_plot.png", dpi=150)
    plt.close(fig)

    
    fig_geo = plot_bar_geometry(x_sofa, u_sofa)
    fig_geo.savefig("results/bar_geometry_distributed.png", dpi=150)
    plt.close(fig_geo)
 