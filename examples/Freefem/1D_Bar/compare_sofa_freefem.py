"""
1D Bar Simulation - Traction Load - Comparison File
"""
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sofa_bar_traction import sofaRun
from pyfreefem import FreeFemRunner


def _l2(a, b):
    return np.linalg.norm(a - b) / a.size


def _write_freefem_results(path, x, u):
    x_final = x + u
    with open(path, 'w') as f:
        f.write(f"{'x_initial':>12}  {'x_final':>12}  {'u_x':>12}\n")
        f.write("-" * 42 + "\n")
        for xi, xf, ui in zip(x, x_final, u):
            f.write(f"{xi:12.6f}  {xf:12.6f}  {ui:12.6f}\n")


if __name__ == "__main__":

    # --- Read parameters from JSON file ---
    config_file = sys.argv[1] if len(sys.argv) > 1 else "params.json"
    with open(config_file) as f:
        cfg = json.load(f)

    length        = float(cfg["length"])
    nx            = int(cfg["nx"])
    force         = float(cfg["force"])
    young_modulus = float(cfg["youngModulus"])
    poisson_ratio = float(cfg["poissonRatio"])

    # --- Run FreeFEM ---
    runner = FreeFemRunner("freefem_bar_traction.edp")
    exports = runner.execute({
        'youngModulus': young_modulus,
        'force': force,
        'nx': nx,
        'length': length,
    })
    x_ff = exports['xcoords']
    u_ff = exports['u[]']

    os.makedirs("results", exist_ok=True)
    _write_freefem_results(os.path.join("results", "freefem_results.txt"), x_ff, u_ff)

    # --- Run SOFA ---
    x_sofa, u_sofa = sofaRun(length=length
            , force=force
            , young_modulus=young_modulus
            , poisson_ratio=poisson_ratio
            , nx=nx)

    # --- Analytical solution: u(x) = F * x / E  (unit cross-section) ---
    u_exact = (force / young_modulus) * x_ff

    # --- Compare Results ---
    with open("results/comparison_results.txt", 'w') as f:
        header = f"{'x':>10}  {'u_exact':>12}  {'u_ff':>12}  {'u_sofa':>12}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for x, ue, uff, us in zip(x_ff, u_exact, u_ff, u_sofa):
            f.write(f"{x:10.4f}  {ue:12.6f}  {uff:12.6f}  {us:12.6f}\n")

        f.write("\n")
        f.write("L2 error norms\n")
        f.write("-" * 40 + "\n")
        f.write(f"  ||u_sofa - u_ff||_2     = {_l2(u_sofa, u_ff):.6e}\n")
        f.write(f"  ||u_sofa - u_exact||_2  = {_l2(u_sofa, u_exact):.6e}\n")
        f.write(f"  ||u_ff   - u_exact||_2  = {_l2(u_ff,   u_exact):.6e}\n")

    # --- Plot ---
    fig, ax = plt.subplots()
    ax.plot(x_ff,   u_exact, label="Analytical", linestyle="--", color="black")
    ax.plot(x_ff,   u_ff,    label="FreeFEM",    marker="o", markersize=4, linestyle="none")
    ax.plot(x_sofa, u_sofa,  label="SOFA",       marker="x", markersize=5, linestyle="none")
    ax.set_xlabel("x")
    ax.set_ylabel("Displacement u(x)")
    ax.set_title("1D Bar — Displacement Comparison")
    ax.legend()
    fig.savefig("results/comparison_plot.png", dpi=150)
    plt.close(fig)

