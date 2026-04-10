"""
1D Bar Simulation - Comparison File
Solution manufacturée : u(x) = sin(pi*x/(2*L))
"""
import json
import math
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sofa_bar_traction import sofaRun
from pyfreefem import FreeFemRunner

RESULTS_DIR = "results"


def _l2(a, b):
    return np.linalg.norm(a - b) / np.sqrt(len(a))


def _write_freefem_results(path, x, u):
    x_final = x + u
    with open(path, 'w') as f:
        f.write(f"{'x_initial':>12}  {'x_final':>12}  {'u_x':>12}\n")
        f.write("-" * 42 + "\n")
        for xi, xf, ui in zip(x, x_final, u):
            f.write(f"{xi:12.6f}  {xf:12.6f}  {ui:12.6f}\n")


if __name__ == "__main__":

    config_file = sys.argv[1] if len(sys.argv) > 1 else "params.json"
    with open(config_file) as f:
        cfg = json.load(f)

    length        = float(cfg["length"])
    young_modulus = float(cfg["youngModulus"])
    poisson_ratio = float(cfg["poissonRatio"])
    nx            = int(cfg["nx"])
    mesh_file     = os.path.join(RESULTS_DIR, cfg.get("meshfile", "bar1d.msh"))
    pi            = math.pi

    # --- Run FreeFEM ---
    print("\n" + "="*60)

    runner = FreeFemRunner("freefem_bar_traction.edp")
    exports = runner.execute({
        'youngModulus': young_modulus,
        'length':       length,
        'meshfile':     mesh_file,
        'nx':           nx
    })
    x_ff = exports['xcoords']
    u_ff = exports['u[]']

    idx_ff = np.argsort(x_ff)
    x_ff = x_ff[idx_ff]
    u_ff = u_ff[idx_ff]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    _write_freefem_results(os.path.join(RESULTS_DIR, "freefem_results.txt"), x_ff, u_ff)
    print(f"FreeFEM: u_max = {np.max(u_ff):.8f}")

    # --- Run SOFA ---
    x_sofa, u_sofa = sofaRun(nx=nx,
                             length=length,
                             young_modulus=young_modulus,
                             poisson_ratio=poisson_ratio)

    idx_sofa = np.argsort(x_sofa)
    x_sofa = x_sofa[idx_sofa]
    u_sofa = u_sofa[idx_sofa]
    print(f"SOFA: u_max = {np.max(u_sofa):.8f}")

    # --- Solution exacte ---
    u_exact = np.sin(pi * x_ff / (2.0 * length))
    print(f"Exact: u_max = {np.max(u_exact):.8f}")

    # --- Interpolation pour comparaison ---
    u_sofa_interp = np.interp(x_ff, x_sofa, u_sofa)

    # --- Écriture comparaison ---
    with open(os.path.join(RESULTS_DIR, "comparison_results.txt"), 'w') as f:
        header = f"{'x':>10}  {'u_exact':>12}  {'u_ff':>12}  {'u_sofa':>12}  {'diff_ff':>12}  {'diff_sofa':>12}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for x, ue, uff, us in zip(x_ff, u_exact, u_ff, u_sofa_interp):
            diff_ff = abs(uff - ue)
            diff_sofa = abs(us - ue)
            f.write(f"{x:10.4f}  {ue:12.6f}  {uff:12.6f}  {us:12.6f}  {diff_ff:12.6e}  {diff_sofa:12.6e}\n")

        f.write("\n")
        f.write("L2 error norms  [u_exact = sin(pi*x/(2L))]\n")
        f.write("-" * 50 + "\n")
        
        l2_ff_exact = _l2(u_ff, u_exact)
        l2_sofa_exact = _l2(u_sofa_interp, u_exact)
        l2_sofa_ff = _l2(u_sofa_interp, u_ff)
        
        f.write(f"  ||u_ff   - u_exact||_2  = {l2_ff_exact:.6e}\n")
        f.write(f"  ||u_sofa - u_exact||_2  = {l2_sofa_exact:.6e}\n")
        f.write(f"  ||u_sofa - u_ff||_2     = {l2_sofa_ff:.6e}\n")
        
        if np.max(u_ff) > 1e-10:
            scale = np.max(u_sofa) / np.max(u_ff)
            f.write(f"\n  Facteur SOFA/FreeFEM = {scale:.8f}\n")
            print(f"\n  Facteur SOFA/FreeFEM = {scale:.8f}")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(x_ff, u_exact, 'k--', linewidth=2, label="Exacte")
    ax1.plot(x_ff, u_ff, 'o-', markersize=4, label="FreeFEM", alpha=0.7)
    ax1.plot(x_sofa, u_sofa, 'x-', markersize=5, label="SOFA", alpha=0.7)
    ax1.set_xlabel("x")
    ax1.set_ylabel("u(x)")
    ax1.set_title("Comparaison des solutions")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.semilogy(x_ff, np.abs(u_ff - u_exact), 'o-', markersize=4, label="FreeFEM")
    ax2.semilogy(x_ff, np.abs(u_sofa_interp - u_exact), 'x-', markersize=5, label="SOFA")
    ax2.set_xlabel("x")
    ax2.set_ylabel("Erreur absolue")
    ax2.set_title("Erreurs")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f"Solution manufacturée u(x) = sin(πx/(2L))")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "comparison_plot.png"), dpi=150)
    plt.close()
    