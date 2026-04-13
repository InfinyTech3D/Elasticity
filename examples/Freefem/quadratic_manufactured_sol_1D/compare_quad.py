"""
MMS solution QUADRATIQUE : u(x) = x*(L-x)/L^2
"""

import importlib
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyfreefem import FreeFemRunner



def _import_sofa_run():
    for module_name in ("sofa_bar_quad", "Sofa_bar_quad"):
        try:
            mod = importlib.import_module(module_name)
            return mod.sofaRun
        except ModuleNotFoundError:
            continue
    raise ImportError(
    )

sofaRun = _import_sofa_run()



def u_exact(x, L):
    return x * (L - x) / L**2


def du_exact(x, L):
    return (L - 2.0 * x) / L**2


def l2_error_discrete(u_num, u_ref):
    u_num = np.asarray(u_num).flatten()
    u_ref = np.asarray(u_ref).flatten()
    if len(u_num) != len(u_ref):
        raise ValueError(
            f"Dimensions incompatibles : u_num={len(u_num)}, u_ref={len(u_ref)}."
        )
    return np.linalg.norm(u_num - u_ref) / np.sqrt(len(u_ref))


def convergence_rate(h_arr, err_arr, floor=1e-13):
    h_arr   = np.asarray(h_arr)
    err_arr = np.asarray(err_arr)
    mask = err_arr > floor
    if mask.sum() < 2:
        return np.nan
    return np.polyfit(np.log(h_arr[mask]), np.log(err_arr[mask]), 1)[0]


if __name__ == "__main__":

    config_file = sys.argv[1] if len(sys.argv) > 1 else "params_mms.json"

    if os.path.exists(config_file):
        with open(config_file) as f:
            cfg = json.load(f)
    else:
        cfg = {}

    L       = float(cfg.get("length",       1.0))
    E       = float(cfg.get("youngModulus", 1000.0))
    nu      = float(cfg.get("poissonRatio", 0.3))
    nx_list = list(map(int, cfg.get("nx_list", [5, 10, 20, 40, 80, 160])))

    os.makedirs("results_quad", exist_ok=True)

    print(f"\n  Solution : u(x) = x*(L-x)/L^2 ")
    print(f"  Terme source : f = 2E/L^2 = {2*E/L**2:.1f}  cnst ")
    print(f"  Neumann : sigma(L) = -E/L = {-E/L:.1f}\n")

    
    h_list          = []
    err_nodal_list  = []
    err_l2_ff_list  = []
    err_h1_ff_list  = []
    err_sofa_list   = []

    x_fine      = None
    u_ff_fine   = None
    u_sofa_fine = None

    runner = FreeFemRunner("freefem_bar_quad.edp")


    # conv loop 
    for nx in nx_list:
        h = L / nx

        # FreeFem 
        exports = runner.execute({
            "youngModulus": E,
            "length":       L,
            "nx":           nx,
        })

        x_ff   = exports["xcoords"]
        u_ff   = exports["u[]"]
        errors = exports["errors"]

        err_nodal = float(errors[0])
        err_l2_ff = float(errors[1])
        err_h1_ff = float(errors[2])

        #  SOFA 
        x_sofa, u_sofa = sofaRun(
            length=L,
            young_modulus=E,
            poisson_ratio=nu,
            nx=nx,
        )

        u_sofa_interp = np.interp(x_ff, x_sofa, u_sofa)
        u_ex_ff       = u_exact(x_ff, L)
        err_sofa      = l2_error_discrete(u_sofa_interp, u_ex_ff)

        # Stockage  
        h_list.append(h)
        err_nodal_list.append(err_nodal)
        err_l2_ff_list.append(err_l2_ff)
        err_h1_ff_list.append(err_h1_ff)
        err_sofa_list.append(err_sofa)


        # Diagnostic 
        du_sofa_L = ((u_sofa[-1] - u_sofa[-2]) /
                     (x_sofa[-1] - x_sofa[-2]))
        print(f"         FF   u_max={np.max(u_ff):.6f} @ x={x_ff[np.argmax(u_ff)]:.3f}"
              f"  u(L)={u_ff[-1]:.2e}")
        print(f"         SOFA u_max={np.max(u_sofa):.6f} @ x={x_sofa[np.argmax(u_sofa)]:.3f}"
              f"  sigma(L)={E*du_sofa_L:.2f} (exact={-E/L:.2f})")
        print()

        if nx == nx_list[-1]:
            x_fine      = x_ff
            u_ff_fine   = u_ff
            u_sofa_fine = u_sofa_interp

    # Taux de convergence 
    h_arr         = np.array(h_list)
    err_nodal_arr = np.array(err_nodal_list)
    err_l2_ff_arr = np.array(err_l2_ff_list)
    err_h1_ff_arr = np.array(err_h1_ff_list)
    err_sofa_arr  = np.array(err_sofa_list)

    rate_nodal = convergence_rate(h_arr, err_nodal_arr)
    rate_l2_ff = convergence_rate(h_arr, err_l2_ff_arr)
    rate_h1_ff = convergence_rate(h_arr, err_h1_ff_arr)
    rate_sofa  = convergence_rate(h_arr, err_sofa_arr)




    #  Plot 1 : Displacment 
    u_ex_fine = u_exact(x_fine, L)
    u_max_ex  = np.max(u_ex_fine)  # = L^2/(4*L^2) = 1/4 = 0.25 pour L=1

    fig1, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Displacment compare 
    ax = axes[0]
    ax.plot(x_fine, u_ex_fine,
            label="Exact  u(x) = x(L-x)/L²",
            color="black", linestyle="--", linewidth=1.8)
    ax.plot(x_fine, u_ff_fine,
            label=f"FreeFEM  (nx={nx_list[-1]})",
            marker="o", markersize=4, linestyle="none", color="steelblue")
    ax.plot(x_fine, u_sofa_fine,
            label=f"SOFA     (nx={nx_list[-1]})",
            marker="x", markersize=5, linestyle="none", color="tomato")
    ax.set_xlabel("x")
    ax.set_ylabel("Displacment u(x)")
    ax.set_title("Solution quadratique\nu(x) = x(L-x)/L²")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Error abs 
    ax2 = axes[1]
    ax2.plot(x_fine, np.abs(u_ff_fine   - u_ex_fine),
             label="FreeFEM", color="steelblue", linewidth=1.5)
    ax2.plot(x_fine, np.abs(u_sofa_fine - u_ex_fine),
             label="SOFA",    color="tomato",    linewidth=1.5)
    ax2.set_xlabel("x")
    ax2.set_ylabel("|u_h - u_exact|")
    ax2.set_title(f"Erreur absolue (nx={nx_list[-1]})")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    fig1.tight_layout()
    fig1.savefig("results_quad/plot_displacement_quad.png", dpi=150)
    plt.close(fig1)
    

    # Convergence log-log 
    fig2, ax2 = plt.subplots(figsize=(7, 5))

    ax2.loglog(h_arr, err_nodal_arr,
               marker="s", linestyle="-",  color="gray",
               label=f"FreeFEM nodal  (rate={rate_nodal:.2f})")
    ax2.loglog(h_arr, err_l2_ff_arr,
               marker="o", linestyle="-",  color="steelblue",
               label=f"FreeFEM L2     (rate={rate_l2_ff:.2f})")
    ax2.loglog(h_arr, err_h1_ff_arr,
               marker="^", linestyle="-",  color="royalblue",
               label=f"FreeFEM H1     (rate={rate_h1_ff:.2f})")
    ax2.loglog(h_arr, err_sofa_arr,
               marker="x", linestyle="-",  color="tomato",
               label=f"SOFA L2        (rate={rate_sofa:.2f})")

    # refs 
    ref2 = err_l2_ff_arr[0] * (h_arr / h_arr[0]) ** 2
    ref1 = err_h1_ff_arr[0] * (h_arr / h_arr[0]) ** 1
    ax2.loglog(h_arr, ref2, linestyle="--", color="gray",
               alpha=0.6, label="O(h²) ref")
    ax2.loglog(h_arr, ref1, linestyle="--", color="lightblue",
               alpha=0.8, label="O(h¹) ref")

    ax2.set_xlabel("h  (taille d'élément)")
    ax2.set_ylabel("Erreur")
    ax2.set_title("Convergence — Solution quadratique\n"
                  "u(x) = x(L-x)/L^2  —  f = 2E/L^2 \n")
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("results_quad/plot_convergence_quad.png", dpi=150)
    plt.close(fig2)
    

    
    summary_path = "results_quad/comparison_results_quad.txt"
    with open(summary_path, "w") as f:
        f.write("  MMS Solution QUADRATIQUE — Resume de convergence\n")
        f.write(f"  u_exact(x) = x*(L-x)/L^2   L={L}  E={E}\n")
        f.write(f"  f(x) = 2E/L^2 = {2*E/L**2:.2f}  const \n")
        f.write(f"{'nx':>6}  {'h':>9}  "
                f"{'Err_nodal':>13}  {'Err_L2_FF':>13}  "
                f"{'Err_H1_FF':>13}  {'Err_SOFA':>13}\n")
        for nx, h, en, el, eh, es in zip(
                nx_list, h_list,
                err_nodal_list, err_l2_ff_list,
                err_h1_ff_list, err_sofa_list):
            f.write(f"{nx:>6}  {h:>9.5f}  "
                    f"{en:>13.4e}  {el:>13.4e}  "
                    f"{eh:>13.4e}  {es:>13.4e}\n")
        f.write("\n")
        f.write(f"Taux FreeFEM nodal  : {rate_nodal:+.3f}  "
                f"(attendu ~~~~~ 2, PAS de superconvergence)\n")
        f.write(f"Taux FreeFEM L2     : {rate_l2_ff:+.3f}  (attendu ~ 2)\n")
        f.write(f"Taux FreeFEM H1     : {rate_h1_ff:+.3f}  (attendu ~ 1)\n")
        f.write(f"Taux SOFA L2        : {rate_sofa:+.3f}  (attendu ~ 2)\n")
